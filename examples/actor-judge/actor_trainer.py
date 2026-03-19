"""GRPO Actor trainer for Actor-Judge Phase II.

Key design points:

P1  log_prob_old is computed HERE (not fetched from vLLM).
    vLLM only returns text.  Before the first mini-batch update of each step,
    we run a torch.no_grad() forward of the *current* Actor (weights unchanged
    at this point) to get perfectly aligned log_prob_old.

M2  action_mask is THREE-PART: [Prompt=0 | Strategy=1 | Padding=0]
    - PAD tokens (right-padded) are excluded via (input_ids != pad_token_id)
    - Prompt tokens are then forced to 0
    This ensures log_prob and KL only integrate over the generated Strategy tokens.

M3  Zero-variance short-circuit:
    When all K strategies for a question receive identical rewards (std < 1e-4),
    the normalised advantage would be numerically explosive.  We mask it to 0
    (skip the gradient update for that group) instead.

M4  Buffer access is FORBIDDEN here (On-Policy boundary).
    actor_trainer ONLY receives `experiences` from the current rollout.
    buffer.sample() is exclusively called by judge_trainer.

P5  ref_model must be distributed (accelerator.prepare) before being passed in.
    If each rank holds its own full copy, 8×8GB = 64GB of redundant VRAM is wasted.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import PreTrainedTokenizer

from buffer import Experience

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenisation helper
# ---------------------------------------------------------------------------

def tokenise_strategy_batch(
    experiences: List[Experience],
    tokenizer: PreTrainedTokenizer,
    model_max_length: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Tokenise (prompt + strategy) pairs and build three-part action_mask.

    Returns:
        input_ids:      [B, seq_len]
        attention_mask: [B, seq_len]   (0 = PAD)
        action_mask:    [B, seq_len]   (1 = strategy token, 0 = prompt / PAD)
        prompt_lengths: list of prompt token counts (one per sample in batch)
    """
    # Separate encode: get prompt length for each sample
    full_ids_list:        List[List[int]] = []
    prompt_lengths_calc:  List[int]       = []   # correct p_len per sample

    for exp in experiences:
        prompt_text = exp.context_text + "\n\nQuestion: " + exp.question
        full_text   = prompt_text + "\n" + exp.strategy

        p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        f_ids = tokenizer.encode(full_text,   add_special_tokens=True)

        # Truncate from the left if over limit (keep the strategy intact).
        # IMPORTANT: track the *adjusted* prompt length so action_mask is correct
        # even when the left-truncation eats into the prompt region.
        if len(f_ids) > model_max_length:
            overflow = len(f_ids) - model_max_length
            f_ids = f_ids[overflow:]
            p_len = max(0, len(p_ids) - overflow)
        else:
            p_len = len(p_ids)

        full_ids_list.append(f_ids)
        prompt_lengths_calc.append(p_len)   # store now; p_ids is NOT stored

    # Pad to same length (right padding)
    max_len = max(len(ids) for ids in full_ids_list)

    batch_input_ids   = torch.zeros(len(experiences), max_len, dtype=torch.long)
    batch_attn_mask   = torch.zeros(len(experiences), max_len, dtype=torch.long)
    batch_action_mask = torch.zeros(len(experiences), max_len, dtype=torch.long)
    prompt_lengths: List[int] = []

    for i, (f_ids, p_len) in enumerate(zip(full_ids_list, prompt_lengths_calc)):
        seq_len = len(f_ids)

        batch_input_ids[i, :seq_len] = torch.tensor(f_ids, dtype=torch.long)
        batch_attn_mask[i, :seq_len] = 1

        # M2: Three-part action_mask [Prompt=0 | Strategy=1 | Padding=0]
        # p_len already accounts for any left-truncation applied above.
        batch_action_mask[i, p_len:seq_len] = 1

        prompt_lengths.append(p_len)

    return batch_input_ids, batch_attn_mask, batch_action_mask, prompt_lengths


# ---------------------------------------------------------------------------
# Core log-prob computation
# ---------------------------------------------------------------------------

def compute_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Sum of per-token log-probs over the Strategy segment only.

    M2: action_mask must be three-part [Prompt=0 | Strategy=1 | Padding=0]
        so that PAD tokens do not pollute the KL or ratio.

    Args:
        input_ids:      [B, seq_len]
        attention_mask: [B, seq_len]
        action_mask:    [B, seq_len]  — three-part mask

    Returns:
        log_probs: [B]  — scalar per sample
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # [B, seq_len, vocab_size]

    # Shift for next-token prediction
    logits      = logits[:, :-1, :]       # [B, seq-1, vocab]
    labels      = input_ids[:, 1:]        # [B, seq-1]
    action_mask = action_mask[:, 1:]      # [B, seq-1]  (shifted with labels)

    per_token_lp = torch.gather(
        F.log_softmax(logits, dim=-1),
        dim=2,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)                          # [B, seq-1]

    # Sum only over Strategy tokens
    return (per_token_lp * action_mask.float()).sum(dim=-1)   # [B]


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_rewards(
    outcomes: torch.Tensor,       # [B*K]  int  {-1, 0, 1}
    v_judge: torch.Tensor,        # [B*K]  float  Judge sigmoid score
    len_penalties: torch.Tensor,  # [B*K]  float  ≤ 0
    log_prob_actor: torch.Tensor, # [B*K]  with grad
    log_prob_ref: torch.Tensor,   # [B*K]  no grad
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Compute GRPO reward per trajectory.

    r = (1-α)·y + α·σ(v_judge) + len_penalty − β·KL

    where KL ≈ log π_θ − log π_ref  (first-order KL approximation).
    """
    # Clamp outcome to [0, 1] for reward (y=-1 treated as 0)
    y = outcomes.float().clamp(min=0.0)
    kl = log_prob_actor - log_prob_ref   # [B*K], still has grad via log_prob_actor
    reward = (1 - alpha) * y + alpha * v_judge + len_penalties - beta * kl
    return reward   # [B*K]


# ---------------------------------------------------------------------------
# ActorTrainer
# ---------------------------------------------------------------------------

class ActorTrainer:
    def __init__(
        self,
        actor_model: torch.nn.Module,
        ref_model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        accelerator,
        cfg,
    ) -> None:
        self.actor   = actor_model
        self.ref     = ref_model
        self.tok     = tokenizer
        self.accel   = accelerator
        self.cfg     = cfg

        self.optimizer = AdamW(
            [p for p in self.actor.parameters() if p.requires_grad],
            lr=cfg.actor_lr,
        )

    # ------------------------------------------------------------------

    def train_step(
        self,
        experiences: List[Experience],
        judge_model: torch.nn.Module,
        global_step: int = 0,
    ) -> float:
        """One GRPO update step.

        Args:
            experiences: Current-rollout experiences (ON-POLICY ONLY — never
                         pass buffer.sample() output here).
            judge_model: Used to compute dense reward (Judge signal).
            global_step: For logging.

        Returns:
            Scalar loss value (Python float).
        """
        B = self.cfg.train_batch_size
        K = self.cfg.K
        device = self.accel.device

        # Filter out y=-1 (format error) experiences before any tensor ops
        valid_exps = [e for e in experiences if e.outcome >= 0]
        if len(valid_exps) < 2:
            logger.warning("train_step: too few valid experiences (%d), skipping.", len(valid_exps))
            return 0.0

        # Align B*K shape: we may have fewer than B*K valid after filtering
        # Pad/truncate to a multiple of K for clean reshape
        n_groups = max(1, len(valid_exps) // K)
        valid_exps = valid_exps[: n_groups * K]
        BK = len(valid_exps)

        # ── Tokenise ─────────────────────────────────────────────────────
        input_ids, attn_mask, action_mask, _ = tokenise_strategy_batch(
            valid_exps, self.tok
        )
        input_ids   = input_ids.to(device)
        attn_mask   = attn_mask.to(device)
        action_mask = action_mask.to(device)

        # ── P1: compute log_prob_old BEFORE any gradient update ──────────
        # Actor weights are unchanged at this point; use the same tokenizer
        # → perfectly aligned with the training forward pass.
        self.actor.eval()
        with torch.no_grad():
            log_prob_old = compute_log_probs(
                self.accel.unwrap_model(self.actor),
                input_ids, attn_mask, action_mask,
            )   # [BK], no grad, float
        log_prob_old = log_prob_old.detach()
        self.actor.train()

        # ── ref_model forward (KL baseline, also no grad) ────────────────
        # P5: ref_model must be accelerator.prepare()'d before being passed in.
        with torch.no_grad():
            log_prob_ref = compute_log_probs(
                self.accel.unwrap_model(self.ref),
                input_ids, attn_mask, action_mask,
            )   # [BK]
        log_prob_ref = log_prob_ref.detach()

        # ── Judge dense reward ────────────────────────────────────────────
        alpha = self.cfg.dense_reward_alpha
        v_judge = torch.zeros(BK, device=device)
        if alpha > 0.0:
            v_judge = self._get_judge_scores(valid_exps, judge_model, device)

        # ── Outcomes & length penalties ───────────────────────────────────
        outcomes = torch.tensor(
            [e.outcome for e in valid_exps], dtype=torch.float, device=device
        )
        len_penalties = torch.tensor(
            [
                -(self.cfg.length_penalty_coeff
                  * max(0, len(e.strategy.split()) - self.cfg.length_penalty_threshold))
                for e in valid_exps
            ],
            dtype=torch.float,
            device=device,
        )

        # ── GRPO loop (mini-batches) ──────────────────────────────────────
        total_loss = 0.0
        n_mini = max(1, BK // max(1, BK))   # single pass for now; split if OOM

        self.optimizer.zero_grad()

        # Actor forward (with grad)
        log_prob_actor = compute_log_probs(
            self.actor, input_ids, attn_mask, action_mask
        )   # [BK], has grad

        # Reward (note: KL computed inside for grad flow)
        reward_raw = compute_rewards(
            outcomes.long(), v_judge, len_penalties,
            log_prob_actor, log_prob_ref,
            alpha=alpha, beta=self.cfg.kl_penalty_beta,
        )   # [BK]

        # ── Z-Score normalisation within each group of K ──────────────────
        # M3: short-circuit when std < 1e-4 (all-same rewards → no info)
        reward_bk = reward_raw.view(n_groups, K)   # [B_eff, K]
        mean_k    = reward_bk.mean(dim=1, keepdim=True)
        std_k     = reward_bk.std(dim=1, keepdim=True)

        mask_valid_std = (std_k > 1e-4).float()
        advantage = ((reward_bk - mean_k) / (std_k + 1e-8)) * mask_valid_std
        advantage_flat = advantage.view(BK)   # [BK]

        # ── GRPO Clip ─────────────────────────────────────────────────────
        ratio         = torch.exp(log_prob_actor - log_prob_old)
        ratio_clipped = ratio.clamp(
            1 - self.cfg.grpo_epsilon, 1 + self.cfg.grpo_epsilon
        )
        loss = -torch.min(
            ratio * advantage_flat,
            ratio_clipped * advantage_flat,
        ).mean()

        self.accel.backward(loss)
        self.optimizer.step()
        total_loss += loss.item()

        logger.info(
            "ActorTrainer step %d: loss=%.4f BK=%d n_groups=%d",
            global_step, total_loss, BK, n_groups,
        )
        return total_loss

    # ------------------------------------------------------------------

    def _get_judge_scores(
        self,
        experiences: List[Experience],
        judge_model: torch.nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """Run Judge model in no_grad mode to obtain σ(logit) scores."""
        from prompts import build_judge_prompt, JUDGE_TOKEN

        texts = [
            build_judge_prompt(
                fewshot_examples=[],
                question=e.question,
                strategy=e.strategy,
                context_text_raw=e.context_text,   # pass pre-formatted context
            )
            for e in experiences
        ]

        enc = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        judge_model.eval()
        with torch.no_grad():
            logits = judge_model(**enc)   # [BK]
        judge_model.train()

        return torch.sigmoid(logits)     # [BK] in (0, 1)
