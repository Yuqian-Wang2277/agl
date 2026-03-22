"""GRPO Actor trainer for Actor-Judge Phase II.

Key design points:

C1  prompt_text stored in Experience.
    tokenise_strategy_batch uses exp.prompt_text (the exact chat-template
    formatted prompt that vLLM used) instead of a simplified reconstruction.
    This ensures action_mask boundaries and log_prob computations are aligned
    with the actual generation context.

C2  FSDP-wrapped models used directly for no_grad forward passes.
    accelerator.unwrap_model() under FSDP ZeRO-3 returns a module with only
    the local parameter shard — calling forward() on it gives wrong results.
    Using the FSDP-wrapped model (self.actor, self.ref) ensures the all-gather
    is triggered and the full parameters are used.

P1  log_prob_old is computed HERE (not fetched from vLLM).
    vLLM only returns text.  Before the first mini-batch update of each step,
    we run a torch.no_grad() forward of the *current* Actor (weights unchanged
    at this point) to get perfectly aligned log_prob_old.

M2  action_mask is THREE-PART: [Prompt=0 | Strategy=1 | Padding=0]
    Prompt region is [0, p_len), Strategy region is [p_len, seq_len),
    Padding region is [seq_len, max_len).

S3  KL penalty is a separate loss term, NOT part of the advantage.
    reward_for_advantage is computed with detached KL so that the Z-score
    normalisation and PPO-clip operate on a clean advantage signal.

S4  log_prob uses per-token MEAN (not SUM).
    SUM penalises longer strategies more, biasing the policy toward short
    outputs independent of quality.  MEAN removes this length bias.

M3  Zero-variance short-circuit: when all K strategies for a question
    receive identical rewards (std < 1e-4), the advantage is set to 0
    (skipping gradient update for that group).

M4  Buffer access is FORBIDDEN here (On-Policy boundary).
    actor_trainer ONLY receives `experiences` from the current rollout.
    buffer.sample() is exclusively called by judge_trainer.

P5  ref_model must be distributed (accelerator.prepare) before being passed in.
    If each rank holds its own full copy, 8×8GB = 64GB of redundant VRAM is wasted.

L3  Gradient clipping via cfg.max_grad_norm (default 1.0).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import wandb
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import PreTrainedTokenizer

from buffer import Experience

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenisation helper
# ---------------------------------------------------------------------------

def tokenise_strategy_batch(
    experiences: List[Experience],
    tokenizer: PreTrainedTokenizer,
    model_max_length: int = 8192,
    *,
    accel=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Tokenise (prompt + strategy) pairs and build three-part action_mask.

    C1 fix: uses exp.prompt_text — the exact chat-template formatted prompt
    that vLLM used for Stage-1 generation — instead of a simplified
    "context_text + Question:" reconstruction.  This ensures the prompt token
    boundary is in the same position as during generation, so action_mask
    correctly isolates only the strategy tokens.

    S2 / add_special_tokens fix: both prompt and full text are encoded with
    add_special_tokens=False because apply_chat_template already inserts all
    special tokens as literal text (e.g. <|im_start|>).  Using True would
    prepend an extra BOS and shift all token indices by 1.

    Returns:
        input_ids:      [B, seq_len]
        attention_mask: [B, seq_len]   (0 = PAD)
        action_mask:    [B, seq_len]   (1 = strategy token, 0 = prompt / PAD)
        prompt_lengths: list of prompt token counts (one per sample in batch)
    """
    full_ids_list:       List[List[int]] = []
    prompt_lengths_calc: List[int]       = []   # correct p_len per sample

    for exp in experiences:
        # C1 fix: prompt_text is the exact chat-template string stored by rollout_engine.
        # full_text appends the generated strategy directly (no extra separator needed
        # since the chat template already ends with the assistant turn opening).
        prompt_text = exp.prompt_text
        full_text   = prompt_text + exp.strategy

        # S2 fix: add_special_tokens=False for both — chat template handles all specials.
        p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        f_ids = tokenizer.encode(full_text,   add_special_tokens=False)

        # Truncate from the left if over limit (keep the strategy intact).
        # IMPORTANT: track the *adjusted* prompt length so action_mask is correct
        # even when the left-truncation eats into the prompt region.
        if len(f_ids) > model_max_length:
            overflow = len(f_ids) - model_max_length
            # Left truncation drops the start of the prompt — few-shot lives there.
            # If this fires often, reduce fewshot_max or raise actor_max_length (config).
            msg = (
                f"WARNING: Actor left truncation occurred! Lost {overflow} tokens. "
                "Few-shot at the start of the prompt may be damaged."
            )
            logger.warning(msg)
            if accel is not None and accel.is_main_process:
                try:
                    wandb.log({"train/actor_left_truncation_tokens": overflow}, commit=False)
                except Exception:
                    pass
            f_ids = f_ids[overflow:]
            p_len = max(0, len(p_ids) - overflow)
        else:
            p_len = len(p_ids)

        full_ids_list.append(f_ids)
        prompt_lengths_calc.append(p_len)   # store correct p_len; p_ids NOT stored

    # Pad to same length (right padding)
    max_len = max(len(ids) for ids in full_ids_list)
    pad_id  = tokenizer.pad_token_id or 0

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
    """Mean per-token log-prob over the Strategy segment only.

    S4 fix: returns MEAN (not SUM) of per-token log-probs.
    SUM would systematically penalise longer strategies (larger negative
    values) and bias the policy ratio toward shorter outputs regardless of
    quality.  MEAN normalises for sequence length.

    M2: action_mask must be three-part [Prompt=0 | Strategy=1 | Padding=0]
        so that PAD tokens do not pollute the KL or ratio.

    Args:
        input_ids:      [B, seq_len]
        attention_mask: [B, seq_len]
        action_mask:    [B, seq_len]  — three-part mask

    Returns:
        log_probs: [B]  — mean per-strategy-token log-prob per sample
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

    # S4 fix: MEAN over strategy tokens (not SUM)
    n_strategy_tokens = action_mask.float().sum(dim=-1).clamp(min=1)
    return (per_token_lp * action_mask.float()).sum(dim=-1) / n_strategy_tokens  # [B]


# ---------------------------------------------------------------------------
# Reward computation (advantage part only — no KL here)
# ---------------------------------------------------------------------------

def compute_advantage_reward(
    outcomes: torch.Tensor,       # [B*K]  int  {-1, 0, 1}
    v_judge: torch.Tensor,        # [B*K]  float  Judge sigmoid score (detached)
    len_penalties: torch.Tensor,  # [B*K]  float  ≤ 0
    alpha: float,
) -> torch.Tensor:
    """Compute the reward used for advantage normalisation (no KL term).

    S3 fix: KL penalty is NOT included here so that Z-score normalisation
    operates on a clean signal and the advantage is free of gradient.
    The KL term is added as a separate loss in train_step.

    r_adv = (1-α)·y + α·σ(v_judge) + len_penalty
    """
    y = outcomes.float().clamp(min=0.0)   # y=-1 (format error) → 0
    return (1 - alpha) * y + alpha * v_judge + len_penalties   # [B*K]


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
        # L4: cosine annealing LR scheduler
        total_steps = getattr(cfg, 'total_train_steps', 10_000)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=cfg.actor_lr * 0.1,
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

        # ── Judge dense reward (computed over ALL experiences for correct Z-score) ──
        # _get_judge_scores returns [BK] scalars — no large logit tensor, no OOM.
        # All FSDP ranks must call this together (FSDP all-gather is collective).
        alpha   = self.cfg.dense_reward_alpha
        v_judge = torch.zeros(BK, device=device)
        if alpha > 0.0:
            v_judge = self._get_judge_scores(valid_exps, judge_model, device)

        # ── Outcomes & length penalties for ALL experiences ───────────────────────
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

        # ── Advantage (S3 fix: no KL in advantage, Z-score over full batch) ──────
        with torch.no_grad():
            reward_adv = compute_advantage_reward(
                outcomes.long(), v_judge.detach(), len_penalties, alpha=alpha
            )   # [BK], no grad

        # ── Z-Score normalisation within each group of K ──────────────────────────
        # M3: short-circuit when std < 1e-4 (all-same rewards → no info)
        reward_bk = reward_adv.view(n_groups, K)   # [n_groups, K]
        mean_k    = reward_bk.mean(dim=1, keepdim=True)
        std_k     = reward_bk.std(dim=1, keepdim=True)

        mask_valid_std   = (std_k > 1e-4).float()
        advantage        = ((reward_bk - mean_k) / (std_k + 1e-8)) * mask_valid_std
        advantage_flat_all = advantage.view(BK).detach()   # [BK], no grad

        # ── Rank-based data split ─────────────────────────────────────────────────
        # Root cause of OOM: without this, every FSDP rank processes all BK
        # experiences → [BK, seq, vocab=151936] logits in fp32 → ~78 GiB → OOM.
        #
        # Fix: split experiences by question-group (multiples of K) so that each
        # rank processes BK/world_size experiences.  This shrinks the logits tensor
        # by world_size (e.g. from [64,seq,vocab]→[8,seq,vocab]) and brings fp32
        # logit memory from ~78 GiB down to ~9.8 GiB per rank.
        #
        # Advantage must be computed over ALL BK first (Z-score needs all K per
        # group), then the per-rank slice is extracted.
        #
        # All FSDP ranks still participate in NCCL collectives (FSDP all-gather)
        # simultaneously — each rank just feeds its own local data slice.
        # FSDP's gradient all-reduce then correctly averages the per-rank gradients.
        world_size = self.accel.num_processes
        rank       = self.accel.process_index

        if n_groups >= world_size:
            # Normal case: assign contiguous groups to each rank.
            local_groups = n_groups // world_size
            r_start = rank * local_groups
            r_end   = (rank + 1) * local_groups if rank < world_size - 1 else n_groups
        else:
            # Fewer question-groups than ranks: round-robin assignment.
            # Multiple ranks process the same group; FSDP gradient averaging
            # still yields the correct global mean.
            r_start = rank % n_groups
            r_end   = r_start + 1

        local_exps     = valid_exps[r_start * K : r_end * K]
        advantage_flat = advantage_flat_all[r_start * K : r_end * K]

        # ── Tokenise LOCAL slice only ─────────────────────────────────────────────
        amax = int(getattr(self.cfg, "actor_max_length", 8192))
        input_ids, attn_mask, action_mask, _ = tokenise_strategy_batch(
            local_exps,
            self.tok,
            model_max_length=amax,
            accel=self.accel,
        )
        input_ids   = input_ids.to(device)
        attn_mask   = attn_mask.to(device)
        action_mask = action_mask.to(device)

        # ── P1: compute log_prob_old on LOCAL slice BEFORE any gradient update ────
        # C2 fix: use self.actor (FSDP-wrapped) directly — NOT unwrap_model().
        # Under FSDP ZeRO-3, unwrap_model() gives only the local parameter shard.
        self.actor.eval()
        with torch.no_grad():
            log_prob_old = compute_log_probs(
                self.actor,
                input_ids, attn_mask, action_mask,
            )   # [local_BK], no grad
        log_prob_old = log_prob_old.detach()
        self.actor.train()

        # ── ref_model forward on LOCAL slice (KL baseline, also no grad) ─────────
        # C2 fix: use self.ref directly (FSDP-wrapped), not unwrap_model().
        with torch.no_grad():
            log_prob_ref = compute_log_probs(
                self.ref,
                input_ids, attn_mask, action_mask,
            )   # [local_BK]
        log_prob_ref = log_prob_ref.detach()

        # ── Actor forward (with grad) on LOCAL slice ──────────────────────────────
        self.optimizer.zero_grad()

        log_prob_actor = compute_log_probs(
            self.actor, input_ids, attn_mask, action_mask
        )   # [local_BK], has grad

        # ── GRPO Clip loss ────────────────────────────────────────────────────────
        ratio         = torch.exp(log_prob_actor - log_prob_old)
        ratio_clipped = ratio.clamp(
            1 - self.cfg.grpo_epsilon, 1 + self.cfg.grpo_epsilon
        )
        clip_loss = -torch.min(
            ratio * advantage_flat,
            ratio_clipped * advantage_flat,
        ).mean()

        # ── S3 fix: KL penalty as a separate loss term ───────────────────────────
        # KL = log π_θ − log π_ref.  This term has gradient through log_prob_actor.
        # Adding it directly to clip_loss (not to advantage) ensures:
        #   (a) The advantage/Z-score normalisation is unaffected by KL.
        #   (b) Gradient flows correctly through a single path.
        kl_per_sample = log_prob_actor - log_prob_ref   # [local_BK], has grad
        kl_loss       = self.cfg.kl_penalty_beta * kl_per_sample.mean()

        total_loss = clip_loss + kl_loss

        self.accel.backward(total_loss)

        # L3: gradient clipping
        self.accel.clip_grad_norm_(
            self.actor.parameters(),
            getattr(self.cfg, "max_grad_norm", 1.0),
        )

        self.optimizer.step()
        self.scheduler.step()

        logger.info(
            "ActorTrainer step %d: clip_loss=%.4f kl_loss=%.4f BK=%d n_groups=%d local_BK=%d",
            global_step, clip_loss.item(), kl_loss.item(), BK, n_groups, len(local_exps),
        )
        return total_loss.item()

    # ------------------------------------------------------------------

    def _get_judge_scores(
        self,
        experiences: List[Experience],
        judge_model: torch.nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """Run Judge model in no_grad mode to obtain σ(logit) scores."""
        from judge_encode import encode_batch_for_judge
        from prompts import build_judge_prompt_body

        bodies = [
            build_judge_prompt_body(
                [],
                e.question,
                e.strategy,
                context_text_raw=e.context_text,
            )
            for e in experiences
        ]
        jmax = int(getattr(self.cfg, "judge_max_length", 8192))
        enc = {
            k: v.to(device)
            for k, v in encode_batch_for_judge(self.tok, bodies, jmax).items()
        }

        judge_model.eval()
        with torch.no_grad():
            logits = judge_model(**enc)   # [BK], using FSDP-wrapped judge
        judge_model.train()

        return torch.sigmoid(logits)     # [BK] in (0, 1)
