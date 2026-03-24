"""ODVA Judge trainer — Bradley-Terry pairwise loss with UCB-prioritised sampling.

Design:
  - Samples (win, lose) pairs from the UCB Buffer (OFF-POLICY, historical data).
  - Computes Bradley-Terry loss: L = -log σ(logit_win - logit_lose)
  - Adds L2 logit regularisation to prevent gradient saturation (logit drift).
  - Writes back Judge's σ(logit) predictions to the buffer (lazy update, O(1)).

buffer.sample() is ONLY called here, never in actor_trainer (On-Policy boundary).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate.utils import broadcast_object_list
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import PreTrainedTokenizer

from buffer import UCBBuffer
from judge_encode import encode_batch_for_judge
from prompts import build_judge_prompt_body

logger = logging.getLogger(__name__)


class JudgeTrainer:
    def __init__(
        self,
        judge_model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        accelerator,
        cfg,
    ) -> None:
        self.judge    = judge_model
        self.tok      = tokenizer
        self.accel    = accelerator
        self.cfg      = cfg
        self.optimizer = AdamW(
            [p for p in self.judge.parameters() if p.requires_grad],
            lr=cfg.judge_lr,
        )
        # L4: cosine LR scheduler
        total_steps = getattr(cfg, 'total_train_steps', 10_000)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=cfg.judge_lr * 0.1
        )
        # L3: gradient clipping threshold
        self.max_grad_norm: float = getattr(cfg, 'max_grad_norm', 1.0)

    # ------------------------------------------------------------------

    def train_step(
        self,
        buffer: UCBBuffer,
        global_step: int = 0,
    ) -> Tuple[float, Optional[Dict[str, float]]]:
        """One ODVA update step.

        Returns:
            (scalar loss, optional WandB metrics dict), or (0.0, None) if skipped.
        """
        device = self.accel.device

        # ── Sample pairwise data from buffer (OFF-POLICY) ─────────────────
        # D1: sample ONLY on rank 0, then broadcast.  Independent RNG per rank
        # yields different pairs → different token shapes → FSDP collective/OOM
        # failures (e.g. exit code 1 on one rank mid-train with no traceback on 0).
        if self.accel.num_processes > 1:
            if self.accel.is_main_process:
                pairs = buffer.sample_pairwise(self.cfg.judge_batch_size)
            else:
                pairs = None
            pair_container = [pairs]
            broadcast_object_list(pair_container, from_process=0)
            pairs = pair_container[0]
            tsync = [buffer.global_sample_steps if self.accel.is_main_process else 0]
            broadcast_object_list(tsync, from_process=0)
            buffer.global_sample_steps = tsync[0]
        else:
            pairs = buffer.sample_pairwise(self.cfg.judge_batch_size)
        if not pairs:
            logger.debug("JudgeTrainer step %d: buffer too sparse, skipping.", global_step)
            return 0.0, None

        # ── Tokenise win/lose: truncate body only, then append <|judge|> ids ─
        jmax = int(getattr(self.cfg, "judge_max_length", 8192))
        win_bodies = [
            build_judge_prompt_body(
                [], p.exp_win.question, p.exp_win.strategy,
                context_text_raw=p.exp_win.context_text,
            )
            for p in pairs
        ]
        lose_bodies = [
            build_judge_prompt_body(
                [], p.exp_lose.question, p.exp_lose.strategy,
                context_text_raw=p.exp_lose.context_text,
            )
            for p in pairs
        ]

        inputs_win = {
            k: v.to(device) for k, v in encode_batch_for_judge(self.tok, win_bodies, jmax).items()
        }
        inputs_lose = {
            k: v.to(device) for k, v in encode_batch_for_judge(self.tok, lose_bodies, jmax).items()
        }

        # ── Forward pass ──────────────────────────────────────────────────
        self.optimizer.zero_grad()
        self.judge.train()

        logit_win  = self.judge(**inputs_win)    # [batch]
        logit_lose = self.judge(**inputs_lose)   # [batch]

        # ── Bradley-Terry loss ────────────────────────────────────────────
        bt_loss = -F.logsigmoid(logit_win - logit_lose).mean()

        # L2 logit regularisation — prevents logits drifting to ±∞ (saturation)
        # Without this, BT loss only constrains the *difference*, not the scale.
        # Logits at ±100 make sigmoid gradients vanish → Judge stops learning.
        l2_penalty = 0.001 * (logit_win ** 2 + logit_lose ** 2).mean()

        loss = bt_loss + l2_penalty

        self.accel.backward(loss)
        # L3: gradient clipping (return value = total norm before clipping)
        judge_grad_norm = self.accel.clip_grad_norm_(self.judge.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        # ── Write back Judge scores to buffer (O(1) via traj_id) ──────────
        with torch.no_grad():
            scores_win  = torch.sigmoid(logit_win).cpu().tolist()
            scores_lose = torch.sigmoid(logit_lose).cpu().tolist()

        for i, pair in enumerate(pairs):
            buffer.update_v_pred(pair.q_hash, pair.traj_id_win,  scores_win[i])
            buffer.update_v_pred(pair.q_hash, pair.traj_id_lose, scores_lose[i])

        margin = (torch.sigmoid(logit_win) - torch.sigmoid(logit_lose)).mean().item()
        judge_wb: Dict[str, float] = {
            "train/judge_bt_loss": float(bt_loss.item()),
            "train/judge_l2_penalty": float(l2_penalty.item()),
            "train/judge_score_margin": float(margin),
        }
        if judge_grad_norm is not None:
            gn = judge_grad_norm.detach() if hasattr(judge_grad_norm, "detach") else judge_grad_norm
            judge_wb["train/judge_grad_norm"] = float(gn.item() if hasattr(gn, "item") else float(gn))

        logger.info(
            "JudgeTrainer step %d: bt_loss=%.4f l2=%.4f total=%.4f n_pairs=%d",
            global_step, bt_loss.item(), l2_penalty.item(), loss.item(), len(pairs),
        )
        return loss.item(), judge_wb

    # ------------------------------------------------------------------

    def warmup_step(
        self,
        s_gold_texts: List[str],
        s_neg_texts: List[str],
        questions: List[str],
        global_step: int = 0,
        *,
        advance_scheduler: bool = True,
        context_texts: Optional[List[str]] = None,
    ) -> float:
        """Pre-train Judge on (S_gold, S_neg) pairs before Phase II starts.

        s_neg_texts should be perturbed / shuffled versions of s_gold_texts.

        context_texts: optional list of few-shot Q&A context strings, one per
        sample (same format as Experience.context_text built in rollout_engine).
        Passing these aligns the warmup distribution with Phase II, where Judge
        always receives context_text_raw from the rollout experience.  When None
        or all-empty, the prompt reduces to (Q, S) only — acceptable but causes
        a minor distribution shift vs Phase II.

        During Phase II–only warmup, pass ``advance_scheduler=False`` so the
        cosine schedule (sized for main training) is not consumed by warmup steps.
        """
        assert len(s_gold_texts) == len(s_neg_texts) == len(questions)
        device = self.accel.device

        ctxs = context_texts if context_texts is not None else [""] * len(questions)
        jmax = int(getattr(self.cfg, "judge_max_length", 8192))
        win_bodies = [
            build_judge_prompt_body([], q, s, context_text_raw=ctx)
            for q, s, ctx in zip(questions, s_gold_texts, ctxs)
        ]
        lose_bodies = [
            build_judge_prompt_body([], q, s, context_text_raw=ctx)
            for q, s, ctx in zip(questions, s_neg_texts, ctxs)
        ]

        inputs_win = {
            k: v.to(device) for k, v in encode_batch_for_judge(self.tok, win_bodies, jmax).items()
        }
        inputs_lose = {
            k: v.to(device) for k, v in encode_batch_for_judge(self.tok, lose_bodies, jmax).items()
        }

        self.optimizer.zero_grad()
        self.judge.train()

        logit_win  = self.judge(**inputs_win)
        logit_lose = self.judge(**inputs_lose)

        bt_loss    = -F.logsigmoid(logit_win - logit_lose).mean()
        l2_penalty = 0.001 * (logit_win ** 2 + logit_lose ** 2).mean()
        loss = bt_loss + l2_penalty

        self.accel.backward(loss)
        self.accel.clip_grad_norm_(self.judge.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if advance_scheduler:
            self.scheduler.step()

        logger.info("JudgeTrainer warmup step %d: loss=%.4f", global_step, loss.item())
        return loss.item()

    def set_optimizer_lr(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def rebuild_optimizer_for_phase2(self) -> None:
        """Fresh AdamW + cosine schedule for Phase II (do not reuse warmup optimizer state)."""
        self.optimizer = AdamW(
            [p for p in self.judge.parameters() if p.requires_grad],
            lr=self.cfg.judge_lr,
        )
        total_steps = getattr(self.cfg, "total_train_steps", 10_000)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=self.cfg.judge_lr * 0.1
        )
