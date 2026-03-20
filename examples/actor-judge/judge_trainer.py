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
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import PreTrainedTokenizer

from buffer import UCBBuffer
from prompts import build_judge_prompt, JUDGE_TOKEN

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
    ) -> float:
        """One ODVA update step.

        Returns:
            Scalar loss value (Python float), or 0.0 if buffer is too sparse.
        """
        device = self.accel.device

        # ── Sample pairwise data from buffer (OFF-POLICY) ─────────────────
        pairs = buffer.sample_pairwise(self.cfg.judge_batch_size)
        if not pairs:
            logger.debug("JudgeTrainer step %d: buffer too sparse, skipping.", global_step)
            return 0.0

        # ── Tokenise win and lose inputs ──────────────────────────────────
        win_texts  = [
            build_judge_prompt(
                [], p.exp_win.question,  p.exp_win.strategy,
                context_text_raw=p.exp_win.context_text,
            )
            for p in pairs
        ]
        lose_texts = [
            build_judge_prompt(
                [], p.exp_lose.question, p.exp_lose.strategy,
                context_text_raw=p.exp_lose.context_text,
            )
            for p in pairs
        ]

        inputs_win  = self.tok(
            win_texts,  return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(device)
        inputs_lose = self.tok(
            lose_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(device)

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
        # L3: gradient clipping
        self.accel.clip_grad_norm_(self.judge.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        # ── Write back Judge scores to buffer (O(1) via traj_id) ──────────
        with torch.no_grad():
            scores_win  = torch.sigmoid(logit_win).cpu().tolist()
            scores_lose = torch.sigmoid(logit_lose).cpu().tolist()

        for i, pair in enumerate(pairs):
            buffer.update_v_pred(pair.q_hash, pair.traj_id_win,  scores_win[i])
            buffer.update_v_pred(pair.q_hash, pair.traj_id_lose, scores_lose[i])

        logger.info(
            "JudgeTrainer step %d: bt_loss=%.4f l2=%.4f total=%.4f n_pairs=%d",
            global_step, bt_loss.item(), l2_penalty.item(), loss.item(), len(pairs),
        )
        return loss.item()

    # ------------------------------------------------------------------

    def warmup_step(
        self,
        s_gold_texts: List[str],
        s_neg_texts: List[str],
        questions: List[str],
        global_step: int = 0,
        *,
        advance_scheduler: bool = True,
    ) -> float:
        """Pre-train Judge on (S_gold, S_neg) pairs before Phase II starts.

        s_neg_texts should be perturbed / shuffled versions of s_gold_texts.

        During Phase II–only warmup, pass ``advance_scheduler=False`` so the
        cosine schedule (sized for main training) is not consumed by warmup steps.
        """
        assert len(s_gold_texts) == len(s_neg_texts) == len(questions)
        device = self.accel.device

        # Warmup uses (question, strategy) pairs only — no few-shot context available
        win_texts  = [build_judge_prompt([], q, s) for q, s in zip(questions, s_gold_texts)]
        lose_texts = [build_judge_prompt([], q, s) for q, s in zip(questions, s_neg_texts)]
        # context_text_raw left empty intentionally: gold strategies don't have
        # a corresponding few-shot context stored; judge still sees Q+S.

        inputs_win  = self.tok(win_texts,  return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        inputs_lose = self.tok(lose_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

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
