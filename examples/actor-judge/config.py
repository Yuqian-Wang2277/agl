"""ActorJudgeConfig — all hyperparameters, paths, and ablation switches.

Design choices:
- Single dataclass for the whole experiment so every script imports one object.
- Ablation switches are plain booleans/floats; downstream code checks them at
  runtime rather than branching at import time.
- Length-Hacking defence config is included but disabled by default (coeff=0.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ActorJudgeConfig:
    # ── Model paths ───────────────────────────────────────────────────────────
    actor_model_path: str = "/home/test/test16/chenlu/model/Qwen3-4B"
    # Path to the HF-format checkpoint converted from the VERL FSDP shards.
    # Leave empty to start from the base model (slightly lower quality but safe).
    actor_sft_checkpoint: str = ""

    # ── Data paths ────────────────────────────────────────────────────────────
    data_base_path: str = "/home/test/test16/chenlu/projects/LLMReflection/data/"
    train_subdir: str = "train_20k"
    val_subdirs: List[str] = field(
        default_factory=lambda: ["test-id-subtask", "test-ood-task", "test-bbh"]
    )
    # Gold strategies from few-shot repo (used only for Judge warmup)
    strategy_dir: str = "/home/test/test16/chenlu/projects/fs/strategy"

    # ── Rollout hyperparameters ───────────────────────────────────────────────
    K: int = 8                         # strategy samples per question
    fewshot_min: int = 3
    fewshot_max: int = 5
    rollout_temperature: float = 0.7   # stage-1 strategy generation
    answer_temperature: float = 0.0    # stage-2 answer generation (greedy)
    strategy_max_tokens: int = 16384
    answer_max_tokens: int = 8192
    cross_domain_ratio: float = 0.0    # fraction of cross-domain batches

    # vLLM engine settings (P3: ≤ 0.75 to leave room for PyTorch CUDA Context)
    # PyTorch CUDA Context pins ~1-1.5 GB/GPU that empty_cache() cannot free;
    # 0.80 will OOM the moment vLLM starts on an 80 GB A100.
    gpu_memory_utilization: float = 0.75
    tensor_parallel_size: int = 8

    # ── Buffer ────────────────────────────────────────────────────────────────
    buffer_max_size: int = 50_000
    # P4 of Round-4: per-Q cap — prevents hard questions from flooding the buffer
    per_q_max: int = 50
    min_buffer_size: int = 100         # minimum entries before Judge training starts

    # ── UCB coefficients ──────────────────────────────────────────────────────
    lambda_err: float = 1.0            # exploitation term weight
    lambda_exp: float = 1.0            # exploration term weight

    # ── Phase II RL hyperparameters ───────────────────────────────────────────
    alpha: float = 0.3                 # dense reward weight (Judge signal)
    grpo_epsilon: float = 0.2          # PPO clip ε

    # ── Training schedule ─────────────────────────────────────────────────────
    total_epochs: int = 5
    num_train_samples: int = 20_000    # L2: configurable dataset size (was hard-coded)
    train_batch_size: int = 8          # B — questions per rollout batch
    actor_lr: float = 1e-6
    judge_lr: float = 1e-5
    judge_batch_size: int = 16         # pairwise pairs per Judge update step
    # Judge warmup vs Phase II: distinguish ``resume`` (full run state) from
    # ``judge_init_checkpoint`` (weights only, fresh optimizer for co-evolution).
    judge_warmup: bool = True          # If False, forces judge_warmup_mode="cold"
    judge_warmup_mode: str = "always"  # "cold" | "always" | "reuse"
    # Path to judge_model.pt or a directory containing judge_model.pt (reuse mode)
    judge_init_checkpoint: str = ""
    # Where to write weights after a successful always-warmup (for later --reuse)
    judge_warmup_save_dir: str = ""    # empty → {checkpoint_dir}/judge_warmup_latest
    warmup_steps: int = 100            # max Judge BT pre-training steps (may early-stop)
    warmup_seed: int = 42              # reproducible warmup sampling / eval negatives
    # Warmup uses a smaller LR than Phase II to avoid destroying backbone alignment
    judge_warmup_lr: float = 0.0       # 0 → judge_lr * 0.1
    # Hold out held-out pairwise eval for accuracy = P(score_win > score_lose)
    judge_warmup_eval_ratio: float = 0.12
    judge_warmup_eval_every: int = 5   # run eval every N warmup steps
    judge_warmup_early_stop_min_acc: float = 0.75   # stop once eval acc >= this (good enough)
    # Pairwise mean(sigmoid(win) - sigmoid(lose)) on held-out triples; needs high acc *and* margin
    judge_warmup_early_stop_min_margin: float = 0.15
    judge_warmup_overfit_warn_acc: float = 0.95     # warn if eval acc >= this (possible hack)
    judge_warmup_reset_optimizer_after: bool = True  # fresh AdamW for Phase II after warmup
    val_freq: int = 1                  # validate every N epochs
    # L4: total_train_steps for LR scheduler — set automatically in main() if 0
    total_train_steps: int = 0         # 0 = auto-compute from epochs × steps_per_epoch

    # ── Optimisation ──────────────────────────────────────────────────────────
    max_grad_norm: float = 1.0         # L3: gradient clipping (0 = disabled)

    # ── Infrastructure ────────────────────────────────────────────────────────
    checkpoint_dir: str = "./checkpoints_actor_judge"
    save_freq: int = 1                 # save actor every N epochs
    n_gpus: int = 8
    # Shared memory dir for vLLM ↔ FSDP weight hand-off (avoids NVMe bottleneck)
    weight_sync_tmp_dir: str = "/dev/shm/actor_weight_tmp"
    # Resume: set to a checkpoint dir to continue training from that point
    resume_from_checkpoint: str = ""   # L5: e.g. "./checkpoints_actor_judge/epoch_002"

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_project: str = "ActorJudge"
    wandb_run_name: str = "phase2_co_evolution"

    # ── Ablation switches ─────────────────────────────────────────────────────
    # Experiment A: freeze Judge → tests whether co-evolution is necessary
    freeze_judge: bool = False
    # Experiment B: pure sparse reward.
    # IMPORTANT: setting this to 0 must also set freeze_judge=True for a fair
    # comparison (otherwise Judge trains but its output is never used).
    dense_reward_alpha: float = 0.3
    # Experiment D: no KL penalty → tests catastrophic forgetting
    kl_penalty_beta: float = 0.04
    # Experiment E: disable UCB replay → degrades to FIFO uniform sampling
    disable_ucb_replay: bool = False

    # ── Length-Hacking defence (disabled by default) ──────────────────────────
    # Set length_penalty_coeff > 0 (e.g. 0.01) once Average_Strategy_Length
    # rises for 3+ consecutive epochs while Pass@1 stagnates.
    length_penalty_coeff: float = 0.0
    length_penalty_threshold: int = 500   # token count above which penalty kicks in
    length_hack_window: int = 3           # consecutive epochs of length growth to trigger

    # ── Helper ────────────────────────────────────────────────────────────────
    @property
    def start_model_path(self) -> str:
        """Resolved path to the model used for Actor/Ref/Judge initialisation."""
        return self.actor_sft_checkpoint or self.actor_model_path

    def validate(self) -> None:
        """Raise ValueError for obviously wrong configurations."""
        mode = (self.judge_warmup_mode or "always").strip().lower()
        if mode not in ("cold", "always", "reuse"):
            raise ValueError(
                f"judge_warmup_mode must be 'cold', 'always', or 'reuse', got {self.judge_warmup_mode!r}"
            )
        if mode == "reuse" and not (self.judge_init_checkpoint or "").strip():
            raise ValueError(
                "judge_warmup_mode='reuse' requires judge_init_checkpoint pointing to "
                "judge_model.pt or a directory that contains it."
            )
        if self.dense_reward_alpha == 0.0 and not self.freeze_judge:
            raise ValueError(
                "Ablation B (dense_reward_alpha=0) must also set freeze_judge=True "
                "for a fair comparison. Otherwise the Judge trains but is never used."
            )
        if self.K < 2:
            raise ValueError("K must be >= 2 for in-group Z-Score to be meaningful.")
