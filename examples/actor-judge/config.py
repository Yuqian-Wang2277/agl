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
    # Drop samples whose Stage-1 chat prompt (few-shot + template) exceeds this many
    # tokens (0 = disabled). Prevents rare ultra-long tails from dominating left-truncation.
    max_stage1_prompt_tokens: int = 4000
    # When no tokenizer is passed to the dataset, estimate tokens as len(text)/chars_per_token.
    stage1_length_chars_per_token: float = 2.5
    # Append one JSON line per rejected sample (path auto-set under checkpoint_dir when empty).
    dataset_stage1_reject_log: bool = True
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
    # Rollout caps: 8192/8192 with B×K concurrent seqs risks VRAM OOM (FSDP + vLLM share GPUs).
    # 4096/4096 is a safer default; raise only after confirming headroom.
    strategy_max_tokens: int = 4096
    answer_max_tokens: int = 4096
    cross_domain_ratio: float = 0.0    # fraction of cross-domain batches
    # GRPO log_prob: full Stage-1 chat prompt (includes few-shot) + strategy S.
    actor_max_length: int = 8192
    # Split each rank's local GRPO batch into micro-batches (gradient accumulation).
    # Reduces peak activation memory during compute_log_probs; 0 = one micro-batch (full local slice).
    actor_microbatch_size: int = 4
    # Judge: body (Context+Q+S) truncated then <|judge|> appended (see judge_encode).
    judge_max_length: int = 8192

    # vLLM engine settings (P3: headroom for FSDP shards on the SAME physical GPUs)
    # PyTorch CUDA Context + FSDP (~1–2 GiB/GPU) + vLLM KV must fit in 80 GiB.
    # Lower default leaves more KV headroom while FSDP weights stay resident.
    gpu_memory_utilization: float = 0.60
    tensor_parallel_size: int = 8
    # Chunked-prefill cap (vLLM default is often 16384; lower = lower peak VRAM during long prompts).
    vllm_max_num_batched_tokens: int | None = 4096
    # Split rollout generate() into chunks of at most N prompts (0 = one call, legacy).
    # Strongly recommended when B×K is large or max_tokens is high.
    vllm_rollout_prompt_chunk_size: int = 32
    # Disable CUDA Graph + torch.compile in vLLM (enforce_eager=True).
    # Default True: avoids 30-90 min silent JIT compilation on first run.
    # Set False only after verifying the compiled cache is warm (production).
    vllm_enforce_eager: bool = True

    # ── Buffer ────────────────────────────────────────────────────────────────
    buffer_max_size: int = 50_000
    # P4 of Round-4: per-Q cap — prevents hard questions from flooding the buffer
    per_q_max: int = 50
    # Dry-run: set low (e.g. 8) when num_train_samples * K < default, or Judge ODVA never runs.
    min_buffer_size: int = 100         # minimum entries before Judge training starts

    # ── UCB coefficients ──────────────────────────────────────────────────────
    lambda_err: float = 1.0            # exploitation term weight
    lambda_exp: float = 1.0            # exploration term weight

    # ── Phase II RL hyperparameters ───────────────────────────────────────────
    alpha: float = 0.3                 # dense reward weight (Judge signal)
    grpo_epsilon: float = 0.2          # PPO clip ε

    # ── Training schedule ─────────────────────────────────────────────────────
    total_epochs: int = 50
    num_train_samples: int = 20_000    # L2: configurable dataset size (was hard-coded)
    train_batch_size: int = 8          # B — questions per rollout batch
    # Max Phase-A rollout batches per epoch (= Phase-B optimizer steps that epoch).
    # 0 = use full len(train_loader). E.g. 250 with 20k/8≈2500 batches → ~10% data/epoch (dev/tuning).
    rollout_steps_per_epoch: int = 0
    # "shuffle": each epoch islice shuffled DataLoader (batches may repeat across epochs).
    # "stratified": disjoint batch-index partitions by domain; epoch e uses partition e % n_groups.
    #   Requires rollout_steps_per_epoch > 0 and len(train_loader) % rollout_steps_per_epoch == 0.
    rollout_partition_mode: str = "shuffle"
    rollout_partition_seed: int = 42
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
    # Periodic vLLM validation every N optimizer steps (Phase B); 0 = disabled.
    # RL is non-smooth vs SFT — prefer step cadence over epoch boundaries.
    val_steps: int = 50
    # Full checkpoints (HF actor + judge + optimizers) every N steps; 0 = off.
    # Align with val_steps to capture peaks when validation runs.
    save_steps: int = 50
    # Greedy Pass@1 before any RL updates (baseline + fail-fast on val pipeline).
    val_before_train: bool = True
    # Validation generation caps (unified with scripts/validate.sh). Rollout uses strategy_max_tokens.
    val_strategy_max_tokens: int = 4096
    val_answer_max_tokens: int = 4096
    val_num_samples: int = 500         # per val_subdir split
    # Per-item rows in eval_*.json (prompts, generations, outcome, judge score).
    val_save_item_details: bool = True
    # inline: items nested in eval JSON | jsonl: sidecar *.jsonl only | both: both
    val_item_storage: str = "inline"
    # Log a wandb.Table of per-item val rows (sortable in UI); runs after Judge scores.
    val_log_items_wandb_table: bool = False
    val_judge_score_batch_size: int = 16
    # L4: total_train_steps for LR scheduler — set automatically in main() if 0
    # 0 = auto: total_epochs × min(rollout_steps_per_epoch, len(train_loader)) (or full loader if rollout_steps_per_epoch=0)
    total_train_steps: int = 0

    # ── Optimisation ──────────────────────────────────────────────────────────
    max_grad_norm: float = 1.0         # L3: gradient clipping (0 = disabled)

    # ── Infrastructure ────────────────────────────────────────────────────────
    # Resolved at train startup: checkpoint_dir = join(checkpoint_root, run_name)
    checkpoint_root: str = "./checkpoints_actor_judge"
    run_name: str = ""                 # empty → auto timestamp in train.py
    checkpoint_dir: str = "./checkpoints_actor_judge"  # overwritten in main()
    # Retain at most this many latest step_* dirs (plus best-k below).
    keep_last_k_checkpoints: int = 2
    # Also retain the top-k steps by mean Pass@1 (across val_subdirs); 0 = disable.
    keep_best_k_checkpoints: int = 2
    n_gpus: int = 8
    # Shared memory dir for vLLM ↔ FSDP weight hand-off (avoids NVMe bottleneck)
    weight_sync_tmp_dir: str = "/dev/shm/actor_weight_tmp"
    # After first validation, keep the vLLM Ray actor alive and hot-reload weights
    # from weight_sync_tmp_dir instead of kill + full engine init (~minutes saved).
    vllm_reuse_validation_actor: bool = True
    # Resume: set to a checkpoint dir to continue training from that point
    resume_from_checkpoint: str = ""   # L5: e.g. "./checkpoints_actor_judge/step_000100"

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_project: str = "ActorJudge"
    wandb_run_name: str = "phase2_co_evolution"
    # If True and resume_from_checkpoint's training_state.json contains wandb_run_id,
    # wandb.init(resume) continues the same online run (same curves / step axis).
    wandb_resume: bool = True

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
        vis = (self.val_item_storage or "inline").strip().lower()
        if vis not in ("inline", "jsonl", "both"):
            raise ValueError(
                f"val_item_storage must be 'inline', 'jsonl', or 'both', got {self.val_item_storage!r}"
            )
        if self.max_stage1_prompt_tokens < 0:
            raise ValueError("max_stage1_prompt_tokens must be >= 0 (0 disables the gate).")
        if self.stage1_length_chars_per_token <= 0:
            raise ValueError("stage1_length_chars_per_token must be > 0.")
        if self.val_steps < 0:
            raise ValueError("val_steps must be >= 0 (0 disables periodic validation).")
        if self.save_steps < 0:
            raise ValueError("save_steps must be >= 0 (0 disables periodic checkpoints).")
        if self.vllm_rollout_prompt_chunk_size < 0:
            raise ValueError("vllm_rollout_prompt_chunk_size must be >= 0 (0 disables chunking).")
        if self.rollout_steps_per_epoch < 0:
            raise ValueError("rollout_steps_per_epoch must be >= 0 (0 = full train_loader per epoch).")
        pm = (self.rollout_partition_mode or "shuffle").strip().lower()
        if pm not in ("shuffle", "stratified"):
            raise ValueError(
                f"rollout_partition_mode must be 'shuffle' or 'stratified', got {self.rollout_partition_mode!r}"
            )
        if pm == "stratified" and self.rollout_steps_per_epoch <= 0:
            raise ValueError(
                "rollout_partition_mode='stratified' requires rollout_steps_per_epoch > 0 "
                "so batches split into equal disjoint groups."
            )
