#!/usr/bin/env bash
# =============================================================================
# smoke_test_step8.sh — 快速验证 Phase B step 8 不再 OOM / crash
#
# 背景：Phase B 的 actor_trainer 在第 8 个训练步（每次都稳定复现）崩溃，根因是
#   compute_log_probs 同时申请了两个 [B, seq, vocab] logit 张量（~19 GiB），
#   经过 7 步训练后 GPU 内存碎片化，第 8 步触发 OOM（exitcode 1）。
#
# 本脚本用 10 个 rollout 步（Phase A 共 10 步，Phase B 刚好能到第 8 步）+
# 极少的验证样本，把端到端时间从 >1 小时压到 ~15-25 分钟，足以验证修复。
#
# 用法（在 examples/actor-judge 目录执行）：
#   bash scripts/smoke_test_step8.sh /path/to/actor_sft_hf
#
# 或先 export：
#   export SFT_CHECKPOINT=/path/to/actor_sft_hf
#   bash scripts/smoke_test_step8.sh
#
# Judge 默认从 ./judge_warmup_ckpt/2026-03-20/judge_model.pt 加载；
# 如在别处请：  export JUDGE_INIT_DIR=/path/to/judge_dir
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export PYTHONUNBUFFERED=1

SFT="${1:-${SFT_CHECKPOINT:-}}"
JUDGE_INIT_DIR="${JUDGE_INIT_DIR:-$PROJECT_DIR/judge_warmup_ckpt/2026-03-20}"

if [[ -z "$SFT" ]]; then
    echo "Usage: $0 /path/to/actor_sft_hf"
    echo "  or:  export SFT_CHECKPOINT=... && $0"
    exit 1
fi
if [[ ! -f "$JUDGE_INIT_DIR/judge_model.pt" ]]; then
    echo "Missing pre-warmed Judge at $JUDGE_INIT_DIR/judge_model.pt"
    echo "Set JUDGE_INIT_DIR or run warmup_judge.sh first."
    exit 1
fi

RUN_TAG="smoke_step8_$(date +%Y%m%d_%H%M%S)"

echo "========================================================"
echo "Smoke test: 10 rollout steps → 10 Phase-B steps"
echo "Specifically tests that Phase-B step 8 completes (OOM fix)"
echo "Run tag: $RUN_TAG"
echo "========================================================"

# --rollout_steps_per_epoch 10  →  Phase A: 10 batches  →  Phase B: 10 steps
# --num_train_samples 20000     →  use full train set so stratified partition
#                                  picks the same questions as the full run
# --val_steps 0 / --save_steps 0 → skip in-loop validation & checkpointing
# --no_val_before_train          → skip baseline validation (saves ~6 min)
# --dry_run_val_size 10          → if val does run, only 10 samples per split
exec bash "$SCRIPT_DIR/train.sh" \
    --sft_checkpoint "$SFT" \
    --epochs 1 \
    --rollout_steps_per_epoch 10 \
    --rollout_partition_mode stratified \
    --rollout_partition_seed 42 \
    --run_name "$RUN_TAG" \
    --extra "--judge_warmup_mode reuse \
             --judge_init_checkpoint $JUDGE_INIT_DIR \
             --no_val_before_train \
             --val_steps 0 \
             --save_steps 0 \
             --dry_run_val_size 10 \
             --keep_last_k_checkpoints 0 \
             --keep_best_k_checkpoints 0"
