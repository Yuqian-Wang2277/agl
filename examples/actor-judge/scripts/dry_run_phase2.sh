#!/usr/bin/env bash
# =============================================================================
# dry_run_phase2.sh — 10 items × 2 epochs smoke test (Scaling Checkpoints)
#
# - min_buffer_size=8 so Judge ODVA can run (default 100 would skip on tiny data)
# - dry_run_val_size=10 for small val; --val_steps/--save_steps=2 so smoke hits step-level val/save
# - WANDB_MODE=offline (no upload; Tables still logged locally under wandb/)
# - Val: per-item details → eval_*_items.jsonl; wandb.Table for sortable UI
# - Judge: reuse pre-warmed checkpoint (no long warmup on 10 rows)
#
# Usage:
#   export SFT_CHECKPOINT=/path/to/actor_hf
#   bash scripts/dry_run_phase2.sh
#
# Or:
#   bash scripts/dry_run_phase2.sh /path/to/actor_hf
#
# Override judge dir (must contain judge_model.pt):
#   export JUDGE_INIT_DIR="$PROJECT_DIR/judge_warmup_ckpt/2026-03-20"
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
    echo "Set SFT_CHECKPOINT or pass actor SFT dir as first argument."
    exit 1
fi
if [[ ! -f "$JUDGE_INIT_DIR/judge_model.pt" ]]; then
    echo "Missing warmed Judge: $JUDGE_INIT_DIR/judge_model.pt"
    echo "Train one first (warmup_judge.sh) or set JUDGE_INIT_DIR."
    exit 1
fi

RUN_TAG="dry_run_smoke_$(date +%Y%m%d_%H%M%S)"

exec bash "$SCRIPT_DIR/train.sh" \
    --sft_checkpoint "$SFT" \
    --epochs 2 \
    --num_train_samples 10 \
    --min_buffer_size 8 \
    --dry_run_val_size 10 \
    --run_name "$RUN_TAG" \
    --extra "--judge_warmup_mode reuse --judge_init_checkpoint $JUDGE_INIT_DIR --val_item_storage jsonl --val_log_items_wandb_table --val_steps 2 --save_steps 2 --keep_last_k_checkpoints 0 --keep_best_k_checkpoints 0"
