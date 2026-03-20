#!/usr/bin/env bash
# =============================================================================
# warmup_judge.sh — Standalone Judge warmup (pairwise RM pretraining)
#
# Usage:
#   bash scripts/warmup_judge.sh --sft_checkpoint /path/to/actor_hf
#
# Output:
#   ./judge_warmup_ckpt/
#     - judge_model.pt
#     - model.safetensors / config.json ... (HF backbone)
#     - tokenizer files
#     - warmup_meta.json
#     - judge_warmup_pairs.jsonl
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SFT_CHECKPOINT=""
MODEL_PATH="/home/test/test16/chenlu/model/Qwen3-4B"
OUTPUT_DIR="$PROJECT_DIR/judge_warmup_ckpt"
WARMUP_STEPS=100
BATCH_SIZE=16
WARMUP_LR=0
EVAL_RATIO=0.12
EVAL_EVERY=5
EARLY_STOP_ACC=0.75
OVERFIT_WARN_ACC=0.95
WARMUP_SEED=42
NUM_TRAIN_SAMPLES=20000
CUDA_VISIBLE_DEVICES_ARG=""
DEVICE_MAP_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sft_checkpoint)   SFT_CHECKPOINT="$2"; shift 2 ;;
        --model_path)       MODEL_PATH="$2"; shift 2 ;;
        --output_dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --warmup_steps)     WARMUP_STEPS="$2"; shift 2 ;;
        --batch_size)       BATCH_SIZE="$2"; shift 2 ;;
        --warmup_lr)        WARMUP_LR="$2"; shift 2 ;;
        --eval_ratio)       EVAL_RATIO="$2"; shift 2 ;;
        --eval_every)       EVAL_EVERY="$2"; shift 2 ;;
        --early_stop_acc)   EARLY_STOP_ACC="$2"; shift 2 ;;
        --overfit_warn_acc) OVERFIT_WARN_ACC="$2"; shift 2 ;;
        --warmup_seed)      WARMUP_SEED="$2"; shift 2 ;;
        --num_train_samples) NUM_TRAIN_SAMPLES="$2"; shift 2 ;;
        --cuda_visible_devices) CUDA_VISIBLE_DEVICES_ARG="$2"; shift 2 ;;
        --device_map) DEVICE_MAP_ARG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agl

cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"

# Ensure python prints are not buffered (we rely on these logs for debugging).
export PYTHONUNBUFFERED=1

# ── GPU selection (avoid OOM from busy cards) ──────────────────────────
# User override takes priority; otherwise auto-pick the 4 least-used GPUs.
if [[ -z "$CUDA_VISIBLE_DEVICES_ARG" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_VISIBLE_DEVICES_ARG="$(
            nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | sort -t',' -k2,2n \
            | head -n 4 \
            | awk -F',' '{gsub(/ /,"",$1); print $1}' \
            | paste -sd, -
        )"
    fi
fi
if [[ -n "$CUDA_VISIBLE_DEVICES_ARG" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_ARG"
    echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

CMD=(
    python run_judge_warmup.py
    --actor_model_path "$MODEL_PATH"
    --output_dir "$OUTPUT_DIR"
    ${DEVICE_MAP_ARG:+--device_map "$DEVICE_MAP_ARG"}
    --warmup_steps "$WARMUP_STEPS"
    --judge_batch_size "$BATCH_SIZE"
    --judge_warmup_lr "$WARMUP_LR"
    --judge_warmup_eval_ratio "$EVAL_RATIO"
    --judge_warmup_eval_every "$EVAL_EVERY"
    --judge_warmup_early_stop_min_acc "$EARLY_STOP_ACC"
    --judge_warmup_overfit_warn_acc "$OVERFIT_WARN_ACC"
    --warmup_seed "$WARMUP_SEED"
    --num_train_samples "$NUM_TRAIN_SAMPLES"
)

[[ -n "$SFT_CHECKPOINT" ]] && CMD+=("--actor_sft_checkpoint" "$SFT_CHECKPOINT")

echo "========================================================"
echo "Launch command:"
echo "  ${CMD[*]}"
echo "========================================================"

exec "${CMD[@]}"
