#!/usr/bin/env bash
# MetaICL SFT training launcher.
#
# Usage:
#   bash scripts/run_train.sh
#   MAX_STEPS=500 bash scripts/run_train.sh
#   OUTPUT_DIR=./checkpoints/test bash scripts/run_train.sh
#
# Runs from the metaicl/ directory regardless of where it is called from.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── Defaults (override via environment) ───────────────────────────────────────
CONFIG="${CONFIG:-configs/train_qwen3_4b.yaml}"
GPUS="${GPUS:-0,1,2,3}"
N_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

echo "=== MetaICL SFT training ==="
echo "Config   : $CONFIG"
echo "GPUs     : $GPUS  (n=$N_GPUS)"
echo "Extra env: MAX_STEPS=${MAX_STEPS:-unset}  K_SHOT=${K_SHOT:-unset}  LR=${LR:-unset}"
echo

conda run -n agl --no-capture-output \
    torchrun \
        --nproc_per_node="$N_GPUS" \
        --master_port=29500 \
    train_metaicl.py \
        --config "$CONFIG" \
        ${MODEL_PATH:+--model-path "$MODEL_PATH"} \
        ${DATA_DIR:+--data-dir "$DATA_DIR"} \
        ${OUTPUT_DIR:+--output-dir "$OUTPUT_DIR"} \
        ${MAX_STEPS:+--max-steps "$MAX_STEPS"} \
        ${K_SHOT:+--k-shot "$K_SHOT"} \
        ${LR:+--lr "$LR"}

echo "=== Training finished ==="
