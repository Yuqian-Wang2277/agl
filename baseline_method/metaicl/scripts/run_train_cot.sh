#!/usr/bin/env bash
# MetaICL-CoT SFT training launcher.
#
# Usage:
#   bash scripts/run_train_cot.sh
#   THINK_SOURCE=empty bash scripts/run_train_cot.sh
#   THINK_SOURCE=file THINK_FILE=/path/to/labels.jsonl bash scripts/run_train_cot.sh
#   MAX_STEPS=500 bash scripts/run_train_cot.sh
#
# Runs from the metaicl/ directory regardless of where it is called from.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── Defaults (override via environment) ───────────────────────────────────────
CONFIG="${CONFIG:-configs/train_qwen3_4b_cot.yaml}"
GPUS="${GPUS:-0,1,2,3}"
N_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
THINK_SOURCE="${THINK_SOURCE:-empty}"

echo "=== MetaICL-CoT SFT training ==="
echo "Config       : $CONFIG"
echo "GPUs         : $GPUS  (n=$N_GPUS)"
echo "think_source : $THINK_SOURCE"
if [[ "$THINK_SOURCE" == "file" ]]; then
    echo "think_file   : ${THINK_FILE:?THINK_FILE must be set when THINK_SOURCE=file}"
fi
echo "Extra env: MAX_STEPS=${MAX_STEPS:-unset}  K_SHOT=${K_SHOT:-unset}"
echo

conda run -n agl --no-capture-output \
    torchrun \
        --nproc_per_node="$N_GPUS" \
        --master_port=29501 \
    train_metaicl_cot.py \
        --config "$CONFIG" \
        --think-source "$THINK_SOURCE" \
        ${THINK_FILE:+--think-file "$THINK_FILE"} \
        ${MODEL_PATH:+--model-path "$MODEL_PATH"} \
        ${DATA_DIR:+--data-dir "$DATA_DIR"} \
        ${OUTPUT_DIR:+--output-dir "$OUTPUT_DIR"} \
        ${MAX_STEPS:+--max-steps "$MAX_STEPS"} \
        ${K_SHOT:+--k-shot "$K_SHOT"}

echo "=== CoT training finished ==="
