#!/bin/bash
# Evaluate no-strategy baseline with reward v3 accuracy routing.
#
# Key settings:
# - Validation sampling: fixed 20 per subtask json (57 subtasks total => 1140 samples)
# - Seed: 42 for reproducibility
# - No-strategy baseline: do not pass generated strategy to answer model
# - Accuracy metric path: reward v3 (type-routed answer judging)
# - val_only: run validation and exit before any training update
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval_no_strategy_baseline.sh

set -e
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

python -m examples.strategy_extraction.train_strategy_generation \
    --data-base-path /home/test/test16/chenlu/projects/LLMReflection/data/ \
    --train-subdir train_20k \
    --val-subdirs test-id-subtask test-ood-task test-bbh \
    --model-path /home/test/test16/chenlu/model/Qwen3-4B \
    --fewshot-min 3 \
    --fewshot-max 8 \
    --num-train-samples 1 \
    --val-sampling-mode per_subtask_fixed \
    --val-samples-per-subtask 20 \
    --val-sampling-seed 42 \
    --n-runners 4 \
    --n-gpus 8 \
    --reward-version v3 \
    --reward-mode scorer_only \
    --format-weight 0.0 \
    --scorer-weight 0.0 \
    --grounded-proxy-weight 0.0 \
    --correctness-weight 1.0 \
    --val-only \
    --no-strategy-for-answer \
    --answer-prompt-version no_strategy_baseline \
    --wandb-experiment no_strategy_baseline_v3 \
    --checkpoint-dir ./checkpoints_strategy_gen_no_strategy_baseline \
    "$@"
