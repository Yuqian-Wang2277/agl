#!/bin/bash
# Copyright (c) Microsoft. All rights reserved.
#
# Training script for strategy extraction - Stage 1
# This script reads configuration from config.py and runs training

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

# Change to repo root to run the training script
cd "$REPO_ROOT"

# Run training with configuration from config.py
# All parameters use defaults from StrategyConfig unless overridden
# Usage: 
#   bash train.sh                                    # Start new training
#   bash train.sh --resume-from-path <checkpoint>    # Resume from checkpoint
python examples/strategy_extraction/train_strategy.py \
    --data-base-path /home/test/test16/chenlu/projects/LLMReflection/data/ \
    --train-subdir train_20k \
    --val-subdirs test-id-subtask test-ood-task test-bbh \
    --model-path /home/test/test16/chenlu/model/Qwen3-4B \
    --fewshot-min 3 \
    --fewshot-max 8 \
    --num-train-samples 20000 \
    --num-val-samples 500 \
    --n-runners 10 \
    --no-resume-from-checkpoint \
    "$@"

