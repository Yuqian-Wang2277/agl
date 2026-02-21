#!/bin/bash
# Copyright (c) Microsoft. All rights reserved.
#
# Training script for strategy generation — trains strategy extraction
# using answer correctness as the reward signal.
#
# Only strategy-generation tokens receive gradient updates.
# The answer-generation step is un-traced and serves purely as reward evaluator.
#
# Usage:
#   # Run with defaults
#   bash scripts/train_generation.sh
#
#   # Override settings via CLI
#   bash scripts/train_generation.sh --wandb-experiment my_exp --lora
#
#   # Show all available options
#   bash scripts/train_generation.sh --help

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

# Change to repo root to run the training script
cd "$REPO_ROOT"

echo "Starting strategy generation training..."
echo "Only strategy-generation tokens are trained; answer quality is the reward signal."
echo ""

python -m examples.strategy_extraction.train_strategy_generation \
    --data-base-path /home/test/test16/chenlu/projects/LLMReflection/data/ \
    --train-subdir train_20k \
    --val-subdirs test-id-subtask test-ood-task test-bbh \
    --model-path /home/test/test16/chenlu/model/Qwen3-4B \
    --fewshot-min 3 \
    --fewshot-max 8 \
    --num-train-samples 20000 \
    --num-val-samples 500 \
    --n-runners 10 \
    --format-weight 0.2 \
    --correctness-weight 0.8 \
    --strategy-prompt-version v1 \
    --answer-prompt-version v1 \
    --reward-version v1 \
    --wandb-project StrategyGeneration \
    --wandb-experiment strategy_gen \
    "$@"

