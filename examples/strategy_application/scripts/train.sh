#!/bin/bash
# Copyright (c) Microsoft. All rights reserved.
#
# Training script for strategy application - Stage 2
# This script runs training for strategy application with configurable parameters

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

# Change to repo root to run the training script
cd "$REPO_ROOT"

# Default values (can be overridden by command line arguments)
STAGE2_MODE="${STAGE2_MODE:-same_domain}"
STAGE1_MODEL_PATH="${STAGE1_MODEL_PATH:-/home/test/test16/chenlu/model/Qwen3-4B}"
DATA_BASE_PATH="${DATA_BASE_PATH:-/home/test/test16/chenlu/projects/LLMReflection/data/}"
TRAIN_SUBDIR="${TRAIN_SUBDIR:-train_20k}"
MODEL_PATH="${MODEL_PATH:-/home/test/test16/chenlu/model/Qwen3-4B}"

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Train strategy application model (Stage 2)

Required arguments:
  --stage1-model-path PATH    Path to stage 1 model for strategy extraction

Optional arguments:
  --stage2-mode MODE          Training mode: same_domain or cross_domain (default: same_domain)
  --data-base-path PATH       Base path for data directory
  --train-subdir DIR          Training data subdirectory name
  --model-path PATH           Path to the model
  --wandb-project NAME        WandB project name
  --wandb-experiment NAME     WandB experiment/run name
  --embedding-model-path PATH Path to embedding model (for cross-domain mode)
  --help                      Show this help message

Examples:
  # Same domain training
  $0 --stage1-model-path /path/to/stage1/model

  # Cross domain training
  $0 --stage2-mode cross_domain \\
     --stage1-model-path /path/to/stage1/model \\
     --embedding-model-path BAAI/bge-large-en-v1.5

  # With custom paths
  $0 --stage1-model-path /path/to/stage1/model \\
     --data-base-path /path/to/data \\
     --model-path /path/to/model \\
     --wandb-project MyProject \\
     --wandb-experiment stage2_experiment

Environment variables:
  STAGE2_MODE                 Default training mode
  STAGE1_MODEL_PATH           Default stage 1 model path
  DATA_BASE_PATH              Default data base path
  TRAIN_SUBDIR                Default training subdirectory
  MODEL_PATH                  Default model path
EOF
}

# Parse arguments
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            usage
            exit 0
            ;;
        --stage2-mode)
            STAGE2_MODE="$2"
            ARGS+=("--stage2-mode" "$2")
            shift 2
            ;;
        --stage1-model-path)
            STAGE1_MODEL_PATH="$2"
            ARGS+=("--stage1-model-path" "$2")
            shift 2
            ;;
        --data-base-path)
            DATA_BASE_PATH="$2"
            ARGS+=("--data-base-path" "$2")
            shift 2
            ;;
        --train-subdir)
            TRAIN_SUBDIR="$2"
            ARGS+=("--train-subdir" "$2")
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            ARGS+=("--model-path" "$2")
            shift 2
            ;;
        --wandb-project)
            ARGS+=("--wandb-project" "$2")
            shift 2
            ;;
        --wandb-experiment)
            ARGS+=("--wandb-experiment" "$2")
            shift 2
            ;;
        --embedding-model-path)
            ARGS+=("--embedding-model-path" "$2")
            shift 2
            ;;
        *)
            # Pass through other arguments
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$STAGE1_MODEL_PATH" ]]; then
    echo "Error: --stage1-model-path is required"
    echo ""
    usage
    exit 1
fi

# Run training
echo "Starting strategy application training (Stage 2)..."
echo "Mode: $STAGE2_MODE"
echo "Stage 1 model path: $STAGE1_MODEL_PATH"
echo "Data base path: $DATA_BASE_PATH"
echo "Model path: $MODEL_PATH"
echo ""

EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-/home/test/test16/chenlu/projects/agent-lightning/model/paraphrase-multilingual-MiniLM-L12-v2}"

python -m examples.strategy_application.train_strategy_application \
    --stage2-mode "$STAGE2_MODE" \
    --stage1-model-path "$STAGE1_MODEL_PATH" \
    --data-base-path "$DATA_BASE_PATH" \
    --train-subdir "$TRAIN_SUBDIR" \
    --model-path "$MODEL_PATH" \
    "${ARGS[@]}"

