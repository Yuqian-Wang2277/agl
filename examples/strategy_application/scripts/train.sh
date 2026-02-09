#!/bin/bash
# Copyright (c) Microsoft. All rights reserved.
#
# Training script for strategy application - Stage 2
#
# All training parameters (stage2_mode, model paths, data paths, etc.) are
# configured in config.py via StrategyApplicationConfig. This script simply
# launches the Python training module and forwards any extra CLI overrides.
#
# Usage:
#   # Run with defaults from config.py (same_domain / cross_domain controlled there)
#   bash scripts/train.sh
#
#   # Override specific settings via CLI
#   bash scripts/train.sh --wandb-experiment my_exp --lora
#
#   # Show all available options
#   bash scripts/train.sh --help

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

# Change to repo root to run the training script
cd "$REPO_ROOT"

echo "Starting strategy application training (Stage 2)..."
echo "All parameters are read from config.py unless overridden via CLI."
echo ""

python -m examples.strategy_application.train_strategy_application "$@"
