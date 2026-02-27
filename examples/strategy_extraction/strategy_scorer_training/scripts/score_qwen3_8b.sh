#!/bin/bash
# Copyright (c) Microsoft. All rights reserved.
#
# Zero-shot rubric scoring with Qwen3-8B via OpenAI-compatible API.
#
# Usage:
#   bash examples/strategy_extraction/strategy_scorer_training/scripts/score_qwen3_8b.sh
#   bash .../score_qwen3_8b.sh --limit 10

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCORER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$SCORER_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

python -m examples.strategy_extraction.strategy_scorer_training.score_with_qwen \
  --input-jsonl examples/strategy_extraction/strategy_scorer_training/data/example_input.jsonl \
  --output-jsonl examples/strategy_extraction/strategy_scorer_training/data/example_output.jsonl \
  --base-url http://localhost:8100/v1 \
  --model qwen3-8b-scorer \
  --prompt-version v2 \
  --temperature 0.0 \
  --max-tokens 512 \
  --overwrite \
  "$@"

