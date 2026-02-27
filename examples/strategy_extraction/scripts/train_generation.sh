#!/bin/bash
# Copyright (c) Microsoft. All rights reserved.
#
# Training script for strategy generation (v2 — scorer-based reward).
#
# Only strategy-generation tokens receive gradient updates.
# A trained strategy-scorer LLM (Qwen3-8B) directly evaluates strategy quality.
# A fixed answer model (Qwen3-4B) provides monitoring-only correctness signal.
#
# ============================================================
#  GPU allocation (8 GPUs total)
# ============================================================
#   GPU 0-5 : VERL training  (6 GPUs, FSDP + vLLM rollout)
#   GPU 6   : Strategy scorer  (Qwen3-8B, ~16 GB FP16)
#   GPU 7   : Fixed answer model  (Qwen3-4B, ~8 GB FP16)
#
# ============================================================
#  Prerequisites — start BOTH vLLM servers BEFORE training
# ============================================================
#
#   Terminal 1 — scorer (Qwen3-8B):
#     CUDA_VISIBLE_DEVICES=6 python -m vllm.entrypoints.openai.api_server \
#         --model /home/test/test16/chenlu/model/Qwen3-8B \
#         --served-model-name strategy_scorer \
#         --port 8100 \
#         --gpu-memory-utilization 0.90 \
#         --max-model-len 8192
#
#   Terminal 2 — fixed answer model (Qwen3-4B):
#     CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
#         --model /home/test/test16/chenlu/model/Qwen3-4B \
#         --served-model-name Qwen3-4B \
#         --port 8200 \
#         --gpu-memory-utilization 0.90 \
#         --max-model-len 8192
#
#   Verify both are ready:
#     curl -s http://localhost:8100/v1/models | python -m json.tool
#     curl -s http://localhost:8200/v1/models | python -m json.tool
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/train_generation.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/train_generation.sh --help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

echo "========================================="
echo " Strategy Generation Training (v2)"
echo "========================================="
echo " Reward    : scorer (Qwen3-8B) on :8100"
echo " Answer    : fixed  (Qwen3-4B) on :8200  (monitoring only)"
echo " Training  : VERL GRPO, n=8 rollouts/prompt"
echo "========================================="
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
    --n-gpus 6 \
    --reward-version v2 \
    --format-weight 0.1 \
    --scorer-weight 0.9 \
    --correctness-weight 0.0 \
    --strategy-scorer-model-path /home/test/test16/chenlu/model/Qwen3-8B \
    --strategy-scorer-model-name strategy_scorer \
    --strategy-scorer-base-url http://localhost:8100/v1 \
    --strategy-scoring-prompt-version v2 \
    --answer-model-path /home/test/test16/chenlu/model/Qwen3-4B \
    --answer-model-base-url http://localhost:8200/v1 \
    --answer-model-name Qwen3-4B \
    --strategy-prompt-version v1 \
    --answer-prompt-version v1 \
    --wandb-project StrategyGeneration \
    --wandb-experiment strategy_gen_v2 \
    "$@"
