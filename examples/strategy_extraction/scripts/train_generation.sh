#!/usr/bin/env bash
# Copyright (c) Microsoft. All rights reserved.
#
# Strategy generation with reward aligned to train_format_answer_v3.sh pattern:
#   scorer_only + reward v3 soft correctness + optional strategy-scorer LLM (four_dim).
# Trainable: strategy model (VERL). Frozen: strategy-scorer vLLM + answer vLLM (un-traced).
#
# Default weights: format 0.1, answer soft correctness 0.7, OC strategy quality 0.2 (sum = 1).
#
# Same data / few-shot / decoding baseline as train_format_answer_v3.sh; this script adds:
#   - Strategy scorer vLLM + --strategy-scoring-prompt-version four_dim (v3 has no scorer server).
#
# ============================================================
#  GPU layout (typical 8 GPUs)
# ============================================================
#   GPU 0-5 : VERL training (FSDP + vLLM rollout)
#   GPU 6   : Strategy scorer vLLM (default: Qwen3-32B — adjust VRAM / max-model-len)
#   GPU 7   : Fixed answer model vLLM (default: Qwen3-8B, same as v3)
#
#   Terminal 1 — scorer (example 32B):
#     CUDA_VISIBLE_DEVICES=6 python -m vllm.entrypoints.openai.api_server \
#         --model /path/to/Qwen3-32B \
#         --served-model-name strategy_scorer \
#         --port 8100 \
#         --gpu-memory-utilization 0.90 \
#         --max-model-len 16384
#
#   Terminal 2 — answer model:
#     CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
#         --model /path/to/Qwen3-8B \
#         --served-model-name Qwen3-8B \
#         --port 8200 \
#         --gpu-memory-utilization 0.90 \
#         --max-model-len 32768
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/train_generation.sh
#   FORMAT_WEIGHT=0.1 SCORER_WEIGHT=0.2 CORRECTNESS_WEIGHT=0.7 bash scripts/train_generation.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# --- Same defaults as train_format_answer_v3.sh (override with env vars) ---
CZJ_TRAIN_ROOT="${CZJ_TRAIN_ROOT:-/home/test/test16/chenlu/projects/LLMReflection-czj/data}"
TRAIN_SUBDIR="${TRAIN_SUBDIR:-train_all_project_suitable}"
VAL_DATA_ROOT="${VAL_DATA_ROOT:-/home/test/test16/chenlu/projects/LLMReflection/data/}"
STRATEGY_MODEL_PATH="${STRATEGY_MODEL_PATH:-/home/test/test16/chenlu/model/Qwen3-4B}"
ANSWER_MODEL_PATH="${ANSWER_MODEL_PATH:-/home/test/test16/chenlu/model/Qwen3-8B}"
ANSWER_MODEL_BASE_URL="${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
ANSWER_MODEL_NAME="${ANSWER_MODEL_NAME:-Qwen3-8B}"

# --- Scorer server + reward weights (soft correctness via reward v3; not hybrid_grounded) ---
STRATEGY_SCORER_MODEL_PATH="${STRATEGY_SCORER_MODEL_PATH:-/home/test/test16/chenlu/model/Qwen3-32B}"
STRATEGY_SCORER_MODEL_NAME="${STRATEGY_SCORER_MODEL_NAME:-strategy_scorer}"
STRATEGY_SCORER_BASE_URL="${STRATEGY_SCORER_BASE_URL:-http://localhost:8100/v1}"
STRATEGY_SCORING_PROMPT_VERSION="${STRATEGY_SCORING_PROMPT_VERSION:-four_dim}"
FORMAT_WEIGHT="${FORMAT_WEIGHT:-0.1}"
SCORER_WEIGHT="${SCORER_WEIGHT:-0.2}"
CORRECTNESS_WEIGHT="${CORRECTNESS_WEIGHT:-0.7}"
GROUNDED_PROXY_K="${GROUNDED_PROXY_K:-1}"
OC_FOUR_DIM_WEIGHTS="${OC_FOUR_DIM_WEIGHTS:-0.3,0.3,0.3,0.1}"
STRATEGY_SCORER_TIMEOUT_SEC="${STRATEGY_SCORER_TIMEOUT_SEC:-180}"

echo "========================================="
echo " Strategy generation — scorer_only (soft + OC)"
echo "========================================="
echo " Train data : ${CZJ_TRAIN_ROOT}/${TRAIN_SUBDIR}"
echo " Val root   : ${VAL_DATA_ROOT}"
echo " Scorer     : ${STRATEGY_SCORER_BASE_URL} (${STRATEGY_SCORER_MODEL_NAME})"
echo " Rubric     : ${STRATEGY_SCORING_PROMPT_VERSION}"
echo " Reward w   : format=${FORMAT_WEIGHT} scorer=${SCORER_WEIGHT} correctness(soft)=${CORRECTNESS_WEIGHT}"
echo " OC weights : ${OC_FOUR_DIM_WEIGHTS}"
echo " K answers  : ${GROUNDED_PROXY_K} (mean soft if K>1, like v3 multi-sample path)"
echo "========================================="
echo ""

python -m examples.strategy_extraction.train_strategy_generation \
    --data-base-path "${CZJ_TRAIN_ROOT}" \
    --train-subdir "${TRAIN_SUBDIR}" \
    --val-data-base-path "${VAL_DATA_ROOT}" \
    --val-subdirs test-id-subtask test-ood-task test-bbh \
    --val-sampling-mode per_subtask_fixed \
    --val-samples-per-subtask 20 \
    --val-sampling-seed 42 \
    --model-path "${STRATEGY_MODEL_PATH}" \
    --fewshot-min 3 \
    --fewshot-max 5 \
    --num-train-samples 20000 \
    --num-val-samples 500 \
    --n-runners 10 \
    --n-gpus 6 \
    --reward-version v3 \
    --reward-mode scorer_only \
    --format-weight "${FORMAT_WEIGHT}" \
    --scorer-weight "${SCORER_WEIGHT}" \
    --correctness-weight "${CORRECTNESS_WEIGHT}" \
    --grounded-proxy-k "${GROUNDED_PROXY_K}" \
    --strategy-scorer-model-path "${STRATEGY_SCORER_MODEL_PATH}" \
    --strategy-scorer-model-name "${STRATEGY_SCORER_MODEL_NAME}" \
    --strategy-scorer-base-url "${STRATEGY_SCORER_BASE_URL}" \
    --strategy-scoring-prompt-version "${STRATEGY_SCORING_PROMPT_VERSION}" \
    --oc-four-dim-weights "${OC_FOUR_DIM_WEIGHTS}" \
    --strategy-scorer-timeout-sec "${STRATEGY_SCORER_TIMEOUT_SEC}" \
    --answer-model-path "${ANSWER_MODEL_PATH}" \
    --answer-model-base-url "${ANSWER_MODEL_BASE_URL}" \
    --answer-model-name "${ANSWER_MODEL_NAME}" \
    --answer-no-think \
    --strategy-no-think \
    ${ANSWER_MAX_TOKENS:+--answer-max-tokens "${ANSWER_MAX_TOKENS}"} \
    --strategy-prompt-version repetition_controls_2026-04-01 \
    --answer-prompt-version v1 \
    --val-answer-temperature 0.0 \
    --wandb-project StrategyGeneration \
    --wandb-experiment strategy_gen_scorer_soft_four_dim \
    --strategy-repetition-penalty 1.1 \
    --answer-request-retries 3 \
    --answer-retry-delay-sec 1.0 \
    "$@"
