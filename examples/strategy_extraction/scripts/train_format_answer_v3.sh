#!/usr/bin/env bash
# Copyright (c) Microsoft. All rights reserved.
#
# Strategy generation training with reward = format (strategy tags) + answer correctness (reward/v3).
# Trainable: strategy model (VERL). Frozen: separate answer vLLM (un-traced).
#
# Disable a signal by setting its weight to 0 (--format-weight 0 or --correctness-weight 0).
#
# ============================================================
#  Prerequisites — start the ANSWER vLLM before training
# ============================================================
#  Strategy rollout/training uses VERL's vLLM actor (no extra server for "strategy only"
#  unless you use an external setup). The ANSWER model must be served separately, e.g.:
#
#    CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
#        --model /path/to/answer/model \
#        --served-model-name Qwen3-4B \
#        --port 8200 \
#        --gpu-memory-utilization 0.90 \
#        --max-model-len 32768
#
#  Override URLs/names with ANSWER_MODEL_BASE_URL / ANSWER_MODEL_NAME below.
#
# ============================================================
#  Data layout
# ============================================================
#  Training data:   LLMReflection-czj/data/<TRAIN_SUBDIR>/  (problem-type subdirs + JSON)
#  Validation data: LLMReflection/data/<val-subdirs>/       (same layout as eval_no_verl.sh)
#  With --val-sampling-mode per_subtask_fixed, validation size is
#    (val-samples-per-subtask) * (number of subtask JSON files per split), NOT --num-val-samples.
#  Leave --num-val-samples as-is unless you switch to global/balanced val sampling.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_format_answer_v3.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_format_answer_v3.sh \
#       --format-weight 0 --correctness-weight 1.0
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# --- Roots (override with env vars if needed) ---
CZJ_TRAIN_ROOT="${CZJ_TRAIN_ROOT:-/home/test/test16/chenlu/projects/LLMReflection-czj/data}"
TRAIN_SUBDIR="${TRAIN_SUBDIR:-train_all_project_suitable}"
VAL_DATA_ROOT="${VAL_DATA_ROOT:-/home/test/test16/chenlu/projects/LLMReflection/data/}"

# --- Models (override with env vars or "$@") ---
STRATEGY_MODEL_PATH="${STRATEGY_MODEL_PATH:-/home/test/test16/chenlu/model/Qwen3-4B}"
ANSWER_MODEL_PATH="${ANSWER_MODEL_PATH:-/home/test/test16/chenlu/model/Qwen3-8B}"
ANSWER_MODEL_BASE_URL="${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
ANSWER_MODEL_NAME="${ANSWER_MODEL_NAME:-Qwen3-8B}"

echo "========================================="
echo " Strategy gen — format + answer (reward v3)"
echo "========================================="
echo " Train data : ${CZJ_TRAIN_ROOT}/${TRAIN_SUBDIR}"
echo " Val root   : ${VAL_DATA_ROOT}"
echo " Reward     : scorer_only + reward v3 (no strategy-scorer LLM)"
echo " Weights    : format + correctness (set either to 0 to disable)"
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
    --n-gpus 8 \
    --reward-version v3 \
    --reward-mode scorer_only \
    --format-weight 0.3 \
    --correctness-weight 0.7 \
    --grounded-proxy-k 1 \
    --answer-model-path "${ANSWER_MODEL_PATH}" \
    --answer-model-base-url "${ANSWER_MODEL_BASE_URL}" \
    --answer-model-name "${ANSWER_MODEL_NAME}" \
    --strategy-prompt-version strategy_update_2026-03-09 \
    --answer-prompt-version v1 \
    --wandb-project StrategyGeneration \
    --wandb-experiment strategy_format_answer_v3 \
    "$@"
