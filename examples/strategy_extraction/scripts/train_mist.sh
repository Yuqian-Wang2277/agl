#!/usr/bin/env bash
# MIST (EIR × (1 + β·ICR)) reward training script.
#
# Supports two training modes via TRAINING_MODE env var:
#
#   mist (default) — Decoupled architecture: a separate, frozen answer model
#     evaluates answer quality.  Reward signal is stable because the answer
#     model weights do not change during training.
#     Requires a running answer-model vLLM server (Terminal 1 below).
#
#   mist_e2e — End-to-end / coupled architecture (ablation experiment):
#     the strategy training model itself is reused as the answer evaluator
#     (answer_model_base_url left empty → agent falls back to rollout URL).
#     All reward hyper-parameters are identical to mist.  The purpose is to
#     demonstrate that the decoupled design outperforms the coupled baseline
#     (ablation study for the decoupled architecture).
#     No separate answer-model server needed — only Terminal 2.
#
# GPU layout (typical 8 GPUs):
#   GPU 0-3 : VERL training (FSDP + vLLM rollout)
#   GPU 4-7 : Fixed answer model vLLM (Qwen3-8B, port 8200) — mist mode only
#   (No strategy scorer GPU needed — MIST uses answer logprobs only)
#
# === mist (decoupled) mode ===
# Terminal 1 — answer model (required for mist mode):
#   CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
#       --model /path/to/Qwen3-8B \
#       --served-model-name Qwen3-8B \
#       --port 8200 \
#       --gpu-memory-utilization 0.90 \
#       --max-model-len 32768
#
# Terminal 2 — training:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_mist.sh
#
# === mist_e2e (end-to-end / coupled, ablation) mode ===
# Terminal 2 only — no answer model server needed:
#   TRAINING_MODE=mist_e2e BASELINE_CACHE_PATH=./baseline_cache.json \
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_mist.sh
#
# Quick-test example (small batch, 1 epoch, frequent eval):
#   TRAIN_BATCH_SIZE=8 TOTAL_EPOCHS=1 TEST_FREQ=10 SAVE_FREQ=10 \
#   BETA=0.5 EIR_K=10.0 \
#   BASELINE_CACHE_PATH=./baseline_cache.json \
#   FILTER_TO_BASELINE_CACHE=1 \
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_mist.sh
#
# To disable cache filtering (enumerate all train problems, cache-miss → reward=0):
#   FILTER_TO_BASELINE_CACHE=0 bash scripts/train_mist.sh
#
set -euo pipefail

# --- Training mode ---
# mist     : decoupled architecture (strategy model + separate frozen answer model)
# mist_e2e : end-to-end / coupled architecture (strategy model reused as answer model; ablation)
TRAINING_MODE="${TRAINING_MODE:-mist}"
if [ "${TRAINING_MODE}" != "mist" ] && [ "${TRAINING_MODE}" != "mist_e2e" ]; then
    echo "ERROR: TRAINING_MODE must be 'mist' or 'mist_e2e', got '${TRAINING_MODE}'" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# --- Data / model paths ---
CZJ_TRAIN_ROOT="${CZJ_TRAIN_ROOT:-/home/test/test16/chenlu/projects/LLMReflection-czj/data}"
TRAIN_SUBDIR="${TRAIN_SUBDIR:-train_all_project_suitable}"
VAL_DATA_ROOT="${VAL_DATA_ROOT:-/home/test/test16/chenlu/projects/LLMReflection/data/}"
STRATEGY_MODEL_PATH="${STRATEGY_MODEL_PATH:-/home/test/test16/chenlu/projects/fs/sft/sft_output/train_qwen4b_trainall_sig/1/checkpoint-1000_merged_hf}"

# --- Answer model: mode-dependent ---
# mist     : dedicated frozen Qwen3-8B server (decoupled, stable reward signal)
# mist_e2e : empty URL → agent falls back to rollout/training model URL (coupled)
if [ "${TRAINING_MODE}" = "mist_e2e" ]; then
    ANSWER_MODEL_PATH=""
    ANSWER_MODEL_BASE_URL=""
    ANSWER_MODEL_NAME=""
    WANDB_EXPERIMENT="${WANDB_EXPERIMENT:-strategy_gen_mist_e2e}"
else
    ANSWER_MODEL_PATH="${ANSWER_MODEL_PATH:-/home/test/test16/chenlu/model/Qwen3-8B}"
    ANSWER_MODEL_BASE_URL="${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
    ANSWER_MODEL_NAME="${ANSWER_MODEL_NAME:-Qwen3-8B}"
    WANDB_EXPERIMENT="${WANDB_EXPERIMENT:-strategy_gen_mist}"
fi

# --- MIST reward parameters (identical across both modes) ---
BETA="${BETA:-0.5}"                                    # β: ICR amplification weight
EIR_K="${EIR_K:-10.0}"                                # k: log-smoothing scale for EIR
ALPHA_UP="${ALPHA_UP:-0.7}"                            # α when a_curr >= a_base (improving)
ALPHA_DOWN="${ALPHA_DOWN:-0.3}"                        # α when a_curr <  a_base (regressing)
BASELINE_CACHE_PATH="${BASELINE_CACHE_PATH:-}"         # path to baseline_cache.json (required for MIST)
FILTER_TO_BASELINE_CACHE="${FILTER_TO_BASELINE_CACHE:-1}"  # 1=filter train set to cached problems only (recommended)

# --- Quick-test / training schedule overrides ---
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-24}"       # VERL train_batch_size (problems per step)
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"                # total training epochs
TEST_FREQ="${TEST_FREQ:-50}"                     # validation every N steps
SAVE_FREQ="${SAVE_FREQ:-50}"                     # checkpoint every N steps

# --- Fixed: M=1 for ICR, no scorer ---
GROUNDED_PROXY_K=1
# 格式阶段可把 FORMAT_WEIGHT 调大（如 1.0），并去掉 BASELINE_CACHE_PATH 以关闭 MIST，仅用 v3 的 format 信号训练。
FORMAT_WEIGHT="${FORMAT_WEIGHT:-0.0}"

echo "========================================="
if [ "${TRAINING_MODE}" = "mist_e2e" ]; then
    echo " MIST E2E Ablation (coupled architecture)"
    echo " — strategy model reused as answer model —"
else
    echo " MIST Reward Training (EIR × (1 + β·ICR))"
    echo " — decoupled: separate frozen answer model —"
fi
echo "========================================="
echo " Mode       : ${TRAINING_MODE}"
echo " Train data : ${CZJ_TRAIN_ROOT}/${TRAIN_SUBDIR}"
echo " Val root   : ${VAL_DATA_ROOT}"
echo " β (beta)   : ${BETA}"
echo " k (eir_k)  : ${EIR_K}"
echo " α_up       : ${ALPHA_UP}"
echo " α_down     : ${ALPHA_DOWN}"
echo " Baseline   : ${BASELINE_CACHE_PATH:-<not set — MIST disabled>}"
echo " Filter     : ${FILTER_TO_BASELINE_CACHE} (1=filter train set to cached problems)"
echo " Schedule   : batch=${TRAIN_BATCH_SIZE}, epochs=${TOTAL_EPOCHS}, test=${TEST_FREQ}, save=${SAVE_FREQ}"
if [ "${TRAINING_MODE}" = "mist_e2e" ]; then
    echo " Answer     : <training model rollout URL> (coupled / no separate server)"
else
    echo " Answer     : ${ANSWER_MODEL_BASE_URL} (${ANSWER_MODEL_NAME})"
fi
echo " WandB exp  : ${WANDB_EXPERIMENT}"
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
    --fewshot-max 3 \
    --train-sampling-mode per_subtask_exhaustive \
    --train-new-problems-per-sample 1 \
    --num-val-samples 500 \
    --n-runners 10 \
    --n-gpus 4 \
    --reward-version v3 \
    --reward-mode scorer_only \
    --format-weight "${FORMAT_WEIGHT}" \
    --scorer-weight 0.0 \
    --correctness-weight 0.0 \
    --grounded-proxy-k "${GROUNDED_PROXY_K}" \
    --eval-problems-per-subtask 0 \
    ${ANSWER_MODEL_PATH:+--answer-model-path "${ANSWER_MODEL_PATH}"} \
    ${ANSWER_MODEL_BASE_URL:+--answer-model-base-url "${ANSWER_MODEL_BASE_URL}"} \
    ${ANSWER_MODEL_NAME:+--answer-model-name "${ANSWER_MODEL_NAME}"} \
    --answer-no-think \
    --strategy-prompt-version repetition_controls_2026-04-01 \
    --answer-prompt-version v1 \
    --val-answer-temperature 0.0 \
    --wandb-project StrategyGeneration \
    --wandb-experiment "${WANDB_EXPERIMENT}" \
    --strategy-repetition-penalty 1.1 \
    --answer-request-retries 3 \
    --answer-retry-delay-sec 1.0 \
    --beta "${BETA}" \
    --eir-k "${EIR_K}" \
    --alpha-up "${ALPHA_UP}" \
    --alpha-down "${ALPHA_DOWN}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --total-epochs "${TOTAL_EPOCHS}" \
    --test-freq "${TEST_FREQ}" \
    --save-freq "${SAVE_FREQ}" \
    ${BASELINE_CACHE_PATH:+--baseline-cache-path "${BASELINE_CACHE_PATH}"} \
    $([ "${FILTER_TO_BASELINE_CACHE:-0}" = "1" ] && echo --filter-to-baseline-cache) \
    "$@"
