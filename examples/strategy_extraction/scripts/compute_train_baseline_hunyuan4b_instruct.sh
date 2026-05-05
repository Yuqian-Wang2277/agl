#!/usr/bin/env bash
# Compute train-set baseline cache for Hunyuan-4B-Instruct strategy model.
#
# Uses stratified per-subtask sampling (N=20/subtask, 1 inference/problem)
# → runs eval_no_verl + build_baseline_cache.py → baseline_cache_hunyuan4b_instruct.json
#
# Note: Hunyuan-4B-Instruct is a 4B instruction-tuned model.
# 4B model can serve with TP=2 on 2 GPUs (GPU 0-1), freeing GPU 2-7 for other use.
#
# 数据：TRAIN_DATA_BASE_PATH / TRAIN_SUBDIR = 训练集。eval_no_verl 的 --val-* 为采样器参数名，
# 会作用在 TRAIN 子目录上，不是项目的 test 验证集。
#
# ============================================================
#  前置步骤：在两个终端分别启动 vLLM 服务
# ============================================================
#
#   终端 1 — Hunyuan-4B-Instruct 策略模型（port 8100，TP=2，只需 2 卡）：
#     CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
#         --model /home/test/test16/chenlu/model/Hunyuan-4B-Instruct \
#         --served-model-name Hunyuan-4B-Instruct \
#         --tensor-parallel-size 2 \
#         --port 8100 \
#         --gpu-memory-utilization 0.90 \
#         --max-model-len 32768
#
#   终端 2 — Qwen3-8B 答题模型（port 8200，TP=4）：
#     CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
#         --model /home/test/test16/chenlu/model/Qwen3-8B \
#         --served-model-name Qwen3-8B \
#         --tensor-parallel-size 4 \
#         --port 8200 \
#         --gpu-memory-utilization 0.90 \
#         --max-model-len 32768
#
# 运行：
#   bash examples/strategy_extraction/scripts/compute_train_baseline_hunyuan4b_instruct.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TRAIN_DATA_BASE_PATH="${TRAIN_DATA_BASE_PATH:-/home/test/test16/chenlu/projects/LLMReflection-czj/data}"
TRAIN_SUBDIR="${TRAIN_SUBDIR:-train_all_project_suitable}"
STRATEGY_MODEL_NAME="${STRATEGY_MODEL_NAME:-Hunyuan-4B-Instruct}"
STRATEGY_MODEL_BASE_URL="${STRATEGY_MODEL_BASE_URL:-http://localhost:8100/v1}"
ANSWER_MODEL_NAME="${ANSWER_MODEL_NAME:-Qwen3-8B}"
ANSWER_MODEL_BASE_URL="${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints_baseline_train_hunyuan4b_instruct}"
VAL_SAMPLES_PER_SUBTASK="${VAL_SAMPLES_PER_SUBTASK:-20}"
OUTPUT="${OUTPUT:-./baseline_cache_hunyuan4b_instruct.json}"

export TRAIN_DATA_BASE_PATH TRAIN_SUBDIR
export STRATEGY_MODEL_NAME STRATEGY_MODEL_BASE_URL
export ANSWER_MODEL_NAME ANSWER_MODEL_BASE_URL
export CHECKPOINT_DIR VAL_SAMPLES_PER_SUBTASK
export NUM_SAMPLES_PER_PROBLEM=1

echo "[INFO] 从训练集子目录读数据: ${TRAIN_DATA_BASE_PATH}/${TRAIN_SUBDIR}"
bash "${SCRIPT_DIR}/compute_train_baseline.sh" "$@"

echo ""
echo "[INFO] 生成 ${OUTPUT} ..."
python "${REPO_ROOT}/examples/strategy_extraction/scripts/build_baseline_cache.py" \
    --input-dir "${CHECKPOINT_DIR}" \
    --output "${OUTPUT}"

echo ""
echo "[DONE] 下一步启动训练："
echo "  BASELINE_CACHE_PATH=${OUTPUT} \\"
echo "  STRATEGY_MODEL_PATH=/home/test/test16/chenlu/model/Hunyuan-4B-Instruct \\"
echo "  bash examples/strategy_extraction/scripts/train_mist.sh"
