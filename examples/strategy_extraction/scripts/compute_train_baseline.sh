#!/bin/bash
# 训练集基线分计算脚本（一次性离线运行）
#
# 使用策略模型 + Qwen3-8B 答题模型，对 TRAIN_DATA_BASE_PATH/TRAIN_SUBDIR（默认
# train_all_project_suitable）做分层/全量推理，记录 soft 分，供 MIST 基线 / R_delta 使用。
#
# 注意：内部调用 eval_no_verl，其 CLI 沿用 --val-subdirs、--val-sampling-mode 等名称；
# 这些参数表示「要评测的数据子目录 + 子任务定 quota 的采样器」，**不一定**是验证集。
# 本脚本显式把 TRAIN_SUBDIR 传给 --val-subdirs，即始终在「训练分片」上跑。
#
#（旧注释）之前示例曾写每题 3 次以观察 pass@k；NUM_SAMPLES_PER_PROBLEM 现常用 1，
# 仅当需要统计 build_baseline_cache 里的 pass@* 打印时再设大。
#
# 输出：./checkpoints_baseline_train/ 下的 validation_step*.json
# 后续运行 build_baseline_cache.py 将其转换为 baseline_cache.json。
#
# ============================================================
#  前置步骤：分别在两个终端启动 vLLM 服务
# ============================================================
#
#   终端 1 — SFT 4B 策略模型（port 8100，TP=2）：
#     CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
#         --model /home/test/test16/chenlu/projects/fs/sft/sft_output/train_qwen4b_trainall_sig/1/checkpoint-1000_merged_hf \
#         --served-model-name SFT-4B \
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
#   验证服务已就绪：
#     curl -s http://localhost:8100/v1/models | python -m json.tool
#     curl -s http://localhost:8200/v1/models | python -m json.tool
#
# 运行示例：
#   bash examples/strategy_extraction/scripts/compute_train_baseline.sh
#
# 可通过环境变量覆盖默认值：
#   STRATEGY_MODEL_BASE_URL=http://localhost:8100/v1 \
#   STRATEGY_MODEL_NAME=SFT-4B \
#   ANSWER_MODEL_BASE_URL=http://localhost:8200/v1 \
#   ANSWER_MODEL_NAME=Qwen3-8B \
#   EVAL_CONCURRENCY=64 \
#   NUM_SAMPLES_PER_PROBLEM=3 \
#   bash examples/strategy_extraction/scripts/compute_train_baseline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# --- 可覆盖的默认值 ---
STRATEGY_MODEL_BASE_URL="${STRATEGY_MODEL_BASE_URL:-http://localhost:8100/v1}"
STRATEGY_MODEL_NAME="${STRATEGY_MODEL_NAME:-SFT-4B}"
ANSWER_MODEL_BASE_URL="${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
ANSWER_MODEL_NAME="${ANSWER_MODEL_NAME:-Qwen3-8B}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-64}"
NUM_SAMPLES_PER_PROBLEM="${NUM_SAMPLES_PER_PROBLEM:-3}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints_baseline_train}"

# 训练数据路径（与训练脚本保持一致）
TRAIN_DATA_BASE_PATH="${TRAIN_DATA_BASE_PATH:-/home/test/test16/chenlu/projects/LLMReflection-czj/data}"
TRAIN_SUBDIR="${TRAIN_SUBDIR:-train_all_project_suitable}"

# --- 采样模式开关 ---
# FULL_DATASET=0（默认）：分层采样，每 subtask 取 VAL_SAMPLES_PER_SUBTASK 条（默认 20）
# FULL_DATASET=1        ：全量枚举，每道题恰好跑一次（~123k 题）
FULL_DATASET="${FULL_DATASET:-0}"
VAL_SAMPLES_PER_SUBTASK="${VAL_SAMPLES_PER_SUBTASK:-20}"

SAMPLING_MODE_ARGS=()
if [[ "${FULL_DATASET}" == "1" ]]; then
    SAMPLING_MODE_ARGS+=(--val-sampling-mode all)
    SAMPLING_DESC="全量枚举 (~123k 题)"
else
    SAMPLING_MODE_ARGS+=(--val-sampling-mode per_subtask_fixed --val-samples-per-subtask "${VAL_SAMPLES_PER_SUBTASK}")
    SAMPLING_DESC="分层采样，每 subtask ${VAL_SAMPLES_PER_SUBTASK} 条"
fi

echo "========================================="
echo " 训练集基线分计算 (SFT-4B + 8B)"
echo "========================================="
echo " 训练数据 : ${TRAIN_DATA_BASE_PATH}/${TRAIN_SUBDIR}"
echo " 策略模型 : ${STRATEGY_MODEL_BASE_URL} (${STRATEGY_MODEL_NAME})"
echo " 答题模型 : ${ANSWER_MODEL_BASE_URL} (${ANSWER_MODEL_NAME})"
echo " 并发数   : ${EVAL_CONCURRENCY}"
echo " 每题采样 : ${NUM_SAMPLES_PER_PROBLEM} 次 (pass@1/2/${NUM_SAMPLES_PER_PROBLEM})"
echo " 采样模式 : ${SAMPLING_DESC}"
echo " 输出目录 : ${CHECKPOINT_DIR}"
echo "========================================="
echo ""

# 检查 vLLM 服务可达性
python - <<PY
import sys, urllib.request, json

for name, url in [
    ("策略模型 (SFT-4B)", "${STRATEGY_MODEL_BASE_URL}/models"),
    ("答题模型 (8B)",     "${ANSWER_MODEL_BASE_URL}/models"),
]:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        models = [m.get("id", "?") for m in data.get("data", [])]
        print(f"[OK] {name} — models: {models}")
    except Exception as e:
        print(f"[ERROR] 无法连接 {name} ({url}): {e}")
        print("       请先在对应终端启动 vLLM 服务（见脚本头注释）。")
        sys.exit(1)
PY

echo ""
echo "[INFO] 开始推理..."

python -m examples.strategy_extraction.eval_no_verl \
  --data-base-path "${TRAIN_DATA_BASE_PATH}" \
  --val-subdirs "${TRAIN_SUBDIR}" \
  --strategy-model-base-url "${STRATEGY_MODEL_BASE_URL}" \
  --strategy-model-name "${STRATEGY_MODEL_NAME}" \
  --answer-model-base-url "${ANSWER_MODEL_BASE_URL}" \
  --answer-model-name "${ANSWER_MODEL_NAME}" \
  --concurrency "${EVAL_CONCURRENCY}" \
  --llm-seed 42 \
  --num-samples-per-problem "${NUM_SAMPLES_PER_PROBLEM}" \
  --strategy-no-think \
  --answer-no-think \
  --strategy-prompt-version repetition_controls_2026-04-01 \
  --strategy-repetition-penalty 1.1 \
  --reward-version v3 \
  --answer-request-retries 3 \
  --answer-retry-delay-sec 1.0 \
  --output-dir "${CHECKPOINT_DIR}" \
  "${SAMPLING_MODE_ARGS[@]}" \
  "$@"

echo ""
echo "[DONE] 推理完成，结果保存在 ${CHECKPOINT_DIR}/"
echo "       运行以下命令生成 baseline_cache.json："
echo "       python examples/strategy_extraction/scripts/build_baseline_cache.py \\"
echo "           --input-dir ${CHECKPOINT_DIR} \\"
echo "           --output ./baseline_cache.json"
