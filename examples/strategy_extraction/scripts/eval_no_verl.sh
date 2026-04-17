#!/bin/bash
# Lightweight evaluation entrypoint for StrategyGenerationAgent WITHOUT VERL/Ray.
# This reuses the same data path and model defaults as the training scripts,
# but runs rollouts concurrently via eval_no_verl.py.
#
# 前置步骤：分别在两个终端启动两个 vLLM 服务
#
#   终端 1 — 策略生成模型（Qwen3-4B，端口 8100）：
#     CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
#         --model /home/test/test16/chenlu/model/Qwen3-4B \
#         --served-model-name Qwen3-4B \
#         --tensor-parallel-size 2 \
#         --port 8100 \
#         --gpu-memory-utilization 0.90 \
#         --max-model-len 32768
#
#   策略 vLLM：建议将默认采样设为 repetition_penalty=1.1（具体参数名见 vllm serve --help，随版本而异）。
#   eval_no_verl.py 默认在请求的 extra_body 中发送 repetition_penalty=1.1；传 --strategy-repetition-penalty 0
#   可省略该字段、完全依赖服务端默认。
#
#   终端 2 — 答案生成模型（Qwen3-8B，端口 8200）：
#     CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
#         --model /home/test/test16/chenlu/model/Qwen3-8B \
#         --served-model-name Qwen3-8B \
#         --tensor-parallel-size 4 \
#         --port 8200 \
#         --gpu-memory-utilization 0.90 \
#         --max-model-len 32768
#
# 运行示例：
#   bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   bash examples/strategy_extraction/scripts/eval_no_verl.sh --max-samples 64
#
# pass@k 说明：
#   默认 NUM_SAMPLES_PER_PROBLEM=3，每道题独立推理 3 次（种子依次为 42/43/44），
#   评测结束后输出 pass@1 / pass@2 / pass@3（hard 和 soft 两种指标）。
#   当 num-samples-per-problem > 1 且 --temperature 未指定时，脚本内部自动切换为
#   temperature=0.7 以保证多次采样的多样性；可通过 --temperature 显式覆盖。
#   若只需单次确定性推理，可传 --num-samples-per-problem 1 --temperature 0。
#
# 可通过环境变量覆盖默认值，例如：
#   STRATEGY_MODEL_BASE_URL=http://localhost:8100/v1 \
#   STRATEGY_MODEL_NAME=Qwen3-4B \
#   ANSWER_MODEL_BASE_URL=http://localhost:8200/v1 \
#   ANSWER_MODEL_NAME=Qwen3-8B \
#   EVAL_CONCURRENCY=64 \
#   NUM_SAMPLES_PER_PROBLEM=3 \
#   bash examples/strategy_extraction/scripts/eval_no_verl.sh
#
# 采样模式开关：
#   FULL_DATASET=0（默认）  — 分层采样，每个 subtask JSON 取 VAL_SAMPLES_PER_SUBTASK 条（默认 20）
#   FULL_DATASET=1          — 全量枚举，每道题恰好跑一次（~123k 题，约为默认的 17 倍）
#
# 示例：
#   VAL_SAMPLES_PER_SUBTASK=50 bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   FULL_DATASET=1 bash examples/strategy_extraction/scripts/eval_no_verl.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# --- 采样模式 ---
FULL_DATASET="${FULL_DATASET:-0}"
VAL_SAMPLES_PER_SUBTASK="${VAL_SAMPLES_PER_SUBTASK:-20}"

SAMPLING_MODE_ARGS=()
if [[ "${FULL_DATASET}" == "1" ]]; then
    SAMPLING_MODE_ARGS+=(--val-sampling-mode all)
    echo "[INFO] 采样模式：全量枚举（FULL_DATASET=1）"
else
    SAMPLING_MODE_ARGS+=(--val-sampling-mode per_subtask_fixed --val-samples-per-subtask "${VAL_SAMPLES_PER_SUBTASK}")
    echo "[INFO] 采样模式：分层采样，每 subtask ${VAL_SAMPLES_PER_SUBTASK} 条（FULL_DATASET=0）"
fi

python -m examples.strategy_extraction.eval_no_verl \
  --data-base-path /home/test/test16/chenlu/projects/LLMReflection/data/ \
  --val-subdirs test-id-subtask test-ood-task test-bbh \
  --model-path /home/test/test16/chenlu/model/Qwen3-4B \
  --strategy-model-base-url "${STRATEGY_MODEL_BASE_URL:-http://localhost:8100/v1}" \
  --strategy-model-name "${STRATEGY_MODEL_NAME:-Qwen3-4B}" \
  --answer-model-path /home/test/test16/chenlu/model/Qwen3-4B \
  --answer-model-base-url "${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}" \
  --answer-model-name "${ANSWER_MODEL_NAME:-Qwen3-4B}" \
  --concurrency "${EVAL_CONCURRENCY:-64}" \
  --llm-seed 42 \
  --num-samples-per-problem "${NUM_SAMPLES_PER_PROBLEM:-3}" \
  --strategy-no-think \
  --answer-no-think \
  --strategy-prompt-version repetition_controls_2026-04-01 \
  --strategy-repetition-penalty 1.1 \
  --reward-version v3 \
  --answer-request-retries 3 \
  --answer-retry-delay-sec 1.0 \
  "${SAMPLING_MODE_ARGS[@]}" \
  "$@"

