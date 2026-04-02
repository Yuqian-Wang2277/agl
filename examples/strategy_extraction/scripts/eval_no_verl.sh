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
# 可通过环境变量覆盖默认值，例如：
#   STRATEGY_MODEL_BASE_URL=http://localhost:8100/v1 \
#   STRATEGY_MODEL_NAME=Qwen3-4B \
#   ANSWER_MODEL_BASE_URL=http://localhost:8200/v1 \
#   ANSWER_MODEL_NAME=Qwen3-8B \
#   EVAL_CONCURRENCY=64 \
#   bash examples/strategy_extraction/scripts/eval_no_verl.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

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
  --temperature 0 \
  --llm-seed 42 \
  --strategy-no-think \
  --answer-no-think \
  --strategy-prompt-version repetition_controls_2026-04-01 \
  --strategy-repetition-penalty 1.1 \
  --reward-version v3 \
  --answer-request-retries 3 \
  --answer-retry-delay-sec 1.0 \
  "$@"

