#!/bin/bash
# Lightweight evaluation entrypoint for StrategyGenerationAgent WITHOUT VERL/Ray.
# This reuses the same data path and model defaults as the training scripts,
# but runs rollouts sequentially via eval_no_verl.py.
#
# Recommended usage:
#   1) 启动固定答案模型（例如 Qwen3-8B）vLLM 服务：
#        CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
#            --model /home/test/test16/chenlu/model/Qwen3-8B \
#            --served-model-name Qwen3-8B \
#            --tensor-parallel-size 4 \
#            --port 8200 \
#            --gpu-memory-utilization 0.90 \
#            --max-model-len 32768
#   2) 可选：单独启动用于策略生成的 Qwen3-4B vLLM 服务（如果不想复用 8200）；
#      否则 eval_no_verl 会默认把 strategy 调用也发到 --answer-model-base-url。
#   3) 本脚本默认使用 LLMReflection 的 data 路径和 3 个验证子集。
#
# 运行示例：
#   CUDA_VISIBLE_DEVICES=0,1 bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   CUDA_VISIBLE_DEVICES=0,1 bash examples/strategy_extraction/scripts/eval_no_verl.sh --max-samples 64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

python -m examples.strategy_extraction.eval_no_verl \
  --data-base-path /home/test/test16/chenlu/projects/LLMReflection/data/ \
  --val-subdirs test-id-subtask test-ood-task test-bbh \
  --model-path /home/test/test16/chenlu/model/Qwen3-4B \
  --answer-model-path /home/test/test16/chenlu/model/Qwen3-8B \
  --answer-model-base-url "${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}" \
  --answer-model-name "${ANSWER_MODEL_NAME:-Qwen3-8B}" \
  "$@"

