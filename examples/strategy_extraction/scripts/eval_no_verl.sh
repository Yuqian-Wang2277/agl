#!/bin/bash
# Lightweight evaluation entrypoint for StrategyGenerationAgent WITHOUT VERL/Ray.
# This reuses the same data path and model defaults as the training scripts,
# but runs rollouts concurrently via eval_no_verl.py.
#
# ============================================================
#  模式选择（MODE 环境变量）
# ============================================================
#
#  MODE=few-shot（仅答案模型，ICL in-context learning）
#    使用 prompt: answer_generation/ICL(few-shot).toml
#    只需启动答案模型（Qwen3-8B，端口 8200），无需策略模型。
#    --skip-strategy-generation 跳过策略生成调用。
#    --fewshot-min 3 --fewshot-max 3 控制每道题的 few-shot 示例数。
#
#  MODE=MIST（默认）（策略模型 + 答案模型）
#    使用 prompt: strategy_generation/repetition_controls_2026-04-01.toml
#                 answer_generation/v1.toml
#    需要同时启动策略模型（Qwen3-4B，端口 8100）和答案模型（Qwen3-8B，端口 8200）。
#
#  MODE=mist-inline（单模型 train-free MIST）
#    同一模型先提取两层策略（FIRST_ORDER + SECOND_ORDER），再用策略回答新题。
#    无需单独策略模型，适合闭源 API（GPT-4o、Claude 等）。
#    只需启动答案模型（端口 8200）或配置闭源 API。
#    使用 prompt: answer_generation/mist_inline_strategy.toml（策略提取）
#                 answer_generation/mist_inline_answer.toml（策略作答）
#
# ============================================================
#  重要 setting 速查（实际生效值，含显式参数 + 隐式 default）
# ============================================================
#
#  [temperature & 采样多样性]
#    默认: --temperature 0.0
#    实际: NUM_SAMPLES_PER_PROBLEM=3 > 1 且 temperature==0 → 自动切换为 temperature=0.7
#          (eval_no_verl.py 内部逻辑，确保多次采样输出互不相同)
#    显式覆盖: --temperature 0.7（或其他值）可绕过自动切换
#    单次确定性推理: --num-samples-per-problem 1 --temperature 0
#
#  [pass@k & seeds]
#    NUM_SAMPLES_PER_PROBLEM=3：每道题独立推理 3 次
#    种子: base_seed=42 → 3 次 rollout 依次使用 seed=42, 43, 44
#    输出指标: pass@1 / pass@2 / pass@3（hard 0/1 和 soft F1 各一份）
#
#  [模型 — MIST 模式]
#    策略模型: Qwen3-4B  端口 8100  no-think  repetition_penalty=1.1
#              prompt_version=repetition_controls_2026-04-01
#    答案模型: Qwen3-8B  端口 8200  no-think  answer_prompt_version=v1
#              grounded-proxy-k=1（每道题仅调用 answer model 1 次）
#
#  [模型 — few-shot 模式]
#    答案模型: Qwen3-8B  端口 8200  no-think  answer_prompt_version=ICL(few-shot)
#
#  [数据集]
#    val-subdirs: test-id-subtask + test-ood-task + test-bbh
#    采样模式 (FULL_DATASET=0): per_subtask_fixed，每 subtask 20 条，val-sampling-seed=42
#    采样模式 (FULL_DATASET=1): 全量枚举，~123k 题
#
#  [reward / 评测逻辑]
#    reward-mode=scorer_only，correctness-weight=1.0，format-weight=0.0，scorer-weight=0.0
#    → 纯答案正确率评测，不含 format tag / strategy scorer 评分
#    reward-version=v3，answer-request-retries=3
#
#  [并发]
#    EVAL_CONCURRENCY=64（64 个异步 worker 并发推理）
#
# ============================================================
#  前置步骤
# ============================================================
#
#  [MIST 模式] 分别在两个终端启动两个 vLLM 服务：
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
#  [few-shot / mist-inline 模式] 只需启动答案模型（终端 2，同上）。
#  [mist-inline 闭源 API] 无需本地 vLLM，配置 ANSWER_MODEL_BASE_URL 和 ANSWER_MODEL_NAME 即可。
#
# 运行示例：
#   MODE=MIST         bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   MODE=few-shot     bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   MODE=mist-inline  bash examples/strategy_extraction/scripts/eval_no_verl.sh
#   MODE=few-shot     bash examples/strategy_extraction/scripts/eval_no_verl.sh --max-samples 64
#
# 可通过环境变量覆盖默认值，例如（MIST 模式）：
#   MODE=MIST \
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

# --- 模式选择 ---
MODE="${MODE:-MIST}"
if [[ "$MODE" != "few-shot" && "$MODE" != "MIST" && "$MODE" != "mist-inline" ]]; then
    echo "[ERROR] MODE 必须为 'few-shot'、'MIST' 或 'mist-inline'（当前值: $MODE）"
    exit 1
fi
echo "[INFO] 运行模式: $MODE"

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

# --- 模式专属参数 ---
MODE_ARGS=()
if [[ "$MODE" == "few-shot" ]]; then
    # few-shot：只有一个答案模型，跳过策略生成，使用 ICL prompt
    MODE_ARGS+=(
        --mode few-shot
        --answer-model-base-url "${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
        --answer-model-name "${ANSWER_MODEL_NAME:-Qwen3-8B}"
        --answer-prompt-version "ICL(few-shot)"
    )
elif [[ "$MODE" == "mist-inline" ]]; then
    # mist-inline：单模型，两次调用（策略提取 + 策略作答），无需策略服务
    MODE_ARGS+=(
        --mode mist-inline
        --answer-model-base-url "${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
        --answer-model-name "${ANSWER_MODEL_NAME:-Qwen3-8B}"
        --inline-strategy-prompt-version "${INLINE_STRATEGY_PROMPT_VERSION:-mist_inline_strategy}"
        --answer-prompt-version "${ANSWER_PROMPT_VERSION:-mist_inline_answer}"
    )
else
    # MIST：策略模型（Qwen3-4B）+ 答案模型（Qwen3-8B）
    MODE_ARGS+=(
        --mode MIST
        --model-path /home/test/test16/chenlu/model/Qwen3-4B
        --strategy-model-base-url "${STRATEGY_MODEL_BASE_URL:-http://localhost:8100/v1}"
        --strategy-model-name "${STRATEGY_MODEL_NAME:-Qwen3-4B}"
        --answer-model-path /home/test/test16/chenlu/model/Qwen3-8B
        --answer-model-base-url "${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
        --answer-model-name "${ANSWER_MODEL_NAME:-Qwen3-8B}"
        --strategy-no-think
        --strategy-prompt-version repetition_controls_2026-04-01
        --strategy-repetition-penalty 1.1
        --answer-prompt-version v1
    )
fi

python -m examples.strategy_extraction.eval_no_verl \
  --data-base-path /home/test/test16/chenlu/projects/LLMReflection/data/ \
  --val-subdirs test-id-subtask test-ood-task test-bbh \
  --fewshot-min 3 \
  --fewshot-max 3 \
  --concurrency "${EVAL_CONCURRENCY:-64}" \
  --llm-seed 42 \
  --num-samples-per-problem "${NUM_SAMPLES_PER_PROBLEM:-3}" \
  --answer-no-think \
  --reward-version v3 \
  --answer-request-retries 3 \
  --answer-retry-delay-sec 1.0 \
  "${MODE_ARGS[@]}" \
  "${SAMPLING_MODE_ARGS[@]}" \
  "$@"

