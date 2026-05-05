#!/bin/bash
# HARDMath2 evaluation launch script.
#
# ════════════════════════════════════════════════════════════════════
#  模式 (MODE 环境变量)
# ════════════════════════════════════════════════════════════════════
#
#  MODE=few-shot  — 仅答案模型 (ICL)
#    answer model 学习同类型 few-shot 示例后直接回答目标题目。
#    只需启动答案模型（端口 8200）。
#    prompt: answer_generation/hardmath_few_shot.toml
#
#  MODE=MIST (默认)  — 策略模型 + 答案模型
#    strategy model 从 few-shot 示例中提取策略，
#    answer model 将策略应用到新题上生成答案。
#    需同时启动策略模型（端口 8100）和答案模型（端口 8200）。
#    prompt: strategy_generation/repetition_controls_2026-04-01.toml
#            answer_generation/v1.toml
#
#  MODE=mist-inline  — 单模型 train-free MIST
#    同一模型先提取两层策略（FIRST_ORDER + SECOND_ORDER），
#    再用策略回答新题。无需单独策略模型，适合闭源 API。
#    只需启动答案模型（端口 8200）。
#    prompt: answer_generation/mist.toml
#            answer_generation/mist_inline_answer.toml
#
# ════════════════════════════════════════════════════════════════════
#  前置步骤：启动 vLLM 服务
# ════════════════════════════════════════════════════════════════════
#
#  [MIST 模式] 终端 1 — 策略生成模型 (Qwen3-4B，端口 8100)：
#    CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
#        --model /home/test/test16/chenlu/model/Qwen3-4B \
#        --served-model-name Qwen3-4B \
#        --tensor-parallel-size 2 --port 8100 \
#        --gpu-memory-utilization 0.90 --max-model-len 32768
#
#  [三种模式] 终端 2 — 答案生成模型 (Qwen3-8B，端口 8200)：
#    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
#        --model /home/test/test16/chenlu/model/Qwen3-8B \
#        --served-model-name Qwen3-8B \
#        --tensor-parallel-size 4 --port 8200 \
#        --gpu-memory-utilization 0.90 --max-model-len 32768
#
# ════════════════════════════════════════════════════════════════════
#  运行示例
# ════════════════════════════════════════════════════════════════════
#
#   # 默认 MIST 模式，完整数据集，pass@1/2/3
#   MODE=MIST bash eval_hardmath.sh
#
#   # few-shot 模式
#   MODE=few-shot bash eval_hardmath.sh
#
#   # mist-inline 模式（单闭源模型，无需策略服务）
#   MODE=mist-inline bash eval_hardmath.sh
#
#   # 指定自定义模型（模型名须与 vLLM --served-model-name 一致）
#   MODE=MIST \
#   STRATEGY_MODEL_NAME=Qwen3-4B \
#   ANSWER_MODEL_NAME=Qwen3-14B \
#   ANSWER_MODEL_BASE_URL=http://localhost:8300/v1 \
#   bash eval_hardmath.sh
#
#   # 快速冒烟测试（仅 10 道题，单次采样）
#   MODE=few-shot MAX_SAMPLES=10 NUM_SAMPLES=1 bash eval_hardmath.sh
#
#   # 自定义 few-shot 数量
#   FEWSHOT_K=5 MODE=MIST bash eval_hardmath.sh
#
#   # 传递额外参数直接到 eval_hardmath.py
#   MODE=few-shot bash eval_hardmath.sh --save-details
#
# ════════════════════════════════════════════════════════════════════
#  环境变量速查（均有默认值，无需全部设置）
# ════════════════════════════════════════════════════════════════════
#
#  MODE                       few-shot | MIST | mist-inline（默认 MIST）
#  ANSWER_MODEL_NAME          答案模型名（默认 Qwen3-8B）
#  ANSWER_MODEL_BASE_URL      答案模型 URL（默认 http://localhost:8200/v1）
#  STRATEGY_MODEL_NAME        策略模型名，仅 MIST（默认 Qwen3-4B）
#  STRATEGY_MODEL_BASE_URL    策略模型 URL，仅 MIST（默认 http://localhost:8100/v1）
#  INLINE_STRATEGY_PROMPT_VERSION  策略提取 prompt 版本，仅 mist-inline
#                             （默认 mist）
#  FEWSHOT_K                  few-shot 示例数（默认 3）
#  NUM_SAMPLES                每道题采样次数，用于 pass@k（默认 3）
#  CONCURRENCY                并发 LLM 调用数（默认 32）
#  MAX_SAMPLES                限制评测题目数，空则全量（211 道题）
#  ANSWER_PROMPT_VERSION      答案 prompt 版本（few-shot 默认 hardmath_few_shot，
#                             MIST 默认 v1，mist-inline 默认 mist_inline_answer）
#  STRATEGY_PROMPT_VERSION    策略 prompt 版本（默认 repetition_controls_2026-04-01）
#  OUTPUT_DIR                 结果保存目录（默认 ./results）
#  DATA_DIR                   HARDMath2 data 目录（默认 ./data）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 模式检查 ─────────────────────────────────────────────────────────────────
MODE="${MODE:-MIST}"
if [[ "$MODE" != "few-shot" && "$MODE" != "MIST" && "$MODE" != "mist-inline" ]]; then
    echo "[ERROR] MODE 必须为 'few-shot'、'MIST' 或 'mist-inline'（当前值: $MODE）"
    exit 1
fi
echo "[INFO] 运行模式: $MODE"

# ── 模型配置 ──────────────────────────────────────────────────────────────────
ANSWER_MODEL_NAME="${ANSWER_MODEL_NAME:-Qwen3-8B}"
ANSWER_MODEL_BASE_URL="${ANSWER_MODEL_BASE_URL:-http://localhost:8200/v1}"
STRATEGY_MODEL_NAME="${STRATEGY_MODEL_NAME:-Qwen3-4B}"
STRATEGY_MODEL_BASE_URL="${STRATEGY_MODEL_BASE_URL:-http://localhost:8100/v1}"

# ── 评测参数 ──────────────────────────────────────────────────────────────────
FEWSHOT_K="${FEWSHOT_K:-3}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
CONCURRENCY="${CONCURRENCY:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"

# ── Prompt 版本 ───────────────────────────────────────────────────────────────
if [[ "$MODE" == "few-shot" ]]; then
    ANSWER_PROMPT_VERSION="${ANSWER_PROMPT_VERSION:-hardmath_few_shot}"
elif [[ "$MODE" == "mist-inline" ]]; then
    ANSWER_PROMPT_VERSION="${ANSWER_PROMPT_VERSION:-mist_inline_answer}"
else
    ANSWER_PROMPT_VERSION="${ANSWER_PROMPT_VERSION:-v1}"
fi
STRATEGY_PROMPT_VERSION="${STRATEGY_PROMPT_VERSION:-repetition_controls_2026-04-01}"
INLINE_STRATEGY_PROMPT_VERSION="${INLINE_STRATEGY_PROMPT_VERSION:-mist}"

# ── 构建模式专属参数 ───────────────────────────────────────────────────────────
MODE_ARGS=()
if [[ "$MODE" == "few-shot" ]]; then
    MODE_ARGS+=(
        --mode few-shot
        --answer-model-name "$ANSWER_MODEL_NAME"
        --answer-model-base-url "$ANSWER_MODEL_BASE_URL"
        --answer-prompt-version "$ANSWER_PROMPT_VERSION"
    )
    echo "[INFO] 答案模型: $ANSWER_MODEL_NAME @ $ANSWER_MODEL_BASE_URL"
    echo "[INFO] 答案 prompt: $ANSWER_PROMPT_VERSION"
elif [[ "$MODE" == "mist-inline" ]]; then
    MODE_ARGS+=(
        --mode mist-inline
        --answer-model-name "$ANSWER_MODEL_NAME"
        --answer-model-base-url "$ANSWER_MODEL_BASE_URL"
        --inline-strategy-prompt-version "$INLINE_STRATEGY_PROMPT_VERSION"
        --answer-prompt-version "$ANSWER_PROMPT_VERSION"
        --answer-no-think
    )
    echo "[INFO] 答案模型: $ANSWER_MODEL_NAME @ $ANSWER_MODEL_BASE_URL"
    echo "[INFO] 策略 prompt: $INLINE_STRATEGY_PROMPT_VERSION"
    echo "[INFO] 答案 prompt: $ANSWER_PROMPT_VERSION"
else
    MODE_ARGS+=(
        --mode MIST
        --strategy-model-name "$STRATEGY_MODEL_NAME"
        --strategy-model-base-url "$STRATEGY_MODEL_BASE_URL"
        --answer-model-name "$ANSWER_MODEL_NAME"
        --answer-model-base-url "$ANSWER_MODEL_BASE_URL"
        --strategy-prompt-version "$STRATEGY_PROMPT_VERSION"
        --answer-prompt-version "$ANSWER_PROMPT_VERSION"
        --strategy-repetition-penalty 1.1
    )
    echo "[INFO] 策略模型: $STRATEGY_MODEL_NAME @ $STRATEGY_MODEL_BASE_URL"
    echo "[INFO] 答案模型:  $ANSWER_MODEL_NAME @ $ANSWER_MODEL_BASE_URL"
    echo "[INFO] 策略 prompt: $STRATEGY_PROMPT_VERSION"
    echo "[INFO] 答案 prompt:  $ANSWER_PROMPT_VERSION"
fi

# ── 可选参数 ──────────────────────────────────────────────────────────────────
OPTIONAL_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    OPTIONAL_ARGS+=(--max-samples "$MAX_SAMPLES")
    echo "[INFO] 题目上限: $MAX_SAMPLES"
fi

echo "[INFO] Few-shot k:    $FEWSHOT_K"
echo "[INFO] 采样次数/题:   $NUM_SAMPLES (pass@1/2/3)"
echo "[INFO] 并发数:        $CONCURRENCY"
echo "[INFO] 结果保存:      $OUTPUT_DIR"
echo ""

python "$SCRIPT_DIR/eval_hardmath.py" \
    --fewshot-k "$FEWSHOT_K" \
    --num-samples "$NUM_SAMPLES" \
    --concurrency "$CONCURRENCY" \
    --output-dir "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" \
    "${MODE_ARGS[@]}" \
    "${OPTIONAL_ARGS[@]}" \
    "$@"
