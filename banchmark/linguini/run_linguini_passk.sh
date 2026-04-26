#!/usr/bin/env bash
# Linguini benchmark pass@k evaluation launch script.
#
# ════════════════════════════════════════════════════════════════════
#  模式 (MODE 环境变量)
# ════════════════════════════════════════════════════════════════════
#
#  MODE=few-shot  — 仅答案模型 (ICL)
#    answer model 看同类型 SHOT_NUM 个 few-shot 示例后直接回答新题。
#    只需单一模型 API（--backend openai 或 openai_compatible）。
#    prompt: answer_generation/linguini-few-shot.toml
#
#  MODE=MIST      — 策略模型 + 答案模型
#    strategy model 从 few-shot 示例提取策略，
#    answer model 将策略应用到新题上。
#    需要策略模型（STRATEGY_MODEL/STRATEGY_API_BASE）和答案模型。
#    prompt: strategy_generation/repetition_controls_2026-04-01.toml
#            answer_generation/v1.toml
#
#  MODE=mist-inline  — 单模型 train-free MIST（默认）
#    同一模型先提取两层策略（FIRST_ORDER + SECOND_ORDER），
#    再用策略回答新题。无需单独策略模型，适合闭源 API。
#    prompt: answer_generation/mist_inline_strategy.toml
#            answer_generation/mist_inline_answer.toml
#
# ════════════════════════════════════════════════════════════════════
#  运行示例
# ════════════════════════════════════════════════════════════════════
#
#   # mist-inline 模式（默认），GPT-4o
#   export OPENAI_API_KEY=sk-...
#   bash run_linguini_passk.sh
#
#   # few-shot 模式
#   MODE=few-shot bash run_linguini_passk.sh
#
#   # MIST 模式（策略模型用 vLLM，答案模型用 OpenAI）
#   MODE=MIST \
#   STRATEGY_MODEL=Qwen3-4B \
#   STRATEGY_API_BASE=http://localhost:8100/v1 \
#   BACKEND=openai_compatible \
#   MODEL=Qwen3-8B \
#   API_BASE=http://localhost:8200/v1 \
#   bash run_linguini_passk.sh
#
#   # 只评特定 task_type
#   TASK_TYPE=translation bash run_linguini_passk.sh
#
#   # pass@3（3 次独立采样）
#   NUM_SAMPLES=3 TEMPERATURE=0.7 bash run_linguini_passk.sh
#
# ════════════════════════════════════════════════════════════════════
#  环境变量速查（均有默认值）
# ════════════════════════════════════════════════════════════════════
#
#  MODE                     few-shot | MIST | mist-inline（默认 mist-inline）
#  MODEL                    答案模型名（默认 gpt-4o）
#  BACKEND                  openai | openrouter | openai_compatible | ollama（默认 openai）
#  API_KEY                  答案模型 API key（默认 $OPENAI_API_KEY）
#  API_BASE                 openai_compatible 模式下的 base URL
#  TEMPERATURE              采样温度（默认 0.7）
#  MAX_TOKENS               最大生成 token 数（默认 4096）
#  SLEEP_TIME               请求间隔秒（默认 0.5）
#  STRATEGY_MODEL           策略模型名，仅 MIST（默认与 MODEL 相同）
#  STRATEGY_API_BASE        策略模型 URL，仅 MIST
#  STRATEGY_API_KEY         策略模型 API key，仅 MIST
#  STRATEGY_TEMPERATURE     策略模型采样温度（默认 0.7）
#  SHOT_NUM                 few-shot 示例数（默认 3）
#  SHOT_SEED                采样随机种子（默认 42）
#  NUM_SAMPLES              每题独立采样次数，用于 pass@k（默认 3）
#  PASS_THRESHOLD           pass 阈值（默认 1.0）
#  TASK_TYPE                ALL | translation | fill_blanks | match_letters |
#                           text_to_num | num_to_text（默认 ALL）
#  OUTPUT_DIR               结果目录（默认 results/passk）
#  OUTPUT_FILE              输出文件名（默认自动生成）
#  PROMPT_DIR               TOML prompt 目录

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 模式检查 ─────────────────────────────────────────────────────────────────
MODE="${MODE:-mist-inline}"
if [[ "$MODE" != "few-shot" && "$MODE" != "MIST" && "$MODE" != "mist-inline" ]]; then
    echo "[ERROR] MODE 必须为 'few-shot'、'MIST' 或 'mist-inline'（当前值: $MODE）"
    exit 1
fi
echo "[INFO] 运行模式: $MODE"

# ── 模型配置 ──────────────────────────────────────────────────────────────────
MODEL="${MODEL:-gpt-4o}"
BACKEND="${BACKEND:-openai}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
API_BASE="${API_BASE:-}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
SLEEP_TIME="${SLEEP_TIME:-0.5}"

STRATEGY_MODEL="${STRATEGY_MODEL:-}"
STRATEGY_API_BASE="${STRATEGY_API_BASE:-}"
STRATEGY_API_KEY="${STRATEGY_API_KEY:-}"
STRATEGY_TEMPERATURE="${STRATEGY_TEMPERATURE:-0.7}"

# ── 评测参数 ──────────────────────────────────────────────────────────────────
SHOT_NUM="${SHOT_NUM:-3}"
SHOT_SEED="${SHOT_SEED:-42}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
PASS_THRESHOLD="${PASS_THRESHOLD:-1.0}"
TASK_TYPE="${TASK_TYPE:-ALL}"

# ── 输出 / prompt 路径 ────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-results/passk}"
OUTPUT_FILE="${OUTPUT_FILE:-}"
PROMPT_DIR="${PROMPT_DIR:-/home/test/test16/chenlu/projects/agent-lightning/examples/strategy_extraction/prompt}"

# ── 打印配置摘要 ───────────────────────────────────────────────────────────────
echo "[INFO] 答案模型: $MODEL  (backend=$BACKEND, T=$TEMPERATURE)"
if [[ "$MODE" == "MIST" ]]; then
    SM="${STRATEGY_MODEL:-$MODEL}"
    echo "[INFO] 策略模型: $SM  (T=$STRATEGY_TEMPERATURE)"
fi
echo "[INFO] Shot num:  $SHOT_NUM  seed=$SHOT_SEED"
echo "[INFO] 采样次数:  $NUM_SAMPLES (pass@1/2/3)"
echo "[INFO] Task type: $TASK_TYPE"
echo "[INFO] 结果保存:  $OUTPUT_DIR"
echo ""

# ── 构建参数数组 ───────────────────────────────────────────────────────────────
ARGS=(
    --model       "${MODEL}"
    --mode        "${MODE}"
    --backend     "${BACKEND}"
    --temperature "${TEMPERATURE}"
    --max_tokens  "${MAX_TOKENS}"
    --sleep_time  "${SLEEP_TIME}"
    --shot_num    "${SHOT_NUM}"
    --shot_seed   "${SHOT_SEED}"
    --context_diversity_mode "diverse"
    --num_samples "${NUM_SAMPLES}"
    --pass_threshold "${PASS_THRESHOLD}"
    --task_type   "${TASK_TYPE}"
    --output_dir  "${OUTPUT_DIR}"
    --prompt_dir  "${PROMPT_DIR}"
)

[[ -n "${API_KEY}" ]]     && ARGS+=(--api_key  "${API_KEY}")
[[ -n "${API_BASE}" ]]    && ARGS+=(--api_base "${API_BASE}")
[[ -n "${OUTPUT_FILE}" ]] && ARGS+=(--output_file "${OUTPUT_FILE}")

if [[ "$MODE" == "MIST" ]]; then
    [[ -n "${STRATEGY_MODEL}" ]]       && ARGS+=(--strategy_model       "${STRATEGY_MODEL}")
    [[ -n "${STRATEGY_API_BASE}" ]]    && ARGS+=(--strategy_api_base    "${STRATEGY_API_BASE}")
    [[ -n "${STRATEGY_API_KEY}" ]]     && ARGS+=(--strategy_api_key     "${STRATEGY_API_KEY}")
    ARGS+=(--strategy_temperature "${STRATEGY_TEMPERATURE}")
fi

exec python3 run_linguini_passk.py "${ARGS[@]}" "$@"
