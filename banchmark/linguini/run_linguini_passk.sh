#!/usr/bin/env bash
# Linguini benchmark few-shot pass@1 evaluation — GPT-4 专用。
#
# 评测方式：给 GPT-4 展示 SHOT_NUM 个同 task_type 的示例（含 context/query/answers），
# 让它回答新题，计算 pass@1（全部子答案正确才算通过）。
#
# 用法：
#   export OPENAI_API_KEY=sk-...
#   bash run_linguini_passk.sh
#
#   # 通过环境变量覆盖参数：
#   MODEL=gpt-4-turbo bash run_linguini_passk.sh
#   SHOT_NUM=5 TASK_TYPE=translation bash run_linguini_passk.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
#  Settings
# ============================================================

# ── 模型（GPT-4 系列）────────────────────────────────────────
MODEL="${MODEL:-gpt-4o}"
# API key 优先取环境变量 OPENAI_API_KEY，也可用 API_KEY 覆盖
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
# pass@1 用 greedy decoding（temperature=0）；若需要随机采样可设为 0.7
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
# GPT-4 API 限速缓冲；若遇到 429 可适当调大
SLEEP_TIME="${SLEEP_TIME:-1.0}"

# ── Few-shot 配置 ─────────────────────────────────────────────
# 每题展示 SHOT_NUM 个同 task_type 示例（leave-one-out）
SHOT_NUM="${SHOT_NUM:-3}"
SHOT_SEED="${SHOT_SEED:-42}"

# ── 评测范围 ──────────────────────────────────────────────────
# ALL | translation | fill_blanks | match_letters | text_to_num | num_to_text
TASK_TYPE="${TASK_TYPE:-ALL}"
# 全部子答案正确才算 pass（符合官方评测语义）
PASS_THRESHOLD="${PASS_THRESHOLD:-1.0}"

# ── 输出 ──────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-results/passk}"
OUTPUT_FILE="${OUTPUT_FILE:-}"
PROMPT_DIR="${PROMPT_DIR:-/home/test/test16/chenlu/projects/agent-lightning/examples/strategy_extraction/prompt}"

# ============================================================
#  打印配置摘要
# ============================================================
echo "============================================================"
echo "  Linguini few-shot pass@1 evaluation (GPT-4)"
echo "  Model     : ${MODEL}  (backend=openai  T=${TEMPERATURE})"
echo "  Shot num  : ${SHOT_NUM}  seed=${SHOT_SEED}"
echo "  Task type : ${TASK_TYPE}"
echo "  Pass thresh: ${PASS_THRESHOLD}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "============================================================"

# ============================================================
#  构建 Python 调用参数
# ============================================================
ARGS=(
    --model                  "${MODEL}"
    --mode                   "few-shot"
    --backend                "openai"
    --api_key                "${API_KEY}"
    --temperature            "${TEMPERATURE}"
    --max_tokens             "${MAX_TOKENS}"
    --sleep_time             "${SLEEP_TIME}"
    --shot_num               "${SHOT_NUM}"
    --shot_seed              "${SHOT_SEED}"
    --context_diversity_mode "reproducible"
    --num_samples            1
    --task_type              "${TASK_TYPE}"
    --pass_threshold         "${PASS_THRESHOLD}"
    --output_dir             "${OUTPUT_DIR}"
    --prompt_dir             "${PROMPT_DIR}"
)

[[ -n "${OUTPUT_FILE}" ]] && ARGS+=(--output_file "${OUTPUT_FILE}")

exec python3 run_linguini_passk.py "${ARGS[@]}" "$@"
