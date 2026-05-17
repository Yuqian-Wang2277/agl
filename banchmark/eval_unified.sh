#!/usr/bin/env bash
# Unified evaluation launcher for MATH-500, StrategyQA, and ReClor.
#
# Usage examples:
#   MODE=FS   bash eval_unified.sh
#   MODE=MIST bash eval_unified.sh
#   MODE=FS BENCHMARKS="math500 strategyqa" MAX_PROBLEMS=10 bash eval_unified.sh
#
# Override any variable inline:
#   ANSWER_MODEL=gpt-4o ANSWER_API_BASE=https://api.openai.com/v1 bash eval_unified.sh

set -euo pipefail

# ── Mode ──────────────────────────────────────────────────────────────────────
MODE="${MODE:-FS}"                        # FS | MIST

# ── Benchmarks ────────────────────────────────────────────────────────────────
BENCHMARKS="${BENCHMARKS:-math500 strategyqa reclor}"

# ── Answer model ──────────────────────────────────────────────────────────────
ANSWER_MODEL="${ANSWER_MODEL:-Qwen3-8B}"
ANSWER_API_BASE="${ANSWER_API_BASE:-http://localhost:8200/v1}"

# ── Strategy model (MIST only) ────────────────────────────────────────────────
STRATEGY_MODEL="${STRATEGY_MODEL:-Qwen3-4B}"
STRATEGY_API_BASE="${STRATEGY_API_BASE:-http://localhost:8100/v1}"
STRATEGY_PROMPT_VERSION="${STRATEGY_PROMPT_VERSION:-repetition_controls_2026-04-01}"
# Repetition penalty for strategy model (set ≤0 to disable)
STRATEGY_REP_PENALTY="${STRATEGY_REP_PENALTY:-1.1}"

# ── Few-shot ──────────────────────────────────────────────────────────────────
SHOT_K="${SHOT_K:-3}"
SHOT_SEED="${SHOT_SEED:-42}"

# ── Sampling / pass@k ─────────────────────────────────────────────────────────
NUM_SAMPLES="${NUM_SAMPLES:-3}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-4096}"

# ── Qwen3 no-think flags (set to "--answer-no-think" or "" to toggle) ─────────
ANSWER_NO_THINK="${ANSWER_NO_THINK:-}"
STRATEGY_NO_THINK="${STRATEGY_NO_THINK:-}"

# ── Execution ─────────────────────────────────────────────────────────────────
CONCURRENCY="${CONCURRENCY:-32}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"         # leave empty to evaluate all problems

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/unified}"

# ── Build command ──────────────────────────────────────────────────────────────
CMD=(
  python "${SCRIPT_DIR}/eval_unified.py"
  --mode "${MODE}"
  --benchmarks ${BENCHMARKS}
  --answer-model "${ANSWER_MODEL}"
  --answer-api-base "${ANSWER_API_BASE}"
  --shot-k "${SHOT_K}"
  --shot-seed "${SHOT_SEED}"
  --num-samples "${NUM_SAMPLES}"
  --temperature "${TEMPERATURE}"
  --seed "${SEED}"
  --max-tokens "${MAX_TOKENS}"
  --concurrency "${CONCURRENCY}"
  --output-dir "${OUTPUT_DIR}"
  --save-details
)

if [[ "${MODE}" == "MIST" ]]; then
  CMD+=(
    --strategy-model "${STRATEGY_MODEL}"
    --strategy-api-base "${STRATEGY_API_BASE}"
    --strategy-prompt-version "${STRATEGY_PROMPT_VERSION}"
    --strategy-rep-penalty "${STRATEGY_REP_PENALTY}"
  )
fi

[[ -n "${ANSWER_NO_THINK}" ]]   && CMD+=("--answer-no-think")
[[ -n "${STRATEGY_NO_THINK}" ]] && CMD+=("--strategy-no-think")
[[ -n "${MAX_PROBLEMS}" ]]       && CMD+=(--max-problems "${MAX_PROBLEMS}")

# ── Run ───────────────────────────────────────────────────────────────────────
echo "================================================================"
echo " eval_unified  mode=${MODE}  benchmarks=[${BENCHMARKS}]"
echo " answer_model=${ANSWER_MODEL}  (@${ANSWER_API_BASE})"
if [[ "${MODE}" == "MIST" ]]; then
  echo " strategy_model=${STRATEGY_MODEL}  (@${STRATEGY_API_BASE})"
  echo " strategy_prompt=${STRATEGY_PROMPT_VERSION}  rep_penalty=${STRATEGY_REP_PENALTY}"
fi
echo " shot_k=${SHOT_K}  num_samples=${NUM_SAMPLES}  temperature=${TEMPERATURE}"
echo " output_dir=${OUTPUT_DIR}"
echo "================================================================"

mkdir -p "${OUTPUT_DIR}"

exec "${CMD[@]}"
