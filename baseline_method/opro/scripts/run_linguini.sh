#!/bin/bash
# OPRO baseline — Linguini
#
# Usage:
#   NUM_STEPS=10 bash scripts/run_linguini.sh                        # Exp-1: Qwen3-4B
#   NUM_STEPS=10 ANSWER_MODEL=gemini-3-flash-preview bash scripts/run_linguini.sh  # Exp-2: Gemini
#
# Required:
#   GEMINI_API_KEY   — Gemini API key (no default)
#
# Optional:
#   NUM_STEPS        — 5 (quick) / 10 (default) / 20 (deep)
#   ANSWER_MODEL     — vLLM model name (default: Qwen3-4B)
#   ANSWER_BASE_URL  — vLLM endpoint (default: http://localhost:8200/v1)
#   OPTIMIZER_BASE_URL
#   CONCURRENCY      — concurrent task_type optimizations (default: 4)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Iteration steps ──────────────────────────────────────────────────────────
NUM_STEPS=${NUM_STEPS:-10}

# ── Answer model ─────────────────────────────────────────────────────────────
ANSWER_MODEL=${ANSWER_MODEL:-"Qwen3-4B"}
ANSWER_BASE_URL=${ANSWER_BASE_URL:-"http://localhost:8200/v1"}

# ── Optimizer (fixed Gemini) ─────────────────────────────────────────────────
OPTIMIZER_MODEL="gemini-3-flash-preview"
OPTIMIZER_BASE_URL=${OPTIMIZER_BASE_URL:-"https://generativelanguage.googleapis.com/v1beta/openai"}
GEMINI_API_KEY=${GEMINI_API_KEY:?}

# ── Concurrency ──────────────────────────────────────────────────────────────
CONCURRENCY=${CONCURRENCY:-4}

echo "[INFO] NUM_STEPS=${NUM_STEPS}  ANSWER_MODEL=${ANSWER_MODEL}  CONCURRENCY=${CONCURRENCY}"

python "${SCRIPT_DIR}/../run_opro_linguini.py" \
    --num-steps          "${NUM_STEPS}" \
    --answer-model       "${ANSWER_MODEL}" \
    --answer-base-url    "${ANSWER_BASE_URL}" \
    --optimizer-model    "${OPTIMIZER_MODEL}" \
    --optimizer-base-url "${OPTIMIZER_BASE_URL}" \
    --gemini-api-key     "${GEMINI_API_KEY}" \
    --concurrency        "${CONCURRENCY}" \
    "$@"
