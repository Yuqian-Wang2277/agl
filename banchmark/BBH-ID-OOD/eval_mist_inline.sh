#!/bin/bash
# Standalone MIST-inline evaluation for BBH / ID / OOD benchmarks.
#
# Calls a closed-source model API (OpenAI-compatible) in mist-inline mode:
#   Call 1 — strategy extraction (mist_inline_strategy.toml)
#   Call 2 — answer generation   (mist_inline_answer.toml)
#
# No local vLLM required. No agent-lightning training dependencies.
#
# ── Modes ──────────────────────────────────────────────────────────────────
#
#   mist-inline (default)
#     Two API calls per problem:
#       Call 1 — extract 2-layer strategy from few-shot examples
#       Call 2 — apply strategy to answer new problem
#     Prompts: mist_inline_strategy.toml + mist_inline_answer.toml
#
#   few-shot
#     Single API call per problem:
#       The model sees 3 solved examples and directly answers the new problem.
#     Prompt: ICL(few-shot).toml
#
# ── Quick start ────────────────────────────────────────────────────────────
#
#   # MIST-inline (default):
#   OPENAI_API_KEY=sk-...  ANSWER_MODEL_NAME=gpt-4o  bash eval_mist_inline.sh
#
#   # Few-shot:
#   MODE=few-shot  OPENAI_API_KEY=sk-...  ANSWER_MODEL_NAME=gpt-4o  bash eval_mist_inline.sh
#
#   # Anthropic Claude (via OpenAI-compatible proxy / LiteLLM):
#   MODE=mist-inline \
#   ANSWER_MODEL_BASE_URL=https://your-proxy/v1 \
#   ANSWER_MODEL_NAME=claude-opus-4-7 \
#   OPENAI_API_KEY=sk-ant-... \
#   bash eval_mist_inline.sh
#
#   # Deepseek / any OpenAI-compatible endpoint:
#   MODE=few-shot \
#   ANSWER_MODEL_BASE_URL=https://api.deepseek.com/v1 \
#   ANSWER_MODEL_NAME=deepseek-chat \
#   OPENAI_API_KEY=sk-... \
#   bash eval_mist_inline.sh
#
# ── Configuration (env vars, all optional) ────────────────────────────────
#
#   MODE                      mist-inline | few-shot    (default: mist-inline)
#   ANSWER_MODEL_BASE_URL     OpenAI-compatible endpoint
#                             (default: https://api.openai.com/v1)
#   ANSWER_MODEL_NAME         Model name  (default: gpt-4o)
#   OPENAI_API_KEY            API key
#   EVAL_CONCURRENCY          Concurrent async workers  (default: 4)
#   VAL_SAMPLES_PER_SUBTASK   Samples per subtask file  (default: 20)
#   VAL_SAMPLING_SEED         Data sampling seed        (default: 42)
#   TEMPERATURE               Sampling temperature      (default: 0.0)
#   LLM_SEED                  OpenAI request seed       (default: 42)
#   FEWSHOT_K                 Few-shot examples/problem (default: 3)
#   OUTPUT_DIR                Results directory         (default: ./results)
#   MAX_RETRIES               API retries per call      (default: 3)
#
# ── Examples ──────────────────────────────────────────────────────────────
#
#   # Run both modes to compare:
#   MODE=mist-inline  OPENAI_API_KEY=sk-... bash eval_mist_inline.sh
#   MODE=few-shot     OPENAI_API_KEY=sk-... bash eval_mist_inline.sh
#
#   # Pass extra args directly to eval_mist_inline.py:
#   MODE=few-shot bash eval_mist_inline.sh --max-tokens 8192
#
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────
MODE="${MODE:-mist-inline}"
ANSWER_MODEL_BASE_URL="${ANSWER_MODEL_BASE_URL:-https://api.openai.com/v1}"
ANSWER_MODEL_NAME="${ANSWER_MODEL_NAME:-gpt-4o}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-4}"
VAL_SAMPLES_PER_SUBTASK="${VAL_SAMPLES_PER_SUBTASK:-20}"
VAL_SAMPLING_SEED="${VAL_SAMPLING_SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.0}"
LLM_SEED="${LLM_SEED:-42}"
FEWSHOT_K="${FEWSHOT_K:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"
MAX_RETRIES="${MAX_RETRIES:-3}"

if [[ "$MODE" != "mist-inline" && "$MODE" != "few-shot" ]]; then
    echo "[ERROR] MODE must be 'mist-inline' or 'few-shot' (got: $MODE)"
    exit 1
fi

echo "[INFO] mode:              ${MODE}"
echo "[INFO] model:             ${ANSWER_MODEL_NAME}"
echo "[INFO]   endpoint:         ${ANSWER_MODEL_BASE_URL}"
echo "[INFO]   concurrency:      ${EVAL_CONCURRENCY}"
echo "[INFO]   samples/subtask:  ${VAL_SAMPLES_PER_SUBTASK}"
echo "[INFO]   fewshot_k:        ${FEWSHOT_K}"
echo "[INFO]   temperature:      ${TEMPERATURE}"
echo "[INFO]   output_dir:       ${OUTPUT_DIR}"

python "${SCRIPT_DIR}/eval_mist_inline.py" \
    --mode                     "${MODE}" \
    --answer-model-base-url    "${ANSWER_MODEL_BASE_URL}" \
    --answer-model-name        "${ANSWER_MODEL_NAME}" \
    --concurrency              "${EVAL_CONCURRENCY}" \
    --samples-per-subtask      "${VAL_SAMPLES_PER_SUBTASK}" \
    --val-sampling-seed        "${VAL_SAMPLING_SEED}" \
    --temperature              "${TEMPERATURE}" \
    --llm-seed                 "${LLM_SEED}" \
    --fewshot-k                "${FEWSHOT_K}" \
    --output-dir               "${OUTPUT_DIR}" \
    --max-retries              "${MAX_RETRIES}" \
    "$@"
