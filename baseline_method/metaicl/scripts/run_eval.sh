#!/usr/bin/env bash
# MetaICL unified evaluation launcher.
#
# Starts a vLLM server for the checkpoint, runs eval.py, then shuts down vLLM.
#
# Key environment variables:
#   CHECKPOINT   — path to the checkpoint directory (default: ./checkpoints/metaicl-qwen3-4b/best)
#   MODEL_NAME   — model name to serve (default: metaicl)
#   BENCHMARK    — space-separated benchmarks: id-ood bbh hardmath linguini all
#                  (default: all)
#   VLLM_GPUS    — CUDA devices for vLLM server (default: 0,1)
#   VLLM_PORT    — API port (default: 8300)
#   K_SHOT       — few-shot k (default: 4, must match training)
#   OUTPUT_DIR   — results output directory (default: ./results)
#   CONCURRENCY  — async concurrency (default: 32)
#   MAX_SAMPLES  — cap problems per benchmark (useful for smoke tests)
#   TENSOR_PARALLEL — tensor parallel size for vLLM (default: N_VLLM_GPUS)
#   VLLM_START_TIMEOUT — seconds to wait for /health after launching vLLM (default: 300).
#                        First cold start with TP>1 often needs 2–3+ minutes (compile + cudagraph).
#   SKIP_VLLM=1        — do not start vLLM; only run eval.py against MODEL_URL (existing server).
#
# Examples:
#   bash scripts/run_eval.sh
#   BENCHMARK=id-ood bash scripts/run_eval.sh
#   BENCHMARK="bbh hardmath" bash scripts/run_eval.sh
#   BENCHMARK=all CHECKPOINT=./checkpoints/metaicl-cot-qwen3-4b/best MODEL_NAME=metaicl-cot bash scripts/run_eval.sh
#   MAX_SAMPLES=10 bash scripts/run_eval.sh    # smoke test
#   SKIP_VLLM=1 bash scripts/run_eval.sh       # vLLM already on VLLM_PORT (default 8300)
#
# Environment: conda env 'agl' (has openai, vllm, torch, transformers).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── Parameters ────────────────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-./checkpoints/metaicl-qwen3-4b/best}"
MODEL_NAME="${MODEL_NAME:-metaicl}"
BENCHMARK="${BENCHMARK:-all}"
VLLM_GPUS="${VLLM_GPUS:-0,1}"
VLLM_PORT="${VLLM_PORT:-8300}"
K_SHOT="${K_SHOT:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
CONCURRENCY="${CONCURRENCY:-32}"

N_VLLM_GPUS=$(echo "$VLLM_GPUS" | tr ',' '\n' | wc -l)
TENSOR_PARALLEL="${TENSOR_PARALLEL:-$N_VLLM_GPUS}"

MODEL_URL="http://localhost:${VLLM_PORT}/v1"
SKIP_VLLM="${SKIP_VLLM:-0}"

echo "=== MetaICL evaluation ==="
echo "Checkpoint      : $CHECKPOINT"
echo "Model name      : $MODEL_NAME"
echo "Benchmark(s)    : $BENCHMARK"
echo "vLLM GPUs       : $VLLM_GPUS  (tensor_parallel=$TENSOR_PARALLEL)"
echo "vLLM port       : $VLLM_PORT"
if [[ "$SKIP_VLLM" == "1" ]]; then
    echo "SKIP_VLLM       : 1 (using existing server, not starting vLLM)"
else
    echo "SKIP_VLLM       : 0"
fi
echo "k-shot          : $K_SHOT"
echo "Output dir      : $OUTPUT_DIR"
echo

# ── Verify checkpoint exists (only when we launch vLLM from this path) ────────
if [[ "$SKIP_VLLM" != "1" ]]; then
    if [[ ! -d "$CHECKPOINT" ]]; then
        echo "[ERROR] Checkpoint directory not found: $CHECKPOINT"
        exit 1
    fi
fi

mkdir -p "$OUTPUT_DIR"

if [[ "$SKIP_VLLM" != "1" ]]; then
    # ── Start vLLM server ────────────────────────────────────────────────────
    VLLM_LOG="${OUTPUT_DIR}/vllm_server.log"
    echo "Starting vLLM server (log: $VLLM_LOG) ..."
    CUDA_VISIBLE_DEVICES="$VLLM_GPUS" conda run -n agl --no-capture-output \
        python -m vllm.entrypoints.openai.api_server \
            --model "$CHECKPOINT" \
            --served-model-name "$MODEL_NAME" \
            --tensor-parallel-size "$TENSOR_PARALLEL" \
            --port "$VLLM_PORT" \
            --disable-log-requests \
        > "$VLLM_LOG" 2>&1 &

    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    _cleanup() {
        echo "Stopping vLLM server (PID $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        echo "vLLM stopped."
    }
    trap _cleanup EXIT

    echo "Waiting for vLLM server to be ready on port $VLLM_PORT ..."
    MAX_WAIT="${VLLM_START_TIMEOUT:-300}"
    WAITED=0
    until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
        if [[ $WAITED -ge $MAX_WAIT ]]; then
            echo "[ERROR] vLLM server did not start within ${MAX_WAIT}s. Check $VLLM_LOG"
            exit 1
        fi
        sleep 3
        WAITED=$((WAITED + 3))
    done
    echo "vLLM server ready (waited ${WAITED}s)."
    echo
else
    echo "Checking existing vLLM at $MODEL_URL ..."
    if ! curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[ERROR] No healthy server on port $VLLM_PORT. Start vLLM first or unset SKIP_VLLM."
        exit 1
    fi
    echo "Server OK. Running evaluation only."
    echo
fi

# ── Run evaluation ────────────────────────────────────────────────────────────
# Split BENCHMARK string into separate --benchmark args
read -ra BENCHMARK_ARGS <<< "$BENCHMARK"

conda run -n agl --no-capture-output \
    python eval.py \
        --benchmark "${BENCHMARK_ARGS[@]}" \
        --model-url "$MODEL_URL" \
        --model-name "$MODEL_NAME" \
        --k-shot "$K_SHOT" \
        --output-dir "$OUTPUT_DIR" \
        --concurrency "$CONCURRENCY" \
        ${MAX_SAMPLES:+--max-samples "$MAX_SAMPLES"}

echo
echo "=== Evaluation finished. Results in: $OUTPUT_DIR ==="
