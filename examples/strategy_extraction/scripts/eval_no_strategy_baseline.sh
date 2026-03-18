#!/bin/bash
# Evaluate no-strategy baseline with reward v3 accuracy routing.
#
# Key settings:
# - Validation sampling: fixed 20 per subtask json (57 subtasks total => 1140 samples)
# - Seed: 42 for reproducibility
# - No-strategy baseline: skip strategy generation, answer directly with Qwen3-8B
# - Accuracy metric path: reward v3 (type-routed answer judging)
# - val_only: run validation and exit before any training update
# - Answer model: Qwen3-8B on 4 GPUs (think mode disabled)
#
# Safety checks:
# - Ensure nvidia-smi is available
# - Ensure --n-gpus does not exceed visible GPU count
# - Ensure each visible GPU has at least MIN_FREE_MEM_MB free memory
#
# ============================================================
#  Recommended: Multi-terminal setup (faster, no GPU contention)
# ============================================================
#
#   Terminal 1 — start dedicated answer model server (Qwen3-8B, GPUs 4-7, think disabled):
#     CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
#         --model /home/test/test16/chenlu/model/Qwen3-8B \
#         --served-model-name Qwen3-8B \
#         --tensor-parallel-size 4 \
#         --port 8200 \
#         --gpu-memory-utilization 0.90 \
#         --max-model-len 32768
#
#   Verify it is ready:
#     curl -s http://localhost:8200/v1/models | python -m json.tool
#
#   Terminal 2 — run this eval (GPUs 0-3 for veRL model, GPUs 4-7 for answer server):
#     CUDA_VISIBLE_DEVICES=0,1,2,3 N_GPUS=4 \
#       ANSWER_MODEL_BASE_URL=http://localhost:8200/v1 \
#       ANSWER_MODEL_NAME=Qwen3-8B \
#       bash scripts/eval_no_strategy_baseline.sh
#
# Usage (single-terminal fallback, slower due to GPU contention):
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 N_GPUS=4 bash scripts/eval_no_strategy_baseline.sh

set -euo pipefail
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export PYTHONUNBUFFERED=1
# Optional rollout diagnostics for baseline trace/reward path.
export AGL_DEBUG_BASELINE="${AGL_DEBUG_BASELINE:-0}"
# Isolate Ray temp artifacts per user to avoid stale permission issues in /tmp/ray.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER}}"

cleanup_ray_state() {
    echo "[INFO] Cleaning stale Ray state..."
    ray stop >/dev/null 2>&1 || true
    mkdir -p "$RAY_TMPDIR"
    chmod 700 "$RAY_TMPDIR" || true
    rm -rf "$RAY_TMPDIR"/session_* "$RAY_TMPDIR"/runtime_resources "$RAY_TMPDIR"/logs || true
    rm -f "$RAY_TMPDIR"/prom_metrics_service_discovery.json "$RAY_TMPDIR"/tmp_prom_metrics_service_discovery.json || true
    echo "[INFO] Ray temp dir: $RAY_TMPDIR"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

if [[ "${CLEAN_RAY_STATE:-1}" == "1" ]]; then
    cleanup_ray_state
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found. Please run on a GPU-enabled node."
    exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[ERROR] CUDA_VISIBLE_DEVICES is empty."
    echo "        Example: CUDA_VISIBLE_DEVICES=0,1,2,3 N_GPUS=4 bash scripts/eval_no_strategy_baseline.sh"
    exit 1
fi

VISIBLE_GPU_COUNT="$(python - <<'PY'
import os
visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
print(len(visible))
PY
)"
N_GPUS="${N_GPUS:-4}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-10000}"
# Dedicated answer model server (recommended: start separately in another terminal).
# If empty, falls back to the VERL vLLM (slow, GPU contention).
ANSWER_MODEL_BASE_URL="${ANSWER_MODEL_BASE_URL:-}"
ANSWER_MODEL_NAME="${ANSWER_MODEL_NAME:-Qwen3-8B}"
# Throughput knobs for validation.
N_RUNNERS="${N_RUNNERS:-$((N_GPUS * 2))}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
FORMAL_VAL_SAMPLES_PER_SUBTASK="${FORMAL_VAL_SAMPLES_PER_SUBTASK:-20}"
FORMAL_MIN_VAL_TRACE_COUNT="${FORMAL_MIN_VAL_TRACE_COUNT:-64}"
FORMAL_MIN_VAL_FIRST_BATCH_RATIO="${FORMAL_MIN_VAL_FIRST_BATCH_RATIO:-0.5}"
FORMAL_MAX_RETRIES="${FORMAL_MAX_RETRIES:-2}"
ACTOR_LOOKUP_FAIL_THRESHOLD="${ACTOR_LOOKUP_FAIL_THRESHOLD:-20}"

if (( N_GPUS > VISIBLE_GPU_COUNT )); then
    echo "[ERROR] N_GPUS=$N_GPUS exceeds visible GPU count=$VISIBLE_GPU_COUNT (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
    exit 1
fi

python - <<'PY'
import os
import subprocess
import sys

visible = [x.strip() for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
min_free = int(os.environ.get("MIN_FREE_MEM_MB", "10000"))

try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
except Exception as e:
    print(f"[ERROR] Failed to query GPU memory: {e}")
    sys.exit(1)

free_map = {}
for line in out.strip().splitlines():
    if not line.strip():
        continue
    idx, mem = [x.strip() for x in line.split(",")]
    free_map[idx] = int(mem)

bad = [(idx, free_map.get(idx, -1)) for idx in visible if free_map.get(idx, -1) < min_free]
if bad:
    print("[ERROR] Not enough free memory on visible GPUs:")
    for idx, mem in bad:
        print(f"  - GPU {idx}: free={mem} MiB < required={min_free} MiB")
    print("Hint: stop conflicting vLLM/Ray jobs, or lower CUDA_VISIBLE_DEVICES/N_GPUS.")
    sys.exit(2)

print(f"[INFO] GPU precheck passed: visible={len(visible)}, min_free_mem={min_free} MiB")
PY

echo "[INFO] Launching no-strategy baseline with N_GPUS=$N_GPUS (visible=$CUDA_VISIBLE_DEVICES)"
echo "[INFO] Throughput config: N_RUNNERS=$N_RUNNERS, VAL_BATCH_SIZE=$VAL_BATCH_SIZE"
echo "[INFO] Formal guardrail: max_retries=$FORMAL_MAX_RETRIES, actor_lookup_fail_threshold=$ACTOR_LOOKUP_FAIL_THRESHOLD, min_val_trace_count=$FORMAL_MIN_VAL_TRACE_COUNT, min_val_first_batch_ratio=$FORMAL_MIN_VAL_FIRST_BATCH_RATIO"
echo "[INFO] Baseline debug: AGL_DEBUG_BASELINE=$AGL_DEBUG_BASELINE"

if [[ -n "$ANSWER_MODEL_BASE_URL" ]]; then
    echo "[INFO] Answer model server: $ANSWER_MODEL_BASE_URL (model=$ANSWER_MODEL_NAME)"
    # Health-check the answer server before starting.
    python - <<PY
import sys, urllib.request, json
url = "${ANSWER_MODEL_BASE_URL}/models"
try:
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())
    models = [m.get("id","?") for m in data.get("data", [])]
    print(f"[INFO] Answer server healthy at ${ANSWER_MODEL_BASE_URL} — models: {models}")
except Exception as e:
    print(f"[ERROR] Cannot reach answer server at ${ANSWER_MODEL_BASE_URL}: {e}")
    print("        Start it first (see script header for the command), then re-run.")
    sys.exit(1)
PY
else
    echo "[WARN] ANSWER_MODEL_BASE_URL is not set — answer calls will use the VERL vLLM (slower)."
    echo "       See script header for how to start a dedicated answer server."
fi

echo "[INFO] Stage 1/1: formal run (1140 validation samples target) ..."
FORMAL_LOG_DIR="./checkpoints_strategy_gen_no_strategy_baseline/formal_logs"
mkdir -p "$FORMAL_LOG_DIR"

attempt=1
while (( attempt <= FORMAL_MAX_RETRIES )); do
    FORMAL_TS="$(date +%Y%m%d_%H%M%S)"
    FORMAL_LOG="$FORMAL_LOG_DIR/formal_${FORMAL_TS}_attempt${attempt}.log"
    echo "[INFO] Formal attempt ${attempt}/${FORMAL_MAX_RETRIES} (log: $FORMAL_LOG)"

    set +e
    python -m examples.strategy_extraction.train_strategy_generation \
        --data-base-path /home/test/test16/chenlu/projects/LLMReflection/data/ \
        --train-subdir train_20k \
        --val-subdirs test-id-subtask test-ood-task test-bbh \
        --model-path /home/test/test16/chenlu/model/Qwen3-4B \
        --fewshot-min 3 \
        --fewshot-max 5 \
        --num-train-samples 1 \
        --val-sampling-mode per_subtask_fixed \
        --val-samples-per-subtask "$FORMAL_VAL_SAMPLES_PER_SUBTASK" \
        --val-sampling-seed 42 \
        --val-batch-size "$VAL_BATCH_SIZE" \
        --min-val-trace-ratio 0.9 \
        --min-val-trace-count "$FORMAL_MIN_VAL_TRACE_COUNT" \
        --min-val-first-batch-ratio "$FORMAL_MIN_VAL_FIRST_BATCH_RATIO" \
        --n-runners "$N_RUNNERS" \
        --n-gpus 4 \
        --reward-version v3 \
        --reward-mode scorer_only \
        --format-weight 0.0 \
        --scorer-weight 0.0 \
        --grounded-proxy-weight 0.0 \
        --grounded-proxy-k 1 \
        --correctness-weight 1.0 \
        --val-only \
        --skip-strategy-generation \
        --no-strategy-for-answer \
        --answer-prompt-version no_strategy_baseline \
        --answer-no-think \
        --answer-model-path /home/test/test16/chenlu/model/Qwen3-8B \
        ${ANSWER_MODEL_BASE_URL:+--answer-model-base-url "$ANSWER_MODEL_BASE_URL"} \
        ${ANSWER_MODEL_NAME:+--answer-model-name "$ANSWER_MODEL_NAME"} \
        --wandb-experiment no_strategy_baseline_v3 \
        --checkpoint-dir ./checkpoints_strategy_gen_no_strategy_baseline \
        "$@" 2>&1 | tee "$FORMAL_LOG"
    RUN_EXIT=${PIPESTATUS[0]}
    set -e

    ACTOR_LOOKUP_FAIL_COUNT="$(grep -c "Failed to look up actor with name" "$FORMAL_LOG" || true)"
    echo "[INFO] Formal attempt $attempt: actor lookup failures=$ACTOR_LOOKUP_FAIL_COUNT, exit_code=$RUN_EXIT"

    if (( ACTOR_LOOKUP_FAIL_COUNT > ACTOR_LOOKUP_FAIL_THRESHOLD )); then
        if (( attempt < FORMAL_MAX_RETRIES )); then
            echo "[WARN] Actor lookup failures exceed threshold; treating as bad run and retrying."
            cleanup_ray_state
            attempt=$((attempt + 1))
            continue
        fi
        echo "[ERROR] Actor lookup failures exceed threshold on final attempt."
        exit 4
    fi

    if (( RUN_EXIT != 0 )); then
        exit "$RUN_EXIT"
    fi

    echo "[INFO] Formal run passed guardrail."
    break
done