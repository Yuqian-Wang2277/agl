#!/bin/bash
# Evaluate no-strategy baseline with reward v3 accuracy routing.
#
# Key settings:
# - Validation sampling: fixed 20 per subtask json (57 subtasks total => 1140 samples)
# - Seed: 42 for reproducibility
# - Strict no-strategy baseline: skip strategy generation entirely and answer directly
# - Accuracy metric path: reward v3 (type-routed answer judging)
# - val_only: run validation and exit before any training update
#
# Safety checks:
# - Ensure nvidia-smi is available
# - Ensure --n-gpus does not exceed visible GPU count
# - Ensure each visible GPU has at least MIN_FREE_MEM_MB free memory
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 N_GPUS=8 bash scripts/eval_no_strategy_baseline.sh

set -euo pipefail
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found. Please run on a GPU-enabled node."
    exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[ERROR] CUDA_VISIBLE_DEVICES is empty."
    echo "        Example: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 N_GPUS=8 bash scripts/eval_no_strategy_baseline.sh"
    exit 1
fi

VISIBLE_GPU_COUNT="$(python - <<'PY'
import os
visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
print(len(visible))
PY
)"
N_GPUS="${N_GPUS:-$VISIBLE_GPU_COUNT}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-10000}"
# Throughput knobs for better GPU saturation in val-only direct baseline.
# Default to 2x runners per GPU and a larger validation batch.
N_RUNNERS="${N_RUNNERS:-$((N_GPUS * 2))}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"

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

python -m examples.strategy_extraction.train_strategy_generation \
    --data-base-path /home/test/test16/chenlu/projects/LLMReflection/data/ \
    --train-subdir train_20k \
    --val-subdirs test-id-subtask test-ood-task test-bbh \
    --model-path /home/test/test16/chenlu/model/Qwen3-4B \
    --fewshot-min 3 \
    --fewshot-max 8 \
    --num-train-samples 1 \
    --val-sampling-mode per_subtask_fixed \
    --val-samples-per-subtask 20 \
    --val-sampling-seed 42 \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --min-val-trace-ratio 0.9 \
    --n-runners "$N_RUNNERS" \
    --n-gpus "$N_GPUS" \
    --reward-version v3 \
    --reward-mode scorer_only \
    --format-weight 0.0 \
    --scorer-weight 0.0 \
    --grounded-proxy-weight 0.0 \
    --grounded-proxy-k 1 \
    --correctness-weight 1.0 \
    --val-only \
    --no-strategy-for-answer \
    --strict-no-strategy-baseline \
    --answer-prompt-version no_strategy_baseline \
    --wandb-experiment no_strategy_baseline_v3 \
    --checkpoint-dir ./checkpoints_strategy_gen_no_strategy_baseline \
    "$@"
