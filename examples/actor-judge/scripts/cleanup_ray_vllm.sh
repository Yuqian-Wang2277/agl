#!/usr/bin/env bash
# =============================================================================
# cleanup_ray_vllm.sh — free GPU / RAM after a crashed Phase II run
#
# PyTorch or Ray crashes often leave vLLM / Ray workers alive and holding VRAM.
# Run this before starting another training job.
#
# WARNING: pkill patterns may terminate other user processes that match.
# Review the patterns below if you run multiple Ray/vLLM jobs on the same host.
# =============================================================================
set -euo pipefail

echo "[1/4] ray stop (best-effort)..."
if command -v ray &>/dev/null; then
    ray stop --force 2>/dev/null || ray stop -f 2>/dev/null || true
else
    echo "  (ray CLI not found, skip)"
fi

echo "[2/4] SIGTERM Ray / vLLM-related Python children..."
# Common leftovers from this example stack
pkill -TERM -f "ray::" 2>/dev/null || true
pkill -TERM -f "VLLMActor" 2>/dev/null || true
pkill -TERM -f "vllm.entrypoints" 2>/dev/null || true
pkill -TERM -f "vllm.engine" 2>/dev/null || true
sleep 2

echo "[3/4] SIGKILL if still alive..."
pkill -KILL -f "ray::" 2>/dev/null || true
pkill -KILL -f "VLLMActor" 2>/dev/null || true
pkill -KILL -f "vllm.entrypoints" 2>/dev/null || true
pkill -KILL -f "vllm.engine" 2>/dev/null || true

echo "[4/4] Clear shared-memory hand-off dir (same as train.sh)..."
rm -rf /dev/shm/actor_weight_tmp 2>/dev/null || true

echo "Done. Check: nvidia-smi"
echo "If GPUs still show python, note PIDs and: kill -9 <pid>"
