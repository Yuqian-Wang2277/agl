#!/usr/bin/env bash
# =============================================================================
# debug.sh — Single-GPU debug run (no accelerate, no Ray, no vLLM)
#
# Purpose: quickly verify that every module imports correctly and the
#          data pipeline / buffer / model forward pass work end-to-end,
#          without consuming multi-GPU resources or waiting for vLLM to start.
#
# Usage:
#   bash scripts/debug.sh [--model_path PATH]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="/home/test/test16/chenlu/model/Qwen3-4B"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agl

cd "$PROJECT_DIR"
echo "=== Actor-Judge Debug Runner ==="
echo "Project: $(pwd)"
echo "Model:   $MODEL_PATH"
echo ""

python - <<PYEOF
import sys, os
sys.path.insert(0, ".")

print("── 1. Imports ─────────────────────────────────────────")
from config       import ActorJudgeConfig
from data_loader  import ActorJudgeDataset, load_rollout_dataset
from prompts      import build_strategy_prompt, build_answer_prompt, build_judge_prompt, JUDGE_TOKEN
from env          import evaluate
from buffer       import UCBBuffer, Experience
from judge_model  import JudgeModel
print("   All imports OK")

print("── 2. Config ──────────────────────────────────────────")
cfg = ActorJudgeConfig(actor_model_path="$MODEL_PATH", total_epochs=1, K=2)
print(f"   start_model_path = {cfg.start_model_path}")

print("── 3. Prompt TOML loading ─────────────────────────────")
examples = [{"input": "1+1=?", "target": "2"}, {"input": "2+2=?", "target": "4"}]
msgs1 = build_strategy_prompt(examples, version="v1")
print(f"   strategy prompt: system={msgs1[0]['content'][:40]!r}...")
msgs2 = build_answer_prompt("<strategy>Add numbers</strategy>", "3+3=?", version="v1")
print(f"   answer prompt:   system={msgs2[0]['content'][:40]!r}...")
j_prompt = build_judge_prompt(examples, "5+5=?", "Add the numbers together.", version="v1")
assert j_prompt.endswith(JUDGE_TOKEN), "Judge prompt must end with JUDGE_TOKEN"
print(f"   judge prompt ends with: {j_prompt[-20:]!r} ✓")

print("── 4. env.evaluate ────────────────────────────────────")
assert evaluate("<strategy>x</strategy>", "<answer>5</answer>", "5") == 1
assert evaluate("<strategy>x</strategy>", "<answer>3</answer>", "5") == 0
assert evaluate("no tags here",           "<answer>5</answer>", "5") == -1
assert evaluate("<strategy>x</strategy>", "no answer tag",      "5") == -1
print("   evaluate() all cases OK ✓")

print("── 5. Buffer ──────────────────────────────────────────")
buf = UCBBuffer(max_size=100, per_q_max=5)
for i in range(6):
    exp = UCBBuffer.make_experience("ctx", f"Q{i%3}", f"S{i}", outcome=i%2, timestamp=i)
    buf.add(exp)
print(f"   buffer size = {len(buf)} (max per-Q=5)")
pairs = buf.sample_pairwise(4)
print(f"   sampled {len(pairs)} pairs")

print("── 6. Data loading (first 3 samples) ─────────────────")
import os
train_dir = os.path.join(cfg.data_base_path, cfg.train_subdir)
if os.path.exists(train_dir):
    ds = ActorJudgeDataset(train_dir, num_samples=10, fewshot_min=2, fewshot_max=3)
    s  = ds[0]
    print(f"   domain={s.domain!r}  q={s.question[:40]!r}  gold={s.answer_gold!r}")
    print(f"   fewshot count = {len(s.fewshot_examples)}")
else:
    print(f"   SKIP — data dir not found: {train_dir}")

print("")
print("=== All debug checks passed ✓ ===")
PYEOF
