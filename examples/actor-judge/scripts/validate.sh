#!/usr/bin/env bash
# =============================================================================
# validate.sh — Run Pass@1 validation on a trained Actor checkpoint
#
# Usage:
#   bash scripts/validate.sh --checkpoint PATH [--split test-bbh] [--n 500]
#
# Examples:
#   bash scripts/validate.sh \
#       --checkpoint ./checkpoints_actor_judge/epoch_004 \
#       --split test-bbh
#
#   # Evaluate all three validation splits:
#   for split in test-id-subtask test-ood-task test-bbh; do
#       bash scripts/validate.sh --checkpoint ./checkpoints_actor_judge/epoch_004 \
#           --split $split
#   done
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CHECKPOINT=""
SPLIT="test-bbh"
N_SAMPLES=500
MODEL_PATH="/home/test/test16/chenlu/model/Qwen3-4B"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)  CHECKPOINT="$2";  shift 2 ;;
        --split)       SPLIT="$2";       shift 2 ;;
        --n)           N_SAMPLES="$2";   shift 2 ;;
        --model_path)  MODEL_PATH="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: --checkpoint is required."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agl

cd "$PROJECT_DIR"

python - <<PYEOF
import sys, os, torch
sys.path.insert(0, ".")

from config      import ActorJudgeConfig
from data_loader import ActorJudgeDataset
from prompts     import (apply_chat_template, build_strategy_prompt,
                         build_answer_prompt, STRATEGY_CLOSE, JUDGE_TOKEN)
from env         import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT   = "$CHECKPOINT"
SPLIT        = "$SPLIT"
N_SAMPLES    = int("$N_SAMPLES")
MODEL_PATH   = "$MODEL_PATH"
DATA_BASE    = "/home/test/test16/chenlu/projects/LLMReflection/data/"

print(f"Validating checkpoint : {CHECKPOINT}")
print(f"Split                 : {SPLIT}")
print(f"N samples             : {N_SAMPLES}")
print()

# ── Load model ────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=device,
)
model.eval()
print("Model loaded.")

# ── Load validation data ─────────────────────────────────────────────────────
val_dir = os.path.join(DATA_BASE, SPLIT)
if not os.path.exists(val_dir):
    print(f"ERROR: val directory not found: {val_dir}")
    sys.exit(1)

ds = ActorJudgeDataset(val_dir, num_samples=N_SAMPLES, fewshot_min=3, fewshot_max=5,
                       cross_domain_ratio=0.0)

# ── Greedy eval ───────────────────────────────────────────────────────────────
correct = total = 0
strategy_lengths = []

for i, sample in enumerate(ds):
    # Stage-1: generate strategy
    msgs1   = build_strategy_prompt(sample.fewshot_examples)
    prompt1 = apply_chat_template(tokenizer, msgs1)
    enc1    = tokenizer(prompt1, return_tensors="pt").to(device)
    with torch.no_grad():
        out1 = model.generate(**enc1, max_new_tokens=512, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id)
    s_text = tokenizer.decode(out1[0][enc1["input_ids"].shape[1]:],
                               skip_special_tokens=False) + STRATEGY_CLOSE

    strategy_lengths.append(len(s_text.split()))

    # Stage-2: generate answer
    msgs2   = build_answer_prompt(s_text, sample.question)
    prompt2 = apply_chat_template(tokenizer, msgs2)
    enc2    = tokenizer(prompt2, return_tensors="pt").to(device)
    with torch.no_grad():
        out2 = model.generate(**enc2, max_new_tokens=256, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id)
    a_text = tokenizer.decode(out2[0][enc2["input_ids"].shape[1]:],
                               skip_special_tokens=False) + "</answer>"

    outcome = evaluate(s_text, a_text, sample.answer_gold)
    if outcome == 1:
        correct += 1
    if outcome >= 0:
        total += 1

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(ds)}] running pass@1 = {correct}/{total} = "
              f"{correct/max(total,1):.3f}")

pass1    = correct / max(total, 1)
avg_len  = sum(strategy_lengths) / max(len(strategy_lengths), 1)

print()
print(f"=== Results [{SPLIT}] ===")
print(f"Pass@1              : {pass1:.4f}  ({correct}/{total})")
print(f"Avg strategy length : {avg_len:.0f} tokens")
PYEOF
