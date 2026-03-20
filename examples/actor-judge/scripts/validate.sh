#!/usr/bin/env bash
# =============================================================================
# validate.sh — Greedy Pass@1 (strict denominator) on a saved Actor checkpoint
#
# Token limits match train-time validation (config defaults: 2048 / 512).
# Pass@1 = correct / N for all N items; format errors count as incorrect.
#
# Usage:
#   bash scripts/validate.sh --checkpoint RUN_DIR/epoch_004 [--split test-bbh] [--n 500]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CHECKPOINT=""
SPLIT="test-bbh"
N_SAMPLES=500
STRAT_TOK=2048
ANS_TOK=512
DATA_BASE="/home/test/test16/chenlu/projects/LLMReflection/data/"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)   CHECKPOINT="$2";  shift 2 ;;
        --split)        SPLIT="$2";       shift 2 ;;
        --n)            N_SAMPLES="$2";   shift 2 ;;
        --strat_tokens) STRAT_TOK="$2";   shift 2 ;;
        --ans_tokens)   ANS_TOK="$2";     shift 2 ;;
        --data_base)    DATA_BASE="$2";   shift 2 ;;
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
import os, sys, torch

sys.path.insert(0, ".")

from data_loader import ActorJudgeDataset
from env import evaluate
from prompts import (
    apply_chat_template,
    build_answer_prompt,
    build_strategy_prompt,
    STRATEGY_CLOSE,
    STRATEGY_OPEN,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT = "$CHECKPOINT"
SPLIT = "$SPLIT"
N_SAMPLES = int("$N_SAMPLES")
STRAT_TOK = int("$STRAT_TOK")
ANS_TOK = int("$ANS_TOK")
DATA_BASE = "$DATA_BASE"

print(f"Checkpoint      : {CHECKPOINT}")
print(f"Split           : {SPLIT}")
print(f"N samples       : {N_SAMPLES}")
print(f"strat/ans tokens: {STRAT_TOK} / {ANS_TOK} (match train.py val defaults)")
print()

device = "cuda" if torch.cuda.is_available() else "cpu"
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

val_dir = os.path.join(DATA_BASE, SPLIT)
if not os.path.exists(val_dir):
    print(f"ERROR: val directory not found: {val_dir}")
    sys.exit(1)

ds = ActorJudgeDataset(
    val_dir,
    num_samples=N_SAMPLES,
    fewshot_min=3,
    fewshot_max=5,
    cross_domain_ratio=0.0,
    max_stage1_prompt_tokens=4000,
    tokenizer=tokenizer,
    stage1_chars_per_token=2.5,
    stage1_reject_log_path="",
)
n_total = len(ds)
correct = 0
strategy_lengths = []

for i, sample in enumerate(ds):
    msgs1 = build_strategy_prompt(sample.fewshot_examples)
    prompt1 = apply_chat_template(tokenizer, msgs1)
    enc1 = tokenizer(prompt1, return_tensors="pt").to(device)
    with torch.no_grad():
        out1 = model.generate(
            **enc1,
            max_new_tokens=STRAT_TOK,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    s_text = (
        tokenizer.decode(out1[0][enc1["input_ids"].shape[1] :], skip_special_tokens=False)
        + STRATEGY_CLOSE
    )
    strategy_lengths.append(len(s_text.split()))

    a_text = ""
    if STRATEGY_OPEN in s_text and STRATEGY_CLOSE in s_text:
        msgs2 = build_answer_prompt(s_text, sample.question)
        prompt2 = apply_chat_template(tokenizer, msgs2)
        enc2 = tokenizer(prompt2, return_tensors="pt").to(device)
        with torch.no_grad():
            out2 = model.generate(
                **enc2,
                max_new_tokens=ANS_TOK,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        a_text = (
            tokenizer.decode(out2[0][enc2["input_ids"].shape[1] :], skip_special_tokens=False)
            + "</answer>"
        )

    outcome = evaluate(s_text, a_text, sample.answer_gold)
    if outcome == 1:
        correct += 1

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{n_total}] strict pass@1 = {correct}/{i+1} = {correct/(i+1):.4f}")

pass1 = correct / max(n_total, 1)
avg_len = sum(strategy_lengths) / max(len(strategy_lengths), 1)

print()
print(f"=== Results [{SPLIT}] (strict) ===")
print(f"Pass@1              : {pass1:.4f}  ({correct}/{n_total})")
print(f"Avg strategy length : {avg_len:.0f} words (approx)")
PYEOF
