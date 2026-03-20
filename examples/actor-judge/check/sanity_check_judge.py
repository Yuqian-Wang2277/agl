#!/usr/bin/env python3
"""Quick qualitative checks for a warmed-up Judge before Phase II.

Loads a warmup output directory (tokenizer + HF backbone + judge_model.pt), scores
pairs of strategies on the same question, and prints logits / sigmoid probs / margin.

Usage:
  cd examples/actor-judge
  python check/sanity_check_judge.py --warmup_dir judge_warmup_ckpt/2026-03-20

Optional custom cases (JSON list):
  [{"name": "my_case", "question": "...", "strategy_a": "<strategy>...</strategy>",
    "strategy_b": "<strategy>...</strategy>", "expect": "a"}]
  expect: \"a\" if strategy_a should score higher than strategy_b.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Script lives in check/; judge_model.py and prompts.py are in parent actor-judge/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from transformers import AutoModel, AutoTokenizer

from judge_encode import encode_batch_for_judge
from judge_model import JudgeModel
from prompts import build_judge_prompt_body


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def resolve_warmup_dir(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_dir() and (p / "judge_model.pt").is_file():
        return p
    if p.is_file() and p.name == "judge_model.pt":
        return p.parent
    raise FileNotFoundError(
        f"Expected a directory containing judge_model.pt, got {path!r}"
    )


def score_strategy(
    judge: JudgeModel,
    tokenizer,
    question: str,
    strategy: str,
    max_length: int,
    model_device: torch.device,
) -> Tuple[float, float]:
    body = build_judge_prompt_body([], question, strategy)
    batch = encode_batch_for_judge(tokenizer, [body], max_length)
    batch = _batch_to_device(batch, model_device)
    with torch.no_grad():
        logit = judge(**batch).float().reshape(-1)[0]
    prob = torch.sigmoid(logit).item()
    return float(logit.item()), float(prob)


def builtin_cases() -> List[Dict[str, str]]:
    return [
        {
            "name": "easy_negative",
            "question": "What is 17 + 25?\nOptions:\n(A) 40\n(B) 42\n(C) 43",
            "strategy_a": (
                "<strategy>\nAdd the two numbers digit-wise from the right, carrying when needed. "
                "Verify by subtracting one operand from the result.\n</strategy>"
            ),
            "strategy_b": "<strategy>\nzqxw $$@@ blorp fnord not a plan\n</strategy>",
            "expect": "a",
        },
        {
            "name": "length_bias",
            "question": "Which option rhymes with 'moon'?\nOptions:\n(A) soon\n(B) door\n(C) run",
            "strategy_a": (
                "<strategy>\nMatch the ending vowel and consonant sounds (rime) of the target word; "
                "choose the option that shares that ending sound.\n</strategy>"
            ),
            "strategy_b": (
                "<strategy>\nFirst, provide an exhaustive historical survey of rhyme schemes in "
                "nineteenth-century criticism, then argue at length that longer answers are more "
                "trustworthy regardless of phonetics, and therefore prefer the most verbose option "
                "even if it does not rhyme, because verbosity implies rigor.\n</strategy>"
            ),
            "expect": "a",
        },
        {
            "name": "off_topic_verbose",
            "question": "A farmer has 12 sheep and sells 4. How many remain?",
            "strategy_a": (
                "<strategy>\nSubtract the sold count from the initial count; the remainder is the answer.\n</strategy>"
            ),
            "strategy_b": (
                "<strategy>\nAnalyze sentiment polarity of customer reviews using a fine-tuned BERT "
                "classifier, then cluster embeddings with k-means to pick a product category unrelated "
                "to arithmetic word problems.\n</strategy>"
            ),
            "expect": "a",
        },
    ]


def run_case(
    judge: JudgeModel,
    tokenizer,
    case: Dict[str, Any],
    max_length: int,
    model_device: torch.device,
) -> bool:
    name = case["name"]
    q = case["question"]
    sa = case["strategy_a"]
    sb = case["strategy_b"]
    expect = (case.get("expect") or "a").lower()

    la, pa = score_strategy(judge, tokenizer, q, sa, max_length, model_device)
    lb, pb = score_strategy(judge, tokenizer, q, sb, max_length, model_device)
    margin = pa - pb
    ok = (la > lb) if expect == "a" else (lb > la)

    print(f"\n=== case={name!r} expect_higher={'A' if expect == 'a' else 'B'} ===")
    print(f"  A  logit={la:+.4f}  prob={pa:.6f}")
    print(f"  B  logit={lb:+.4f}  prob={pb:.6f}")
    print(f"  margin(A-B) prob = {margin:+.6f}  (logit margin {la - lb:+.4f})")
    print(f"  pass={ok}")
    return bool(ok)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge warmup sanity check (qualitative)")
    parser.add_argument(
        "--warmup_dir",
        type=str,
        required=True,
        help="Directory from run_judge_warmup (contains judge_model.pt, tokenizer, HF backbone)",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="HF device_map for backbone (e.g. auto, cuda:0, or balanced low-memory settings)",
    )
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument(
        "--cases_json",
        type=str,
        default="",
        help="JSON file: list of {name, question, strategy_a, strategy_b, expect}",
    )
    parser.add_argument(
        "--no_builtin",
        action="store_true",
        help="Skip built-in cases; use only --cases_json (must be non-empty)",
    )
    args = parser.parse_args()

    ckpt_dir = resolve_warmup_dir(args.warmup_dir)
    judge_pt = ckpt_dir / "judge_model.pt"

    cases: List[Dict[str, Any]] = []
    if not args.no_builtin:
        cases.extend(builtin_cases())
    if args.cases_json:
        path = Path(args.cases_json).expanduser()
        with open(path, encoding="utf-8") as f:
            extra = json.load(f)
        if not isinstance(extra, list):
            print("--cases_json must contain a JSON list", file=sys.stderr)
            sys.exit(1)
        cases.extend(extra)
    if not cases:
        print("No cases to run (use built-ins or --cases_json)", file=sys.stderr)
        sys.exit(1)

    print(f"[sanity] warmup_dir={ckpt_dir}")
    print(f"[sanity] judge weights={judge_pt}")
    print(f"[sanity] device_map={args.device_map!r}")

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModel.from_pretrained(
        str(ckpt_dir),
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    hidden = backbone.config.hidden_size
    judge = JudgeModel(backbone, hidden)
    try:
        state = torch.load(judge_pt, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(judge_pt, map_location="cpu")
    missing, unexpected = judge.load_state_dict(state, strict=False)
    print(f"[sanity] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    judge.eval()

    model_device = judge.transformer.get_input_embeddings().weight.device
    print(f"[sanity] input embedding device={model_device}")

    passed = 0
    for c in cases:
        if run_case(judge, tokenizer, c, args.max_length, model_device):
            passed += 1

    print(f"\n[sanity] passed {passed}/{len(cases)} checks (qualitative only)")
    if passed < len(cases):
        sys.exit(1)


if __name__ == "__main__":
    main()
