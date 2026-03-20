#!/usr/bin/env python3
"""Rough token-length stats for Stage-1 prompts (few-shot + template + Q).

Run from ``examples/actor-judge`` (recommended):

  python scripts/estimate_prompt_lengths.py
  python scripts/estimate_prompt_lengths.py --model_path /path/to/Qwen3-4B
  python scripts/estimate_prompt_lengths.py --data_base_path /data/LLMReflection/data --train_subdir train_20k

Uses ``ActorJudgeConfig`` defaults unless overridden by flags. Few-shot lives inside
the Stage-1 chat prompt; left truncation in GRPO removes the **start** of that prompt
first when over ``actor_max_length``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from transformers import AutoTokenizer  # noqa: E402

from config import ActorJudgeConfig  # noqa: E402
from data_loader import ActorJudgeDataset, load_val_dataset  # noqa: E402
from prompts import apply_chat_template, build_strategy_prompt  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Stage-1 prompt token length stats (few-shot + template).")
    p.add_argument(
        "--model_path",
        type=str,
        default="",
        help="HF dir for tokenizer (default: cfg.start_model_path from ActorJudgeConfig)",
    )
    p.add_argument("--data_base_path", type=str, default="", help="Override cfg.data_base_path")
    p.add_argument("--train_subdir", type=str, default="", help="Override cfg.train_subdir")
    p.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="Cap train rows scanned (default: min(500, cfg.num_train_samples))",
    )
    p.add_argument(
        "--max_val_samples",
        type=int,
        default=200,
        help="Max rows per val split (default: 200)",
    )
    args = p.parse_args()

    cfg = ActorJudgeConfig()
    if args.data_base_path:
        cfg.data_base_path = args.data_base_path
    if args.train_subdir:
        cfg.train_subdir = args.train_subdir
    tok_path = (args.model_path or "").strip() or cfg.start_model_path
    print(f"Tokenizer: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    def stats_for_samples(samples, label: str, n: int) -> None:
        lens: list[int] = []
        for s in samples[:n]:
            msgs = build_strategy_prompt(s.fewshot_examples)
            p = apply_chat_template(tokenizer, msgs)
            lens.append(len(tokenizer.encode(p, add_special_tokens=False)))
        if not lens:
            print(f"{label}: no samples")
            return
        lens.sort()
        print(
            f"{label} (n={len(lens)}): "
            f"min={lens[0]} p50={lens[len(lens)//2]} "
            f"p90={lens[int(0.9 * len(lens)) - 1]} max={lens[-1]}"
        )

    if args.max_train_samples > 0:
        n_train = min(args.max_train_samples, cfg.num_train_samples)
    else:
        n_train = min(500, cfg.num_train_samples)
    train_ds = ActorJudgeDataset(
        str(Path(cfg.data_base_path) / cfg.train_subdir),
        num_samples=n_train,
        fewshot_min=cfg.fewshot_min,
        fewshot_max=cfg.fewshot_max,
        cross_domain_ratio=cfg.cross_domain_ratio,
        max_stage1_prompt_tokens=0,
        tokenizer=tokenizer,
        stage1_chars_per_token=float(cfg.stage1_length_chars_per_token),
    )
    stats_for_samples(list(train_ds), "train Stage-1 prompt", n_train)

    for sub in cfg.val_subdirs[:3]:
        try:
            vds = load_val_dataset(
                cfg,
                sub,
                tokenizer=tokenizer,
                max_stage1_prompt_tokens=0,
            )
            vs = list(vds)[: max(1, args.max_val_samples)]
            stats_for_samples(vs, f"val[{sub}] Stage-1 prompt", len(vs))
        except Exception as exc:
            print(f"val[{sub}]: skip ({exc})")

    print(
        f"\nConfig reference: actor_max_length={cfg.actor_max_length} "
        f"judge_max_length={cfg.judge_max_length} "
        f"fewshot_min/max={cfg.fewshot_min}/{cfg.fewshot_max} "
        f"rollout strategy_max_tokens={cfg.strategy_max_tokens} "
        f"val_strategy_max_tokens={cfg.val_strategy_max_tokens} "
        f"val_answer_max_tokens={cfg.val_answer_max_tokens}"
    )
    print(
        "GRPO tokenises [Stage-1 prompt + generated strategy]; if total > actor_max_length, "
        "left truncation drops the beginning of the prompt (few-shot is at the left)."
    )


if __name__ == "__main__":
    main()
