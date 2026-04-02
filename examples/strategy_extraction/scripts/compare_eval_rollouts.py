#!/usr/bin/env python3
"""Merge pre/post eval_no_verl JSON shards by rollout_id and emit quadrant stats + Scheme E report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_rollouts(root: Path) -> Dict[str, Dict[str, Any]]:
    """Load all validation_step*.json under root; index by rollout_id (last wins if duplicate)."""
    by_id: Dict[str, Dict[str, Any]] = {}
    files = sorted(root.rglob("validation_step*.json"))
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for item in data:
            rid = item.get("rollout_id")
            if not rid:
                continue
            by_id[str(rid)] = item
    return by_id


def extract_row(item: Dict[str, Any], label: str) -> Dict[str, Any]:
    reward = item.get("reward") or {}
    inp = item.get("input") or {}
    out = item.get("output") or {}
    meta = item.get("metadata") or {}
    tm = inp.get("task_meta") or {}
    strat = (out.get("strategy_extracted") or "").strip()
    hard = int(reward.get("hard_correct") or 0)
    fmt = float(reward.get("format") or 0.0)
    return {
        "rollout_id": item.get("rollout_id"),
        "label": label,
        "hard_correct": hard,
        "format_ok": 1 if fmt >= 1.0 else 0,
        "format": fmt,
        "correctness": float(reward.get("correctness") or 0.0),
        "problem_type": item.get("problem_type") or "",
        "source_problem_type": item.get("source_problem_type") or "",
        "validation_split": tm.get("validation_split") or "",
        "strategy_empty": len(strat) == 0,
        "strategy_len": len(strat),
        "answer_extracted": (out.get("answer_extracted") or "").strip(),
        "ground_truth": (inp.get("ground_truth") or ""),
        "model_meta": meta.get("model") or "",
    }


def top_keys(counter: Counter, k: int = 5) -> List[Tuple[str, int]]:
    return counter.most_common(k)


def stratify(rows: List[Dict[str, Any]], key: str) -> Counter:
    c: Counter = Counter()
    for r in rows:
        v = r.get(key) or "(empty)"
        c[v] += 1
    return c


def pick_examples(
    merged: List[Dict[str, Any]],
    rollout_ids: set,
    k: int = 3,
) -> List[Dict[str, Any]]:
    """Pick up to k rows whose rollout_id is in rollout_ids, preferring diverse problem_type."""
    pool = [m for m in merged if m.get("rollout_id") in rollout_ids]
    seen_types: set = set()
    out: List[Dict[str, Any]] = []
    for m in pool:
        t = m.get("problem_type") or ""
        if t not in seen_types and len(out) < k:
            seen_types.add(t)
            out.append(m)
    i = 0
    while len(out) < k and i < len(pool):
        if pool[i] not in out:
            out.append(pool[i])
        i += 1
    return out[:k]


def strategy_snippet(s: str, max_len: int = 400) -> str:
    s = (s or "").replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pre-root",
        type=Path,
        default=Path(
            "/home/test/test16/chenlu/projects/agent-lightning/checkpoints_eval_no_verl/20260329_213924"
        ),
    )
    parser.add_argument(
        "--post-root",
        type=Path,
        default=Path(
            "/home/test/test16/chenlu/projects/agent-lightning/checkpoints_eval_no_verl/eval_global_step250"
        ),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path(
            "/home/test/test16/chenlu/projects/agent-lightning/checkpoints_eval_no_verl/scheme_e_case_analysis.md"
        ),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path(
            "/home/test/test16/chenlu/projects/agent-lightning/checkpoints_eval_no_verl/merged_pre_post_rollouts.csv"
        ),
    )
    args = parser.parse_args()

    pre = load_rollouts(args.pre_root)
    post = load_rollouts(args.post_root)
    pre_ids = set(pre.keys())
    post_ids = set(post.keys())
    only_pre = pre_ids - post_ids
    only_post = post_ids - pre_ids
    common = sorted(pre_ids & post_ids)

    merged: List[Dict[str, Any]] = []
    for rid in common:
        a = extract_row(pre[rid], "pre")
        b = extract_row(post[rid], "post")
        merged.append(
            {
                "rollout_id": rid,
                "pre_hard": a["hard_correct"],
                "post_hard": b["hard_correct"],
                "pre_format": a["format"],
                "post_format": b["format"],
                "pre_strategy_empty": a["strategy_empty"],
                "post_strategy_empty": b["strategy_empty"],
                "problem_type": a["problem_type"],
                "source_problem_type": a["source_problem_type"],
                "validation_split": a["validation_split"],
                "pre_answer": a["answer_extracted"],
                "post_answer": b["answer_extracted"],
                "ground_truth": a["ground_truth"],
                "pre_model": a["model_meta"],
                "post_model": b["model_meta"],
            }
        )

    n = len(merged)
    imp = [m for m in merged if m["pre_hard"] == 0 and m["post_hard"] == 1]
    reg = [m for m in merged if m["pre_hard"] == 1 and m["post_hard"] == 0]
    persist_bad = [m for m in merged if m["pre_hard"] == 0 and m["post_hard"] == 0]
    persist_good = [m for m in merged if m["pre_hard"] == 1 and m["post_hard"] == 1]

    pre_acc = sum(m["pre_hard"] for m in merged) / n if n else 0.0
    post_acc = sum(m["post_hard"] for m in merged) / n if n else 0.0

    def row_csv(m: Dict[str, Any]) -> str:
        def esc(x: Any) -> str:
            s = str(x).replace('"', '""')
            if "," in s or "\n" in s or '"' in s:
                return f'"{s}"'
            return s

        keys = [
            "rollout_id",
            "pre_hard",
            "post_hard",
            "pre_format",
            "post_format",
            "pre_strategy_empty",
            "post_strategy_empty",
            "validation_split",
            "problem_type",
            "source_problem_type",
            "pre_answer",
            "post_answer",
            "ground_truth",
        ]
        return ",".join(esc(m[k]) for k in keys)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = "rollout_id,pre_hard,post_hard,pre_format,post_format,pre_strategy_empty,post_strategy_empty,validation_split,problem_type,source_problem_type,pre_answer,post_answer,ground_truth\n"
    args.out_csv.write_text(header + "\n".join(row_csv(m) for m in merged), encoding="utf-8")

    def strategy_stage_fail(m: Dict[str, Any], prefix: str) -> bool:
        if prefix == "pre":
            return m["pre_format"] < 1.0 or m["pre_strategy_empty"]
        return m["post_format"] < 1.0 or m["post_strategy_empty"]

    persist_strategy_fail = [
        m for m in persist_bad if strategy_stage_fail(m, "pre") and strategy_stage_fail(m, "post")
    ]
    fail_ids = {m["rollout_id"] for m in persist_strategy_fail}
    persist_semantic = [m for m in persist_bad if m["rollout_id"] not in fail_ids]

    imp_split = stratify(imp, "validation_split")
    imp_pt = stratify(imp, "problem_type")
    reg_split = stratify(reg, "validation_split")
    reg_pt = stratify(reg, "problem_type")
    bad_split = stratify(persist_bad, "validation_split")
    bad_pt = stratify(persist_bad, "problem_type")
    good_split = stratify(persist_good, "validation_split")
    good_pt = stratify(persist_good, "problem_type")

    def top3_line(c: Counter, n: int = 3) -> str:
        parts = [f"{name} ({cnt})" for name, cnt in top_keys(c, n)]
        return ", ".join(parts) if parts else "—"

    # Load full strategy text for a few examples (from original dicts)
    def full_items(rid: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return pre[rid], post[rid]

    def strat_from_item(it: Dict[str, Any]) -> str:
        o = it.get("output") or {}
        return (o.get("strategy_extracted") or "").strip()

    lines: List[str] = []
    lines.append("# Scheme E: Pre- vs Post-training Case Analysis (eval_no_verl)\n")
    lines.append("Generated by `examples/strategy_extraction/scripts/compare_eval_rollouts.py`.\n")
    lines.append("## Setup\n")
    lines.append(
        f"- **Pre-training (4B base)**: root `{args.pre_root}` — shards merged by `rollout_id`.\n"
    )
    lines.append(
        f"- **Post-training (4B step250)**: root `{args.post_root}` — same merge rule.\n"
    )
    lines.append(
        "- **Answer model**: frozen Qwen3-8B (logged as `answer_model_version` in raw JSON); only strategy generator differs.\n"
    )
    lines.append(
        "- **Correctness**: `reward.hard_correct` ∈ {0,1}; alignment key is `rollout_id`.\n"
    )
    if only_pre or only_post:
        lines.append(
            f"- **Warning**: `rollout_id` only in pre: {len(only_pre)}, only in post: {len(only_post)} (excluded from merged table).\n"
        )
    lines.append(f"- **Merged n**: {n} | **Pre accuracy**: {pre_acc:.4f} | **Post accuracy**: {post_acc:.4f} | **Δ**: {post_acc - pre_acc:+.4f}\n")
    lines.append("\n### Attribution scope (frozen answer model)\n\n")
    lines.append("```mermaid\nflowchart LR\n")
    lines.append("  subgraph held [Held_fixed]\n")
    lines.append("    Answer8B[Answer_Qwen3_8B]\n")
    lines.append("    Judge[hard_correct]\n")
    lines.append("  end\n")
    lines.append("  StrategyPre[Strategy_Qwen3_4B_base]\n")
    lines.append("  StrategyPost[Strategy_Qwen3_4B_step250]\n")
    lines.append("  StrategyPre --> Answer8B\n")
    lines.append("  StrategyPost --> Answer8B\n")
    lines.append("  Answer8B --> Judge\n")
    lines.append("```\n")

    lines.append("\n## Master table (four quadrants)\n\n")
    lines.append("| Quadrant | Count | Share of n | Top `validation_split` (3) | Top `problem_type` (3) |\n")
    lines.append("|----------|------:|-----------:|------------------------------|-------------------------|\n")
    lines.append(
        f"| I — Post-training improvements (0→1) | {len(imp)} | {len(imp)/n:.2%} | {top3_line(imp_split)} | {top3_line(imp_pt)} |\n"
    )
    lines.append(
        f"| II — Regressions (1→0) | {len(reg)} | {len(reg)/n:.2%} | {top3_line(reg_split)} | {top3_line(reg_pt)} |\n"
    )
    lines.append(
        f"| III — Persistent failures (0→0) | {len(persist_bad)} | {len(persist_bad)/n:.2%} | {top3_line(bad_split)} | {top3_line(bad_pt)} |\n"
    )
    lines.append(
        f"| IV — Consistently correct (1→1) | {len(persist_good)} | {len(persist_good)/n:.2%} | {top3_line(good_split)} | {top3_line(good_pt)} |\n"
    )
    sum_pre = sum(m["pre_hard"] for m in merged)
    sum_post = sum(m["post_hard"] for m in merged)
    lines.append(
        f"\n**Consistency check**: I − II = {len(imp) - len(reg)}; "
        f"Σ post `hard_correct` − Σ pre = {sum_post - sum_pre} (equal to I−II for binary labels).\n"
    )

    lines.append("\n### III refinement: pipeline vs semantic (persistent wrong)\n\n")
    lines.append(
        f"- **Both runs strategy/format failure** (pre **and** post: `format<1` or empty `strategy_extracted`): **{len(persist_strategy_fail)}**.\n"
    )
    lines.append(
        f"- **Other persistent wrong** (remaining 0→0 rows): **{len(persist_semantic)}** — includes cases where at least one run had parseable strategy but answer stayed wrong.\n"
    )

    def section_case_analysis(
        title: str,
        subset: List[Dict[str, Any]],
        strat_split: Counter,
        strat_pt: Counter,
        headline: str,
        implication: str,
    ) -> None:
        lines.append(f"\n## {title}\n")
        lines.append(f"### Headline\n{headline}\n")
        lines.append("### Stratification\n")
        lines.append(f"- By `validation_split`: {top3_line(strat_split, 8)}\n")
        lines.append(f"- By `problem_type`: {top3_line(strat_pt, 8)}\n")
        lines.append("### Mechanism sketch (micro case boxes)\n")
        if not subset:
            lines.append("_No rows in this quadrant._\n")
            lines.append("### Implication\n")
            lines.append(implication + "\n")
            return
        ids = {m["rollout_id"] for m in subset}
        examples = pick_examples(merged, ids, k=3)
        for m in examples:
            rid = m["rollout_id"]
            pit, pot = full_items(rid)
            spre = strat_from_item(pit)
            spost = strat_from_item(pot)
            lines.append(
                f"\n| Field | Value |\n|-------|-------|\n"
                f"| `rollout_id` | `{rid}` |\n"
                f"| `validation_split` | {m['validation_split']} |\n"
                f"| `problem_type` | {m['problem_type']} |\n"
                f"| pre `hard_correct` → post | {m['pre_hard']} → {m['post_hard']} |\n"
                f"| pre `format` → post | {m['pre_format']} → {m['post_format']} |\n"
                f"| pre strategy empty? → post | {m['pre_strategy_empty']} → {m['post_strategy_empty']} |\n"
                f"| `ground_truth` | {strategy_snippet(str(m['ground_truth']), 200)} |\n"
                f"| pre `answer_extracted` | {strategy_snippet(m['pre_answer'], 200)} |\n"
                f"| post `answer_extracted` | {strategy_snippet(m['post_answer'], 200)} |\n"
            )
            lines.append("\n**Strategy delta (extracted, truncated)**\n\n")
            lines.append(f"- **Pre**: {strategy_snippet(spre, 500)}\n")
            lines.append(f"- **Post**: {strategy_snippet(spost, 500)}\n")
        lines.append("\n### Implication\n")
        lines.append(implication + "\n")

    section_case_analysis(
        "Case Analysis I: Post-training Improvements (Negative → Positive)",
        imp,
        imp_split,
        imp_pt,
        headline=(
            f"Training yields **{len(imp)}** rollouts that flip from wrong to right "
            f"({len(imp)/n:.1%} of merged validation), concentrated in: "
            f"{top3_line(imp_split)} / {top3_line(imp_pt)}."
        ),
        implication=(
            "With 8B fixed, these flips are **consistent with** more parseable strategies and "
            "better-aligned FORMAT/CHECK/SECOND_ORDER guidance from the 4B policy. "
            "Causal claims should stay associative unless ablations isolate strategy-only effects."
        ),
    )

    section_case_analysis(
        "Case Analysis II: Post-training Regressions (Positive → Negative)",
        reg,
        reg_split,
        reg_pt,
        headline=(
            f"**{len(reg)}** rollouts regress ({len(reg)/n:.1%} of n); "
            f"hotspots: {top3_line(reg_split)} / {top3_line(reg_pt)}."
        ),
        implication=(
            "Regressions may reflect **distribution shift in strategy text** (over-constraint, wrong task framing) "
            "that misleads the frozen answer model. Report alongside I for net trade-off."
        ),
    )

    section_case_analysis(
        "Case Analysis III: Persistent Failure Modes",
        persist_bad,
        bad_split,
        bad_pt,
        headline=(
            f"**{len(persist_bad)}** rollouts remain wrong under both checkpoints ({len(persist_bad)/n:.1%}). "
            f"Rough split: **{len(persist_strategy_fail)}** with repeated strategy/format path issues vs "
            f"**{len(persist_semantic)}** other persistent errors."
        ),
        implication=(
            "Persistent failures bound **what the current objective/data do not repair**: "
            "either systematic strategy-generation limits, 8B reasoning limits, or task instances outside training coverage."
        ),
    )

    section_case_analysis(
        "Case Analysis IV: Consistently Strong Behaviors",
        persist_good,
        good_split,
        good_pt,
        headline=(
            f"**{len(persist_good)}** rollouts are correct in both runs ({len(persist_good)/n:.1%}), "
            f"dominated by: {top3_line(good_split)} / {top3_line(good_pt)}."
        ),
        implication=(
            "These are **low-risk regions** under strategy perturbation: downstream already succeeds; "
            "training mainly should avoid breaking them (contrast with quadrant II)."
        ),
    )

    lines.append("\n## Artifacts\n\n")
    lines.append(f"- Merged CSV: `{args.out_csv}`\n")

    args.out_md.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_md} ({n} rows)")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
