#!/usr/bin/env python3
"""L1 validation for outcome-conditioned scorer quality.

This script implements the Phase-1.5 L1 checks on validation rollouts:
- boundary slice (problem types with mean correctness in [0.20, 0.45])
- stratified natural slice
- scorer/correctness monotonicity and spread diagnostics
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from scorer_correctness_auditor import load_rollout_records, spearman_rank_correlation

JsonDict = Dict[str, Any]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * q
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _reward_dict(record: JsonDict) -> JsonDict:
    reward = record.get("reward")
    return reward if isinstance(reward, dict) else {}


def _problem_type(record: JsonDict) -> str:
    value = record.get("problem_type")
    return value if isinstance(value, str) and value else "unknown"


def _correctness(record: JsonDict) -> float:
    reward = _reward_dict(record)
    for key in ("correctness", "correctness_reward"):
        if key in reward:
            return _to_float(reward[key], default=0.0)
    return 0.0


def _oc_score(record: JsonDict) -> float:
    reward = _reward_dict(record)
    if "oc_final_score_100" in reward:
        return max(0.0, min(1.0, _to_float(reward["oc_final_score_100"]) / 100.0))
    for key in ("oc_scorer", "scorer", "scorer_reward"):
        if key in reward:
            return max(0.0, min(1.0, _to_float(reward[key])))
    return 0.0


def _dim_a(record: JsonDict) -> Optional[float]:
    reward = _reward_dict(record)
    dims = reward.get("oc_dimension_scores")
    if not isinstance(dims, dict):
        return None
    if "A" in dims:
        return _to_float(dims.get("A"), default=0.0)
    return None


def _has_cap_or_penalty(record: JsonDict) -> Dict[str, bool]:
    reward = _reward_dict(record)
    cap_flags = reward.get("oc_cap_flags")
    penalties = reward.get("oc_penalties")
    has_cap = bool(cap_flags) if isinstance(cap_flags, (dict, list, tuple, set)) else False
    has_penalty = bool(penalties) if isinstance(penalties, (dict, list, tuple, set)) else False
    return {"has_cap": has_cap, "has_penalty": has_penalty}


def _group_by_problem_type(records: Sequence[JsonDict]) -> Dict[str, List[JsonDict]]:
    grouped: Dict[str, List[JsonDict]] = defaultdict(list)
    for record in records:
        grouped[_problem_type(record)].append(record)
    return grouped


def build_boundary_slice(
    records: Sequence[JsonDict],
    min_size: int,
    max_size: int,
    seed: int,
) -> List[JsonDict]:
    grouped = _group_by_problem_type(records)
    boundary_types: List[str] = []
    for problem_type, items in grouped.items():
        mean_corr = _mean(_correctness(x) for x in items)
        if 0.20 <= mean_corr <= 0.45:
            boundary_types.append(problem_type)

    candidates: List[JsonDict] = []
    for problem_type in boundary_types:
        candidates.extend(grouped[problem_type])

    rng = random.Random(seed)
    rng.shuffle(candidates)
    size = min(max_size, max(min_size, len(candidates)))
    return candidates[:size]


def build_natural_slice(
    records: Sequence[JsonDict],
    target_size: int,
    seed: int,
) -> List[JsonDict]:
    grouped = _group_by_problem_type(records)
    total = sum(len(v) for v in grouped.values())
    if total == 0:
        return []

    rng = random.Random(seed)
    out: List[JsonDict] = []
    for items in grouped.values():
        items_copy = list(items)
        rng.shuffle(items_copy)
        quota = max(1, round(target_size * len(items_copy) / total))
        out.extend(items_copy[: min(quota, len(items_copy))])

    rng.shuffle(out)
    return out[:target_size]


def _bootstrap_spearman_ci(
    scores: Sequence[float],
    labels: Sequence[float],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Optional[float]]:
    if len(scores) < 5 or len(scores) != len(labels):
        return {"low": None, "high": None}

    rng = random.Random(seed)
    boots: List[float] = []
    n = len(scores)
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        s = [scores[i] for i in idxs]
        y = [labels[i] for i in idxs]
        rho = spearman_rank_correlation(s, y)
        if rho is not None:
            boots.append(float(rho))
    if not boots:
        return {"low": None, "high": None}
    boots.sort()
    return {
        "low": _percentile(boots, 0.025),
        "high": _percentile(boots, 0.975),
    }


def evaluate_slice(
    records: Sequence[JsonDict],
    n_bootstrap: int,
    seed: int,
) -> JsonDict:
    scores = [_oc_score(r) for r in records]
    correctness = [_correctness(r) for r in records]
    rho = spearman_rank_correlation(scores, correctness)
    score_std = _std(scores)

    threshold = _percentile(scores, 0.8)
    top_mask = [s >= threshold for s in scores]
    top_corr = [c for c, keep in zip(correctness, top_mask) if keep]
    overall_corr = _mean(correctness)
    top_uplift = _mean(top_corr) - overall_corr if top_corr else 0.0

    a_correct = []
    a_incorrect = []
    for record, corr in zip(records, correctness):
        a = _dim_a(record)
        if a is None:
            continue
        if corr >= 1.0:
            a_correct.append(a)
        else:
            a_incorrect.append(a)

    caps = 0
    penalties = 0
    for record in records:
        flags = _has_cap_or_penalty(record)
        caps += 1 if flags["has_cap"] else 0
        penalties += 1 if flags["has_penalty"] else 0

    ci = _bootstrap_spearman_ci(scores, correctness, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "n": len(records),
        "spearman_new_oc_correctness": rho,
        "top_20pct_correctness_uplift": top_uplift,
        "new_score_std": score_std,
        "a_outcome_support_gap": (_mean(a_correct) - _mean(a_incorrect)) if a_correct and a_incorrect else None,
        "cap_trigger_rate": caps / len(records) if records else 0.0,
        "penalty_trigger_rate": penalties / len(records) if records else 0.0,
        "bootstrap_ci95_spearman": ci,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-1.5 L1: validate OC-scorer output quality.")
    parser.add_argument("--val-dir", required=True, help="Validation outputs directory.")
    parser.add_argument("--output", default="analysis/oc_scorer_validation_report.json", help="Output report path.")
    parser.add_argument("--boundary-min-size", type=int, default=300)
    parser.add_argument("--boundary-max-size", type=int, default=500)
    parser.add_argument("--natural-size", type=int, default=800)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_rollout_records(args.val_dir)
    if not records:
        raise SystemExit(f"No validation records found in: {args.val_dir}")

    boundary = build_boundary_slice(
        records,
        min_size=args.boundary_min_size,
        max_size=args.boundary_max_size,
        seed=args.seed,
    )
    natural = build_natural_slice(records, target_size=args.natural_size, seed=args.seed + 1)

    report = {
        "val_dir": str(Path(args.val_dir).resolve()),
        "total_records": len(records),
        "boundary_slice": evaluate_slice(boundary, n_bootstrap=args.bootstrap, seed=args.seed + 11),
        "natural_slice": evaluate_slice(natural, n_bootstrap=args.bootstrap, seed=args.seed + 17),
        "pass_thresholds": {
            "boundary_spearman_gt": 0.15,
            "natural_spearman_gt": 0.10,
            "top_20pct_uplift_gte": 0.08,
            "new_score_std_gt": 0.12,
            "a_gap_gte": 0.8,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    b = report["boundary_slice"]
    n = report["natural_slice"]
    print(f"[L1] total={report['total_records']}, boundary={b['n']}, natural={n['n']}")
    print(f"[L1] spearman(boundary)={b['spearman_new_oc_correctness']}, spearman(natural)={n['spearman_new_oc_correctness']}")
    print(f"[L1] report written to {output_path}")


if __name__ == "__main__":
    main()
