#!/usr/bin/env python3
"""L2.5 paired uplift auditor for with/without strategy evaluations.

Expected input format (JSON file):
[
  {"seed": 1, "accuracy": 0.41},
  {"seed": 2, "accuracy": 0.39}
]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def _normal_pvalue_from_t(t_value: float) -> float:
    # Two-sided normal approximation fallback.
    return math.erfc(abs(t_value) / math.sqrt(2.0))


def _bootstrap_ci(values: Sequence[float], n_bootstrap: int = 2000, seed: int = 42) -> Tuple[float, float]:
    import random

    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def _load_seed_accuracy(path: str) -> Dict[int, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON: {path}")
    out: Dict[int, float] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        seed = item.get("seed")
        acc = item.get("accuracy")
        if isinstance(seed, int) and isinstance(acc, (int, float)):
            out[seed] = float(acc)
    if not out:
        raise ValueError(f"No valid {{seed, accuracy}} rows found in: {path}")
    return out


def compute_uplift(
    with_strategy: Dict[int, float],
    without_strategy: Dict[int, float],
    bootstrap: int,
) -> Dict[str, object]:
    common = sorted(set(with_strategy) & set(without_strategy))
    if len(common) < 2:
        raise ValueError("Need at least 2 shared seeds for paired uplift audit.")

    deltas = [with_strategy[s] - without_strategy[s] for s in common]
    mu = _mean(deltas)
    sd = _std(deltas)
    n = len(deltas)
    se = sd / math.sqrt(n) if n > 0 else 0.0
    t_value = (mu / se) if se > 0 else 0.0
    pvalue = _normal_pvalue_from_t(t_value)
    ci_low, ci_high = _bootstrap_ci(deltas, n_bootstrap=bootstrap)
    effect_size_dz = (mu / sd) if sd > 0 else 0.0

    return {
        "n_seeds": n,
        "shared_seeds": common,
        "deltas": deltas,
        "strategy_uplift_mean": mu,
        "strategy_uplift_ci95": {"low": ci_low, "high": ci_high},
        "strategy_uplift_pvalue": pvalue,
        "strategy_uplift_effect_size_dz": effect_size_dz,
        "thresholds": {
            "strategy_uplift_mean_gt": 0.0,
            "strategy_uplift_pvalue_lt": 0.05,
            "strategy_uplift_ci95_low_gt": 0.0,
            "strategy_uplift_effect_size_dz_gte": 0.2,
        },
        "pvalue_method": "normal_approx_from_paired_t_stat",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-1.5 L2.5: paired strategy uplift audit.")
    parser.add_argument("--with-strategy-json", required=True, help="JSON file with per-seed accuracy (with strategy).")
    parser.add_argument(
        "--without-strategy-json",
        required=True,
        help="JSON file with per-seed accuracy (without strategy).",
    )
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap rounds for uplift CI.")
    parser.add_argument("--output", default="analysis/strategy_uplift_report.json", help="Output report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with_strategy = _load_seed_accuracy(args.with_strategy_json)
    without_strategy = _load_seed_accuracy(args.without_strategy_json)
    report = compute_uplift(with_strategy, without_strategy, bootstrap=args.bootstrap)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        "[L2.5] mean={:.4f}, pvalue={:.4g}, ci95=({:.4f}, {:.4f})".format(
            report["strategy_uplift_mean"],
            report["strategy_uplift_pvalue"],
            report["strategy_uplift_ci95"]["low"],
            report["strategy_uplift_ci95"]["high"],
        )
    )
    print(f"[L2.5] report written to {output_path}")


if __name__ == "__main__":
    main()
