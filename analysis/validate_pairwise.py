#!/usr/bin/env python3
"""L2 validation for in-group ranking and effective gradient signal.

This script evaluates whether reward values induce useful within-group ranking
for GRPO-style optimization.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from scorer_correctness_auditor import load_rollout_records

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


def _reward(record: JsonDict, key: str) -> float:
    reward = record.get("reward")
    if not isinstance(reward, dict):
        return 0.0
    return _to_float(reward.get(key), 0.0)


def _correctness(record: JsonDict) -> float:
    reward = record.get("reward")
    if not isinstance(reward, dict):
        return 0.0
    if "grounded_proxy" in reward:
        return _to_float(reward.get("grounded_proxy"), 0.0)
    if "correctness" in reward:
        return _to_float(reward.get("correctness"), 0.0)
    return _to_float(reward.get("correctness_reward"), 0.0)


def _group_key(record: JsonDict) -> Tuple[str, str]:
    payload = record.get("input")
    if not isinstance(payload, dict):
        return ("", "")
    problem = payload.get("problem", "")
    examples = payload.get("user_prompt", "")
    return (str(problem), str(examples))


def _pairwise_accuracy_for_group(records: Sequence[JsonDict], reward_key: str) -> Tuple[int, int]:
    right = 0
    total = 0
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            yi = _correctness(records[i])
            yj = _correctness(records[j])
            if yi == yj:
                continue
            ri = _reward(records[i], reward_key)
            rj = _reward(records[j], reward_key)
            total += 1
            if (yi > yj and ri > rj) or (yj > yi and rj > ri):
                right += 1
            elif ri == rj:
                right += 0
    return right, total


def _pairwise_margin_calibration(records: Sequence[JsonDict], reward_key: str) -> Optional[float]:
    margins: List[float] = []
    labels: List[float] = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            yi = _correctness(records[i])
            yj = _correctness(records[j])
            if yi == yj:
                continue
            ri = _reward(records[i], reward_key)
            rj = _reward(records[j], reward_key)
            margins.append(abs(ri - rj))
            labels.append(abs(yi - yj))
    if not margins:
        return None
    mean_margin = _mean(margins)
    mean_label_gap = _mean(labels)
    return mean_margin / (mean_label_gap + 1e-8)


def evaluate(
    records: Sequence[JsonDict],
    reward_key: str,
    old_reward_key: str,
    std_eps: float,
) -> JsonDict:
    groups: Dict[Tuple[str, str], List[JsonDict]] = defaultdict(list)
    for record in records:
        groups[_group_key(record)].append(record)

    eligible = [g for g in groups.values() if len(g) >= 2]
    if not eligible:
        raise ValueError("No in-group multi-rollout samples found. Need >=2 records per group.")

    total_right = 0
    total_pairs = 0
    old_right = 0
    old_pairs = 0
    std_values = []
    old_std_values = []
    effective_groups = 0
    margin_scores: List[float] = []

    for group in eligible:
        right, pairs = _pairwise_accuracy_for_group(group, reward_key)
        total_right += right
        total_pairs += pairs

        o_right, o_pairs = _pairwise_accuracy_for_group(group, old_reward_key)
        old_right += o_right
        old_pairs += o_pairs

        s = _std([_reward(x, reward_key) for x in group])
        old_s = _std([_reward(x, old_reward_key) for x in group])
        std_values.append(s)
        old_std_values.append(old_s)
        if s > std_eps:
            effective_groups += 1

        m = _pairwise_margin_calibration(group, reward_key)
        if m is not None:
            margin_scores.append(m)

    return {
        "n_total_records": len(records),
        "n_groups_total": len(groups),
        "n_groups_eligible": len(eligible),
        "pairwise_accuracy_in_group": (total_right / total_pairs) if total_pairs else None,
        "pairwise_accuracy_in_group_old_reward": (old_right / old_pairs) if old_pairs else None,
        "pairwise_margin_calibration": _mean(margin_scores) if margin_scores else None,
        "effective_gradient_group_rate": effective_groups / len(eligible),
        "reward_std_within_group": _mean(std_values),
        "reward_std_within_group_old_reward": _mean(old_std_values),
        "thresholds": {
            "pairwise_accuracy_in_group_gt": 0.60,
            "effective_gradient_group_rate_gt": 0.55,
        },
        "notes": {
            "cross_sample_pairwise": "Not treated as gate metric; in-group metrics are primary.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-1.5 L2: validate pairwise in-group ranking.")
    parser.add_argument("--val-dir", required=True, help="Validation outputs directory.")
    parser.add_argument("--reward-key", default="final", help="Primary reward key in record['reward'].")
    parser.add_argument("--old-reward-key", default="scorer", help="Baseline reward key for side-by-side comparison.")
    parser.add_argument("--std-eps", type=float, default=1e-6, help="Std threshold for effective-gradient groups.")
    parser.add_argument("--output", default="analysis/pairwise_validation_report.json", help="Output report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_rollout_records(args.val_dir)
    if not records:
        raise SystemExit(f"No validation records found in: {args.val_dir}")

    report = evaluate(
        records,
        reward_key=args.reward_key,
        old_reward_key=args.old_reward_key,
        std_eps=args.std_eps,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[L2] groups={report['n_groups_eligible']}/{report['n_groups_total']}")
    print(f"[L2] pairwise_acc={report['pairwise_accuracy_in_group']}")
    print(f"[L2] report written to {output_path}")


if __name__ == "__main__":
    main()
