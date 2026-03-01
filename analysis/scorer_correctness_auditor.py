#!/usr/bin/env python3
"""Offline audit for scorer/correctness alignment on validation outputs.

Usage:
    python analysis/scorer_correctness_auditor.py \
        --val-dir checkpoints_strategy_gen/<run>/validation_outputs/<run> \
        --output-dir analysis
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


JsonDict = Dict[str, Any]
AnswerModelFn = Callable[[str, str], str]


def load_json_robust(filepath: Path) -> List[JsonDict]:
    """Load a JSON array file, tolerating trailing content."""
    content = filepath.read_text(encoding="utf-8")
    first_bracket = content.find("[")
    if first_bracket == -1:
        return []

    stack = 0
    in_string = False
    escape = False
    for idx, char in enumerate(content[first_bracket:], first_bracket):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "[":
            stack += 1
        elif char == "]":
            stack -= 1
            if stack == 0:
                payload = content[first_bracket : idx + 1]
                loaded = json.loads(payload)
                return loaded if isinstance(loaded, list) else []
    return []


def _discover_validation_files(val_dir: Path) -> List[Path]:
    global_pattern = re.compile(r"validation_global_step(\d+)\.json$")
    shard_pattern = re.compile(r"validation_step(\d+)_worker\d+\.json$")

    global_files: List[Tuple[int, Path]] = []
    shard_groups: Dict[int, List[Path]] = defaultdict(list)

    for path in sorted(val_dir.iterdir()):
        if not path.is_file():
            continue
        global_match = global_pattern.match(path.name)
        if global_match:
            global_files.append((int(global_match.group(1)), path))
            continue
        shard_match = shard_pattern.match(path.name)
        if shard_match:
            shard_groups[int(shard_match.group(1))].append(path)

    if global_files:
        return [path for _, path in sorted(global_files)]

    return [path for _, paths in sorted(shard_groups.items()) for path in sorted(paths)]


def load_rollout_records(val_dir: str) -> List[JsonDict]:
    """Load validation records from merged global files or worker shards."""
    base_dir = Path(val_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"validation directory not found: {base_dir}")

    records: List[JsonDict] = []
    for path in _discover_validation_files(base_dir):
        loaded = load_json_robust(path)
        if loaded:
            records.extend(item for item in loaded if isinstance(item, dict))
    return records


def infer_default_val_dir(repo_root: Optional[Path] = None) -> Path:
    """Infer the newest strategy-generation validation output directory."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]

    checkpoints_root = repo_root / "checkpoints_strategy_gen"
    if not checkpoints_root.exists():
        raise FileNotFoundError(
            "could not infer --val-dir: checkpoints_strategy_gen directory does not exist"
        )

    candidates: List[Path] = []
    for run_dir in checkpoints_root.iterdir():
        if not run_dir.is_dir():
            continue
        candidate = run_dir / "validation_outputs" / run_dir.name
        if candidate.is_dir() and _discover_validation_files(candidate):
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            "could not infer --val-dir: no validation output directories were found under "
            f"{checkpoints_root}"
        )

    return sorted(candidates)[-1]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _reward_value(record: JsonDict, *keys: str, default: float = 0.0) -> float:
    reward = record.get("reward")
    if not isinstance(reward, dict):
        return default
    for key in keys:
        if key in reward:
            return _to_float(reward.get(key), default=default)
    return default


def _strategy_text(record: JsonDict) -> str:
    output = record.get("output")
    if not isinstance(output, dict):
        return ""
    value = output.get("strategy_extracted")
    return value if isinstance(value, str) else ""


def _problem_type(record: JsonDict) -> str:
    value = record.get("problem_type")
    return value if isinstance(value, str) and value else "unknown"


def _rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n != len(y) or n < 2:
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sum_xy = 0.0
    sum_xx = 0.0
    sum_yy = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mean_x
        dy = yi - mean_y
        sum_xy += dx * dy
        sum_xx += dx * dx
        sum_yy += dy * dy
    if sum_xx <= 0.0 or sum_yy <= 0.0:
        return None
    return sum_xy / math.sqrt(sum_xx * sum_yy)


def spearman_rank_correlation(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Compute Spearman rho without scipy."""
    if len(x) != len(y) or len(x) < 2:
        return None
    return _pearson(_rankdata(x), _rankdata(y))


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def _distribution(values: Sequence[float], precision: int = 2) -> Dict[str, int]:
    counter = Counter(f"{value:.{precision}f}" for value in values)
    return dict(sorted(counter.items(), key=lambda item: float(item[0])))


def _top_problem_types(records: Sequence[JsonDict], limit: int = 10) -> List[JsonDict]:
    grouped: Dict[str, List[JsonDict]] = defaultdict(list)
    for record in records:
        grouped[_problem_type(record)].append(record)

    rows: List[JsonDict] = []
    for problem_type, items in grouped.items():
        scorer_values = [_reward_value(item, "scorer", "scorer_reward") for item in items]
        correctness_values = [
            _reward_value(item, "correctness", "correctness_reward") for item in items
        ]
        rows.append(
            {
                "problem_type": problem_type,
                "n_records": len(items),
                "mean_scorer": round(_mean(scorer_values), 4),
                "mean_correctness": round(_mean(correctness_values), 4),
                "spearman_scorer_correctness": _round_or_none(
                    spearman_rank_correlation(scorer_values, correctness_values)
                ),
            }
        )

    rows.sort(key=lambda row: (-int(row["n_records"]), str(row["problem_type"])))
    return rows[:limit]


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def run_audit(val_dir: str, output_dir: str = "analysis") -> JsonDict:
    """Run the offline audit and save ``audit_report.json``."""
    records = load_rollout_records(val_dir)
    scorer_scores = [_reward_value(record, "scorer", "scorer_reward") for record in records]
    correctnesses = [_reward_value(record, "correctness", "correctness_reward") for record in records]
    final_rewards = [_reward_value(record, "final", "final_reward") for record in records]
    parse_ok = sum(1 for record in records if _strategy_text(record).strip())

    top_bin_correctnesses = [
        correctness
        for scorer, correctness in zip(scorer_scores, correctnesses)
        if scorer > 0.8
    ]
    report: JsonDict = {
        "val_dir": str(Path(val_dir).resolve()),
        "n_records": len(records),
        "parse_rate": round(parse_ok / len(records), 4) if records else 0.0,
        "spearman_scorer_correctness": _round_or_none(
            spearman_rank_correlation(scorer_scores, correctnesses)
        ),
        "spearman_final_reward_correctness": _round_or_none(
            spearman_rank_correlation(final_rewards, correctnesses)
        ),
        "mean_scorer": round(_mean(scorer_scores), 4),
        "mean_correctness": round(_mean(correctnesses), 4),
        "mean_final_reward": round(_mean(final_rewards), 4),
        "top_scorer_bin_threshold": 0.8,
        "top_scorer_bin_size": len(top_bin_correctnesses),
        "top_scorer_bin_correctness": round(_mean(top_bin_correctnesses), 4),
        "approx_grounded_proxy_note": "Phase 0 uses single-sample correctness as a grounded_proxy approximation.",
        "approx_grounded_proxy_distribution": _distribution(correctnesses),
        "scorer_distribution": _distribution(scorer_scores),
        "top_problem_types": _top_problem_types(records),
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "audit_report.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rho = report["spearman_scorer_correctness"]
    print(f"[Audit] n={report['n_records']}, parse_rate={report['parse_rate']:.2%}")
    print(f"[Audit] Spearman(scorer, correctness) = {rho if rho is not None else 'N/A'}")
    print(
        "[Audit] Top scorer(>0.8) actual correct = "
        f"{report['top_scorer_bin_correctness']:.3f} (n={report['top_scorer_bin_size']})"
    )
    print(f"[Audit] Report written to {output_path}")
    return report


def load_answer_model_fn(spec: str) -> AnswerModelFn:
    """Load an answer model callable from ``module:function``."""
    if ":" not in spec:
        raise ValueError("--answer-model-fn must use module:function format")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise ValueError(f"callable not found: {spec}")
    return func


def counterfactual_audit(
    val_records: Sequence[JsonDict],
    answer_model_fn: AnswerModelFn,
    n_sample: int = 200,
    seed: int = 0,
) -> JsonDict:
    """Estimate whether strategies help or hurt answer performance."""
    if not val_records:
        return {"n_records": 0, "harm_rate": 0.0, "help_rate": 0.0, "neutral_rate": 0.0}

    rng = random.Random(seed)
    sample_size = min(n_sample, len(val_records))
    sample = rng.sample(list(val_records), sample_size)
    harm_count = 0
    help_count = 0
    neutral_count = 0

    for record in sample:
        input_payload = record.get("input", {})
        output_payload = record.get("output", {})
        if not isinstance(input_payload, dict) or not isinstance(output_payload, dict):
            neutral_count += 1
            continue

        problem = input_payload.get("problem", "")
        ground_truth = input_payload.get("ground_truth", "")
        strategy = output_payload.get("strategy_extracted", "")
        if not isinstance(problem, str) or not isinstance(ground_truth, str) or not isinstance(strategy, str):
            neutral_count += 1
            continue

        correct_with = answer_model_fn(strategy, problem) == ground_truth
        correct_without = answer_model_fn("", problem) == ground_truth

        if correct_without and not correct_with:
            harm_count += 1
        elif not correct_without and correct_with:
            help_count += 1
        else:
            neutral_count += 1

    return {
        "n_records": sample_size,
        "harm_rate": round(harm_count / sample_size, 4) if sample_size else 0.0,
        "help_rate": round(help_count / sample_size, 4) if sample_size else 0.0,
        "neutral_rate": round(neutral_count / sample_size, 4) if sample_size else 0.0,
        "seed": seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit scorer/correctness alignment from validation outputs."
    )
    parser.add_argument(
        "--val-dir",
        default=None,
        help=(
            "Path to validation output directory. "
            "If omitted, auto-detect the newest checkpoints_strategy_gen run."
        ),
    )
    parser.add_argument("--output-dir", default="analysis", help="Directory to write audit_report.json")
    parser.add_argument(
        "--counterfactual",
        action="store_true",
        help="Run optional strategy help/harm audit using an injected answer model callable",
    )
    parser.add_argument(
        "--answer-model-fn",
        default="",
        help="Callable in module:function format, used only with --counterfactual",
    )
    parser.add_argument(
        "--counterfactual-sample-size",
        type=int,
        default=200,
        help="Sample size for counterfactual audit",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for counterfactual sampling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val_dir = args.val_dir
    if not val_dir:
        val_dir = str(infer_default_val_dir())
        print(f"[Audit] Auto-selected val_dir={val_dir}")

    report = run_audit(val_dir, args.output_dir)

    if args.counterfactual:
        if not args.answer_model_fn:
            raise SystemExit("--counterfactual requires --answer-model-fn module:function")
        answer_model_fn = load_answer_model_fn(args.answer_model_fn)
        cf_report = counterfactual_audit(
            load_rollout_records(val_dir),
            answer_model_fn=answer_model_fn,
            n_sample=args.counterfactual_sample_size,
            seed=args.seed,
        )
        report["counterfactual"] = cf_report
        output_path = Path(args.output_dir) / "audit_report.json"
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[Counterfactual] help={cf_report['help_rate']:.1%} harm={cf_report['harm_rate']:.1%}")


if __name__ == "__main__":
    main()
