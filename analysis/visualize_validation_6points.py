#!/usr/bin/env python3
"""Visualize validation outputs at 6 evenly spaced steps (static PNG).

Usage:
    python visualize_validation_6points.py \
        /path/to/validation_outputs/20260221_190204 \
        --steps 0,150,300,450,600,750 \
        --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid")
except Exception:  # pragma: no cover
    sns = None


def load_json_robust(filepath: Path) -> List[Dict[str, Any]]:
    """Load a JSON file that may contain a single array (possibly with trailing data)."""
    content = filepath.read_text(encoding="utf-8")
    first_bracket = content.find("[")
    if first_bracket == -1:
        return []

    stack = 0
    in_string = False
    escape = False
    for i, char in enumerate(content[first_bracket:], first_bracket):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "[":
            stack += 1
        elif char == "]":
            stack -= 1
            if stack == 0:
                return json.loads(content[first_bracket : i + 1])
    return []


def parse_steps(text: str) -> List[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def is_missing_text(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def load_step_data(validation_dir: Path, step: int) -> List[Dict[str, Any]]:
    global_path = validation_dir / f"validation_global_step{step}.json"
    if global_path.exists():
        return load_json_robust(global_path)

    shard_glob = f"validation_step{step}_worker*.json"
    shard_paths = sorted(validation_dir.glob(shard_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No global or shard files found for step {step}")
    merged: List[Dict[str, Any]] = []
    for p in shard_paths:
        merged.extend(load_json_robust(p))
    return merged


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not math.isnan(float(x))


def compute_step_metrics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    sums = defaultdict(float)
    counts = defaultdict(int)

    failure_counts = defaultdict(int)
    missing_strategy = 0
    missing_answer = 0

    per_pt_sum = defaultdict(float)
    per_pt_cnt = defaultdict(int)
    per_pt_corr_sum = defaultdict(float)

    scatter_points: List[Tuple[float, float, float]] = []  # scorer, correctness, format
    strat_len_points: List[Tuple[int, float]] = []  # len(strategy), final
    ans_len_points: List[Tuple[int, float]] = []  # len(answer), correctness

    for item in data:
        reward = item.get("reward") or {}
        fmt = reward.get("format")
        scorer = reward.get("scorer")
        correctness = reward.get("correctness")
        final = reward.get("final")

        if is_number(fmt):
            sums["format"] += float(fmt)
            counts["format"] += 1
        if is_number(scorer):
            sums["scorer"] += float(scorer)
            counts["scorer"] += 1
        if is_number(correctness):
            sums["correctness"] += float(correctness)
            counts["correctness"] += 1
        if is_number(final):
            sums["final"] += float(final)
            counts["final"] += 1

        output = item.get("output") or {}
        strategy_extracted = output.get("strategy_extracted")
        answer_extracted = output.get("answer_extracted")

        if is_missing_text(strategy_extracted):
            missing_strategy += 1
        if is_missing_text(answer_extracted):
            missing_answer += 1

        if is_number(fmt) and float(fmt) == 0:
            failure_counts["format_fail"] += 1
        elif is_number(fmt) and float(fmt) == 1:
            if is_missing_text(answer_extracted):
                failure_counts["parse_fail"] += 1
            else:
                if is_number(correctness) and float(correctness) == 1:
                    failure_counts["correct"] += 1
                elif is_number(correctness) and float(correctness) == 0.5:
                    failure_counts["partial"] += 1
                else:
                    failure_counts["wrong_parseable"] += 1
        else:
            failure_counts["unknown"] += 1

        if is_number(scorer) and is_number(correctness) and is_number(fmt):
            scatter_points.append((float(scorer), float(correctness), float(fmt)))

        if isinstance(strategy_extracted, str) and is_number(final):
            strat_len_points.append((len(strategy_extracted), float(final)))
        if isinstance(answer_extracted, str) and is_number(correctness):
            ans_len_points.append((len(answer_extracted), float(correctness)))

        problem_type = item.get("problem_type")
        if problem_type and is_number(final):
            per_pt_sum[problem_type] += float(final)
            per_pt_cnt[problem_type] += 1
        if problem_type and is_number(correctness):
            per_pt_corr_sum[problem_type] += float(correctness)

    means = {
        k: (sums[k] / counts[k] if counts[k] else float("nan"))
        for k in ("final", "format", "scorer", "correctness")
    }

    n = len(data)
    missing = {
        "missing_strategy": missing_strategy,
        "missing_answer": missing_answer,
        "missing_strategy_rate": missing_strategy / n if n else float("nan"),
        "missing_answer_rate": missing_answer / n if n else float("nan"),
    }

    per_pt_mean = {
        p: per_pt_sum[p] / per_pt_cnt[p] for p in per_pt_cnt if per_pt_cnt[p]
    }
    per_pt_corr_mean = {
        p: per_pt_corr_sum[p] / per_pt_cnt[p] for p in per_pt_cnt if per_pt_cnt[p]
    }

    return {
        "n": n,
        "means": means,
        "failure_counts": dict(failure_counts),
        "missing": missing,
        "per_pt_mean": per_pt_mean,
        "per_pt_corr_mean": per_pt_corr_mean,
        "scatter_points": scatter_points,
        "strat_len_points": strat_len_points,
        "ans_len_points": ans_len_points,
    }


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_metric_trends(steps: List[int], metrics: Dict[int, Dict[str, float]], out: Path) -> None:
    labels = ["final", "format", "scorer", "correctness"]
    plt.figure(figsize=(9, 5))
    for label in labels:
        ys = [metrics[s][label] for s in steps]
        plt.plot(steps, ys, marker="o", linewidth=2, label=label)
    plt.ylim(0, 1.05)
    plt.xlabel("Step")
    plt.ylabel("Mean")
    plt.title("Metric Trends (6 evenly spaced steps)")
    plt.legend(loc="lower right")
    save_fig(out / "metric_trends.png")


def plot_failure_buckets(steps: List[int], failures: Dict[int, Dict[str, int]], out: Path) -> None:
    order = ["correct", "partial", "wrong_parseable", "parse_fail", "format_fail", "unknown"]
    colors = {
        "correct": "#2ca02c",
        "partial": "#1f77b4",
        "wrong_parseable": "#ff7f0e",
        "parse_fail": "#9467bd",
        "format_fail": "#d62728",
        "unknown": "#7f7f7f",
    }
    totals = []
    for s in steps:
        totals.append(sum(failures[s].get(k, 0) for k in order))

    plt.figure(figsize=(10, 5))
    bottoms = np.zeros(len(steps))
    for k in order:
        vals = [failures[s].get(k, 0) / totals[i] if totals[i] else 0 for i, s in enumerate(steps)]
        plt.bar(steps, vals, bottom=bottoms, label=k, color=colors[k])
        bottoms += np.array(vals)
    plt.ylim(0, 1.05)
    plt.xlabel("Step")
    plt.ylabel("Fraction")
    plt.title("Failure Bucket Composition")
    plt.legend(loc="upper right")
    save_fig(out / "failure_buckets.png")


def plot_missing_rates(steps: List[int], missing: Dict[int, Dict[str, float]], out: Path) -> None:
    x = np.arange(len(steps))
    w = 0.35
    strat = [missing[s]["missing_strategy_rate"] for s in steps]
    ans = [missing[s]["missing_answer_rate"] for s in steps]

    plt.figure(figsize=(9, 5))
    plt.bar(x - w / 2, strat, width=w, label="strategy_extracted missing")
    plt.bar(x + w / 2, ans, width=w, label="answer_extracted missing")
    plt.xticks(x, steps)
    plt.ylim(0, 1.05)
    plt.xlabel("Step")
    plt.ylabel("Missing rate")
    plt.title("Extraction Missing Rates")
    plt.legend(loc="upper right")
    save_fig(out / "missing_rates.png")


def plot_heatmap(
    steps: List[int],
    pt_means: Dict[int, Dict[str, float]],
    out: Path,
    metric_name: str = "final",
) -> None:
    problem_types = sorted(
        {p for step in pt_means for p in pt_means[step].keys()}
    )
    # Order by last step performance
    last = steps[-1]
    problem_types.sort(key=lambda p: pt_means[last].get(p, float("nan")), reverse=True)

    matrix = np.full((len(problem_types), len(steps)), np.nan, dtype=float)
    for i, p in enumerate(problem_types):
        for j, s in enumerate(steps):
            matrix[i, j] = pt_means[s].get(p, float("nan"))

    plt.figure(figsize=(10, max(6, len(problem_types) * 0.25)))
    if sns:
        sns.heatmap(
            matrix,
            cmap="viridis",
            cbar=True,
            yticklabels=problem_types,
            xticklabels=steps,
            vmin=0,
            vmax=1,
        )
    else:
        plt.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        plt.colorbar()
        plt.yticks(np.arange(len(problem_types)), problem_types)
        plt.xticks(np.arange(len(steps)), steps)
    plt.xlabel("Step")
    plt.ylabel("Problem type")
    plt.title(f"Problem Type Heatmap (mean {metric_name})")
    save_fig(out / f"heatmap_problem_type_{metric_name}.png")


def plot_dumbbell(
    step_a: int,
    step_b: int,
    pt_means: Dict[int, Dict[str, float]],
    out: Path,
    top_n: int = 20,
) -> None:
    vals = []
    for p in pt_means.get(step_a, {}):
        if p not in pt_means.get(step_b, {}):
            continue
        a = pt_means[step_a][p]
        b = pt_means[step_b][p]
        vals.append((abs(b - a), p, a, b))
    vals.sort(reverse=True)
    vals = vals[:top_n]

    labels = [v[1] for v in vals][::-1]
    a_vals = [v[2] for v in vals][::-1]
    b_vals = [v[3] for v in vals][::-1]

    y = np.arange(len(labels))
    plt.figure(figsize=(9, max(5, len(labels) * 0.4)))
    for i in range(len(labels)):
        plt.plot([a_vals[i], b_vals[i]], [y[i], y[i]], color="#999999", zorder=1)
    plt.scatter(a_vals, y, color="#1f77b4", label=f"Step {step_a}", zorder=2)
    plt.scatter(b_vals, y, color="#d62728", label=f"Step {step_b}", zorder=2)
    plt.yticks(y, labels)
    plt.xlim(0, 1.0)
    plt.xlabel("Mean final")
    plt.title(f"Top {top_n} Problem Types by |Δ| (Step {step_a} → {step_b})")
    plt.legend(loc="lower right")
    save_fig(out / f"dumbbell_delta_{step_a}_{step_b}.png")


def plot_scorer_correctness(scatter_points: List[Tuple[float, float, float]], out: Path) -> None:
    if not scatter_points:
        return
    scorer = np.array([p[0] for p in scatter_points])
    correctness = np.array([p[1] for p in scatter_points])
    fmt = np.array([p[2] for p in scatter_points])
    jitter = (np.random.rand(len(correctness)) - 0.5) * 0.05
    y = correctness + jitter

    plt.figure(figsize=(9, 5))
    mask_fmt1 = fmt == 1
    plt.scatter(scorer[mask_fmt1], y[mask_fmt1], s=12, alpha=0.25, label="format=1")
    plt.scatter(scorer[~mask_fmt1], y[~mask_fmt1], s=12, alpha=0.5, label="format=0", color="#d62728")
    plt.yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Scorer")
    plt.ylabel("Correctness (jittered)")
    plt.title("Scorer vs Correctness (all selected steps)")
    plt.legend(loc="lower right")
    save_fig(out / "scatter_scorer_correctness.png")


def plot_strategy_len_vs_final(points: List[Tuple[int, float]], out: Path) -> None:
    if not points:
        return
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    plt.figure(figsize=(8, 5))
    plt.hexbin(x, y, gridsize=40, cmap="Blues", mincnt=1)
    plt.colorbar(label="count")
    plt.xlabel("len(strategy_extracted)")
    plt.ylabel("final")
    plt.ylim(0, 1.05)
    plt.title("Strategy Length vs Final")
    save_fig(out / "hexbin_strategylen_final.png")


def plot_answer_len_vs_correctness(points: List[Tuple[int, float]], out: Path) -> None:
    if not points:
        return
    groups: Dict[float, List[int]] = defaultdict(list)
    for length, corr in points:
        groups[corr].append(length)

    order = [0.0, 0.5, 1.0]
    data = [groups.get(k, []) for k in order]
    labels = [str(k) for k in order]

    plt.figure(figsize=(7, 5))
    if sns:
        sns.boxplot(data=data)
        plt.xticks(range(len(labels)), labels)
    else:
        plt.boxplot(data, labels=labels)
    plt.xlabel("Correctness")
    plt.ylabel("len(answer_extracted)")
    plt.title("Answer Length by Correctness")
    save_fig(out / "box_answerlen_correctness.png")


def write_csvs(
    steps: List[int],
    metrics: Dict[int, Dict[str, float]],
    failures: Dict[int, Dict[str, int]],
    missing: Dict[int, Dict[str, float]],
    pt_means: Dict[int, Dict[str, float]],
    pt_corr_means: Dict[int, Dict[str, float]],
    out: Path,
) -> None:
    with open(out / "overall_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "n", "final", "format", "scorer", "correctness"])
        for s in steps:
            w.writerow([s, metrics[s]["n"], metrics[s]["final"], metrics[s]["format"], metrics[s]["scorer"], metrics[s]["correctness"]])

    with open(out / "failure_buckets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "bucket", "count"])
        for s in steps:
            for k, v in failures[s].items():
                w.writerow([s, k, v])

    with open(out / "missing_rates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "missing_strategy_rate", "missing_answer_rate"])
        for s in steps:
            w.writerow([s, missing[s]["missing_strategy_rate"], missing[s]["missing_answer_rate"]])

    with open(out / "problem_type_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["problem_type", "step", "mean_final", "mean_correctness"])
        all_pts = sorted({p for s in pt_means for p in pt_means[s]})
        for p in all_pts:
            for s in steps:
                w.writerow([p, s, pt_means[s].get(p, ""), pt_corr_means[s].get(p, "")])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_dir", type=str, help="Path to validation_outputs/<run_id>")
    parser.add_argument(
        "--steps",
        type=str,
        default="0,150,300,450,600,750",
        help="Comma-separated steps (default: 0,150,300,450,600,750)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help=(
            "Output directory for PNG and CSV "
            "(default: <validation_dir>/analysis_charts_6points)"
        ),
    )
    parser.add_argument(
        "--dumbbell-top",
        type=int,
        default=20,
        help="Top-N problem types by |delta| for dumbbell plot",
    )

    args = parser.parse_args()
    validation_dir = Path(args.validation_dir)
    steps = parse_steps(args.steps)

    if not validation_dir.exists():
        raise SystemExit(f"validation_dir not found: {validation_dir}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else validation_dir / "analysis_charts_6points"
    )
    ensure_output_dir(output_dir)

    metrics: Dict[int, Dict[str, float]] = {}
    failures: Dict[int, Dict[str, int]] = {}
    missing: Dict[int, Dict[str, float]] = {}
    pt_means: Dict[int, Dict[str, float]] = {}
    pt_corr_means: Dict[int, Dict[str, float]] = {}
    scatter_points: List[Tuple[float, float, float]] = []
    strat_len_points: List[Tuple[int, float]] = []
    ans_len_points: List[Tuple[int, float]] = []

    for step in steps:
        data = load_step_data(validation_dir, step)
        summary = compute_step_metrics(data)
        metrics[step] = {"n": summary["n"], **summary["means"]}
        failures[step] = summary["failure_counts"]
        missing[step] = summary["missing"]
        pt_means[step] = summary["per_pt_mean"]
        pt_corr_means[step] = summary["per_pt_corr_mean"]
        scatter_points.extend(summary["scatter_points"])
        strat_len_points.extend(summary["strat_len_points"])
        ans_len_points.extend(summary["ans_len_points"])

    plot_metric_trends(steps, metrics, output_dir)
    plot_failure_buckets(steps, failures, output_dir)
    plot_missing_rates(steps, missing, output_dir)
    plot_heatmap(steps, pt_means, output_dir, metric_name="final")
    plot_dumbbell(steps[0], steps[-1], pt_means, output_dir, top_n=args.dumbbell_top)
    plot_scorer_correctness(scatter_points, output_dir)
    plot_strategy_len_vs_final(strat_len_points, output_dir)
    plot_answer_len_vs_correctness(ans_len_points, output_dir)

    write_csvs(steps, metrics, failures, missing, pt_means, pt_corr_means, output_dir)

    print(f"Saved charts and CSVs to: {output_dir}")


if __name__ == "__main__":
    main()
