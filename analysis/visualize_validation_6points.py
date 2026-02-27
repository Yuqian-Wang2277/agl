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
import re
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


ID_TYPES = {
    "contextual_parametric_knowledge_conflicts",
    "cryptonite",
    "disfl_qa",
    "elementary_math_qa",
    "fact_checker",
    "language_identification",
    "matrixshapes",
    "mnist_ascii",
    "movie_dialog_same_or_different",
    "vitaminc_fact_verification",
    "word_unscrambling",
}

OOD_TYPES = {
    "arithmetic",
    "ascii_word_recognition",
    "chess_state_tracking",
    "discourse_marker_prediction",
    "goal_step_wikihow",
    "hyperbaton",
    "implicatures",
    "intersect_geometry",
    "linguistic_mappings",
    "modified_arithmetic",
    "nonsense_words_grammar",
    "real_or_fake_text",
    "simp_turing_concept",
    "snarks",
    "unnatural_in_context_learning",
}

HARD_TYPES = {
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects",
    "web_of_lies",
    "word_sorting",
}

SPLIT_ORDER = ["ID", "OOD", "HARD", "UNKNOWN"]


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


def classify_problem(problem_type: str) -> str:
    if problem_type in ID_TYPES:
        return "ID"
    if problem_type in OOD_TYPES:
        return "OOD"
    if problem_type in HARD_TYPES:
        return "HARD"
    return "UNKNOWN"

def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def normalize_text(value: Any) -> str:
    text = to_text(value).strip().lower()
    # Lenient normalization: keep only alphanumeric characters.
    return re.sub(r"[^0-9a-z]+", "", text)


def mean_ci(values: List[float]) -> Tuple[float, float, int]:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(arr.mean())
    if n == 1:
        return mean, 0.0, 1
    std = float(arr.std(ddof=1))
    ci = 1.96 * std / math.sqrt(n)
    return mean, ci, n


def compute_step_metrics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    sums = defaultdict(float)
    counts = defaultdict(int)

    failure_counts = defaultdict(int)
    missing_strategy = 0
    missing_answer = 0

    metric_names = ["final", "format", "scorer", "correctness"]
    metric_values: Dict[str, List[float]] = {m: [] for m in metric_names}

    per_pt_metric_sum: Dict[str, Dict[str, float]] = {
        m: defaultdict(float) for m in metric_names
    }
    per_pt_metric_cnt: Dict[str, Dict[str, int]] = {
        m: defaultdict(int) for m in metric_names
    }
    per_pt_total_cnt: Dict[str, int] = defaultdict(int)
    per_pt_correct_cnt: Dict[str, int] = defaultdict(int)
    split_metric_sum: Dict[str, Dict[str, float]] = {
        s: defaultdict(float) for s in SPLIT_ORDER
    }
    split_metric_cnt: Dict[str, Dict[str, int]] = {
        s: defaultdict(int) for s in SPLIT_ORDER
    }

    scatter_points: List[Tuple[float, float, float]] = []  # scorer, correctness, format
    scorer_correctness_pt: List[Tuple[float, float, str]] = []  # scorer, correctness, problem_type
    strat_len_points: List[Tuple[int, float]] = []  # len(strategy), final
    ans_len_points: List[Tuple[int, float]] = []  # len(answer), correctness
    strat_len_scorer_points: List[Tuple[int, float]] = []  # len(strategy), scorer
    calibration_points: List[Tuple[float, float, float]] = []  # scorer, correctness, format
    quadrant_counts = defaultdict(int)
    scorer_format_sum = 0.0
    scorer_format_cnt = 0

    exact_correct = 0
    norm_correct = 0
    total_count = 0

    for item in data:
        reward = item.get("reward") or {}
        fmt = reward.get("format")
        scorer = reward.get("scorer")
        correctness = reward.get("correctness")
        final = reward.get("final")

        for name, value in (
            ("format", fmt),
            ("scorer", scorer),
            ("correctness", correctness),
            ("final", final),
        ):
            if is_number(value):
                value_f = float(value)
                sums[name] += value_f
                counts[name] += 1
                metric_values[name].append(value_f)

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

        problem_type = item.get("problem_type")

        if is_number(scorer) and is_number(correctness) and is_number(fmt):
            scatter_points.append((float(scorer), float(correctness), float(fmt)))
            calibration_points.append((float(scorer), float(correctness), float(fmt)))
            if problem_type:
                scorer_correctness_pt.append((float(scorer), float(correctness), problem_type))

        if isinstance(strategy_extracted, str) and is_number(final):
            strat_len_points.append((len(strategy_extracted), float(final)))
        if isinstance(answer_extracted, str) and is_number(correctness):
            ans_len_points.append((len(answer_extracted), float(correctness)))
        if isinstance(strategy_extracted, str) and is_number(scorer):
            strat_len_scorer_points.append((len(strategy_extracted), float(scorer)))

        if problem_type:
            per_pt_total_cnt[problem_type] += 1
            if is_number(correctness) and float(correctness) == 1:
                per_pt_correct_cnt[problem_type] += 1
            for name, value in (
                ("final", final),
                ("format", fmt),
                ("scorer", scorer),
                ("correctness", correctness),
            ):
                if is_number(value):
                    per_pt_metric_sum[name][problem_type] += float(value)
                    per_pt_metric_cnt[name][problem_type] += 1
            split = classify_problem(problem_type)
            for name, value in (
                ("final", final),
                ("format", fmt),
                ("scorer", scorer),
                ("correctness", correctness),
            ):
                if is_number(value):
                    split_metric_sum[split][name] += float(value)
                    split_metric_cnt[split][name] += 1

        # Exact vs normalized accuracy (based on extracted answer vs ground truth)
        total_count += 1
        gt_text = to_text((item.get("input") or {}).get("ground_truth"))
        ans_text = to_text(answer_extracted)
        if ans_text.strip() and gt_text.strip() and ans_text.strip() == gt_text.strip():
            exact_correct += 1
        if normalize_text(ans_text) and normalize_text(gt_text) and normalize_text(ans_text) == normalize_text(gt_text):
            norm_correct += 1

        if is_number(scorer) and is_number(fmt):
            scorer_format_sum += float(scorer) * float(fmt)
            scorer_format_cnt += 1

        if is_number(scorer) and is_number(correctness):
            is_correct = float(correctness) == 1.0
            high_score = float(scorer) >= 0.7
            if high_score and is_correct:
                quadrant_counts["high_score_correct"] += 1
            elif high_score and not is_correct:
                quadrant_counts["high_score_wrong"] += 1
            elif (not high_score) and is_correct:
                quadrant_counts["low_score_correct"] += 1
            else:
                quadrant_counts["low_score_wrong"] += 1

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

    per_pt_metric_mean: Dict[str, Dict[str, float]] = {}
    for name in metric_names:
        per_pt_metric_mean[name] = {
            p: per_pt_metric_sum[name][p] / per_pt_metric_cnt[name][p]
            for p in per_pt_metric_cnt[name]
            if per_pt_metric_cnt[name][p]
        }
    split_metric_mean: Dict[str, Dict[str, float]] = {}
    for split in SPLIT_ORDER:
        split_metric_mean[split] = {}
        for name in metric_names:
            if split_metric_cnt[split].get(name):
                split_metric_mean[split][name] = (
                    split_metric_sum[split][name] / split_metric_cnt[split][name]
                )

    return {
        "n": n,
        "means": means,
        "failure_counts": dict(failure_counts),
        "missing": missing,
        "per_pt_metric_mean": per_pt_metric_mean,
        "split_metric_mean": split_metric_mean,
        "per_pt_total_cnt": dict(per_pt_total_cnt),
        "per_pt_correct_cnt": dict(per_pt_correct_cnt),
        "metric_values": metric_values,
        "scatter_points": scatter_points,
        "scorer_correctness_pt": scorer_correctness_pt,
        "strat_len_points": strat_len_points,
        "ans_len_points": ans_len_points,
        "strat_len_scorer_points": strat_len_scorer_points,
        "calibration_points": calibration_points,
        "quadrant_counts": dict(quadrant_counts),
        "scorer_format_mean": (
            scorer_format_sum / scorer_format_cnt if scorer_format_cnt else float("nan")
        ),
        "exact_acc": exact_correct / total_count if total_count else float("nan"),
        "norm_acc": norm_correct / total_count if total_count else float("nan"),
    }


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_metric_trends_simple(
    steps: List[int],
    metrics: Dict[int, Dict[str, float]],
    out: Path,
) -> None:
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


def plot_metric_trends_by_split(
    steps: List[int],
    split_metric_mean_by_step: Dict[int, Dict[str, Dict[str, float]]],
    out: Path,
) -> None:
    metrics = ["final", "correctness", "format", "scorer"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for split in SPLIT_ORDER:
            ys = []
            for step in steps:
                ys.append(split_metric_mean_by_step[step].get(split, {}).get(metric, float("nan")))
            if all(math.isnan(v) for v in ys):
                continue
            ax.plot(steps, ys, marker="o", linewidth=2, label=split)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        if idx in (2, 3):
            ax.set_xlabel("Step")
        if idx in (0, 2):
            ax.set_ylabel("Mean")
        if idx == 0:
            ax.legend(loc="lower right")

    plt.suptitle("Metric Trends by Validation Split", y=1.02)
    save_fig(out / "metric_trends_by_split.png")


def plot_scorer_correctness_by_problem_type(
    points: List[Tuple[float, float, str]],
    out: Path,
) -> None:
    if not points:
        return
    problem_types = sorted({p for _, _, p in points})
    colors = plt.cm.hsv(np.linspace(0, 1, len(problem_types), endpoint=False))
    color_map = {p: colors[i] for i, p in enumerate(problem_types)}

    plt.figure(figsize=(9, 6))
    for scorer, correctness, ptype in points:
        plt.scatter(scorer, correctness, s=10, alpha=0.5, color=color_map[ptype])

    plt.xlabel("Scorer")
    plt.ylabel("Correctness")
    plt.yticks([0, 1], ["0", "1"])
    plt.ylim(-0.1, 1.1)
    plt.title("Scorer vs Correctness by Problem Type")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[p], markersize=6)
        for p in problem_types
    ]
    plt.legend(
        handles,
        problem_types,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        fontsize=6,
        ncol=1,
    )
    save_fig(out / "scatter_scorer_correctness_by_problem_type.png")


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


def plot_step_scatter(
    step_a: int,
    step_b: int,
    pt_means: Dict[int, Dict[str, float]],
    pt_counts: Dict[int, Dict[str, int]],
    out: Path,
) -> None:
    pts = sorted(
        set(pt_means.get(step_a, {}).keys()) | set(pt_means.get(step_b, {}).keys())
    )
    xs = []
    ys = []
    sizes = []
    deltas = []
    for p in pts:
        x = pt_means.get(step_a, {}).get(p, float("nan"))
        y = pt_means.get(step_b, {}).get(p, float("nan"))
        if math.isnan(x) or math.isnan(y):
            continue
        xs.append(x)
        ys.append(y)
        deltas.append(y - x)
        sizes.append(pt_counts.get(step_b, {}).get(p, 1))

    sizes = np.array(sizes, dtype=float)
    sizes = 40 * (sizes / sizes.max() if sizes.max() else 1.0) + 20

    plt.figure(figsize=(7, 7))
    sc = plt.scatter(xs, ys, c=deltas, s=sizes, cmap="RdBu", vmin=-0.5, vmax=0.5, alpha=0.7)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    plt.xlabel(f"Mean final @ step {step_a}")
    plt.ylabel(f"Mean final @ step {step_b}")
    plt.title(f"Problem Type Transfer: Step {step_a} vs Step {step_b}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    cbar = plt.colorbar(sc)
    cbar.set_label("Δ (step_b - step_a)")
    save_fig(out / "scatter_step0_step750.png")


def plot_accuracy_heatmap(
    steps: List[int],
    pt_correctness: Dict[int, Dict[str, float]],
    out: Path,
) -> None:
    problem_types = sorted({p for s in pt_correctness for p in pt_correctness[s]})
    last = steps[-1]
    problem_types.sort(key=lambda p: pt_correctness[last].get(p, float("nan")), reverse=True)

    matrix = np.full((len(problem_types), len(steps)), np.nan, dtype=float)
    for i, p in enumerate(problem_types):
        for j, s in enumerate(steps):
            matrix[i, j] = pt_correctness[s].get(p, float("nan"))

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
    plt.title("Problem Type Accuracy Heatmap (correctness)")
    save_fig(out / "heatmap_problem_type_accuracy.png")


def plot_easy_hard_curves(
    steps: List[int],
    pt_correctness: Dict[int, Dict[str, float]],
    out: Path,
) -> None:
    base = steps[0]
    easy = []
    hard = []
    for p, acc in pt_correctness.get(base, {}).items():
        if acc > 0.5:
            easy.append(p)
        elif acc < 0.2:
            hard.append(p)

    def mean_for_group(step: int, group: List[str]) -> float:
        vals = [pt_correctness[step].get(p, float("nan")) for p in group]
        vals = [v for v in vals if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    easy_curve = [mean_for_group(s, easy) for s in steps]
    hard_curve = [mean_for_group(s, hard) for s in steps]

    plt.figure(figsize=(9, 5))
    plt.plot(steps, easy_curve, marker="o", linewidth=2, label=f"easy (>50%), n={len(easy)}")
    plt.plot(steps, hard_curve, marker="s", linewidth=2, label=f"hard (<20%), n={len(hard)}")
    plt.ylim(0, 1.05)
    plt.xlabel("Step")
    plt.ylabel("Accuracy (correctness)")
    plt.title("Easy vs Hard Tasks (defined at step 0)")
    plt.legend(loc="lower right")
    save_fig(out / "accuracy_easy_hard.png")


def plot_scorer_hist_steps(
    scorer_values_by_step: Dict[int, List[float]],
    out: Path,
) -> None:
    steps = list(scorer_values_by_step.keys())
    bins = np.linspace(0, 1, 21)
    plt.figure(figsize=(9, 5))
    for step in steps:
        vals = scorer_values_by_step[step]
        if not vals:
            continue
        plt.hist(vals, bins=bins, density=True, alpha=0.35, label=f"step {step}")
    plt.xlabel("Scorer")
    plt.ylabel("Density")
    plt.title("Scorer Distribution (selected steps)")
    plt.legend(loc="upper left")
    save_fig(out / "scorer_hist_steps.png")


def plot_final_reward_decomposition(
    steps: List[int],
    metrics: Dict[int, Dict[str, float]],
    scorer_format_mean: Dict[int, float],
    out: Path,
) -> None:
    format_base = [metrics[s]["format"] * 0.1 for s in steps]
    scorer_comp = [scorer_format_mean.get(s, float("nan")) * 0.9 for s in steps]
    correctness_comp = [0.0 for _ in steps]

    plt.figure(figsize=(9, 5))
    plt.bar(steps, format_base, label="format base (0.1 * format)")
    plt.bar(steps, scorer_comp, bottom=format_base, label="scorer component (0.9 * scorer * format)")
    plt.bar(
        steps,
        correctness_comp,
        bottom=np.array(format_base) + np.array(scorer_comp),
        label="correctness (0 weight)",
    )
    plt.xlabel("Step")
    plt.ylabel("Mean contribution")
    plt.title("Final Reward Decomposition")
    plt.legend(loc="upper left")
    save_fig(out / "final_reward_decomposition.png")


def plot_strategy_len_vs_scorer(points: List[Tuple[int, float]], out: Path) -> None:
    if not points:
        return
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    plt.figure(figsize=(8, 5))
    plt.hexbin(x, y, gridsize=40, cmap="Purples", mincnt=1)
    plt.colorbar(label="count")
    plt.xlabel("len(strategy_extracted)")
    plt.ylabel("scorer")
    plt.ylim(0, 1.05)
    plt.title("Strategy Length vs Scorer")
    save_fig(out / "hexbin_strategylen_scorer.png")


def plot_quadrant_counts(
    steps: List[int],
    quadrant_counts: Dict[int, Dict[str, int]],
    out: Path,
) -> None:
    order = ["high_score_correct", "high_score_wrong", "low_score_correct", "low_score_wrong"]
    colors = {
        "high_score_correct": "#2ca02c",
        "high_score_wrong": "#d62728",
        "low_score_correct": "#1f77b4",
        "low_score_wrong": "#ff7f0e",
    }
    plt.figure(figsize=(10, 5))
    bottoms = np.zeros(len(steps))
    for key in order:
        vals = [quadrant_counts.get(s, {}).get(key, 0) for s in steps]
        plt.bar(steps, vals, bottom=bottoms, label=key, color=colors[key])
        bottoms += np.array(vals)
    plt.xlabel("Step")
    plt.ylabel("Count")
    plt.title("Scorer/Correctness Quadrants (threshold=0.7)")
    plt.legend(loc="upper right")
    save_fig(out / "quadrant_counts.png")


def plot_scorer_calibration(
    calibration_by_step: Dict[int, List[Tuple[float, float, float]]],
    step_a: int,
    step_b: int,
    out: Path,
    bins: int = 10,
) -> None:
    def bucket_curve(points: List[Tuple[float, float, float]]) -> Tuple[List[float], List[float]]:
        if not points:
            return [], []
        edges = np.linspace(0, 1, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        bucket_vals = [[] for _ in range(bins)]
        for s, c, f in points:
            if f != 1:
                continue
            idx = min(int(s * bins), bins - 1)
            bucket_vals[idx].append(c)
        means = [float(np.mean(v)) if v else float("nan") for v in bucket_vals]
        return centers.tolist(), means

    plt.figure(figsize=(8, 5))
    for step in (step_a, step_b):
        centers, means = bucket_curve(calibration_by_step.get(step, []))
        plt.plot(centers, means, marker="o", linewidth=2, label=f"step {step}")
    plt.xlabel("Scorer (binned)")
    plt.ylabel("Mean correctness")
    plt.title("Scorer Calibration (format=1)")
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    save_fig(out / "scorer_calibration.png")


def plot_exact_normalized_gap(
    steps: List[int],
    exact_acc: Dict[int, float],
    norm_acc: Dict[int, float],
    out: Path,
) -> None:
    exact = [exact_acc[s] for s in steps]
    norm = [norm_acc[s] for s in steps]
    plt.figure(figsize=(9, 5))
    plt.plot(steps, exact, marker="o", linewidth=2, label="exact accuracy")
    plt.plot(steps, norm, marker="s", linewidth=2, label="normalized accuracy")
    plt.fill_between(steps, exact, norm, alpha=0.2)
    plt.ylim(0, 1.05)
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Exact vs Normalized Accuracy (gap shaded)")
    plt.legend(loc="lower right")
    save_fig(out / "exact_vs_normalized_accuracy.png")


def plot_topk_error_contrib(
    step: int,
    pt_counts: Dict[int, Dict[str, int]],
    pt_correct: Dict[int, Dict[str, int]],
    out: Path,
    top_k: int = 15,
) -> None:
    total = pt_counts.get(step, {})
    correct = pt_correct.get(step, {})
    errors = {p: total.get(p, 0) - correct.get(p, 0) for p in total}
    errors = {p: v for p, v in errors.items() if v > 0}
    top = sorted(errors.items(), key=lambda x: x[1], reverse=True)[:top_k]
    if not top:
        return
    labels = [p for p, _ in top][::-1]
    vals = [v for _, v in top][::-1]
    total_errors = sum(errors.values()) or 1
    perc = [v / total_errors * 100 for v in vals]

    plt.figure(figsize=(9, max(5, len(labels) * 0.4)))
    bars = plt.barh(labels, vals, color="#ff7f0e")
    plt.xlabel("Error count")
    plt.title(f"Top-{top_k} Error-Contributing Tasks (step {step})")
    for b, p in zip(bars, perc):
        plt.text(
            b.get_width() + 1,
            b.get_y() + b.get_height() / 2,
            f"{p:.1f}%",
            va="center",
            fontsize=9,
        )
    save_fig(out / "topk_error_contrib.png")


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
    per_pt_metric_mean: Dict[int, Dict[str, Dict[str, float]]],
    metric_values: Dict[int, Dict[str, List[float]]],
    split_metric_mean_by_step: Dict[int, Dict[str, Dict[str, float]]],
    exact_acc: Dict[int, float],
    norm_acc: Dict[int, float],
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
        w.writerow(["problem_type", "step", "mean_final", "mean_correctness", "mean_format", "mean_scorer"])
        all_pts = sorted({p for s in per_pt_metric_mean for p in per_pt_metric_mean[s].get("final", {})})
        for p in all_pts:
            for s in steps:
                w.writerow([
                    p,
                    s,
                    per_pt_metric_mean[s].get("final", {}).get(p, ""),
                    per_pt_metric_mean[s].get("correctness", {}).get(p, ""),
                    per_pt_metric_mean[s].get("format", {}).get(p, ""),
                    per_pt_metric_mean[s].get("scorer", {}).get(p, ""),
                ])

    with open(out / "split_metric_trends.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "split", "final", "correctness", "format", "scorer"])
        for s in steps:
            for split in SPLIT_ORDER:
                w.writerow([
                    s,
                    split,
                    split_metric_mean_by_step[s].get(split, {}).get("final", ""),
                    split_metric_mean_by_step[s].get(split, {}).get("correctness", ""),
                    split_metric_mean_by_step[s].get(split, {}).get("format", ""),
                    split_metric_mean_by_step[s].get(split, {}).get("scorer", ""),
                ])

    with open(out / "metric_trends_micro_macro.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "metric", "micro_mean", "micro_ci", "macro_mean", "macro_ci", "micro_n", "macro_n"])
        for s in steps:
            for metric in ("final", "correctness", "format", "scorer"):
                micro_mean, micro_ci, micro_n = mean_ci(metric_values[s].get(metric, []))
                macro_vals = list(per_pt_metric_mean[s].get(metric, {}).values())
                macro_mean, macro_ci, macro_n = mean_ci(macro_vals)
                w.writerow([s, metric, micro_mean, micro_ci, macro_mean, macro_ci, micro_n, macro_n])

    with open(out / "exact_vs_normalized_accuracy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "exact_accuracy", "normalized_accuracy"])
        for s in steps:
            w.writerow([s, exact_acc[s], norm_acc[s]])


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
    per_pt_metric_mean: Dict[int, Dict[str, Dict[str, float]]] = {}
    per_pt_counts: Dict[int, Dict[str, int]] = {}
    per_pt_correct: Dict[int, Dict[str, int]] = {}
    split_metric_mean_by_step: Dict[int, Dict[str, Dict[str, float]]] = {}
    metric_values: Dict[int, Dict[str, List[float]]] = {}
    calibration_by_step: Dict[int, List[Tuple[float, float, float]]] = {}
    scorer_correctness_pt_all: List[Tuple[float, float, str]] = []
    strat_len_scorer_points: List[Tuple[int, float]] = []
    quadrant_counts_by_step: Dict[int, Dict[str, int]] = {}
    scorer_format_mean_by_step: Dict[int, float] = {}
    exact_acc: Dict[int, float] = {}
    norm_acc: Dict[int, float] = {}
    strat_len_points: List[Tuple[int, float]] = []
    ans_len_points: List[Tuple[int, float]] = []

    for step in steps:
        data = load_step_data(validation_dir, step)
        summary = compute_step_metrics(data)
        metrics[step] = {"n": summary["n"], **summary["means"]}
        failures[step] = summary["failure_counts"]
        missing[step] = summary["missing"]
        per_pt_metric_mean[step] = summary["per_pt_metric_mean"]
        per_pt_counts[step] = summary["per_pt_total_cnt"]
        per_pt_correct[step] = summary["per_pt_correct_cnt"]
        split_metric_mean_by_step[step] = summary["split_metric_mean"]
        metric_values[step] = summary["metric_values"]
        calibration_by_step[step] = summary["calibration_points"]
        scorer_correctness_pt_all.extend(summary["scorer_correctness_pt"])
        strat_len_scorer_points.extend(summary["strat_len_scorer_points"])
        quadrant_counts_by_step[step] = summary["quadrant_counts"]
        scorer_format_mean_by_step[step] = summary["scorer_format_mean"]
        exact_acc[step] = summary["exact_acc"]
        norm_acc[step] = summary["norm_acc"]
        strat_len_points.extend(summary["strat_len_points"])
        ans_len_points.extend(summary["ans_len_points"])

    scorer_hist_steps = [0, 200, 500, 750]
    scorer_values_by_step: Dict[int, List[float]] = {}
    for step in scorer_hist_steps:
        if step in metric_values:
            scorer_values_by_step[step] = metric_values[step].get("scorer", [])
            continue
        try:
            data = load_step_data(validation_dir, step)
        except FileNotFoundError:
            continue
        summary = compute_step_metrics(data)
        scorer_values_by_step[step] = summary["metric_values"].get("scorer", [])

    pt_final_means = {s: per_pt_metric_mean[s].get("final", {}) for s in steps}
    pt_correctness = {s: per_pt_metric_mean[s].get("correctness", {}) for s in steps}

    plot_metric_trends_simple(steps, metrics, output_dir)
    plot_metric_trends_by_split(steps, split_metric_mean_by_step, output_dir)
    plot_scorer_correctness_by_problem_type(scorer_correctness_pt_all, output_dir)
    plot_step_scatter(steps[0], steps[-1], pt_final_means, per_pt_counts, output_dir)
    plot_accuracy_heatmap(steps, pt_correctness, output_dir)
    plot_easy_hard_curves(steps, pt_correctness, output_dir)
    plot_scorer_calibration(calibration_by_step, steps[0], steps[-1], output_dir)
    plot_scorer_hist_steps(scorer_values_by_step, output_dir)
    plot_final_reward_decomposition(steps, metrics, scorer_format_mean_by_step, output_dir)
    plot_exact_normalized_gap(steps, exact_acc, norm_acc, output_dir)
    plot_topk_error_contrib(steps[-1], per_pt_counts, per_pt_correct, output_dir)
    plot_strategy_len_vs_scorer(strat_len_scorer_points, output_dir)
    plot_quadrant_counts(steps, quadrant_counts_by_step, output_dir)

    write_csvs(
        steps,
        metrics,
        failures,
        missing,
        per_pt_metric_mean,
        metric_values,
        split_metric_mean_by_step,
        exact_acc,
        norm_acc,
        output_dir,
    )

    print(f"Saved charts and CSVs to: {output_dir}")


if __name__ == "__main__":
    main()
