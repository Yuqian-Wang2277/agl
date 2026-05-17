#!/usr/bin/env python3
"""
3_compute_metrics.py — 计算五项认知指标。

给定标注结果（annotations/pre_annotations.json 和 post_annotations.json），
对每条轨迹 T = [u_1,...,u_n] 计算：

  M1 ID  = |{P,R}| / n
  M2 IE  = (1 - P_idx[0]/n) if P_idx else 0
  M3 PIR = |{P before first D or R}| / max(|P|, 1)
  M4 RGR = |{R with some P before it}| / max(|R|, 1)
  M5 AHS = mean(abstractness_score for P units) if P_idx else 0

输出：
  metrics/all_metrics.json    — 每条轨迹的五项指标
  metrics/summary_stats.json  — Pre/Post 均值、标准差、Mann-Whitney U p 值

用法：
  python 3_compute_metrics.py
"""

import json
import os
import statistics

import numpy as np
from scipy.stats import mannwhitneyu

BASE = os.path.dirname(os.path.abspath(__file__))
ANNO_DIR = os.path.join(BASE, "annotations")
OUT_DIR = os.path.join(BASE, "metrics")

METRIC_NAMES = ["ID", "IE", "PIR", "RGR", "AHS"]


# ---------------------------------------------------------------------------
# Core metric computation (mirrors plan pseudocode exactly)
# ---------------------------------------------------------------------------

def compute_metrics(task_record: dict) -> dict | None:
    units = task_record.get("units", [])
    final_labels = task_record.get("final_labels", [])

    if not units or not final_labels:
        return None

    n = len(final_labels)
    if n == 0:
        return None

    P_idx = [i for i, lbl in enumerate(final_labels) if lbl == "P"]
    R_idx = [i for i, lbl in enumerate(final_labels) if lbl == "R"]
    exec_idx = next(
        (i for i, lbl in enumerate(final_labels) if lbl in ("D", "R")), n
    )

    # M1 · Inductive Density
    ID = len([lbl for lbl in final_labels if lbl in ("P", "R")]) / n

    # M2 · Inductive Earliness
    IE = (1 - P_idx[0] / n) if P_idx else 0.0

    # M3 · Pre-execution Induction Ratio
    P_before_exec = len([i for i in P_idx if i < exec_idx])
    PIR = P_before_exec / max(len(P_idx), 1)

    # M4 · Rule Grounding Rate
    grounded_R = len([i for i in R_idx if any(j < i for j in P_idx)])
    RGR = grounded_R / max(len(R_idx), 1)

    # M5 · Abstraction Hierarchy Score (from Prompt 2 AHS scoring)
    ahs_scores = []
    for i, (ur, lbl) in enumerate(zip(units, final_labels)):
        if lbl == "P":
            ahs_info = ur.get("ahs") or {}
            if isinstance(ahs_info, dict) and "abstractness_score" in ahs_info:
                ahs_scores.append(int(ahs_info["abstractness_score"]))
    AHS = statistics.mean(ahs_scores) if ahs_scores else 0.0

    return {
        "task_id": task_record["task_id"],
        "problem_type": task_record["problem_type"],
        "split": task_record.get("split", ""),
        "n_units": n,
        "n_P": len(P_idx),
        "n_R": len(R_idx),
        "n_E": sum(1 for lbl in final_labels if lbl == "E"),
        "n_D": sum(1 for lbl in final_labels if lbl == "D"),
        "exec_idx": exec_idx,
        "kappa": task_record.get("kappa"),
        "ID": round(ID, 4),
        "IE": round(IE, 4),
        "PIR": round(PIR, 4),
        "RGR": round(RGR, 4),
        "AHS": round(AHS, 4),
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(pre_metrics: list, post_metrics: list) -> dict:
    summary = {}
    for metric in METRIC_NAMES:
        pre_vals = [m[metric] for m in pre_metrics if m is not None]
        post_vals = [m[metric] for m in post_metrics if m is not None]

        pre_mean = float(np.mean(pre_vals)) if pre_vals else 0.0
        post_mean = float(np.mean(post_vals)) if post_vals else 0.0
        pre_std = float(np.std(pre_vals)) if pre_vals else 0.0
        post_std = float(np.std(post_vals)) if post_vals else 0.0

        p_value = None
        if len(pre_vals) >= 2 and len(post_vals) >= 2:
            try:
                _, p_value = mannwhitneyu(pre_vals, post_vals, alternative="two-sided")
                p_value = float(p_value)
            except Exception:
                p_value = None

        delta = post_mean - pre_mean
        sig = ""
        if p_value is not None:
            if p_value < 0.001:
                sig = "***"
            elif p_value < 0.01:
                sig = "**"
            elif p_value < 0.05:
                sig = "*"

        summary[metric] = {
            "pre_mean": round(pre_mean, 4),
            "pre_std": round(pre_std, 4),
            "post_mean": round(post_mean, 4),
            "post_std": round(post_std, 4),
            "delta": round(delta, 4),
            "p_value": round(p_value, 6) if p_value is not None else None,
            "significance": sig,
        }
    return summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pre_anno_path = os.path.join(ANNO_DIR, "pre_annotations.json")
    post_anno_path = os.path.join(ANNO_DIR, "post_annotations.json")

    if not os.path.exists(pre_anno_path):
        raise FileNotFoundError(f"Pre annotations not found: {pre_anno_path}\n"
                                "Run 2_annotate_units.py --split pre first.")
    if not os.path.exists(post_anno_path):
        raise FileNotFoundError(f"Post annotations not found: {post_anno_path}\n"
                                "Run 2_annotate_units.py --split post first.")

    with open(pre_anno_path, encoding="utf-8") as f:
        pre_anno = json.load(f)
    with open(post_anno_path, encoding="utf-8") as f:
        post_anno = json.load(f)

    # Compute per-trajectory metrics
    pre_metrics = []
    for rec in pre_anno:
        rec["split"] = "pre"
        m = compute_metrics(rec)
        if m:
            pre_metrics.append(m)

    post_metrics = []
    for rec in post_anno:
        rec["split"] = "post"
        m = compute_metrics(rec)
        if m:
            post_metrics.append(m)

    all_metrics = pre_metrics + post_metrics

    # Save per-trajectory metrics
    all_metrics_path = os.path.join(OUT_DIR, "all_metrics.json")
    with open(all_metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"Per-trajectory metrics saved: {all_metrics_path}")
    print(f"  Pre: {len(pre_metrics)} trajectories, Post: {len(post_metrics)} trajectories")

    # Summary statistics
    summary = compute_summary(pre_metrics, post_metrics)
    summary_path = os.path.join(OUT_DIR, "summary_stats.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Print comparison table
    print("\n" + "=" * 65)
    print(f"  {'Metric':<8} {'Pre':>10} {'Post':>10} {'Delta':>10} {'p':>10}  Sig")
    print("=" * 65)
    for metric in METRIC_NAMES:
        s = summary[metric]
        p_str = f"{s['p_value']:.4f}" if s["p_value"] is not None else "N/A"
        print(f"  {metric:<8} {s['pre_mean']:>10.4f} {s['post_mean']:>10.4f} "
              f"{s['delta']:>+10.4f} {p_str:>10}  {s['significance']}")
    print("=" * 65)
    print(f"\nSummary saved: {summary_path}")

    # Label distribution
    print("\n=== Label Distribution ===")
    for split_name, metrics_list, anno in [("Pre", pre_metrics, pre_anno),
                                             ("Post", post_metrics, post_anno)]:
        total_n = sum(m["n_units"] for m in metrics_list)
        total_P = sum(m["n_P"] for m in metrics_list)
        total_R = sum(m["n_R"] for m in metrics_list)
        total_E = sum(m["n_E"] for m in metrics_list)
        total_D = sum(m["n_D"] for m in metrics_list)
        kappas = [r.get("kappa") for r in anno if r.get("kappa") is not None]
        mean_k = sum(kappas) / len(kappas) if kappas else 0.0
        print(f"  {split_name}: n={total_n}  "
              f"E={total_E}({100*total_E/max(total_n,1):.1f}%)  "
              f"D={total_D}({100*total_D/max(total_n,1):.1f}%)  "
              f"P={total_P}({100*total_P/max(total_n,1):.1f}%)  "
              f"R={total_R}({100*total_R/max(total_n,1):.1f}%)  "
              f"kappa={mean_k:.3f}")


if __name__ == "__main__":
    main()
