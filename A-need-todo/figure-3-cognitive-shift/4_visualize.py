#!/usr/bin/env python3
"""
4_visualize.py — 认知迁移散点图（Figure-3 主图）。

图表元素：
  主图：散点 X=PIR, Y=ID
    形状 = IE  (IE>=0.7 -> star, 0.3<=IE<0.7 -> circle, IE<0.3 -> triangle)
    大小  = RGR (归一化到 40-220 pt^2)
    颜色深度 = AHS (同色系，越深越抽象)
    红色=Pre-GIST, 紫色=Post-GIST
  顶部：Pre/Post 的 PIR 核密度分布曲线
  右下嵌入表格：Cognitive Fingerprint (5 指标 pre/post/delta/sig)
  椭圆：各 95% 置信椭圆（协方差）

数据点过滤（只保留"符合课题内容"的点）：
  - n_units >= 5（轨迹足够长）
  - ID > 0 或 AHS > 0（有归纳行为发生）
  - PIR 或 ID 不同时为零（非纯执行型）

用法：
  python 4_visualize.py
  python 4_visualize.py --metrics metrics/all_metrics.json --out results/cognitive_shift.pdf
"""

import argparse
import json
import os

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.stats import mannwhitneyu

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

BASE = os.path.dirname(os.path.abspath(__file__))

# Color palettes: Pre=red family, Post=purple family
PRE_BASE = "#E05C5C"
POST_BASE = "#7B5EA7"


# ---------------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------------

def load_metrics(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def filter_records(records: list) -> list:
    kept = []
    for r in records:
        if r.get("n_units", 0) < 5:
            continue
        if r.get("ID", 0) == 0 and r.get("AHS", 0) == 0:
            continue
        kept.append(r)
    return kept


def load_summary(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def ie_to_marker(ie: float) -> str:
    if ie >= 0.7:
        return "*"      # star
    elif ie >= 0.3:
        return "o"      # circle
    else:
        return "^"      # triangle


def rgr_to_size(rgr: float, s_min: float = 40.0, s_max: float = 220.0) -> float:
    return s_min + rgr * (s_max - s_min)


def ahs_to_alpha(ahs: float, ahs_max: float = 3.0,
                  alpha_min: float = 0.35, alpha_max: float = 0.95) -> float:
    return alpha_min + (ahs / max(ahs_max, 1e-9)) * (alpha_max - alpha_min)


# ---------------------------------------------------------------------------
# Confidence ellipse
# ---------------------------------------------------------------------------

def confidence_ellipse(xs: np.ndarray, ys: np.ndarray, ax,
                        n_std: float = 2.0, edgecolor: str = "black",
                        facecolor: str = "none", linestyle: str = "--",
                        linewidth: float = 1.5, alpha: float = 0.6):
    if len(xs) < 3:
        return
    cov = np.cov(xs, ys)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.abs(eigenvalues))
    ellipse = Ellipse(
        xy=(np.mean(xs), np.mean(ys)),
        width=width, height=height, angle=angle,
        edgecolor=edgecolor, facecolor=facecolor,
        linestyle=linestyle, linewidth=linewidth, alpha=alpha,
    )
    ax.add_patch(ellipse)


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

def draw_scatter(ax, records_pre: list, records_post: list):
    for split_records, base_color, label_prefix in [
        (records_pre, PRE_BASE, "M_pre"),
        (records_post, POST_BASE, "M_post"),
    ]:
        marker_groups: dict[str, tuple] = {"*": ([], [], [], []), "o": ([], [], [], []), "^": ([], [], [], [])}
        for r in split_records:
            pir = r.get("PIR", 0.0)
            id_ = r.get("ID", 0.0)
            ie = r.get("IE", 0.0)
            rgr = r.get("RGR", 0.0)
            ahs = r.get("AHS", 0.0)
            marker = ie_to_marker(ie)
            size = rgr_to_size(rgr)
            alpha = ahs_to_alpha(ahs)
            xs, ys, ss, als = marker_groups[marker]
            xs.append(pir)
            ys.append(id_)
            ss.append(size)
            als.append(alpha)

        for marker, (xs, ys, ss, als) in marker_groups.items():
            if not xs:
                continue
            for x, y, s, a in zip(xs, ys, ss, als):
                ax.scatter(x, y, s=s, marker=marker, color=base_color,
                           alpha=a, edgecolors="none", zorder=3)

        # Confidence ellipse
        all_x = [r.get("PIR", 0.0) for r in split_records]
        all_y = [r.get("ID", 0.0) for r in split_records]
        if len(all_x) >= 3:
            confidence_ellipse(np.array(all_x), np.array(all_y), ax,
                               n_std=2.0, edgecolor=base_color,
                               linestyle="--", linewidth=1.8, alpha=0.7)

        # Centroid cross
        cx, cy = float(np.mean(all_x)), float(np.mean(all_y))
        ax.scatter(cx, cy, marker="x", s=120, color=base_color,
                   linewidths=2.5, zorder=5)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Pre-execution Induction Ratio (PIR) →", fontsize=11)
    ax.set_ylabel("Inductive Density (ID) →", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    # Legend: shape = IE, color = split
    legend_elements = [
        mpatches.Patch(facecolor=PRE_BASE, label=r"$M_{pre}$"),
        mpatches.Patch(facecolor=POST_BASE, label=r"$M_{post}$"),
        plt.Line2D([0], [0], marker="*", color="gray", linestyle="None",
                   markersize=10, label="IE ≥ 0.7"),
        plt.Line2D([0], [0], marker="o", color="gray", linestyle="None",
                   markersize=8, label="0.3 ≤ IE < 0.7"),
        plt.Line2D([0], [0], marker="^", color="gray", linestyle="None",
                   markersize=8, label="IE < 0.3"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8.5,
              framealpha=0.85, edgecolor="0.7")

    # Cognitive shift arrow
    pre_cx = float(np.mean([r.get("PIR", 0) for r in records_pre]))
    pre_cy = float(np.mean([r.get("ID", 0) for r in records_pre]))
    post_cx = float(np.mean([r.get("PIR", 0) for r in records_post]))
    post_cy = float(np.mean([r.get("ID", 0) for r in records_post]))
    ax.annotate("cognitive shift",
                xy=(post_cx, post_cy), xytext=(pre_cx + 0.05, pre_cy + 0.02),
                arrowprops=dict(arrowstyle="->", color="0.3", lw=1.4),
                fontsize=8.5, color="0.3", fontstyle="italic")


# ---------------------------------------------------------------------------
# Density panel (top)
# ---------------------------------------------------------------------------

def draw_density(ax, records_pre: list, records_post: list):
    pre_pir = [r.get("PIR", 0.0) for r in records_pre]
    post_pir = [r.get("PIR", 0.0) for r in records_post]
    if pre_pir:
        sns.kdeplot(pre_pir, ax=ax, color=PRE_BASE, fill=True, alpha=0.35, linewidth=1.5)
    if post_pir:
        sns.kdeplot(post_pir, ax=ax, color=POST_BASE, fill=True, alpha=0.35, linewidth=1.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.grid(True, linestyle=":", alpha=0.3)


# ---------------------------------------------------------------------------
# Cognitive Fingerprint table (inset)
# ---------------------------------------------------------------------------

def draw_fingerprint_table(ax_main, summary: dict, metrics_order=None):
    if not summary:
        return
    if metrics_order is None:
        metrics_order = ["ID", "IE", "PIR", "RGR", "AHS"]

    rows = []
    for m in metrics_order:
        s = summary.get(m, {})
        pre_v = s.get("pre_mean", 0.0)
        post_v = s.get("post_mean", 0.0)
        delta = s.get("delta", 0.0)
        sig = s.get("significance", "")
        delta_str = f"{delta:+.2f}{sig}"
        rows.append([m, f"{pre_v:.2f}", f"{post_v:.2f}", delta_str])

    col_labels = ["", "pre", "post", "Δ"]
    table = ax_main.table(
        cellText=rows, colLabels=col_labels,
        loc="lower right", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#DDDDDD")
        table[0, j].set_text_props(weight="bold")

    # Color delta column
    for i, row in enumerate(rows, start=1):
        delta_val = summary.get(row[0], {}).get("delta", 0.0)
        color = "#D9EAD3" if delta_val >= 0 else "#FCE5CD"
        table[i, 3].set_facecolor(color)
        table[i, 3].set_text_props(color="#CC0000" if delta_val >= 0 else "#666666",
                                    weight="bold")

    # Add title above table
    ax_main.text(0.99, 0.02, "Cognitive Fingerprint",
                  transform=ax_main.transAxes,
                  ha="right", va="bottom", fontsize=8.5, fontstyle="italic",
                  color="0.3")


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot Figure-3: cognitive shift scatter.")
    parser.add_argument("--metrics", default=os.path.join(BASE, "metrics", "all_metrics.json"))
    parser.add_argument("--summary", default=os.path.join(BASE, "metrics", "summary_stats.json"))
    parser.add_argument("--out", default=os.path.join(BASE, "results", "cognitive_shift.pdf"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable data point filtering (include all trajectories)")
    args = parser.parse_args()

    all_records = load_metrics(args.metrics)
    summary = load_summary(args.summary)

    pre_all = [r for r in all_records if r.get("split") == "pre"]
    post_all = [r for r in all_records if r.get("split") == "post"]

    if args.no_filter:
        pre_plot = pre_all
        post_plot = post_all
    else:
        pre_plot = filter_records(pre_all)
        post_plot = filter_records(post_all)

    print(f"Pre: {len(pre_all)} total -> {len(pre_plot)} plotted")
    print(f"Post: {len(post_all)} total -> {len(post_plot)} plotted")

    # Figure layout: top density (small) + bottom scatter (main)
    fig = plt.figure(figsize=(7.5, 8.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])

    draw_density(ax_top, pre_plot, post_plot)
    draw_scatter(ax_main, pre_plot, post_plot)
    draw_fingerprint_table(ax_main, summary)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")

    # Also save PNG
    png_out = args.out.replace(".pdf", ".png")
    fig.savefig(png_out, dpi=args.dpi, bbox_inches="tight")

    print(f"\nFigure saved: {args.out}")
    print(f"PNG saved:    {png_out}")

    # Print quick stats
    if pre_plot and post_plot:
        print("\n=== Quick Stats (filtered points) ===")
        for metric in ["ID", "IE", "PIR", "RGR", "AHS"]:
            pre_v = [r.get(metric, 0) for r in pre_plot]
            post_v = [r.get(metric, 0) for r in post_plot]
            pre_m = float(np.mean(pre_v))
            post_m = float(np.mean(post_v))
            print(f"  {metric}: Pre={pre_m:.3f}  Post={post_m:.3f}  Δ={post_m-pre_m:+.3f}")


if __name__ == "__main__":
    main()
