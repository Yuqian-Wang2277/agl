#!/usr/bin/env python3
"""将 compute_train_baseline.sh 的推理结果汇总为 baseline_cache.json。

baseline_cache.json 格式：
    {
        "<md5(problem)[:16]>": <avg_answer_soft>,
        ...
    }

每道题的 key 为问题文本的 MD5 前 16 位，value 为多次推理 answer_soft 的均值。
该 cache 在 RL 训练中用于计算增量奖励 R_delta = max(0, current_soft - baseline_soft)。

用法：
    python examples/strategy_extraction/scripts/build_baseline_cache.py \\
        --input-dir ./checkpoints_baseline_train \\
        --output ./baseline_cache.json

    # 查看统计但不保存：
    python examples/strategy_extraction/scripts/build_baseline_cache.py \\
        --input-dir ./checkpoints_baseline_train \\
        --output ./baseline_cache.json \\
        --dry-run
"""

import argparse
import glob
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build baseline_cache.json from eval_no_verl output JSONs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./checkpoints_baseline_train",
        help="Directory containing validation_step*.json files (searched recursively).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./baseline_cache.json",
        help="Output path for baseline_cache.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without saving the cache file.",
    )
    return parser.parse_args()


def problem_key(problem_text: str) -> str:
    """Stable 16-char hex key derived from problem text (MD5 prefix)."""
    return hashlib.md5(problem_text.encode("utf-8")).hexdigest()[:16]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021).

    n: total attempts, c: correct attempts, k: k value.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def main() -> None:
    args = parse_args()

    pattern = str(Path(args.input_dir) / "**" / "validation_step*.json")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print(f"[ERROR] 未找到 validation_step*.json 文件（搜索路径：{pattern}）", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 找到 {len(files)} 个 JSON 文件，开始解析...")

    # key → 多次推理的 soft / hard 列表
    soft_by_key: dict[str, list[float]] = defaultdict(list)
    hard_by_key: dict[str, list[int]] = defaultdict(list)
    problem_text_by_key: dict[str, str] = {}
    parse_errors = 0

    for fpath in files:
        try:
            entries = json.load(open(fpath, encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] 无法读取 {fpath}: {e}", file=sys.stderr)
            parse_errors += 1
            continue

        for e in entries:
            try:
                problem = e["input"]["problem"]
                soft = float(e["reward"]["answer_soft"])
                # hard_correct_list 记录每次采样的 0/1；若不存在则退化为单值
                hard_list = e["reward"].get(
                    "hard_correct_list",
                    [int(e["reward"].get("hard_correct", e["reward"].get("answer_hard", 0)))],
                )
                key = problem_key(problem)
                soft_by_key[key].append(soft)
                hard_by_key[key].extend([int(x) for x in hard_list])
                problem_text_by_key[key] = problem  # 仅用于 debug
            except (KeyError, TypeError, ValueError) as e2:
                parse_errors += 1
                continue

    if not soft_by_key:
        print("[ERROR] 未解析到任何有效条目，请检查 JSON 文件结构。", file=sys.stderr)
        sys.exit(1)

    n_problems = len(soft_by_key)
    print(f"[INFO] 解析完成：{n_problems} 道题，{parse_errors} 条解析错误")

    # ---- baseline cache（avg soft）----
    cache: dict[str, float] = {k: sum(v) / len(v) for k, v in soft_by_key.items()}

    # ---- 统计输出 ----
    avg_soft = sum(cache.values()) / n_problems

    # pass@k 统计（基于每题的 hard_correct 列表）
    rollout_counts = [len(v) for v in hard_by_key.values()]
    avg_rollouts = sum(rollout_counts) / n_problems
    max_k = min(3, min(rollout_counts)) if rollout_counts else 1

    passk_stats = {}
    for k in range(1, max_k + 1):
        vals = [
            pass_at_k(n=len(hard_by_key[key]), c=sum(hard_by_key[key]), k=k)
            for key in hard_by_key
        ]
        passk_stats[k] = sum(vals) / len(vals)

    difficulty_dist = {
        "easy (soft>0.8)":   sum(1 for v in cache.values() if v > 0.8) / n_problems,
        "mid (0.2<soft≤0.8)": sum(1 for v in cache.values() if 0.2 < v <= 0.8) / n_problems,
        "hard (soft≤0.2)":   sum(1 for v in cache.values() if v <= 0.2) / n_problems,
    }

    print()
    print("=" * 50)
    print(f"  题目数量    : {n_problems}")
    print(f"  平均推理次数 : {avg_rollouts:.1f}")
    print(f"  avg soft    : {avg_soft:.4f}")
    for k, v in passk_stats.items():
        print(f"  pass@{k}      : {v:.4f}")
    print()
    print("  难度分布（基于 SFT 策略 baseline soft 分）：")
    for label, ratio in difficulty_dist.items():
        print(f"    {label}: {ratio:.1%}")
    print("=" * 50)
    print()

    if args.dry_run:
        print("[DRY-RUN] 未保存文件。")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"[DONE] baseline_cache.json 已保存至 {output_path}（{n_problems} 条记录）")
    print()
    print("下一步：在训练脚本中启用增量奖励：")
    print(f"  BASELINE_CACHE_PATH={output_path} INCREMENTAL_WEIGHT=0.3 \\")
    print("    bash examples/strategy_extraction/scripts/train_generation.sh")


if __name__ == "__main__":
    main()
