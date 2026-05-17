#!/usr/bin/env python3
"""
1_extract_tasks.py — 从 Pre-GIST 和 Post-GIST 评测 shard 中分层采样任务。

采样策略：
  Pre-GIST：按 problem_type 分层，每类取 reward.correctness 最低的条目（表现差的案例）
  Post-GIST：按 problem_type 分层，每类取 reward.correctness 最高的条目（表现好的案例）
  两个 split 各取 N=100 条，覆盖相同的 47 种 problem_type。

输出：
  data/pre_gist_tasks.json
  data/post_gist_tasks.json
"""

import json
import os
import glob
import re
import argparse
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
PRE_SHARD_DIR = os.path.join(
    BASE, "../../checkpoints_eval_no_verl/habit/Qwen3-4B/habit/Qwen3-4B"
)
POST_SHARD_DIR = os.path.join(
    BASE, "../../checkpoints_eval_no_verl/MIST/Qwen3-semi/MIST/Qwen3-semi"
)
OUT_DIR = os.path.join(BASE, "data")
N_SAMPLES = 100


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_shards(directory: str) -> list:
    files = sorted(glob.glob(os.path.join(directory, "*.json")))
    if not files:
        raise FileNotFoundError(f"未找到 JSON shard 文件：{directory}")
    rollouts = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  [warn] 跳过损坏 shard {path}: {e}")
                continue
        if isinstance(data, list):
            rollouts.extend(data)
        elif isinstance(data, dict):
            rollouts.append(data)
    return rollouts


def get_correctness(rollout: dict) -> float:
    reward = rollout.get("reward", {})
    val = reward.get("correctness")
    if val is None:
        val = reward.get("final", 0.0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def extract_fewshot_learning(strategy_text: str) -> str:
    """
    从策略全文中提取 FEWSHOT_LEARNING 块。
    匹配 FEWSHOT_LEARNING: 到下一个全大写节标题（如 SECOND_ORDER_STEPS:）之间的文本。
    """
    pattern = re.compile(
        r"FEWSHOT_LEARNING:\s*(.*?)(?=\n[A-Z_]{3,}:|$)",
        re.DOTALL,
    )
    m = pattern.search(strategy_text)
    if m:
        return m.group(1).strip()
    # 回退：返回完整策略
    return strategy_text.strip()


def build_record(rollout: dict, split: str) -> dict:
    inp = rollout.get("input", {})
    strategy_full = rollout.get("output", {}).get("strategy_extracted", "").strip()
    inductive_rule = extract_fewshot_learning(strategy_full)
    examples = inp.get("examples", [])
    return {
        "rollout_id": rollout.get("rollout_id", ""),
        "problem_type": rollout.get("problem_type", "unknown"),
        "reward_correctness": get_correctness(rollout),
        "split": split,
        "system_prompt": inp.get("system_prompt", ""),
        "examples": examples,
        "problem": inp.get("problem", "").strip(),
        "strategy_full": strategy_full,
        "inductive_rule": inductive_rule,
    }


def is_valid(rollout: dict) -> bool:
    """过滤条件：恰好 K=3 个示例，且策略非空。"""
    examples = rollout.get("input", {}).get("examples", [])
    strategy = rollout.get("output", {}).get("strategy_extracted", "").strip()
    return len(examples) == 3 and len(strategy) >= 100


def stratified_select(rollouts: list, n: int, descending: bool) -> list:
    """
    按 problem_type 分层，在每类中按 reward.correctness 排序后取头部条目，
    轮转直至凑满 n 条。

    descending=False → 取最低（Pre-GIST 最差案例）
    descending=True  → 取最高（Post-GIST 最佳案例）
    """
    buckets = defaultdict(list)
    for r in rollouts:
        pt = r.get("problem_type", "unknown")
        buckets[pt].append(r)

    for pt in buckets:
        buckets[pt].sort(key=get_correctness, reverse=descending)

    selected = []
    seen_ids = set()
    bucket_lists = list(buckets.values())
    pointers = [0] * len(bucket_lists)

    while len(selected) < n:
        made_progress = False
        for i, lst in enumerate(bucket_lists):
            if len(selected) >= n:
                break
            while pointers[i] < len(lst):
                candidate = lst[pointers[i]]
                pointers[i] += 1
                rid = candidate.get("rollout_id", id(candidate))
                if rid in seen_ids:
                    continue
                if not is_valid(candidate):
                    continue
                seen_ids.add(rid)
                selected.append(candidate)
                made_progress = True
                break
        if not made_progress:
            break

    return selected


def print_stats(records: list, label: str):
    scores = [r["reward_correctness"] for r in records]
    types = sorted(set(r["problem_type"] for r in records))
    print(f"\n=== {label} ===")
    print(f"  样本数: {len(records)}")
    if scores:
        print(f"  correctness: min={min(scores):.3f}  max={max(scores):.3f}  "
              f"mean={sum(scores)/len(scores):.3f}")
    print(f"  覆盖 problem_type 数: {len(types)}")
    print(f"  类型列表（前10）: {types[:10]}{'...' if len(types) > 10 else ''}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="从评测 shard 中分层采样任务。")
    parser.add_argument("--pre-dir", default=PRE_SHARD_DIR)
    parser.add_argument("--post-dir", default=POST_SHARD_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--n", type=int, default=N_SAMPLES)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Pre-GIST：取正确率最低的 N 条 ----
    print(f"读取 Pre-GIST shard：{args.pre_dir}")
    pre_rollouts = load_shards(args.pre_dir)
    print(f"  加载 rollout 总数：{len(pre_rollouts)}")
    pre_selected = stratified_select(pre_rollouts, args.n, descending=False)
    pre_records = [build_record(r, "pre_gist") for r in pre_selected]
    print_stats(pre_records, "Pre-GIST（最差案例）")
    pre_path = os.path.join(args.out_dir, "pre_gist_tasks.json")
    with open(pre_path, "w", encoding="utf-8") as f:
        json.dump(pre_records, f, ensure_ascii=False, indent=2)
    print(f"  保存至 {pre_path}")

    # ---- Post-GIST：取正确率最高的 N 条 ----
    print(f"\n读取 Post-GIST shard：{args.post_dir}")
    post_rollouts = load_shards(args.post_dir)
    print(f"  加载 rollout 总数：{len(post_rollouts)}")
    post_selected = stratified_select(post_rollouts, args.n, descending=True)
    post_records = [build_record(r, "post_gist") for r in post_selected]
    print_stats(post_records, "Post-GIST（最佳案例）")
    post_path = os.path.join(args.out_dir, "post_gist_tasks.json")
    with open(post_path, "w", encoding="utf-8") as f:
        json.dump(post_records, f, ensure_ascii=False, indent=2)
    print(f"  保存至 {post_path}")

    print("\n完成。数据文件已写入 data/ 目录。")


if __name__ == "__main__":
    main()
