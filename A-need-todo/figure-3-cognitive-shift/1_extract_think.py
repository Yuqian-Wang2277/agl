#!/usr/bin/env python3
"""
1_extract_think.py — 从 Pre/Post OOD 评测 shard 中提取 <think> 轨迹。

过滤条件：validation_split in {"test-ood-task", "test-bbh"}
选取策略：
  Pre：按 problem_type 分层，各类内取 reward.correctness 均值最低的任务（表现差）
  Post：各类内取 reward.correctness 均值最高的任务（表现好）
  共各选 N=100 道题，保留 a0（主分析）+ a1/a2（稳定性）rollout

用法：
  python 1_extract_think.py
  python 1_extract_think.py --pre-dir /path/to/pre --post-dir /path/to/post --n 100
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, "../../checkpoints_eval_no_verl")

PRE_DIR = os.path.join(REPO, "few-shot/Qwen3-4B/open-think/few-shot/Qwen3-4B/open-think")
POST_DIR = os.path.join(
    REPO,
    "few-shot/Qwen3-4B-trained/open-think-all/few-shot/Qwen3-4B-trained/open-think-all",
)
OUT_DIR = os.path.join(BASE, "data")
N_TASKS = 100
OOD_SPLITS = {"test-ood-task", "test-bbh"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_shards(directory: str) -> list:
    files = sorted(glob.glob(os.path.join(directory, "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON shards found: {directory}")
    rollouts = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as e:
                print(f"  [warn] skip {f}: {e}")
                continue
        if isinstance(data, list):
            rollouts.extend(data)
        elif isinstance(data, dict):
            rollouts.append(data)
    return rollouts


def extract_think(answer_raw: str) -> str:
    m = re.search(r"<think>(.*?)</think>", answer_raw, re.DOTALL)
    return m.group(1).strip() if m else ""


def get_task_id(rollout_id: str) -> str:
    # "val-000238-a0" -> "val-000238"
    return "-".join(rollout_id.split("-")[:2])


def get_rollout_suffix(rollout_id: str) -> str:
    # "val-000238-a0" -> "a0"
    parts = rollout_id.split("-")
    return parts[2] if len(parts) >= 3 else "a0"


def is_ood(rollout: dict) -> bool:
    split = rollout.get("input", {}).get("task_meta", {}).get("validation_split", "")
    return split in OOD_SPLITS


def get_correctness(rollout: dict) -> float:
    try:
        return float(rollout.get("reward", {}).get("correctness", 0.0))
    except (TypeError, ValueError):
        return 0.0


def get_problem_type(rollout: dict) -> str:
    return rollout.get("problem_type", "unknown")


def build_task_record(task_id: str, rollouts_by_suffix: dict) -> dict | None:
    """Build structured task record from collected rollouts. Returns None if no think content."""
    a0 = rollouts_by_suffix.get("a0")
    if a0 is None:
        return None

    primary_think = extract_think(a0.get("output", {}).get("answer_raw", ""))
    if not primary_think or len(primary_think) < 50:
        return None

    all_rollouts = []
    for suffix in ["a0", "a1", "a2"]:
        r = rollouts_by_suffix.get(suffix)
        if r is None:
            continue
        think = extract_think(r.get("output", {}).get("answer_raw", ""))
        if not think:
            continue
        inp = r.get("input", {})
        all_rollouts.append({
            "rollout_id": r.get("rollout_id", f"{task_id}-{suffix}"),
            "suffix": suffix,
            "think": think,
            "correctness": get_correctness(r),
        })

    inp = a0.get("input", {})
    examples = inp.get("examples", [])
    examples_summary = "; ".join(
        f"Ex{i+1}: {ex.get('input','')[:80].strip()} → {str(ex.get('target',['']))[0:50]}"
        for i, ex in enumerate(examples[:3])
    )

    return {
        "task_id": task_id,
        "problem_type": get_problem_type(a0),
        "validation_split": a0.get("input", {}).get("task_meta", {}).get("validation_split", ""),
        "mean_correctness": sum(r["correctness"] for r in all_rollouts) / max(len(all_rollouts), 1),
        "problem_text": inp.get("problem", "").strip()[:500],
        "examples_summary": examples_summary,
        "rollouts": all_rollouts,
        "primary_think": primary_think,
    }


def stratified_select(tasks: list, n: int, descending: bool) -> list:
    """
    Group tasks by problem_type, sort each group by mean_correctness,
    then round-robin pick until n tasks selected.
    descending=False  → take lowest correctness (Pre worst cases)
    descending=True   → take highest correctness (Post best cases)
    """
    buckets = defaultdict(list)
    for t in tasks:
        buckets[t["problem_type"]].append(t)

    for pt in buckets:
        buckets[pt].sort(key=lambda x: x["mean_correctness"], reverse=descending)

    selected = []
    seen = set()
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
                if candidate["task_id"] in seen:
                    continue
                seen.add(candidate["task_id"])
                selected.append(candidate)
                made_progress = True
                break
        if not made_progress:
            break

    return selected


def print_stats(records: list, label: str):
    scores = [r["mean_correctness"] for r in records]
    types = sorted(set(r["problem_type"] for r in records))
    splits = sorted(set(r["validation_split"] for r in records))
    print(f"\n=== {label} ===")
    print(f"  Tasks: {len(records)}")
    if scores:
        print(f"  correctness: min={min(scores):.3f}  max={max(scores):.3f}  "
              f"mean={sum(scores)/len(scores):.3f}")
    print(f"  problem_types ({len(types)}): {types[:8]}{'...' if len(types)>8 else ''}")
    print(f"  validation_splits: {splits}")
    total_rollouts = sum(len(r["rollouts"]) for r in records)
    print(f"  total rollouts retained: {total_rollouts}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def process_split(directory: str, n: int, descending: bool, label: str) -> list:
    print(f"\nLoading {label} shards from: {directory}")
    rollouts = load_shards(directory)
    print(f"  Total rollouts loaded: {len(rollouts)}")

    # filter OOD
    ood_rollouts = [r for r in rollouts if is_ood(r)]
    print(f"  OOD rollouts (test-ood-task + test-bbh): {len(ood_rollouts)}")

    # group by task_id
    tasks_map: dict[str, dict] = {}  # task_id -> {suffix -> rollout}
    for r in ood_rollouts:
        rid = r.get("rollout_id", "")
        tid = get_task_id(rid)
        suffix = get_rollout_suffix(rid)
        if tid not in tasks_map:
            tasks_map[tid] = {}
        tasks_map[tid][suffix] = r

    # build task records
    task_records = []
    for tid, by_suffix in tasks_map.items():
        rec = build_task_record(tid, by_suffix)
        if rec is not None:
            task_records.append(rec)
    print(f"  Valid task records (non-empty think): {len(task_records)}")

    # stratified selection
    selected = stratified_select(task_records, n, descending)
    print_stats(selected, label)
    return selected


def main():
    parser = argparse.ArgumentParser(description="Extract <think> trajectories from OOD eval shards.")
    parser.add_argument("--pre-dir", default=PRE_DIR)
    parser.add_argument("--post-dir", default=POST_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--n", type=int, default=N_TASKS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pre_records = process_split(args.pre_dir, args.n, descending=False, label="Pre-GIST (worst)")
    post_records = process_split(args.post_dir, args.n, descending=True, label="Post-GIST (best)")

    pre_path = os.path.join(args.out_dir, "pre_ood_tasks.json")
    post_path = os.path.join(args.out_dir, "post_ood_tasks.json")

    with open(pre_path, "w", encoding="utf-8") as f:
        json.dump(pre_records, f, ensure_ascii=False, indent=2)
    with open(post_path, "w", encoding="utf-8") as f:
        json.dump(post_records, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(pre_records)} Pre tasks -> {pre_path}")
    print(f"Saved {len(post_records)} Post tasks -> {post_path}")
    print("Done.")


if __name__ == "__main__":
    main()
