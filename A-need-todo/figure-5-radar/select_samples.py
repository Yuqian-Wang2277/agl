#!/usr/bin/env python3
"""
select_samples.py — Extract 100 representative strategy samples for Figure-5 radar scoring.

Pre-GIST  (habit):  selects the 100 lowest-correctness rollouts (worst performers),
                    stratified by problem_type.
Post-GIST (MIST):   selects the 100 highest-correctness rollouts (best performers),
                    stratified by problem_type.

Output:
  data/pre_gist_100.json
  data/post_gist_100.json
"""

import json
import os
import glob
import argparse
from collections import defaultdict

# ---------------------------------------------------------------------------
# Defaults — adjust if you move the raw shards
# ---------------------------------------------------------------------------
PRE_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../checkpoints_eval_no_verl/habit/Qwen3-4B/habit/Qwen3-4B",
)
POST_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../checkpoints_eval_no_verl/MIST/Qwen3-semi/MIST/Qwen3-semi",
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
N_SAMPLES = 100


def load_shards(directory: str) -> list[dict]:
    """Load all JSON shard files from a directory into a flat list of rollouts."""
    pattern = os.path.join(directory, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No JSON shards found in: {directory}")
    rollouts = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  [warn] Skipping malformed shard {path}: {e}")
                continue
            if isinstance(data, list):
                rollouts.extend(data)
            elif isinstance(data, dict):
                rollouts.append(data)
    return rollouts


def get_strategy(rollout: dict) -> str:
    """Extract the cleaned strategy text, fall back to raw if extracted is empty."""
    out = rollout.get("output", {})
    strategy = out.get("strategy_extracted", "").strip()
    if not strategy:
        raw = out.get("strategy_raw", "")
        # strip <strategy>...</strategy> tags if present
        import re
        m = re.search(r"<strategy>(.*?)</strategy>", raw, re.DOTALL)
        strategy = m.group(1).strip() if m else raw.strip()
    return strategy


def get_problem_type(rollout: dict) -> str:
    """Return the problem type string."""
    # Top-level field (always present in the sample we inspected)
    pt = rollout.get("problem_type", "")
    if not pt:
        pt = rollout.get("input", {}).get("task_meta", {}).get("problem_type", "unknown")
    return pt or "unknown"


def get_correctness(rollout: dict) -> float:
    """Return the correctness reward (float). Falls back to final reward."""
    reward = rollout.get("reward", {})
    val = reward.get("correctness")
    if val is None:
        val = reward.get("final", 0.0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def build_record(rollout: dict, split: str) -> dict:
    """Build the output record dict."""
    inp = rollout.get("input", {})
    return {
        "rollout_id": rollout.get("rollout_id", ""),
        "problem_type": get_problem_type(rollout),
        "task": inp.get("problem", "").strip(),
        "strategy": get_strategy(rollout),
        "reward_correctness": get_correctness(rollout),
        "split": split,
    }


def stratified_select(rollouts: list[dict], n: int, descending: bool) -> list[dict]:
    """
    Stratify rollouts by problem_type, then pick `n` records total.

    Strategy:
      1. Sort within each bucket by correctness (ascending for worst, descending for best).
      2. Distribute `n` slots evenly across buckets (round-robin top picks).
      3. Fill remaining slots greedily from globally sorted leftovers.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rollouts:
        buckets[get_problem_type(r)].append(r)

    # Sort each bucket
    for pt in buckets:
        buckets[pt].sort(key=get_correctness, reverse=descending)

    # Round-robin: take the best/worst from each bucket in turn
    selected = []
    seen_ids = set()
    bucket_lists = [v for v in buckets.values()]
    pointers = [0] * len(bucket_lists)
    round_idx = 0

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
                strategy = get_strategy(candidate)
                if len(strategy) < 50:  # skip trivially empty strategies
                    continue
                seen_ids.add(rid)
                selected.append(candidate)
                made_progress = True
                break
        round_idx += 1
        if not made_progress:
            break  # exhausted all buckets

    return selected


def main():
    parser = argparse.ArgumentParser(description="Select representative strategy samples.")
    parser.add_argument("--pre-dir", default=PRE_DIR, help="Path to pre-GIST shard directory")
    parser.add_argument("--post-dir", default=POST_DIR, help="Path to post-GIST shard directory")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Output directory for JSON files")
    parser.add_argument("--n", type=int, default=N_SAMPLES, help="Number of samples to select per split")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Pre-GIST: worst performers (lowest correctness) ----
    print(f"Loading pre-GIST shards from: {args.pre_dir}")
    pre_rollouts = load_shards(args.pre_dir)
    print(f"  Total rollouts loaded: {len(pre_rollouts)}")
    pre_selected = stratified_select(pre_rollouts, args.n, descending=False)
    pre_records = [build_record(r, "pre_gist") for r in pre_selected]
    print(f"  Selected {len(pre_records)} samples (worst performers, stratified by problem_type)")
    pre_path = os.path.join(args.out_dir, "pre_gist_100.json")
    with open(pre_path, "w", encoding="utf-8") as f:
        json.dump(pre_records, f, ensure_ascii=False, indent=2)
    print(f"  Saved -> {pre_path}")

    # ---- Post-GIST: best performers (highest correctness) ----
    print(f"\nLoading post-GIST shards from: {args.post_dir}")
    post_rollouts = load_shards(args.post_dir)
    print(f"  Total rollouts loaded: {len(post_rollouts)}")
    post_selected = stratified_select(post_rollouts, args.n, descending=True)
    post_records = [build_record(r, "post_gist") for r in post_selected]
    print(f"  Selected {len(post_records)} samples (best performers, stratified by problem_type)")
    post_path = os.path.join(args.out_dir, "post_gist_100.json")
    with open(post_path, "w", encoding="utf-8") as f:
        json.dump(post_records, f, ensure_ascii=False, indent=2)
    print(f"  Saved -> {post_path}")

    # Summary stats
    print("\n=== Pre-GIST correctness distribution ===")
    scores = [r["reward_correctness"] for r in pre_records]
    print(f"  min={min(scores):.3f}  max={max(scores):.3f}  mean={sum(scores)/len(scores):.3f}")
    types = sorted(set(r["problem_type"] for r in pre_records))
    print(f"  Covered {len(types)} problem types: {types[:10]}{'...' if len(types)>10 else ''}")

    print("\n=== Post-GIST correctness distribution ===")
    scores = [r["reward_correctness"] for r in post_records]
    print(f"  min={min(scores):.3f}  max={max(scores):.3f}  mean={sum(scores)/len(scores):.3f}")
    types = sorted(set(r["problem_type"] for r in post_records))
    print(f"  Covered {len(types)} problem types: {types[:10]}{'...' if len(types)>10 else ''}")


if __name__ == "__main__":
    main()
