"""
MetaICL unified evaluation script.

Evaluates a trained MetaICL (or MetaICL-CoT) checkpoint against all five
benchmark sets.  Connects to a vLLM-served model via OpenAI-compatible API.

Benchmarks selectable via --benchmark:
  id-ood    test-id-subtask + test-ood-task  (from LLMReflection/data/)
  bbh       test-bbh                          (from LLMReflection/data/)
  hardmath  HARDMath2                         (banchmark/HARDMath2/data/)
  linguini  Linguini                          (banchmark/linguini/dataset.jsonl)
  all       all four above

Scoring:
  - Greedy decode: temperature=0, do_sample=False
  - Pass@1 only (consistent across all methods including closed-source models)
  - Reuses judging functions from existing benchmark scripts

Setup (start vLLM before running):
  CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \\
      --model ./checkpoints/metaicl-qwen3-4b/best \\
      --served-model-name metaicl \\
      --tensor-parallel-size 2 --port 8300

Usage:
  python eval.py --benchmark all --model-name metaicl --model-url http://localhost:8300/v1
  BENCHMARK=id-ood bash scripts/run_eval.sh

Environment: conda activate agl  (has openai)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import AsyncOpenAI
except ImportError:
    sys.exit("[ERROR] openai not found — install with: pip install openai")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Repo paths ────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent
_REPO_ROOT = _THIS_DIR.parent.parent
_BANCHMARK = _REPO_ROOT / "banchmark"

# Inject judging-function paths
sys.path.insert(0, str(_BANCHMARK / "BBH-ID-OOD"))
sys.path.insert(0, str(_BANCHMARK / "HARDMath2"))
sys.path.insert(0, str(_BANCHMARK / "linguini"))

from eval_mist_inline import compute_answer_judgement, load_dataset  # noqa: E402
from eval_hardmath import compare_math_answers, load_hardmath_data, select_fewshot  # noqa: E402
from run_linguini_passk import (  # noqa: E402
    load_dataset as load_linguini,
    format_problem_text,
    format_answers_for_example,
    score_answers,
    parse_numbered_answers,
)

sys.path.insert(0, str(_THIS_DIR))
from data_formatter import make_metaicl_prompt  # noqa: E402

# ── Default data paths ────────────────────────────────────────────────────────
_DATA_BASE = _REPO_ROOT / "data"          # contains test-id-subtask / test-ood-task / test-bbh
_HARDMATH_DATA = _BANCHMARK / "HARDMath2" / "data"
_LINGUINI_DATA = _BANCHMARK / "linguini" / "dataset.jsonl"

BENCHMARKS = ["id-ood", "bbh", "hardmath", "linguini"]

# ── Generation config (greedy, pass@1) ───────────────────────────────────────
GENERATION_KWARGS = dict(
    temperature=0.0,
    max_tokens=512,
    seed=42,
)


# ═════════════════════════════════════════════════════════════════════════════
# LLM call helper
# ═════════════════════════════════════════════════════════════════════════════

async def _generate(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    sem: asyncio.Semaphore,
    retries: int = 3,
) -> str:
    for attempt in range(retries):
        try:
            async with sem:
                resp = await client.completions.create(
                    model=model,
                    prompt=prompt,
                    **GENERATION_KWARGS,
                )
            return resp.choices[0].text
        except Exception as e:
            if attempt == retries - 1:
                logger.warning("Generation failed after %d retries: %s", retries, e)
                return ""
            await asyncio.sleep(1.0)
    return ""


# ═════════════════════════════════════════════════════════════════════════════
# BBH / ID-OOD evaluation
# ═════════════════════════════════════════════════════════════════════════════

async def eval_bbh_splits(
    client: AsyncOpenAI,
    model: str,
    subdirs: List[str],
    data_base: Path,
    k: int,
    samples_per_subtask: int,
    seed: int,
    concurrency: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Evaluate on test-id-subtask / test-ood-task / test-bbh splits."""
    dataset = load_dataset(
        test_data_base=data_base,
        subdirs=subdirs,
        fewshot_k=k,
        samples_per_subtask=samples_per_subtask,
        seed=seed,
    )
    logger.info("[BBH] Loaded %d problems from %s", len(dataset), subdirs)

    sem = asyncio.Semaphore(concurrency)

    async def _eval_one(item: dict) -> dict:
        prompt = make_metaicl_prompt(item["examples"], item["problem"], k=k)
        raw_output = await _generate(client, model, prompt, sem)
        judged = compute_answer_judgement(raw_output.strip(), item["ground_truth"])
        return {
            "split": item["split"],
            "subtask": item["subtask"],
            "hard_correct": judged["hard_correct"],
            "soft_score": judged["soft_score"],
            "prediction": raw_output.strip(),
            "ground_truth": item["ground_truth"],
        }

    results = await asyncio.gather(*[_eval_one(item) for item in dataset])

    # ── Aggregate by split ────────────────────────────────────────────────
    by_split: Dict[str, List[int]] = defaultdict(list)
    for r in results:
        by_split[r["split"]].append(r["hard_correct"])

    summary: Dict[str, Any] = {}
    for split, scores in sorted(by_split.items()):
        acc = sum(scores) / len(scores) if scores else 0.0
        summary[split] = {"pass@1": round(acc, 4), "n": len(scores)}
        logger.info("[BBH] %s  pass@1=%.4f  n=%d", split, acc, len(scores))

    all_scores = [r["hard_correct"] for r in results]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    summary["overall"] = {"pass@1": round(overall, 4), "n": len(all_scores)}
    logger.info("[BBH] overall pass@1=%.4f  n=%d", overall, len(all_scores))

    # Save results
    tag = "_".join(s.replace("test-", "") for s in subdirs)
    _save_results(output_dir, f"bbh_{tag}", {"summary": summary, "details": results})
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# HARDMath2 evaluation
# ═════════════════════════════════════════════════════════════════════════════

async def eval_hardmath(
    client: AsyncOpenAI,
    model: str,
    data_dir: Path,
    k: int,
    max_samples: Optional[int],
    seed: int,
    concurrency: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Evaluate on HARDMath2 using MetaICL format."""
    problems_by_type = load_hardmath_data(data_dir)
    rng = random.Random(seed)

    # Build flat eval list
    eval_items: List[Tuple[str, int, dict]] = []
    for ptype in sorted(problems_by_type):
        for local_idx, problem in enumerate(problems_by_type[ptype]):
            eval_items.append((ptype, local_idx, problem))

    if max_samples is not None:
        eval_items = eval_items[:max_samples]

    logger.info("[HARDMath2] %d problems", len(eval_items))
    sem = asyncio.Semaphore(concurrency)

    async def _eval_one(ptype: str, local_idx: int, problem: dict) -> dict:
        shots = select_fewshot(problems_by_type[ptype], local_idx, k, rng)
        # MetaICL format: Input=prompt, Output=solution
        shot_dicts = [{"input": s["prompt"], "target": s["solution"]} for s in shots]
        prompt = make_metaicl_prompt(shot_dicts, problem["prompt"], k=k)
        raw_output = await _generate(client, model, prompt, sem)
        hard_correct, soft_score = compare_math_answers(raw_output, problem["solution"])
        return {
            "type": ptype,
            "hard_correct": int(hard_correct),
            "soft_score": float(soft_score),
            "prediction": raw_output.strip(),
            "ground_truth": problem["solution"],
        }

    results = await asyncio.gather(*[_eval_one(*item) for item in eval_items])

    scores = [r["hard_correct"] for r in results]
    acc = sum(scores) / len(scores) if scores else 0.0
    summary = {"pass@1": round(acc, 4), "n": len(scores)}
    logger.info("[HARDMath2] pass@1=%.4f  n=%d", acc, len(scores))

    _save_results(output_dir, "hardmath", {"summary": summary, "details": results})
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Linguini evaluation
# ═════════════════════════════════════════════════════════════════════════════

async def eval_linguini(
    client: AsyncOpenAI,
    model: str,
    dataset_file: Path,
    k: int,
    seed: int,
    concurrency: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Evaluate on Linguini using MetaICL format."""
    problems: List[dict] = load_linguini(dataset_file)

    # Group by task_type for leave-one-out few-shot sampling
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for p in problems:
        by_type[p["task_type"]].append(p)

    logger.info("[Linguini] %d problems across %d task types", len(problems), len(by_type))
    rng = random.Random(seed)
    sem = asyncio.Semaphore(concurrency)

    async def _eval_one(problem: dict) -> dict:
        task_type = problem["task_type"]
        candidates = [p for p in by_type[task_type] if p["id"] != problem["id"]]
        shots = rng.sample(candidates, min(k, len(candidates))) if candidates else []

        # Build MetaICL-style shot dicts
        # Input = "Context:\n{ctx}\n\nTask:\n{query}"
        # Output = numbered answer list
        shot_dicts = []
        for s in shots:
            shot_input = format_problem_text(s)
            shot_output = format_answers_for_example(s)
            shot_dicts.append({"input": shot_input, "target": shot_output})

        test_input = format_problem_text(problem)
        prompt = make_metaicl_prompt(shot_dicts, test_input, k=k)
        raw_output = await _generate(client, model, prompt, sem)

        predicted = parse_numbered_answers(raw_output)
        score = score_answers(predicted, problem["answer"], problem.get("eval_type", "single"))
        hard_correct = int(score >= 1.0)

        return {
            "id": problem["id"],
            "task_type": task_type,
            "hard_correct": hard_correct,
            "soft_score": float(score),
            "prediction": raw_output.strip(),
        }

    results = await asyncio.gather(*[_eval_one(p) for p in problems])

    scores = [r["hard_correct"] for r in results]
    acc = sum(scores) / len(scores) if scores else 0.0
    summary = {"pass@1": round(acc, 4), "n": len(scores)}
    logger.info("[Linguini] pass@1=%.4f  n=%d", acc, len(scores))

    _save_results(output_dir, "linguini", {"summary": summary, "details": results})
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Output helpers
# ═════════════════════════════════════════════════════════════════════════════

def _save_results(output_dir: Path, tag: str, data: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{tag}_{ts}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results saved → %s", path)


def _print_summary(all_results: dict) -> None:
    print("\n" + "=" * 60)
    print("MetaICL Evaluation Summary (pass@1, greedy decode)")
    print("=" * 60)
    for bench, res in all_results.items():
        if isinstance(res, dict) and "overall" in res:
            print(f"  {bench}")
            for split, v in res.items():
                print(f"    {split:25s}  {v['pass@1']:.4f}  (n={v['n']})")
        elif isinstance(res, dict) and "pass@1" in res:
            print(f"  {bench:30s}  {res['pass@1']:.4f}  (n={res['n']})")
    print("=" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MetaICL unified evaluation")
    p.add_argument(
        "--benchmark", "-b", nargs="+",
        choices=BENCHMARKS + ["all"],
        default=["all"],
        help="Benchmark(s) to evaluate. 'all' runs all four.",
    )
    p.add_argument("--model-url", default=os.environ.get("MODEL_URL", "http://localhost:8300/v1"),
                   help="vLLM OpenAI-compatible base URL")
    p.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "metaicl"),
                   help="Model name as served by vLLM")
    p.add_argument("--api-key", default="dummy")
    p.add_argument("--k-shot", type=int, default=int(os.environ.get("K_SHOT", "4")),
                   help="Few-shot examples per problem (must match training k)")
    p.add_argument("--samples-per-subtask", type=int, default=20,
                   help="BBH/ID-OOD: problems sampled per subtask file (default 20)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=32,
                   help="Async concurrent requests to vLLM")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap total problems per benchmark (for smoke tests)")
    p.add_argument("--output-dir", default="./results",
                   help="Directory for JSON result files")
    p.add_argument("--data-base", default=str(_DATA_BASE),
                   help="Root data directory (contains test-* subdirs)")
    p.add_argument("--hardmath-data", default=str(_HARDMATH_DATA))
    p.add_argument("--linguini-data", default=str(_LINGUINI_DATA))
    return p.parse_args()


async def _main() -> None:
    args = _parse_args()

    targets = BENCHMARKS if "all" in args.benchmark else list(dict.fromkeys(args.benchmark))
    output_dir = Path(args.output_dir)

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.model_url)
    data_base = Path(args.data_base)
    all_results: dict = {}

    logger.info("Model: %s @ %s", args.model_name, args.model_url)
    logger.info("Benchmarks: %s  k=%d  concurrency=%d", targets, args.k_shot, args.concurrency)

    if "id-ood" in targets:
        res = await eval_bbh_splits(
            client, args.model_name,
            subdirs=["test-id-subtask", "test-ood-task"],
            data_base=data_base,
            k=args.k_shot,
            samples_per_subtask=args.samples_per_subtask,
            seed=args.seed,
            concurrency=args.concurrency,
            output_dir=output_dir,
        )
        all_results["id-ood"] = res

    if "bbh" in targets:
        res = await eval_bbh_splits(
            client, args.model_name,
            subdirs=["test-bbh"],
            data_base=data_base,
            k=args.k_shot,
            samples_per_subtask=args.samples_per_subtask,
            seed=args.seed,
            concurrency=args.concurrency,
            output_dir=output_dir,
        )
        all_results["bbh"] = res

    if "hardmath" in targets:
        res = await eval_hardmath(
            client, args.model_name,
            data_dir=Path(args.hardmath_data),
            k=args.k_shot,
            max_samples=args.max_samples,
            seed=args.seed,
            concurrency=args.concurrency,
            output_dir=output_dir,
        )
        all_results["hardmath"] = res

    if "linguini" in targets:
        res = await eval_linguini(
            client, args.model_name,
            dataset_file=Path(args.linguini_data),
            k=args.k_shot,
            seed=args.seed,
            concurrency=args.concurrency,
            output_dir=output_dir,
        )
        all_results["linguini"] = res

    _print_summary(all_results)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
