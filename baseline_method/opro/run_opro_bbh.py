"""OPRO baseline evaluation on BBH / ID / OOD (three independent eval sets).

Uses eval_mist_inline.load_dataset() with seed=42 / samples_per_subtask=20,
exactly matching the existing eval script's data loading.
Optimizes one instruction per problem_type (49 total across all three splits).
Evaluates with temperature=0.0, num_samples=1, reports pass@1 per split.
"""

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from openai import AsyncOpenAI

_OPRO_DIR = Path(__file__).parent
_REPO_ROOT = _OPRO_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "banchmark" / "BBH-ID-OOD"))
from eval_mist_inline import TEST_SUBDIRS, load_dataset  # noqa: E402

from opro_optimizer import OPROOptimizer, _load_toml  # noqa: E402
from scorers.bbh_scorer import score as bbh_score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK = "bbh"
_BBH_EVAL_MODULE = _REPO_ROOT / "banchmark" / "BBH-ID-OOD"
DEFAULT_DATA_BASE = _REPO_ROOT / "examples" / "strategy_extraction" / "test"
PROMPTS_DIR = _OPRO_DIR / "prompts"


async def _answer_one(
    client: AsyncOpenAI,
    model: str,
    prompt: dict,
    instruction: str,
    problem_text: str,
    seed: int,
) -> str:
    user_content = prompt["user"]["content"].format(
        instruction=instruction,
        problem=problem_text,
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt["system"]["content"]},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=1024,
        seed=seed,
    )
    return resp.choices[0].message.content or ""


async def evaluate(
    dataset: list[dict],
    best_instructions: dict[str, str],
    answer_client: AsyncOpenAI,
    answer_model: str,
    answer_prompt: dict,
    concurrency: int,
) -> dict:
    """Evaluate best instructions on all three splits; return per-split pass@1."""
    sem = asyncio.Semaphore(concurrency)
    per_split: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    async def _eval_one(item: dict) -> None:
        instruction = best_instructions.get(item["problem_type"], "")
        async with sem:
            ans = await _answer_one(answer_client, answer_model, answer_prompt,
                                    instruction, item["problem"], 42)
        hard, _ = bbh_score(ans, item["ground_truth"])
        split = item["split"]
        per_split[split]["correct"] += int(hard)
        per_split[split]["total"] += 1

    await asyncio.gather(*[_eval_one(item) for item in dataset])

    results: dict = {}
    for split, counts in per_split.items():
        t = counts["total"]
        results[split] = {**counts, "pass1": counts["correct"] / t if t else 0.0}
    return results


async def main(args: argparse.Namespace) -> None:
    # Load dataset (same as eval_mist_inline.py)
    data_base = Path(args.data_base)
    dataset = load_dataset(
        data_base,
        subdirs=TEST_SUBDIRS,
        fewshot_k=3,
        samples_per_subtask=20,
        seed=args.val_sampling_seed,
    )
    logger.info("Loaded %d evaluation samples", len(dataset))

    # Group by problem_type for optimization
    tasks: dict[str, list[dict]] = defaultdict(list)
    for item in dataset:
        tasks[item["problem_type"]].append(item)
    logger.info("Optimizing %d problem_types", len(tasks))

    # Clients
    optimizer_client = AsyncOpenAI(
        base_url=args.optimizer_base_url,
        api_key=args.gemini_api_key,
    )
    answer_client = AsyncOpenAI(
        base_url=args.answer_base_url,
        api_key=args.answer_api_key or "EMPTY",
    )

    # Prompts
    meta_prompt = _load_toml(PROMPTS_DIR / "meta_optimizer.toml")
    answer_prompt = _load_toml(PROMPTS_DIR / "opro_answer_bbh.toml")

    # Optimizer
    optimizer = OPROOptimizer(
        optimizer_client=optimizer_client,
        answer_client=answer_client,
        optimizer_model=args.optimizer_model,
        answer_model=args.answer_model,
        scorer_fn=bbh_score,
        meta_prompt=meta_prompt,
        num_steps=args.num_steps,
        eval_batch_size=5,
        history_top_k=8,
        optimizer_temperature=1.0,
        answer_temperature=0.0,
        checkpoint_dir=_OPRO_DIR / "checkpoints",
        concurrency=args.concurrency,
    )

    # Optimize
    logger.info("=== OPRO Optimization Phase ===")
    best_instructions = await optimizer.optimize_all_tasks(BENCHMARK, dict(tasks))

    # Save best instructions
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inst_path = output_dir / f"best_instructions_{args.answer_model.replace('/', '_')}.json"
    inst_path.write_text(json.dumps(best_instructions, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Best instructions saved to %s", inst_path)

    # Evaluate across all three splits
    logger.info("=== OPRO Evaluation Phase (pass@1, BBH / ID / OOD) ===")
    eval_results = await evaluate(
        dataset=dataset,
        best_instructions=best_instructions,
        answer_client=answer_client,
        answer_model=args.answer_model,
        answer_prompt=answer_prompt,
        concurrency=args.concurrency * 8,
    )

    # Print results
    print("\n=== BBH / ID / OOD OPRO Results (pass@1) ===")
    split_order = ["test-bbh", "test-id-subtask", "test-ood-task"]
    print(f"{'split':<25} {'correct':>8} {'total':>7} {'pass@1':>8}")
    print("-" * 55)
    for split in split_order:
        if split in eval_results:
            r = eval_results[split]
            print(f"{split:<25} {r['correct']:>8} {r['total']:>7} {r['pass1']:>8.4f}")
    for split in sorted(eval_results):
        if split not in split_order:
            r = eval_results[split]
            print(f"{split:<25} {r['correct']:>8} {r['total']:>7} {r['pass1']:>8.4f}")

    results_path = output_dir / f"results_{args.answer_model.replace('/', '_')}.json"
    results_path.write_text(
        json.dumps({"per_split": eval_results, "answer_model": args.answer_model,
                    "num_steps": args.num_steps}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Results saved to %s", results_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OPRO baseline for BBH / ID / OOD")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--answer-model", default="Qwen3-4B")
    p.add_argument("--answer-base-url", default="http://localhost:8200/v1")
    p.add_argument("--answer-api-key", default="EMPTY")
    p.add_argument("--optimizer-model", default="gemini-3-flash-preview")
    p.add_argument("--optimizer-base-url",
                   default="https://generativelanguage.googleapis.com/v1beta/openai")
    p.add_argument("--gemini-api-key", default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--data-base", default=str(DEFAULT_DATA_BASE))
    p.add_argument("--val-sampling-seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(_OPRO_DIR / "results" / "bbh"))
    return p.parse_args()


if __name__ == "__main__":
    import os
    args = parse_args()
    if args.gemini_api_key is None:
        args.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    asyncio.run(main(args))
