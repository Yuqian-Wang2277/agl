"""OPRO baseline evaluation on Linguini.

Loads all 160 problems via run_linguini_passk.load_dataset(), groups by
task_type, runs OPRO optimization, then evaluates best instructions
(num_samples=1, temperature=0.7, pass@1).
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
sys.path.insert(0, str(_REPO_ROOT / "banchmark" / "linguini"))
from run_linguini_passk import (  # noqa: E402
    format_problem_text,
    load_dataset as load_linguini_dataset,
)

from opro_optimizer import OPROOptimizer, _load_toml  # noqa: E402
from scorers.linguini_scorer import score as linguini_score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK = "linguini"
DEFAULT_DATASET_FILE = _REPO_ROOT / "banchmark" / "linguini" / "dataset.jsonl"
PROMPTS_DIR = _OPRO_DIR / "prompts"


def _make_scorer(eval_type: str):
    """Return a scorer_fn bound to the given eval_type."""
    def _score(model_output: str, ground_truth) -> tuple[bool, float]:
        return linguini_score(model_output, ground_truth, eval_type)
    return _score


async def _answer_one(
    client: AsyncOpenAI,
    model: str,
    prompt: dict,
    instruction: str,
    problem: dict,
    temperature: float,
    seed: int,
) -> str:
    problem_text = format_problem_text(problem)
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
        temperature=temperature,
        max_tokens=1024,
        seed=seed,
    )
    return resp.choices[0].message.content or ""


async def evaluate(
    all_problems: dict[str, list[dict]],
    best_instructions: dict[str, str],
    answer_client: AsyncOpenAI,
    answer_model: str,
    answer_prompt: dict,
    temperature: float,
    base_seed: int,
    concurrency: int,
) -> dict:
    """Evaluate best instructions on all task_types; return per-type pass@1."""
    sem = asyncio.Semaphore(concurrency)
    results: dict = {}

    async def _eval_type(task_type: str, problems: list[dict]) -> None:
        instruction = best_instructions.get(task_type, "")
        correct = 0
        total = 0

        async def _one(prob: dict) -> None:
            nonlocal correct, total
            async with sem:
                ans = await _answer_one(answer_client, answer_model, answer_prompt,
                                        instruction, prob, temperature, base_seed)
            eval_type = prob.get("eval_type", "single")
            hard, _ = linguini_score(ans, prob["answer"], eval_type)
            correct += int(hard)
            total += 1

        await asyncio.gather(*[_one(p) for p in problems])
        results[task_type] = {"correct": correct, "total": total,
                               "pass1": correct / total if total else 0.0}

    await asyncio.gather(*[_eval_type(tt, probs) for tt, probs in all_problems.items()])
    return results


async def main(args: argparse.Namespace) -> None:
    # Load dataset
    dataset_file = Path(args.dataset_file)
    problems = load_linguini_dataset(dataset_file)
    logger.info("Loaded %d Linguini problems", len(problems))

    # Group by task_type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for p in problems:
        by_type[p["task_type"]].append(p)

    # Prepare tasks with ground_truth field for optimizer
    tasks: dict[str, list[dict]] = {}
    for tt, probs in by_type.items():
        enriched = []
        for p in probs:
            ep = dict(p)
            # Store answer as JSON string for scorer
            ep["ground_truth"] = json.dumps(p["answer"], ensure_ascii=False)
            enriched.append(ep)
        tasks[tt] = enriched

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
    answer_prompt = _load_toml(PROMPTS_DIR / "opro_answer_linguini.toml")

    # Use a generic scorer for optimization (task_type-aware eval_type via problem field)
    def _opt_score(model_output: str, ground_truth: str) -> tuple[bool, float]:
        return linguini_score(model_output, ground_truth, "single")

    # Optimizer
    optimizer = OPROOptimizer(
        optimizer_client=optimizer_client,
        answer_client=answer_client,
        optimizer_model=args.optimizer_model,
        answer_model=args.answer_model,
        scorer_fn=_opt_score,
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
    best_instructions = await optimizer.optimize_all_tasks(BENCHMARK, tasks)

    # Save best instructions
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inst_path = output_dir / f"best_instructions_{args.answer_model.replace('/', '_')}.json"
    inst_path.write_text(json.dumps(best_instructions, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Best instructions saved to %s", inst_path)

    # Evaluate
    logger.info("=== OPRO Evaluation Phase (pass@1) ===")
    eval_results = await evaluate(
        all_problems=by_type,
        best_instructions=best_instructions,
        answer_client=answer_client,
        answer_model=args.answer_model,
        answer_prompt=answer_prompt,
        temperature=0.7,
        base_seed=42,
        concurrency=args.concurrency * 4,
    )

    # Print results table
    total_correct = sum(v["correct"] for v in eval_results.values())
    total_problems = sum(v["total"] for v in eval_results.values())
    overall = total_correct / total_problems if total_problems else 0.0

    print("\n=== Linguini OPRO Results (pass@1) ===")
    print(f"{'task_type':<20} {'correct':>8} {'total':>7} {'pass@1':>8}")
    print("-" * 50)
    for tt in sorted(eval_results):
        r = eval_results[tt]
        print(f"{tt:<20} {r['correct']:>8} {r['total']:>7} {r['pass1']:>8.4f}")
    print("-" * 50)
    print(f"{'Overall':<20} {total_correct:>8} {total_problems:>7} {overall:>8.4f}")

    results_path = output_dir / f"results_{args.answer_model.replace('/', '_')}.json"
    results_path.write_text(
        json.dumps({"per_type": eval_results, "overall": overall,
                    "answer_model": args.answer_model,
                    "num_steps": args.num_steps}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Results saved to %s", results_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OPRO baseline for Linguini")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--answer-model", default="Qwen3-4B")
    p.add_argument("--answer-base-url", default="http://localhost:8200/v1")
    p.add_argument("--answer-api-key", default="EMPTY")
    p.add_argument("--optimizer-model", default="gemini-3-flash-preview")
    p.add_argument("--optimizer-base-url",
                   default="https://generativelanguage.googleapis.com/v1beta/openai")
    p.add_argument("--gemini-api-key", default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--dataset-file", default=str(DEFAULT_DATASET_FILE))
    p.add_argument("--output-dir", default=str(_OPRO_DIR / "results" / "linguini"))
    return p.parse_args()


if __name__ == "__main__":
    import os
    args = parse_args()
    if args.gemini_api_key is None:
        args.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    asyncio.run(main(args))
