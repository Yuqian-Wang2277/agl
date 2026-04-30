"""OPRO baseline evaluation on HARDMath2.

Loads all 211 problems via eval_hardmath.load_hardmath_data(), groups by
task_type, runs OPRO optimization, then evaluates best instructions on the
full problem set (num_samples=1, temperature=0.7, pass@1).
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from openai import AsyncOpenAI

_OPRO_DIR = Path(__file__).parent
_REPO_ROOT = _OPRO_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "banchmark" / "HARDMath2"))
from eval_hardmath import load_hardmath_data  # noqa: E402

from opro_optimizer import OPROOptimizer, _load_toml  # noqa: E402
from scorers.hardmath_scorer import score as hardmath_score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK = "hardmath"
DEFAULT_DATA_DIR = _REPO_ROOT / "banchmark" / "HARDMath2" / "data"
PROMPTS_DIR = _OPRO_DIR / "prompts"


async def evaluate(
    optimizer: OPROOptimizer,
    best_instructions: dict[str, str],
    all_problems: dict[str, list[dict]],
    answer_client: AsyncOpenAI,
    answer_model: str,
    answer_prompt: dict,
    temperature: float,
    base_seed: int,
    concurrency: int,
) -> dict:
    """Run test-time evaluation with best instructions; return per-type pass@1."""
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, dict] = {}

    async def _eval_type(task_type: str, problems: list[dict]) -> None:
        instruction = best_instructions.get(task_type, "")
        correct = 0
        total = 0
        async with sem:
            tasks_coro = [
                _answer_one(answer_client, answer_model, answer_prompt,
                            instruction, prob, temperature, base_seed)
                for prob in problems
            ]
            answers = await asyncio.gather(*tasks_coro)
        for prob, ans in zip(problems, answers):
            hard, _ = hardmath_score(ans, prob.get("solution", prob.get("answer", "")))
            correct += int(hard)
            total += 1
        results[task_type] = {"correct": correct, "total": total,
                               "pass1": correct / total if total else 0.0}

    await asyncio.gather(*[_eval_type(tt, probs) for tt, probs in all_problems.items()])
    return results


async def _answer_one(
    client: AsyncOpenAI,
    model: str,
    prompt: dict,
    instruction: str,
    problem: dict,
    temperature: float,
    seed: int,
) -> str:
    problem_text = problem.get("prompt", problem.get("problem", ""))
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
        max_tokens=2048,
        seed=seed,
    )
    return resp.choices[0].message.content or ""


async def main(args: argparse.Namespace) -> None:
    # Load data
    data_dir = Path(args.data_dir)
    all_problems = load_hardmath_data(data_dir)
    logger.info("Loaded %d task_types, %d total problems",
                len(all_problems), sum(len(v) for v in all_problems.values()))

    # Flatten to {task_type: [{..., ground_truth: ...}]} compatible with optimizer
    tasks: dict[str, list[dict]] = {}
    for tt, probs in all_problems.items():
        enriched = []
        for p in probs:
            ep = dict(p)
            ep["ground_truth"] = p.get("solution", p.get("answer", ""))
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
    answer_prompt = _load_toml(PROMPTS_DIR / "opro_answer_hardmath.toml")

    # Checkpoint dir
    ckpt_dir = _OPRO_DIR / "checkpoints"

    # Optimizer
    optimizer = OPROOptimizer(
        optimizer_client=optimizer_client,
        answer_client=answer_client,
        optimizer_model=args.optimizer_model,
        answer_model=args.answer_model,
        scorer_fn=hardmath_score,
        meta_prompt=meta_prompt,
        num_steps=args.num_steps,
        eval_batch_size=5,
        history_top_k=8,
        optimizer_temperature=1.0,
        answer_temperature=0.0,
        checkpoint_dir=ckpt_dir,
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
        optimizer=optimizer,
        best_instructions=best_instructions,
        all_problems=all_problems,
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

    print("\n=== HARDMath2 OPRO Results (pass@1) ===")
    print(f"{'task_type':<25} {'correct':>8} {'total':>7} {'pass@1':>8}")
    print("-" * 55)
    for tt in sorted(eval_results):
        r = eval_results[tt]
        print(f"{tt:<25} {r['correct']:>8} {r['total']:>7} {r['pass1']:>8.4f}")
    print("-" * 55)
    print(f"{'Overall':<25} {total_correct:>8} {total_problems:>7} {overall:>8.4f}")

    results_path = output_dir / f"results_{args.answer_model.replace('/', '_')}.json"
    results_path.write_text(
        json.dumps({"per_type": eval_results, "overall": overall,
                    "answer_model": args.answer_model,
                    "num_steps": args.num_steps}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Results saved to %s", results_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OPRO baseline for HARDMath2")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--answer-model", default="Qwen3-4B")
    p.add_argument("--answer-base-url", default="http://localhost:8200/v1")
    p.add_argument("--answer-api-key", default="EMPTY")
    p.add_argument("--optimizer-model", default="gemini-3-flash-preview")
    p.add_argument("--optimizer-base-url",
                   default="https://generativelanguage.googleapis.com/v1beta/openai")
    p.add_argument("--gemini-api-key", default=None,
                   help="Gemini API key (or set GEMINI_API_KEY env var)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output-dir", default=str(_OPRO_DIR / "results" / "hardmath"))
    return p.parse_args()


if __name__ == "__main__":
    import os
    args = parse_args()
    if args.gemini_api_key is None:
        args.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    asyncio.run(main(args))
