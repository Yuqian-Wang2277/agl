import argparse
import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import agentlightning as agl

from examples.strategy_extraction.strategy_generation_agent import (
    StrategyGenerationAgent,
    StrategyGenerationTask,
)
from examples.strategy_extraction.train_strategy_generation import (
    _create_strategy_generation_dataset,
    _create_strategy_generation_dataset_per_subtask,
)


logger = logging.getLogger(__name__)


@dataclass
class _SimpleAttempt:
    attempt_id: str


@dataclass
class _SimpleRollout:
    rollout_id: str
    attempt: _SimpleAttempt
    mode: str = "val"


class _DummyLLM:
    """Minimal LLM stub to satisfy StrategyGenerationAgent.rollout_async."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "dummy-key",
        temperature: float = 0.7,
        max_tokens: int = 16384,
    ) -> None:
        self.model = model
        self._base_url = base_url
        self.api_key = api_key
        self.sampling_parameters: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def get_base_url(self, rollout_id: str, attempt_id: str) -> str:  # noqa: D401
        """Return fixed base URL for all rollouts."""
        return self._base_url


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lightweight evaluation of StrategyGenerationAgent without VERL/Ray.",
    )

    # Data
    parser.add_argument(
        "--data-base-path",
        type=str,
        default="/home/test/test16/chenlu/projects/LLMReflection/data/",
        help="Base path to strategy-extraction data (same as train_strategy_generation).",
    )
    parser.add_argument(
        "--val-subdirs",
        type=str,
        nargs="+",
        default=["test-id-subtask", "test-ood-task", "test-bbh"],
        help="Validation subdirectories under data-base-path.",
    )
    parser.add_argument(
        "--val-sampling-mode",
        type=str,
        choices=["per_subtask_fixed", "global"],
        default="per_subtask_fixed",
        help="Validation sampling mode (matches train_strategy_generation).",
    )
    parser.add_argument(
        "--val-samples-per-subtask",
        type=int,
        default=20,
        help="Samples per subtask JSON when using per_subtask_fixed.",
    )
    parser.add_argument(
        "--num-val-samples",
        type=int,
        default=500,
        help="Global cap when using val-sampling-mode=global.",
    )
    parser.add_argument(
        "--val-sampling-seed",
        type=int,
        default=42,
        help="Seed for validation sampling.",
    )
    parser.add_argument(
        "--fewshot-min",
        type=int,
        default=3,
        help="Minimum number of few-shot examples.",
    )
    parser.add_argument(
        "--fewshot-max",
        type=int,
        default=5,
        help="Maximum number of few-shot examples.",
    )

    # Models / servers
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/test/test16/chenlu/model/Qwen3-4B",
        help="Identifier for strategy-generation base model (logged only).",
    )
    parser.add_argument(
        "--strategy-model-base-url",
        type=str,
        default="",
        help="OpenAI-compatible base URL for strategy-generation model. "
        "If empty, falls back to answer-model-base-url.",
    )
    parser.add_argument(
        "--strategy-model-name",
        type=str,
        default="",
        help="Model name used when calling the strategy server. "
        "If empty, falls back to model-path.",
    )
    parser.add_argument(
        "--answer-model-path",
        type=str,
        default="/home/test/test16/chenlu/model/Qwen3-8B",
        help="Identifier for fixed answer model (logged only).",
    )
    parser.add_argument(
        "--answer-model-base-url",
        type=str,
        default="http://localhost:8200/v1",
        help="OpenAI-compatible base URL for answer model.",
    )
    parser.add_argument(
        "--answer-model-name",
        type=str,
        default="Qwen3-8B",
        help="Model name used when calling the answer server.",
    )

    # Reward / prompts
    parser.add_argument(
        "--reward-version",
        type=str,
        default="v3",
        help="Reward config version (see reward/ package).",
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        choices=["scorer_only", "hybrid_grounded"],
        default="scorer_only",
        help="Reward mode for StrategyGenerationAgent.",
    )
    parser.add_argument(
        "--format-weight",
        type=float,
        default=0.0,
        help="Weight for format reward.",
    )
    parser.add_argument(
        "--scorer-weight",
        type=float,
        default=0.0,
        help="Weight for scorer reward (unused in scorer_only mode).",
    )
    parser.add_argument(
        "--grounded-proxy-weight",
        type=float,
        default=0.0,
        help="Weight for grounded proxy reward (hybrid_grounded mode).",
    )
    parser.add_argument(
        "--correctness-weight",
        type=float,
        default=1.0,
        help="Weight for correctness reward (used in v3 correctness).",
    )
    parser.add_argument(
        "--grounded-proxy-k",
        type=int,
        default=1,
        help="Number of answer samples for grounded proxy (k>1 enables multi-sample).",
    )
    parser.add_argument(
        "--strategy-prompt-version",
        type=str,
        default="strategy_update_2026-03-09",
        help="Strategy prompt version.",
    )
    parser.add_argument(
        "--answer-prompt-version",
        type=str,
        default="v1",
        help="Answer prompt version.",
    )

    # Output / bookkeeping
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints_eval_no_verl",
        help="Directory to write per-worker validation JSON shards (optional).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Optional experiment id (subdirectory under output-dir).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional hard cap on number of validation samples (for quick smoke tests).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent async workers for evaluation. "
        "Each worker owns its own agent instance to avoid shared-state conflicts.",
    )

    return parser


async def _run_eval(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger.info("Loading validation datasets (no VERL)...")
    import itertools

    val_datasets: List[List[Dict[str, Any]]] = []
    for i, vs in enumerate(args.val_subdirs):
        vd = os.path.join(args.data_base_path, vs)
        logger.info("Loading val set %d/%d: %s", i + 1, len(args.val_subdirs), vs)
        if args.val_sampling_mode == "per_subtask_fixed":
            vds = _create_strategy_generation_dataset_per_subtask(
                vd,
                fewshot_min=args.fewshot_min,
                fewshot_max=args.fewshot_max,
                samples_per_subtask=args.val_samples_per_subtask,
                seed=args.val_sampling_seed,
            )
        else:
            vds = _create_strategy_generation_dataset(
                vd,
                args.fewshot_min,
                args.fewshot_max,
                args.num_val_samples,
                seed=args.val_sampling_seed,
            )
        val_datasets.append(vds)
        logger.info("  %s: %d samples", vs, len(vds))

    val_dataset: List[StrategyGenerationTask] = list(
        itertools.chain.from_iterable(val_datasets)
    )

    if args.max_samples is not None:
        val_dataset = val_dataset[: args.max_samples]

    logger.info("Total validation samples: %d", len(val_dataset))
    if not val_dataset:
        raise ValueError("Validation dataset is empty.")

    # Agent setup
    rollout_traces_dir = None
    test_freq = 0

    # Create an experiment-scoped subdirectory under output-dir, similar to training.
    output_root = os.path.abspath(args.output_dir)
    if args.experiment_id is None:
        args.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_root, args.experiment_id)
    os.makedirs(exp_dir, exist_ok=True)

    # Save eval configuration for reproducibility.
    cfg_dir = os.path.join(exp_dir, "eval_configs")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_file = os.path.join(cfg_dir, f"config_{args.experiment_id}.json")
    try:
        import json

        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_id": args.experiment_id,
                    "timestamp": datetime.now().isoformat(),
                    **vars(args),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("Eval config saved: %s", cfg_file)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to save eval config: %s", e)

    validation_output_dir = exp_dir

    # Build per-worker agent instances to avoid shared-state conflicts under concurrency.
    # Each worker writes its own validation shard: validation_step0_worker{worker_id}.json
    strategy_base_url = args.strategy_model_base_url or args.answer_model_base_url
    concurrency = max(1, int(args.concurrency or 1))
    strategy_model_name = args.strategy_model_name or args.model_path

    def _make_worker(worker_idx: int) -> Tuple[StrategyGenerationAgent, agl.NamedResources]:
        agent = StrategyGenerationAgent(
            save_full_output=False,
            rollout_traces_dir=rollout_traces_dir,
            validation_output_dir=validation_output_dir,
            experiment_id=args.experiment_id,
            test_freq=test_freq,
            format_weight=args.format_weight,
            scorer_weight=args.scorer_weight,
            proxy_weight=args.grounded_proxy_weight,
            reward_mode=args.reward_mode,
            grounded_proxy_k=args.grounded_proxy_k,
            correctness_weight=args.correctness_weight,
            strategy_scorer_base_url="",  # disabled by default in this lightweight eval
            strategy_scorer_model="",
            answer_model_base_url=args.answer_model_base_url,
            answer_model_name=args.answer_model_name or args.answer_model_path,
            use_strategy_for_answer=True,
            skip_strategy_generation=False,
            strategy_prompt_version=args.strategy_prompt_version,
            answer_prompt_version=args.answer_prompt_version,
            reward_version=args.reward_version,
        )
        # Ensure shard filenames don't collide across workers in the same process.
        agent._worker_id = f"{os.getpid()}_{worker_idx}"

        main_llm = _DummyLLM(
            model=strategy_model_name,
            base_url=strategy_base_url,
        )
        resources: agl.NamedResources = {"main_llm": main_llm}
        return agent, resources

    # Run rollouts sequentially (no VERL / Ray).
    overall_soft: List[float] = []
    overall_hard: List[float] = []
    by_split_soft: Dict[str, List[float]] = {}
    by_split_hard: Dict[str, List[float]] = {}

    logger.info("Starting lightweight evaluation over %d samples...", len(val_dataset))
    start_ts = datetime.now().isoformat()
    print(f"[{start_ts}] Starting eval_no_verl over {len(val_dataset)} samples")

    queue: asyncio.Queue[Tuple[int, StrategyGenerationTask]] = asyncio.Queue()
    for idx, task in enumerate(val_dataset):
        queue.put_nowait((idx, task))

    stats_lock = asyncio.Lock()
    workers: List[Tuple[StrategyGenerationAgent, agl.NamedResources]] = [
        _make_worker(i) for i in range(concurrency)
    ]

    async def _process_one(
        *,
        idx: int,
        task: StrategyGenerationTask,
        agent: StrategyGenerationAgent,
        resources: agl.NamedResources,
    ) -> None:
        rollout_id = f"val-{idx:06d}"
        rollout = _SimpleRollout(
            rollout_id=rollout_id,
            attempt=_SimpleAttempt(attempt_id="0"),
            mode="val",
        )
        try:
            _ = await agent.rollout_async(task, resources, rollout)
        except Exception as e:  # noqa: BLE001
            logger.warning("Rollout %s failed (non-fatal): %s", rollout_id, e)
            return

        if not agent.validation_outputs:
            return
        entry = agent.validation_outputs[-1]
        reward = entry.get("reward", {}) or {}
        task_meta = entry.get("input", {}).get("task_meta", {}) or {}
        split = str(task_meta.get("validation_split", task.get("problem_type", "unknown")))

        soft = reward.get("correctness", None)
        hard = reward.get("hard_correct", None)
        async with stats_lock:
            if soft is not None:
                try:
                    s = float(soft)
                    overall_soft.append(s)
                    by_split_soft.setdefault(split, []).append(s)
                except Exception:  # noqa: BLE001
                    pass
            if hard is not None:
                try:
                    h = float(hard)
                    overall_hard.append(h)
                    by_split_hard.setdefault(split, []).append(h)
                except Exception:  # noqa: BLE001
                    pass

    async def _worker_loop(worker_idx: int) -> None:
        agent, resources = workers[worker_idx]
        while True:
            try:
                idx, task = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                await _process_one(idx=idx, task=task, agent=agent, resources=resources)
            finally:
                queue.task_done()

        # Flush this worker's buffered validation outputs to disk.
        try:
            saved = agent.save_validation_outputs(0)
            if saved:
                logger.info("[Worker %s] Saved validation shard: %s", agent.worker_id, saved)
        except Exception as e:  # noqa: BLE001
            logger.warning("[Worker %s] Failed to save shard: %s", agent.worker_id, e)

    await asyncio.gather(*[_worker_loop(i) for i in range(concurrency)])

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    print("\n" + "=" * 90)
    print("StrategyGenerationAgent — eval_no_verl summary")
    print("=" * 90)
    print(f"Samples(soft): {len(overall_soft)}")
    print(f"Samples(hard): {len(overall_hard)}")
    if overall_soft:
        print(f"Acc_soft(all): {_mean(overall_soft):.4f}")
    if overall_hard:
        print(f"Acc_hard(all): {_mean(overall_hard):.4f}")

    if by_split_soft or by_split_hard:
        print("\nBy validation split:")
        for split in sorted(set(by_split_soft.keys()) | set(by_split_hard.keys())):
            soft_vals = by_split_soft.get(split, [])
            hard_vals = by_split_hard.get(split, [])
            soft_part = f"acc_soft={_mean(soft_vals):.4f}" if soft_vals else "acc_soft=NA"
            hard_part = f"acc_hard={_mean(hard_vals):.4f}" if hard_vals else "acc_hard=NA"
            n_part = max(len(soft_vals), len(hard_vals))
            print(f"  - {split:20s} n={n_part:4d}  {soft_part}  {hard_part}")
    print("=" * 90 + "\n")


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    asyncio.run(_run_eval(args))


if __name__ == "__main__":
    main()

