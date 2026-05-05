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
        temperature: float = 0.0,
        max_tokens: int = 16384,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self._base_url = base_url
        self.api_key = api_key
        self.sampling_parameters: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            self.sampling_parameters["seed"] = int(seed)

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
        choices=["per_subtask_fixed", "global", "all"],
        default="per_subtask_fixed",
        help="Validation sampling mode: per_subtask_fixed (N samples per subtask JSON), "
        "global (global cap across all subtasks), or all (enumerate every example exactly once).",
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
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for both strategy and answer generation (0.0 = greedy/deterministic).",
    )
    parser.add_argument(
        "--llm-seed",
        type=int,
        default=None,
        help="Optional OpenAI-style request seed forwarded to vLLM for strategy and answer "
        "/chat/completions calls (per-request; not a server startup flag).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent async workers for evaluation. "
        "Each worker owns its own agent instance to avoid shared-state conflicts.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="MIST",
        choices=["few-shot", "MIST", "mist-inline", "habit", "0-shot", "habit-0-shot"],
        help=(
            "Evaluation mode:\n"
            "  few-shot     — ICL direct answer (no strategy generation)\n"
            "  MIST         — separate strategy model + answer model (default)\n"
            "  mist-inline  — train-free single-model MIST: answer model extracts\n"
            "                 a two-layer strategy then solves the problem (two calls,\n"
            "                 same endpoint; suitable for closed-source APIs)\n"
            "  habit        — like mist-inline but the answer call receives both the\n"
            "                 extracted strategy AND the original few-shot examples,\n"
            "                 combining abstract strategy with concrete demonstrations\n"
            "  0-shot       — no few-shot examples, direct answer only (infant baseline:\n"
            "                 no meta-learning ability assumed)\n"
            "  habit-0-shot — no few-shot examples; model first self-generates a two-layer\n"
            "                 strategy from the problem alone (adult baseline: internalized\n"
            "                 meta-inductive ability), then applies it to answer"
        ),
    )
    parser.add_argument(
        "--inline-strategy-prompt-version",
        type=str,
        default="mist_inline_strategy",
        help=(
            "Prompt TOML name under answer_generation/ for inline strategy extraction "
            "(mist-inline mode only). Default: 'mist_inline_strategy'."
        ),
    )
    parser.add_argument(
        "--skip-strategy-generation",
        action="store_true",
        default=False,
        help="Skip strategy generation and evaluate raw answer accuracy (no-strategy baseline). "
        "When set, the strategy model server is not called at all.",
    )
    parser.add_argument(
        "--strategy-no-think",
        action="store_true",
        default=False,
        help="Disable Qwen3 think mode for strategy /chat/completions (same as train --strategy-no-think).",
    )
    parser.add_argument(
        "--strategy-repetition-penalty",
        type=float,
        default=1.1,
        help="vLLM extra_body.repetition_penalty for strategy /chat/completions (default 1.1). "
        "Use 0 or negative to omit and use server defaults only (see train_strategy_generation).",
    )
    parser.add_argument(
        "--answer-no-think",
        action="store_true",
        default=False,
        help="Disable Qwen3 think mode for answer /chat/completions (same as train --answer-no-think).",
    )
    parser.add_argument(
        "--answer-max-tokens",
        type=int,
        default=None,
        help="Override max output tokens for answer /chat/completions. "
        "Defaults to the rollout LLM's max_tokens (16384). "
        "For CoT (think-mode) it is recommended to pass 32768 to avoid truncation of <think> blocks.",
    )
    parser.add_argument(
        "--answer-request-retries",
        type=int,
        default=3,
        help="Retries per endpoint for answer /chat/completions (see train_strategy_generation).",
    )
    parser.add_argument(
        "--answer-retry-delay-sec",
        type=float,
        default=1.0,
        help="Delay between answer retries (seconds).",
    )
    parser.add_argument(
        "--no-answer-fallback-rollout",
        action="store_true",
        default=False,
        help="Do not fall back to the strategy (rollout) model if the answer server fails.",
    )
    parser.add_argument(
        "--num-samples-per-problem",
        type=int,
        default=3,
        help="Number of independent rollouts per problem for pass@k evaluation. "
        "When >1 and temperature==0, auto-switches temperature to 0.7 and uses "
        "incrementing seeds (base_seed, base_seed+1, ...) per attempt.",
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
        elif args.val_sampling_mode == "all":
            # 全量枚举：每道题恰好作为一次 problem_ex（samples_per_subtask=0 为哨兵值）
            vds = _create_strategy_generation_dataset_per_subtask(
                vd,
                fewshot_min=args.fewshot_min,
                fewshot_max=args.fewshot_max,
                samples_per_subtask=0,
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

    strategy_repetition_penalty_effective: float | None = (
        None if args.strategy_repetition_penalty <= 0 else float(args.strategy_repetition_penalty)
    )

    # Auto-switch temperature when running multiple samples per problem.
    num_samples = max(1, int(args.num_samples_per_problem or 1))
    effective_temperature = args.temperature
    if num_samples > 1 and effective_temperature == 0.0:
        logger.warning(
            "num_samples_per_problem=%d with temperature=0; auto-switching temperature to 0.7 "
            "so that each attempt produces a distinct output. Pass --temperature explicitly to override.",
            num_samples,
        )
        effective_temperature = 0.7

    base_seed = args.llm_seed if args.llm_seed is not None else 42

    # Build per-worker agent instances to avoid shared-state conflicts under concurrency.
    # Each worker writes its own validation shard: validation_step0_worker{worker_id}.json
    concurrency = max(1, int(args.concurrency or 1))

    # Resolve mode-derived settings.
    mode = getattr(args, "mode", "MIST")
    _answer_model_name = args.answer_model_name or args.answer_model_path
    if mode in ("few-shot", "0-shot") or args.skip_strategy_generation:
        # No strategy generation: direct answer only.
        # 0-shot differs from few-shot only in that it uses zero examples (fewshot-min/max=0)
        # and a zero-shot answer prompt; the agent-level logic is identical.
        _skip_strategy = True
        _inline_strat_version = ""
        strategy_base_url = args.answer_model_base_url
        strategy_model_name = _answer_model_name
    elif mode in ("mist-inline", "habit", "habit-0-shot"):
        # Inline strategy: same endpoint for both strategy extraction and answer.
        # habit-0-shot uses a problem-only strategy prompt (habit_0shot_strategy.toml)
        # and zero few-shot examples; the agent routing is otherwise identical.
        _skip_strategy = False
        _inline_strat_version = args.inline_strategy_prompt_version
        strategy_base_url = args.answer_model_base_url
        strategy_model_name = _answer_model_name
    else:  # MIST (default)
        _skip_strategy = args.skip_strategy_generation
        _inline_strat_version = ""
        strategy_base_url = args.strategy_model_base_url or args.answer_model_base_url
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
            answer_model_name=_answer_model_name,
            use_strategy_for_answer=True,
            skip_strategy_generation=_skip_strategy,
            strategy_prompt_version=args.strategy_prompt_version,
            answer_prompt_version=args.answer_prompt_version,
            inline_strategy_prompt_version=_inline_strat_version,
            reward_version=args.reward_version,
            strategy_no_think=args.strategy_no_think,
            strategy_repetition_penalty=strategy_repetition_penalty_effective,
            answer_no_think=args.answer_no_think,
            answer_max_tokens=args.answer_max_tokens,
            answer_request_retries=args.answer_request_retries,
            answer_retry_delay_sec=args.answer_retry_delay_sec,
            answer_fallback_rollout_on_failure=not args.no_answer_fallback_rollout,
        )
        # Ensure shard filenames don't collide across workers in the same process.
        agent._worker_id = f"{os.getpid()}_{worker_idx}"

        main_llm = _DummyLLM(
            model=strategy_model_name,
            base_url=strategy_base_url,
            temperature=effective_temperature,
            seed=base_seed,
        )
        resources: agl.NamedResources = {"main_llm": main_llm}
        return agent, resources

    # Run rollouts sequentially (no VERL / Ray).
    overall_soft: List[float] = []
    overall_hard: List[float] = []
    by_split_soft: Dict[str, List[float]] = {}
    by_split_hard: Dict[str, List[float]] = {}

    # Per-problem tracking for pass@k: problem_idx -> list of (soft, hard) per attempt.
    per_problem_soft: Dict[int, List[float]] = {}
    per_problem_hard: Dict[int, List[float]] = {}
    by_split_per_problem_soft: Dict[str, Dict[int, List[float]]] = {}
    by_split_per_problem_hard: Dict[str, Dict[int, List[float]]] = {}

    logger.info(
        "Starting lightweight evaluation over %d samples × %d attempt(s) = %d total rollouts...",
        len(val_dataset),
        num_samples,
        len(val_dataset) * num_samples,
    )
    start_ts = datetime.now().isoformat()
    print(
        f"[{start_ts}] Starting eval_no_verl over {len(val_dataset)} problems "
        f"× {num_samples} attempt(s) (pass@k eval)"
    )

    # Queue items: (problem_idx, attempt_idx, task)
    queue: asyncio.Queue[Tuple[int, int, StrategyGenerationTask]] = asyncio.Queue()
    for problem_idx, task in enumerate(val_dataset):
        for attempt_idx in range(num_samples):
            queue.put_nowait((problem_idx, attempt_idx, task))

    stats_lock = asyncio.Lock()
    workers: List[Tuple[StrategyGenerationAgent, agl.NamedResources]] = [
        _make_worker(i) for i in range(concurrency)
    ]

    async def _process_one(
        *,
        problem_idx: int,
        attempt_idx: int,
        task: StrategyGenerationTask,
        agent: StrategyGenerationAgent,
        resources: agl.NamedResources,
    ) -> None:
        rollout_id = f"val-{problem_idx:06d}-a{attempt_idx}"
        rollout = _SimpleRollout(
            rollout_id=rollout_id,
            attempt=_SimpleAttempt(attempt_id=str(attempt_idx)),
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
                    per_problem_soft.setdefault(problem_idx, []).append(s)
                    by_split_per_problem_soft.setdefault(split, {}).setdefault(problem_idx, []).append(s)
                except Exception:  # noqa: BLE001
                    pass
            if hard is not None:
                try:
                    h = float(hard)
                    overall_hard.append(h)
                    by_split_hard.setdefault(split, []).append(h)
                    per_problem_hard.setdefault(problem_idx, []).append(h)
                    by_split_per_problem_hard.setdefault(split, {}).setdefault(problem_idx, []).append(h)
                except Exception:  # noqa: BLE001
                    pass

    async def _worker_loop(worker_idx: int) -> None:
        agent, resources = workers[worker_idx]
        while True:
            try:
                problem_idx, attempt_idx, task = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            # Give this attempt a unique seed so outputs differ across attempts.
            resources["main_llm"].sampling_parameters["seed"] = base_seed + attempt_idx
            try:
                await _process_one(
                    problem_idx=problem_idx,
                    attempt_idx=attempt_idx,
                    task=task,
                    agent=agent,
                    resources=resources,
                )
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

    def _pass_at_k(per_problem: Dict[int, List[float]], k: int, threshold: float = 0.5) -> Optional[float]:
        """Fraction of problems where at least 1 of the first k attempts passes threshold."""
        if not per_problem:
            return None
        results = [
            any(v >= threshold for v in scores[:k])
            for scores in per_problem.values()
            if scores
        ]
        return sum(results) / len(results) if results else None

    def _pass_at_k_split(
        per_problem: Dict[int, List[float]], k: int, threshold: float = 0.5
    ) -> Optional[str]:
        v = _pass_at_k(per_problem, k, threshold)
        return f"{v:.4f}" if v is not None else "NA"

    print("\n" + "=" * 90)
    print("StrategyGenerationAgent — eval_no_verl summary")
    print("=" * 90)
    print(f"Problems:      {len(val_dataset)}")
    print(f"Attempts/prob: {num_samples}  (temperature={effective_temperature:.2f}, base_seed={base_seed})")
    print(f"Samples(soft): {len(overall_soft)}")
    print(f"Samples(hard): {len(overall_hard)}")
    if overall_soft:
        print(f"Acc_soft(all): {_mean(overall_soft):.4f}")
    if overall_hard:
        print(f"Acc_hard(all): {_mean(overall_hard):.4f}")

    # pass@k overall
    k_values = [k for k in (1, 2, 3) if k <= num_samples]
    if k_values and (per_problem_hard or per_problem_soft):
        print()
        for k in k_values:
            ph = _pass_at_k(per_problem_hard, k)
            ps = _pass_at_k(per_problem_soft, k)
            hard_str = f"{ph:.4f}" if ph is not None else "NA"
            soft_str = f"{ps:.4f}" if ps is not None else "NA"
            print(f"pass@{k}(hard): {hard_str}   pass@{k}(soft): {soft_str}")

    if by_split_soft or by_split_hard:
        print("\nBy validation split:")
        all_splits = sorted(set(by_split_soft.keys()) | set(by_split_hard.keys()))
        for split in all_splits:
            soft_vals = by_split_soft.get(split, [])
            hard_vals = by_split_hard.get(split, [])
            soft_part = f"acc_soft={_mean(soft_vals):.4f}" if soft_vals else "acc_soft=NA"
            hard_part = f"acc_hard={_mean(hard_vals):.4f}" if hard_vals else "acc_hard=NA"
            n_part = max(len(soft_vals), len(hard_vals))
            pass_parts = []
            for k in k_values:
                ph_str = _pass_at_k_split(by_split_per_problem_hard.get(split, {}), k)
                ps_str = _pass_at_k_split(by_split_per_problem_soft.get(split, {}), k)
                pass_parts.append(f"p@{k}(h)={ph_str} p@{k}(s)={ps_str}")
            pass_str = "  " + "  ".join(pass_parts) if pass_parts else ""
            print(f"  - {split:20s} n={n_part:4d}  {soft_part}  {hard_part}{pass_str}")
    print("=" * 90 + "\n")


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    asyncio.run(_run_eval(args))


if __name__ == "__main__":
    main()

