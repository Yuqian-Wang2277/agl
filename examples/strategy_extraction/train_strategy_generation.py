# Copyright (c) Microsoft. All rights reserved.

"""Training script for strategy generation — trains strategy extraction using answer quality as reward.

This script trains a model to generate better problem-solving strategies by using
answer correctness as the reward signal. Only strategy-generation tokens receive
gradient updates; the answer-generation step is un-traced and serves purely as
a reward evaluator.

Example usage:

```bash
python train_strategy_generation.py \\
    --data-base-path /home/test/test16/chenlu/projects/LLMReflection/data/ \\
    --train-subdir train_20k \\
    --model-path /home/test/test16/chenlu/model/Qwen3-4B
```
"""

import argparse
import glob
import json
import logging
import os
import re
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, cast

import agentlightning as agl

# Handle both direct execution and module import
try:
    from .config import StrategyConfig, get_verl_config
    from .prompt import list_versions as list_prompt_versions
    from .reward import list_reward_versions
    from .strategy_generation_agent import StrategyGenerationAgent, StrategyGenerationTask
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from examples.strategy_extraction.config import StrategyConfig, get_verl_config
    from examples.strategy_extraction.prompt import list_versions as list_prompt_versions
    from examples.strategy_extraction.reward import list_reward_versions
    from examples.strategy_extraction.strategy_generation_agent import (
        StrategyGenerationAgent,
        StrategyGenerationTask,
    )

logger = logging.getLogger(__name__)


# ---- Dataset creation (reuses strategy_application data format) ---- #

def _create_strategy_generation_dataset(
    data_dir: str,
    fewshot_min: int,
    fewshot_max: int,
    num_samples: int,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Create a dataset that contains few-shot examples, a problem, and its ground truth.

    This is essentially the same_domain variant of the strategy_application
    dataset, but produced here to avoid an import dependency on that package.
    """
    import random

    from examples.strategy_extraction.data_loader import load_problem_types, sample_fewshot_examples

    random.seed(seed)
    problem_types_data = load_problem_types(data_dir)
    if not problem_types_data:
        raise ValueError(f"No problem types found in {data_dir}")

    problem_type_names = list(problem_types_data.keys())
    exclude_indices: dict[str, set[int]] = {pt: set() for pt in problem_type_names}
    sample_counts: dict[str, int] = {pt: 0 for pt in problem_type_names}
    dataset: list[dict[str, object]] = []

    logger.info(
        f"Creating strategy-generation dataset: {num_samples} samples, "
        f"few-shot [{fewshot_min}, {fewshot_max}]"
    )

    consecutive_failures = 0
    max_consecutive_failures = 1000

    while len(dataset) < num_samples:
        # Balanced sampling
        min_count = min(sample_counts.values())
        candidates = [pt for pt, c in sample_counts.items() if c == min_count]
        problem_type = random.choice(candidates)

        n_shots = random.randint(fewshot_min, fewshot_max)
        examples_pool = problem_types_data[problem_type]

        if n_shots > len(examples_pool):
            continue

        try:
            fewshot, updated = sample_fewshot_examples(
                examples_pool, n_shots, exclude_indices[problem_type]
            )
            exclude_indices[problem_type] = updated

            # Validate few-shot examples
            valid_fewshot = []
            for ex in fewshot:
                ex_input = ex.get("input", "")
                ex_target = ex.get("target", [])
                if ex_input and str(ex_input).strip():
                    if isinstance(ex_target, list):
                        if ex_target and ex_target[0]:
                            valid_fewshot.append(ex)
                    elif ex_target:
                        valid_fewshot.append(ex)
            if not valid_fewshot:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    break
                continue

            # Pick a problem (not from few-shot)
            remaining = [
                ex for i, ex in enumerate(examples_pool)
                if i not in exclude_indices[problem_type]
            ]
            if not remaining:
                exclude_indices[problem_type] = set()
                remaining = examples_pool

            problem_ex = random.choice(remaining)
            problem_text = problem_ex.get("input", "")
            target_val = problem_ex.get("target", [])
            if isinstance(target_val, list):
                ground_truth = str(target_val[0]) if target_val else ""
            else:
                ground_truth = str(target_val)

            if not problem_text or not str(problem_text).strip():
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    break
                continue
            if not ground_truth or not ground_truth.strip():
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    break
                continue

            dataset.append({
                "problem_type": problem_type,
                "examples": valid_fewshot,
                "num_shots": len(valid_fewshot),
                "problem": problem_text,
                "ground_truth": ground_truth,
                "source_problem_type": None,
            })
            sample_counts[problem_type] += 1
            consecutive_failures = 0

        except ValueError as e:
            logger.error(f"Sampling error for {problem_type}: {e}")
            continue

    logger.info(f"Dataset created: {len(dataset)} samples, distribution: {sample_counts}")
    return dataset


# ---- Merge remaining validation files (same as strategy_application) ---- #

def _merge_remaining_validation_files(validation_output_dir: str) -> None:
    """Merge any un-merged per-worker validation files."""
    if not os.path.isdir(validation_output_dir):
        return

    worker_files = glob.glob(os.path.join(validation_output_dir, "validation_step*_worker*.json"))
    step_pattern = re.compile(r"validation_step(\d+)_worker")
    steps: set[int] = set()
    for wf in worker_files:
        m = step_pattern.search(os.path.basename(wf))
        if m:
            steps.add(int(m.group(1)))

    for step in sorted(steps):
        merged_path = os.path.join(validation_output_dir, f"validation_global_step{step}.json")
        if os.path.exists(merged_path):
            continue

        pattern = os.path.join(validation_output_dir, f"validation_step{step}_worker*.json")
        wfiles = sorted(glob.glob(pattern))
        all_outputs: list[dict[str, object]] = []
        for swf in wfiles:
            try:
                with open(swf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_outputs.extend(data)
            except Exception as e:
                logger.warning(f"Failed to load {swf}: {e}")
        if all_outputs:
            try:
                with open(merged_path, "w", encoding="utf-8") as f:
                    json.dump(all_outputs, f, ensure_ascii=False, indent=2)
                logger.info(f"Merged {len(all_outputs)} outputs -> {merged_path}")
            except Exception as e:
                logger.error(f"Merge failed for {merged_path}: {e}")


def find_free_port(start_port: int = 4747, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                logger.info(f"Found available port: {port}")
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port in range {start_port}-{start_port + max_attempts}")


# ------------------------------------------------------------------ #
#  train()
# ------------------------------------------------------------------ #

def train(
    *,
    data_base_path: str,
    train_subdir: str,
    val_subdir: str,
    val_subdirs: list[str] | None = None,
    model_path: str,
    fewshot_min: int,
    fewshot_max: int,
    num_train_samples: int,
    num_val_samples: int,
    n_runners: int,
    lora: bool,
    lora_rank: int,
    external_store_address: str,
    debug: bool,
    wandb_project: str,
    wandb_experiment: str,
    checkpoint_dir: str,
    save_full_output: bool,
    resume_from_checkpoint: bool,
    resume_from_path: str | None,
    format_weight: float,
    correctness_weight: float,
    numeric_tolerance: float,
    f1_threshold: float,
    strategy_prompt_version: str,
    answer_prompt_version: str,
    reward_version: str,
) -> None:
    """Train strategy generation model."""
    original_checkpoint_dir = os.path.abspath(checkpoint_dir)

    log_level = "DEBUG" if debug else "INFO"
    agl.setup_logging(log_level)

    print("=" * 80)
    print("Strategy Generation Training — train strategy with answer-quality reward")
    print("=" * 80)
    print(f"Starting at {datetime.now().isoformat()}")
    print(f"Checkpoint directory: {original_checkpoint_dir}")

    logger.info("=" * 80)
    logger.info("Strategy Generation Training")
    logger.info("=" * 80)

    # Ray cleanup
    try:
        import ray  # type: ignore
        if ray.is_initialized():  # type: ignore
            ray.shutdown()  # type: ignore
        os.system("ray stop > /dev/null 2>&1")
    except Exception as e:
        logger.debug(f"Ray cleanup: {e}")

    # Port
    if not external_store_address:
        free_port = find_free_port()
        os.environ["AGL_SERVER_PORT"] = str(free_port)
        logger.info(f"Using port: {free_port}")

    # Data paths
    train_dir = os.path.join(data_base_path, train_subdir)
    validation_subdirs = val_subdirs if val_subdirs else [val_subdir]
    logger.info(f"Validation sets: {validation_subdirs}")

    # Experiment ID
    experiment_id: str | None = None
    if not resume_from_checkpoint and resume_from_path is None:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(original_checkpoint_dir, experiment_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"New experiment: {experiment_id}")
    else:
        if resume_from_path:
            parts = os.path.normpath(os.path.abspath(resume_from_path)).split(os.sep)
            for part in parts:
                if len(part) == 15 and part[8] == "_" and part[:8].isdigit() and part[9:].isdigit():
                    experiment_id = part
                    checkpoint_dir = os.path.join(original_checkpoint_dir, experiment_id)
                    break
        if experiment_id is None:
            experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join(original_checkpoint_dir, experiment_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Resuming experiment: {experiment_id}")

    # File logging
    log_dir = os.path.join(checkpoint_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{experiment_id}.log")
    agl.setup_logging(
        log_level,
        files={
            "agentlightning": log_file,
            "examples.strategy_extraction": log_file,
        },
    )
    logger.info(f"Log file: {log_file}")
    print(f"Log file: {log_file}")

    # Save config
    if not resume_from_checkpoint and resume_from_path is None:
        cfg_dir = os.path.join(checkpoint_dir, "training_configs")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_file = os.path.join(cfg_dir, f"config_{experiment_id}.json")
        try:
            with open(cfg_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "experiment_id": experiment_id,
                        "timestamp": datetime.now().isoformat(),
                        "training_mode": "strategy_generation",
                        "data_base_path": data_base_path,
                        "train_subdir": train_subdir,
                        "val_subdirs": validation_subdirs,
                        "model_path": model_path,
                        "fewshot_min": fewshot_min,
                        "fewshot_max": fewshot_max,
                        "num_train_samples": num_train_samples,
                        "num_val_samples": num_val_samples,
                        "n_runners": n_runners,
                        "lora": lora,
                        "lora_rank": lora_rank,
                        "format_weight": format_weight,
                        "correctness_weight": correctness_weight,
                        "numeric_tolerance": numeric_tolerance,
                        "f1_threshold": f1_threshold,
                        "strategy_prompt_version": strategy_prompt_version,
                        "answer_prompt_version": answer_prompt_version,
                        "reward_version": reward_version,
                        "wandb_project": wandb_project,
                        "wandb_experiment": wandb_experiment,
                        "checkpoint_dir": checkpoint_dir,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(f"Config saved: {cfg_file}")
        except Exception as e:
            logger.warning(f"Failed to save config: {e}")

    # ---- Load datasets ---- #
    logger.info("Loading training dataset...")
    print(f"[{datetime.now().isoformat()}] Loading training dataset from: {train_dir}")
    train_dataset = _create_strategy_generation_dataset(
        train_dir, fewshot_min, fewshot_max, num_train_samples, seed=42,
    )
    print(f"[{datetime.now().isoformat()}] Training dataset: {len(train_dataset)} samples")

    logger.info("Loading validation datasets...")
    import itertools

    val_datasets = []
    for i, vs in enumerate(validation_subdirs):
        vd = os.path.join(data_base_path, vs)
        logger.info(f"Loading val set {i + 1}/{len(validation_subdirs)}: {vs}")
        vds = _create_strategy_generation_dataset(
            vd, fewshot_min, fewshot_max, num_val_samples, seed=43 + i,
        )
        val_datasets.append(vds)
        logger.info(f"  {vs}: {len(vds)} samples")
    val_dataset = list(itertools.chain.from_iterable(val_datasets))

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Validate
    for i, s in enumerate(train_dataset):
        if not s.get("examples"):
            raise ValueError(f"Train sample {i} has no examples")
        if not s.get("problem"):
            raise ValueError(f"Train sample {i} has no problem")
        if not s.get("ground_truth"):
            raise ValueError(f"Train sample {i} has no ground_truth")
    logger.info("✓ Dataset validation passed")

    train_dataset = cast(agl.Dataset[StrategyGenerationTask], train_dataset)
    val_dataset = cast(agl.Dataset[StrategyGenerationTask], val_dataset)

    # ---- VERL config ---- #
    config = get_verl_config(
        model_path,
        lora=lora,
        lora_rank=lora_rank,
        resume_from_checkpoint=resume_from_checkpoint,
        resume_from_path=resume_from_path,
        checkpoint_dir=checkpoint_dir,
    )
    config["trainer"]["project_name"] = wandb_project
    config["trainer"]["experiment_name"] = wandb_experiment
    config["trainer"]["default_local_dir"] = checkpoint_dir

    algorithm = agl.VERL(config)

    # Store
    store: Optional[agl.LightningStore] = None
    if external_store_address:
        logger.info(f"External store: {external_store_address}")
        store = agl.LightningStoreClient(external_store_address)
    else:
        logger.info("In-memory store")

    # Agent
    rollout_traces_dir = os.path.join(checkpoint_dir, "rollout_traces")
    validation_output_dir = os.path.join(checkpoint_dir, "validation_outputs")
    test_freq = config.get("trainer", {}).get("test_freq", 50)
    agent = StrategyGenerationAgent(
        save_full_output=save_full_output,
        rollout_traces_dir=rollout_traces_dir,
        validation_output_dir=validation_output_dir,
        experiment_id=experiment_id,
        test_freq=test_freq,
        format_weight=format_weight,
        correctness_weight=correctness_weight,
        numeric_tolerance=numeric_tolerance,
        f1_threshold=f1_threshold,
        strategy_prompt_version=strategy_prompt_version,
        answer_prompt_version=answer_prompt_version,
        reward_version=reward_version,
    )

    # Trainer
    trainer = agl.Trainer(algorithm=algorithm, n_runners=n_runners, store=store)

    logger.info("Starting training...")
    print(f"[{datetime.now().isoformat()}] Starting training...")
    print("=" * 80)

    try:
        trainer.fit(agent, train_dataset=train_dataset, val_dataset=val_dataset)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    _merge_remaining_validation_files(validation_output_dir)

    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train strategy generation (strategy tokens trained, answer for reward only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data-base-path", type=str, default=StrategyConfig.data_base_path)
    parser.add_argument("--train-subdir", type=str, default=StrategyConfig.train_subdir)
    parser.add_argument("--val-subdir", type=str, default=StrategyConfig.val_subdir)
    parser.add_argument("--val-subdirs", type=str, nargs="+", default=None)

    # Model
    parser.add_argument("--model-path", type=str, default=StrategyConfig.model_path)

    # Few-shot
    parser.add_argument("--fewshot-min", type=int, default=StrategyConfig.fewshot_min)
    parser.add_argument("--fewshot-max", type=int, default=StrategyConfig.fewshot_max)

    # Dataset size
    parser.add_argument("--num-train-samples", type=int, default=StrategyConfig.num_train_samples)
    parser.add_argument("--num-val-samples", type=int, default=StrategyConfig.num_val_samples)

    # Training
    parser.add_argument("--n-runners", type=int, default=StrategyConfig.n_runners)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=StrategyConfig.lora_rank)

    # WandB
    parser.add_argument("--wandb-project", type=str, default="StrategyGeneration")
    parser.add_argument("--wandb-experiment", type=str, default="strategy_gen")

    # Checkpoint
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints_strategy_gen")
    parser.add_argument("--resume-from-checkpoint", action="store_true", default=False)
    parser.add_argument("--resume-from-path", type=str, default=None)

    # Reward weights
    parser.add_argument("--format-weight", type=float, default=0.2, help="Weight for strategy format reward")
    parser.add_argument("--correctness-weight", type=float, default=0.8, help="Weight for answer correctness reward")
    parser.add_argument("--numeric-tolerance", type=float, default=0.02, help="Numeric tolerance for answer matching")
    parser.add_argument("--f1-threshold", type=float, default=0.5, help="F1 threshold for partial answer match")

    # Prompt / reward versions (see prompt/ and reward/ packages)
    parser.add_argument(
        "--strategy-prompt-version", type=str,
        default=StrategyConfig.strategy_prompt_version,
        help=f"Strategy-generation prompt version (available: {', '.join(list_prompt_versions('strategy_generation'))})",
    )
    parser.add_argument(
        "--answer-prompt-version", type=str,
        default=StrategyConfig.answer_prompt_version,
        help=f"Answer-generation prompt version (available: {', '.join(list_prompt_versions('answer_generation'))})",
    )
    parser.add_argument(
        "--reward-version", type=str,
        default=StrategyConfig.reward_version,
        help=f"Reward version key (available: {', '.join(list_reward_versions())})",
    )

    # Infrastructure
    parser.add_argument("--external-store-address", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-full-output", action="store_true", default=True)

    args = parser.parse_args()

    if args.external_store_address:
        from agentlightning.env_var import LightningEnvVar, resolve_bool_env_var

        if resolve_bool_env_var(LightningEnvVar.AGL_MANAGED_STORE, fallback=True):
            raise ValueError("Set AGL_MANAGED_STORE=0 when using an external store.")

    train(
        data_base_path=args.data_base_path,
        train_subdir=args.train_subdir,
        val_subdir=args.val_subdir,
        val_subdirs=args.val_subdirs if args.val_subdirs else (
            StrategyConfig.val_subdirs if hasattr(StrategyConfig, "val_subdirs") else None
        ),
        model_path=args.model_path,
        fewshot_min=args.fewshot_min,
        fewshot_max=args.fewshot_max,
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
        n_runners=args.n_runners,
        lora=args.lora,
        lora_rank=args.lora_rank,
        external_store_address=args.external_store_address,
        debug=args.debug,
        wandb_project=args.wandb_project,
        wandb_experiment=args.wandb_experiment,
        checkpoint_dir=args.checkpoint_dir,
        save_full_output=args.save_full_output,
        resume_from_checkpoint=args.resume_from_checkpoint,
        resume_from_path=args.resume_from_path,
        format_weight=args.format_weight,
        correctness_weight=args.correctness_weight,
        numeric_tolerance=args.numeric_tolerance,
        f1_threshold=args.f1_threshold,
        strategy_prompt_version=args.strategy_prompt_version,
        answer_prompt_version=args.answer_prompt_version,
        reward_version=args.reward_version,
    )


if __name__ == "__main__":
    main()

