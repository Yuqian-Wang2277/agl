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
                "ground_truths": [ground_truth],
                "task_meta": {"problem_type": problem_type},
                "source_problem_type": None,
            })
            sample_counts[problem_type] += 1
            consecutive_failures = 0

        except ValueError as e:
            logger.error(f"Sampling error for {problem_type}: {e}")
            continue

    logger.info(f"Dataset created: {len(dataset)} samples, distribution: {sample_counts}")
    return dataset


def _create_strategy_generation_dataset_per_subtask(
    data_dir: str,
    fewshot_min: int,
    fewshot_max: int,
    samples_per_subtask: int,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Create validation dataset with fixed quota per subtask JSON file.

    Sampling rule:
    - Iterate all ``<problem_type>/*.json`` files in deterministic order.
    - Sample exactly ``samples_per_subtask`` records per subtask file.
    - Randomness is controlled by a single seed for full reproducibility.
    """
    import random

    rng = random.Random(seed)
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Validation directory not found: {data_dir}")

    dataset: list[dict[str, object]] = []
    subtask_counter = 0
    split_name = data_path.name

    for problem_type_dir in sorted([p for p in data_path.iterdir() if p.is_dir()], key=lambda p: p.name):
        problem_type = problem_type_dir.name
        for subtask_file in sorted(problem_type_dir.glob("*.json"), key=lambda p: p.name):
            try:
                raw = json.loads(subtask_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Skip invalid json file {subtask_file}: {e}")
                continue

            examples_raw = raw.get("examples", [])
            if not isinstance(examples_raw, list):
                logger.warning(f"Skip malformed examples field: {subtask_file}")
                continue

            examples_pool: list[dict[str, object]] = []
            for ex in examples_raw:
                if not isinstance(ex, dict):
                    continue
                ex_input = ex.get("input", "")
                ex_target = ex.get("target", [])
                if not ex_input or not str(ex_input).strip():
                    continue
                if isinstance(ex_target, list):
                    if not ex_target:
                        continue
                    if not str(ex_target[0]).strip():
                        continue
                elif not str(ex_target).strip():
                    continue
                examples_pool.append(ex)

            if len(examples_pool) < 2:
                logger.warning(f"Skip tiny subtask (<2 valid examples): {subtask_file}")
                continue

            subtask_name = subtask_file.stem
            subtask_id = f"{problem_type}/{subtask_name}"

            for _ in range(samples_per_subtask):
                # Need at least one problem sample in addition to few-shot examples.
                effective_fewshot_max = min(fewshot_max, len(examples_pool) - 1)
                if effective_fewshot_max < fewshot_min:
                    n_shots = max(1, effective_fewshot_max)
                else:
                    n_shots = rng.randint(fewshot_min, effective_fewshot_max)

                picked = rng.sample(examples_pool, n_shots + 1)
                fewshot = picked[:-1]
                problem_ex = picked[-1]

                problem_text = str(problem_ex.get("input", "") or "")
                target_val = problem_ex.get("target", [])
                if isinstance(target_val, list):
                    ground_truth = str(target_val[0]) if target_val else ""
                else:
                    ground_truth = str(target_val)
                if not problem_text.strip() or not ground_truth.strip():
                    continue

                dataset.append(
                    {
                        "problem_type": problem_type,
                        "examples": fewshot,
                        "num_shots": len(fewshot),
                        "problem": problem_text,
                        "ground_truth": ground_truth,
                        "ground_truths": [ground_truth],
                        # Used by VERL daemon for val/{data_source}/* WandB metrics (per validation split).
                        "data_source": split_name,
                        "task_meta": {
                            "problem_type": problem_type,
                            "subtask": subtask_name,
                            "subtask_id": subtask_id,
                            "validation_split": split_name,
                        },
                        "source_problem_type": subtask_id,
                    }
                )
            subtask_counter += 1

    logger.info(
        "Per-subtask validation dataset created: %d samples, %d subtasks, %d per subtask",
        len(dataset),
        subtask_counter,
        samples_per_subtask,
    )
    return dataset


def _load_strategy_dataset_from_file(path: str) -> list[dict[str, object]]:
    """Load pre-built dataset from JSON/JSONL file."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    records: list[dict[str, object]] = []
    if dataset_path.suffix.lower() == ".jsonl":
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
    else:
        loaded = json.loads(dataset_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            records = [item for item in loaded if isinstance(item, dict)]
        else:
            raise ValueError(f"JSON dataset must be a list: {dataset_path}")

    for item in records:
        ground_truth = str(item.get("ground_truth", "") or "")
        if "ground_truths" not in item:
            item["ground_truths"] = [ground_truth] if ground_truth else []
        if "task_meta" not in item:
            item["task_meta"] = {"problem_type": item.get("problem_type", "unknown")}
    return records


# ---- Merge remaining validation files (same as strategy_application) ---- #

def _merge_remaining_validation_files(validation_output_dir: str) -> None:
    """Merge any un-merged per-worker validation files."""
    if not os.path.isdir(validation_output_dir):
        return

    # NOTE:
    # `StrategyGenerationAgent` nests worker files under
    #   validation_output_dir / experiment_id / validation_step*_worker*.json
    # so we must scan subdirectories as well.
    worker_files = glob.glob(
        os.path.join(validation_output_dir, "**", "validation_step*_worker*.json"),
        recursive=True,
    )
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

        # Same reason as above: worker shards may live under an experiment_id subdirectory.
        pattern = os.path.join(validation_output_dir, "**", f"validation_step{step}_worker*.json")
        wfiles = sorted(glob.glob(pattern, recursive=True))
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


def _summarize_validation_outputs(saved_path: str) -> None:
    """Print baseline accuracy summary from saved validation outputs."""
    try:
        with open(saved_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load validation outputs for summary: {e}")
        return

    if not isinstance(rows, list) or not rows:
        logger.warning(f"Validation output file is empty: {saved_path}")
        return

    overall_soft_scores: list[float] = []
    overall_hard_scores: list[float] = []
    by_split_soft: dict[str, list[float]] = {}
    by_split_hard: dict[str, list[float]] = {}
    by_source_type_soft: dict[str, list[float]] = {}
    by_source_type_hard: dict[str, list[float]] = {}

    for item in rows:
        if not isinstance(item, dict):
            continue
        reward = item.get("reward", {})
        if not isinstance(reward, dict):
            continue
        task_meta = item.get("input", {}).get("task_meta", {})
        split = "unknown"
        if isinstance(task_meta, dict):
            split = str(task_meta.get("validation_split", "unknown"))

        source_type = item.get("source_problem_type", None)
        source_key = str(source_type).split("/")[0] if source_type is not None else None

        correctness = reward.get("correctness", None)
        if correctness is not None:
            try:
                soft_score = float(correctness)
                overall_soft_scores.append(soft_score)
                by_split_soft.setdefault(split, []).append(soft_score)
                if source_key is not None:
                    by_source_type_soft.setdefault(source_key, []).append(soft_score)
            except Exception:
                pass

        hard_correct = reward.get("hard_correct", None)
        if hard_correct is not None:
            try:
                hard_score = float(hard_correct)
                overall_hard_scores.append(hard_score)
                by_split_hard.setdefault(split, []).append(hard_score)
                if source_key is not None:
                    by_source_type_hard.setdefault(source_key, []).append(hard_score)
            except Exception:
                pass

    if not overall_soft_scores and not overall_hard_scores:
        logger.warning(f"No valid correctness/hard_correct values found in {saved_path}")
        return

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    print("\n" + "=" * 90)
    print("No-Strategy Baseline (v3 correctness) — Validation Summary")
    print("=" * 90)
    print(f"File         : {saved_path}")
    print(f"Samples(soft): {len(overall_soft_scores)}")
    print(f"Samples(hard): {len(overall_hard_scores)}")
    if overall_soft_scores:
        print(f"Acc_soft(all): {_mean(overall_soft_scores):.4f}")
    if overall_hard_scores:
        print(f"Acc_hard(all): {_mean(overall_hard_scores):.4f}")

    if by_split_soft or by_split_hard:
        print("\nBy validation split:")
        for split in sorted(set(by_split_soft.keys()) | set(by_split_hard.keys())):
            soft_vals = by_split_soft.get(split, [])
            hard_vals = by_split_hard.get(split, [])
            soft_part = f"acc_soft={_mean(soft_vals):.4f}" if soft_vals else "acc_soft=NA"
            hard_part = f"acc_hard={_mean(hard_vals):.4f}" if hard_vals else "acc_hard=NA"
            n_part = max(len(soft_vals), len(hard_vals))
            print(f"  - {split:20s} n={n_part:4d}  {soft_part}  {hard_part}")
    elif by_source_type_soft or by_source_type_hard:
        print("\nBy source problem type:")
        for key in sorted(set(by_source_type_soft.keys()) | set(by_source_type_hard.keys())):
            soft_vals = by_source_type_soft.get(key, [])
            hard_vals = by_source_type_hard.get(key, [])
            soft_part = f"acc_soft={_mean(soft_vals):.4f}" if soft_vals else "acc_soft=NA"
            hard_part = f"acc_hard={_mean(hard_vals):.4f}" if hard_vals else "acc_hard=NA"
            n_part = max(len(soft_vals), len(hard_vals))
            print(f"  - {key:24s} n={n_part:4d}  {soft_part}  {hard_part}")
    print("=" * 90 + "\n")


def _recover_validation_outputs_from_rollout_traces(
    *,
    rollout_traces_file: str,
    validation_output_dir: str,
    step: int = 0,
) -> Optional[str]:
    """Recover validation outputs from rollout traces when worker flush did not happen."""
    if not os.path.isfile(rollout_traces_file):
        return None

    recovered_rows: list[dict[str, object]] = []
    try:
        with open(rollout_traces_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                if str(row.get("mode", "")) == "train":
                    continue
                recovered_rows.append(row)
    except Exception as e:
        logger.warning(f"Failed to read rollout traces for recovery: {e}")
        return None

    if not recovered_rows:
        return None

    os.makedirs(validation_output_dir, exist_ok=True)
    recovered_path = os.path.join(validation_output_dir, f"validation_global_step{step}.json")
    try:
        with open(recovered_path, "w", encoding="utf-8") as f:
            json.dump(recovered_rows, f, ensure_ascii=False, indent=2)
        logger.warning(
            f"Recovered {len(recovered_rows)} validation rows from traces -> {recovered_path}"
        )
        return recovered_path
    except Exception as e:
        logger.warning(f"Failed to write recovered validation outputs: {e}")
        return None


def _count_validation_rows(saved_path: str) -> int:
    """Count rows in a saved validation JSON list file."""
    try:
        with open(saved_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            return 0
        return len(rows)
    except Exception:
        return 0


# ------------------------------------------------------------------ #
#  train()
# ------------------------------------------------------------------ #

def train(
    *,
    data_base_path: str,
    val_data_base_path: str,
    train_subdir: str,
    val_subdir: str,
    val_subdirs: list[str] | None = None,
    model_path: str,
    fewshot_min: int,
    fewshot_max: int,
    num_train_samples: int,
    num_val_samples: int,
    val_sampling_mode: str,
    val_samples_per_subtask: int,
    val_sampling_seed: int,
    val_batch_size: int,
    min_val_trace_ratio: float,
    min_val_trace_count: int,
    min_val_first_batch_ratio: float,
    n_runners: int,
    n_gpus: int,
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
    scorer_weight: float,
    grounded_proxy_weight: float,
    reward_mode: str,
    grounded_proxy_k: int,
    correctness_weight: float,
    numeric_tolerance: float,
    f1_threshold: float,
    strategy_prompt_version: str,
    answer_prompt_version: str,
    reward_version: str,
    # Strategy scorer
    strategy_scorer_model_path: str,
    strategy_scorer_model_name: str,
    strategy_scorer_base_url: str,
    strategy_scoring_prompt_version: str,
    # Fixed answer model
    answer_model_path: str,
    answer_model_base_url: str,
    answer_model_name: str,
    use_strategy_for_answer: bool,
    skip_strategy_generation: bool,
    answer_temperature: float | None,
    val_answer_temperature: float | None,
    use_hard_correctness_metric: bool,
    answer_no_think: bool,
    answer_max_tokens: int | None,
    train_dataset_json: str,
    val_only: bool,
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

    # Lightning Store HTTP port (managed client-server store). Default 4748; override with AGL_SERVER_PORT.
    if not external_store_address:
        if not (os.environ.get("AGL_SERVER_PORT") or "").strip():
            os.environ["AGL_SERVER_PORT"] = "4748"
        logger.info(f"Lightning Store port (AGL_SERVER_PORT): {os.environ['AGL_SERVER_PORT']}")

    # Data paths — training and validation roots may differ (e.g. custom train corpus + standard eval).
    train_dir = os.path.join(data_base_path, train_subdir)
    val_root = (val_data_base_path or "").strip() or data_base_path
    validation_subdirs = val_subdirs if val_subdirs else [val_subdir]
    logger.info(f"Train data directory: {train_dir}")
    logger.info(f"Validation data root: {val_root}")
    logger.info(f"Validation sets: {validation_subdirs}")
    if val_sampling_mode == "per_subtask_fixed" and val_samples_per_subtask <= 0:
        raise ValueError("val_samples_per_subtask must be > 0 in per_subtask_fixed mode")
    if not (0.0 <= min_val_trace_ratio <= 1.0):
        raise ValueError("min_val_trace_ratio must be within [0, 1]")
    if min_val_trace_count < 0:
        raise ValueError("min_val_trace_count must be >= 0")
    if not (0.0 <= min_val_first_batch_ratio <= 1.0):
        raise ValueError("min_val_first_batch_ratio must be within [0, 1]")

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
                        "val_data_base_path": val_root,
                        "train_subdir": train_subdir,
                        "val_subdirs": validation_subdirs,
                        "model_path": model_path,
                        "fewshot_min": fewshot_min,
                        "fewshot_max": fewshot_max,
                        "num_train_samples": num_train_samples,
                        "num_val_samples": num_val_samples,
                        "val_batch_size": val_batch_size,
                        "min_val_trace_ratio": min_val_trace_ratio,
                        "min_val_trace_count": min_val_trace_count,
                        "min_val_first_batch_ratio": min_val_first_batch_ratio,
                        "n_runners": n_runners,
                        "lora": lora,
                        "lora_rank": lora_rank,
                        "format_weight": format_weight,
                        "correctness_weight": correctness_weight,
                        "grounded_proxy_weight": grounded_proxy_weight,
                        "reward_mode": reward_mode,
                        "grounded_proxy_k": grounded_proxy_k,
                        "numeric_tolerance": numeric_tolerance,
                        "f1_threshold": f1_threshold,
                        "strategy_prompt_version": strategy_prompt_version,
                        "answer_prompt_version": answer_prompt_version,
                        "reward_version": reward_version,
                        "scorer_weight": scorer_weight,
                        "strategy_scorer_model_path": strategy_scorer_model_path,
                        "strategy_scorer_model_name": strategy_scorer_model_name,
                        "strategy_scorer_base_url": strategy_scorer_base_url,
                        "strategy_scoring_prompt_version": strategy_scoring_prompt_version,
                        "answer_model_path": answer_model_path,
                        "answer_model_base_url": answer_model_base_url,
                        "answer_model_name": answer_model_name,
                        "answer_max_tokens": answer_max_tokens,
                        "skip_strategy_generation": skip_strategy_generation,
                        "answer_temperature": answer_temperature,
                        "use_hard_correctness_metric": use_hard_correctness_metric,
                        "train_dataset_json": train_dataset_json,
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
    train_dataset: list[dict[str, object]]
    if val_only:
        logger.info("val_only=True: skip loading training dataset from disk.")
        train_dataset = []
    else:
        logger.info("Loading training dataset...")
        if train_dataset_json:
            print(f"[{datetime.now().isoformat()}] Loading pre-built training dataset: {train_dataset_json}")
            train_dataset = _load_strategy_dataset_from_file(train_dataset_json)
        else:
            print(f"[{datetime.now().isoformat()}] Loading training dataset from: {train_dir}")
            train_dataset = _create_strategy_generation_dataset(
                train_dir, fewshot_min, fewshot_max, num_train_samples, seed=42,
            )
        print(f"[{datetime.now().isoformat()}] Training dataset: {len(train_dataset)} samples")

    logger.info("Loading validation datasets...")
    import itertools

    val_datasets = []
    for i, vs in enumerate(validation_subdirs):
        vd = os.path.join(val_root, vs)
        logger.info(f"Loading val set {i + 1}/{len(validation_subdirs)}: {vs}")
        if val_sampling_mode == "per_subtask_fixed":
            vds = _create_strategy_generation_dataset_per_subtask(
                vd,
                fewshot_min=fewshot_min,
                fewshot_max=fewshot_max,
                samples_per_subtask=val_samples_per_subtask,
                seed=val_sampling_seed,
            )
        else:
            vds = _create_strategy_generation_dataset(
                vd, fewshot_min, fewshot_max, num_val_samples, seed=43 + i,
            )
        val_datasets.append(vds)
        logger.info(f"  {vs}: {len(vds)} samples")
    val_dataset = list(itertools.chain.from_iterable(val_datasets))

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    if val_only:
        if not val_dataset:
            raise ValueError("Validation dataset is empty in val_only mode.")
        # VERL still builds train_dataloader in val_only init path; provide one
        # guaranteed-short placeholder sample to avoid empty-train assertion.
        train_dataset = [
            {
                "problem_type": "__val_only_placeholder__",
                "examples": [
                    {"input": "1 + 1 = ?", "target": ["2"]},
                    {"input": "2 + 2 = ?", "target": ["4"]},
                    {"input": "3 + 3 = ?", "target": ["6"]},
                ],
                "num_shots": 3,
                "problem": "1 + 2 = ?",
                "ground_truth": "3",
                "ground_truths": ["3"],
                "task_meta": {"problem_type": "__val_only_placeholder__"},
                "source_problem_type": "__val_only_placeholder__",
            }
        ]

    # Validate
    for i, s in enumerate(train_dataset):
        if not s.get("examples"):
            raise ValueError(f"Train sample {i} has no examples")
        if not s.get("problem"):
            raise ValueError(f"Train sample {i} has no problem")
        if not s.get("ground_truth"):
            raise ValueError(f"Train sample {i} has no ground_truth")
        if not s.get("ground_truths"):
            raise ValueError(f"Train sample {i} has no ground_truths")
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
        n_gpus=n_gpus,
    )
    if val_only:
        config["trainer"]["val_only"] = True
        # Ensure trainer init can always form at least one train batch.
        config["data"]["train_batch_size"] = 1
        config["data"]["filter_overlong_prompts"] = False
        # Avoid enqueueing an oversized validation burst that can starve traces.
        config["data"]["val_batch_size"] = max(1, val_batch_size)
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
        scorer_weight=scorer_weight,
        proxy_weight=grounded_proxy_weight,
        reward_mode=reward_mode,
        grounded_proxy_k=grounded_proxy_k,
        correctness_weight=correctness_weight,
        numeric_tolerance=numeric_tolerance,
        f1_threshold=f1_threshold,
        strategy_scorer_base_url=strategy_scorer_base_url,
        strategy_scorer_model=strategy_scorer_model_name or strategy_scorer_model_path,
        strategy_scoring_prompt_version=strategy_scoring_prompt_version,
        answer_model_base_url=answer_model_base_url,
        answer_model_name=answer_model_name or answer_model_path,
        use_strategy_for_answer=use_strategy_for_answer,
        skip_strategy_generation=skip_strategy_generation,
        answer_temperature=answer_temperature,
        val_answer_temperature=val_answer_temperature,
        use_hard_correctness_metric=use_hard_correctness_metric,
        answer_no_think=answer_no_think,
        answer_max_tokens=answer_max_tokens,
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

    if val_only:
        try:
            forced_saved = agent.save_validation_outputs(0)
            if forced_saved:
                logger.info(f"val_only final flush saved validation outputs to: {forced_saved}")
        except Exception as e:
            logger.warning(f"val_only final flush failed (non-fatal): {e}")

    _merge_remaining_validation_files(validation_output_dir)

    if val_only:
        expected_val_count = len(val_dataset)
        first_batch_target = min(expected_val_count, max(1, val_batch_size))
        merged_files = sorted(glob.glob(os.path.join(validation_output_dir, "validation_global_step*.json")))

        def _enforce_val_only_trace_guards(observed_count: int, source: str) -> None:
            if observed_count < min_val_trace_count:
                raise RuntimeError(
                    f"Validation trace count too low ({source}): observed={observed_count} < "
                    f"min_val_trace_count={min_val_trace_count}. "
                    "Likely execution-layer instability (actor/server startup or rollout failures)."
                )
            if min_val_first_batch_ratio > 0:
                min_first_batch_count = int(first_batch_target * min_val_first_batch_ratio + 1e-9)
                if observed_count < min_first_batch_count:
                    raise RuntimeError(
                        f"Validation first-batch trace gate failed ({source}): observed={observed_count} < "
                        f"required_first_batch_count={min_first_batch_count} "
                        f"(first_batch_target={first_batch_target}, "
                        f"min_val_first_batch_ratio={min_val_first_batch_ratio:.4f})."
                    )
            coverage = observed_count / max(1, expected_val_count)
            logger.info(
                f"val_only coverage check ({source}): observed={observed_count}, "
                f"expected={expected_val_count}, ratio={coverage:.4f}"
            )
            if coverage < min_val_trace_ratio:
                raise RuntimeError(
                    f"Validation coverage too low: {coverage:.4f} < min_val_trace_ratio={min_val_trace_ratio:.4f}. "
                    "Results are likely invalid due to missing traces."
                )

        if merged_files:
            latest_path = merged_files[-1]
            logger.info(f"val_only run: using validation outputs at {latest_path}")
            _summarize_validation_outputs(latest_path)
            observed_count = _count_validation_rows(latest_path)
            _enforce_val_only_trace_guards(observed_count, "merged_file")
        else:
            rollout_traces_file = os.path.join(
                rollout_traces_dir,
                experiment_id,
                "rollout_traces.jsonl",
            )
            recovered_path = _recover_validation_outputs_from_rollout_traces(
                rollout_traces_file=rollout_traces_file,
                validation_output_dir=validation_output_dir,
                step=0,
            )
            if recovered_path:
                _summarize_validation_outputs(recovered_path)
                observed_count = _count_validation_rows(recovered_path)
                _enforce_val_only_trace_guards(observed_count, "recovered_traces")
            else:
                logger.warning(
                    "val_only run produced no validation output files and could not recover from traces."
                )

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
    parser.add_argument(
        "--val-data-base-path",
        type=str,
        default="",
        help="Root directory containing val-subdirs (test-id-subtask, etc.). "
        "If empty, defaults to --data-base-path.",
    )
    parser.add_argument("--train-subdir", type=str, default=StrategyConfig.train_subdir)
    parser.add_argument(
        "--train-dataset-json",
        type=str,
        default="",
        help="Use pre-built JSON/JSONL dataset instead of generating from train-subdir.",
    )
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
    parser.add_argument(
        "--val-sampling-mode",
        type=str,
        choices=["problem_type_balanced", "per_subtask_fixed"],
        default="problem_type_balanced",
        help="Validation sampling mode: legacy balanced-by-problem-type or fixed per subtask.",
    )
    parser.add_argument(
        "--val-samples-per-subtask",
        type=int,
        default=20,
        help="Used when --val-sampling-mode=per_subtask_fixed. Samples per subtask JSON.",
    )
    parser.add_argument(
        "--val-sampling-seed",
        type=int,
        default=42,
        help="Random seed for reproducible validation sampling.",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=64,
        help="Validation batch size used by VERL (important for val_only stability).",
    )
    parser.add_argument(
        "--min-val-trace-ratio",
        type=float,
        default=0.9,
        help="Minimum accepted ratio: saved validation rows / expected validation rows in val_only mode.",
    )
    parser.add_argument(
        "--min-val-trace-count",
        type=int,
        default=0,
        help="Absolute minimum number of validation traces required in val_only mode.",
    )
    parser.add_argument(
        "--min-val-first-batch-ratio",
        type=float,
        default=0.0,
        help="Fail-fast guard in val_only mode: observed traces must be >= this ratio * min(expected_val_count, val_batch_size).",
    )

    # Training
    parser.add_argument("--n-runners", type=int, default=StrategyConfig.n_runners)
    parser.add_argument("--n-gpus", type=int, default=8, help="Number of GPUs for VERL training")
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
    parser.add_argument("--format-weight", type=float, default=0.1, help="Weight for strategy format reward")
    parser.add_argument("--scorer-weight", type=float, default=0.9, help="Weight for strategy scorer reward (v2)")
    parser.add_argument(
        "--grounded-proxy-weight",
        type=float,
        default=0.5,
        help="Weight for grounded proxy in hybrid reward mode",
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        choices=["scorer_only", "hybrid_grounded"],
        default="hybrid_grounded",
        help="Reward routing mode",
    )
    parser.add_argument(
        "--grounded-proxy-k",
        type=int,
        default=4,
        help="K samples used to compute grounded proxy",
    )
    parser.add_argument("--correctness-weight", type=float, default=0.0, help="Weight for answer correctness reward (0 = disabled)")
    parser.add_argument("--numeric-tolerance", type=float, default=0.02, help="Numeric tolerance for answer matching")
    parser.add_argument("--f1-threshold", type=float, default=0.5, help="F1 threshold for partial answer match")

    # Strategy scorer LLM (separately trained model that scores strategy quality)
    parser.add_argument("--strategy-scorer-model-path", type=str, default="",
                        help="Path to the trained strategy-scorer model")
    parser.add_argument("--strategy-scorer-model-name", type=str, default="",
                        help="Model name in the scorer vLLM API (must match --served-model-name)")
    parser.add_argument("--strategy-scorer-base-url", type=str, default="",
                        help="Base URL of the vLLM server for the strategy scorer (e.g. http://localhost:8100/v1)")
    parser.add_argument("--strategy-scoring-prompt-version", type=str, default=StrategyConfig.strategy_scoring_prompt_version,
                        help="Prompt version for strategy scoring (see prompt/strategy_scoring/)")

    # Fixed answer-generation model (frozen copy, not trained)
    parser.add_argument("--answer-model-path", type=str, default="",
                        help="Path to the fixed answer-generation model")
    parser.add_argument("--answer-model-base-url", type=str, default="",
                        help="Base URL of the vLLM server for the fixed answer model (e.g. http://localhost:8200/v1)")
    parser.add_argument("--answer-model-name", type=str, default="",
                        help="Model name for the answer API (defaults to answer-model-path if empty)")

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
    strategy_answer_group = parser.add_mutually_exclusive_group()
    strategy_answer_group.add_argument(
        "--use-strategy-for-answer",
        dest="use_strategy_for_answer",
        action="store_true",
        help="Use generated strategy when calling the answer model.",
    )
    strategy_answer_group.add_argument(
        "--no-strategy-for-answer",
        dest="use_strategy_for_answer",
        action="store_false",
        help="Do not pass generated strategy to answer model (no-strategy baseline).",
    )
    parser.set_defaults(use_strategy_for_answer=True)
    parser.add_argument(
        "--skip-strategy-generation",
        action="store_true",
        help="Skip strategy generation entirely and run direct answer-only evaluation.",
    )
    parser.add_argument(
        "--answer-temperature",
        type=float,
        default=None,
        help="Override answer decoding temperature for training rollouts. If unset, uses rollout sampling temperature.",
    )
    parser.add_argument(
        "--val-answer-temperature",
        type=float,
        default=0.0,
        help="Answer decoding temperature used during validation (default: 0.0 = greedy/deterministic). "
             "Set to None to inherit from rollout sampling temperature.",
    )
    parser.add_argument(
        "--use-hard-correctness-metric",
        action="store_true",
        help="Use exact/hard correctness as the correctness metric instead of soft scorer correctness.",
    )
    parser.add_argument(
        "--answer-no-think",
        action="store_true",
        help="Disable think mode for the answer model (sets enable_thinking=False in chat_template_kwargs).",
    )
    parser.add_argument(
        "--answer-max-tokens",
        type=int,
        default=None,
        help="Override max_tokens for answer-model /chat/completions only (default: rollout LLM max_tokens or 16384).",
    )
    parser.add_argument(
        "--strict-no-strategy-baseline",
        action="store_true",
        help="One-shot direct baseline: skip strategy generation, k=1, temperature=0, hard correctness metric.",
    )

    # Infrastructure
    parser.add_argument("--external-store-address", type=str, default="")
    parser.add_argument("--val-only", action="store_true", help="Run validation before training and exit.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-full-output", action="store_true", default=True)

    args = parser.parse_args()

    if args.strict_no_strategy_baseline:
        args.skip_strategy_generation = True
        args.use_strategy_for_answer = False
        args.grounded_proxy_k = 1
        args.answer_temperature = 0.0
        args.use_hard_correctness_metric = True

    if args.external_store_address:
        from agentlightning.env_var import LightningEnvVar, resolve_bool_env_var

        if resolve_bool_env_var(LightningEnvVar.AGL_MANAGED_STORE, fallback=True):
            raise ValueError("Set AGL_MANAGED_STORE=0 when using an external store.")

    train(
        data_base_path=args.data_base_path,
        val_data_base_path=args.val_data_base_path,
        train_subdir=args.train_subdir,
        train_dataset_json=args.train_dataset_json,
        val_subdir=args.val_subdir,
        val_subdirs=args.val_subdirs if args.val_subdirs else (
            StrategyConfig.val_subdirs if hasattr(StrategyConfig, "val_subdirs") else None
        ),
        model_path=args.model_path,
        fewshot_min=args.fewshot_min,
        fewshot_max=args.fewshot_max,
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
        val_sampling_mode=args.val_sampling_mode,
        val_samples_per_subtask=args.val_samples_per_subtask,
        val_sampling_seed=args.val_sampling_seed,
        val_batch_size=args.val_batch_size,
        min_val_trace_ratio=args.min_val_trace_ratio,
        min_val_trace_count=args.min_val_trace_count,
        min_val_first_batch_ratio=args.min_val_first_batch_ratio,
        n_runners=args.n_runners,
        n_gpus=args.n_gpus,
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
        scorer_weight=args.scorer_weight,
        grounded_proxy_weight=args.grounded_proxy_weight,
        reward_mode=args.reward_mode,
        grounded_proxy_k=args.grounded_proxy_k,
        correctness_weight=args.correctness_weight,
        numeric_tolerance=args.numeric_tolerance,
        f1_threshold=args.f1_threshold,
        strategy_prompt_version=args.strategy_prompt_version,
        answer_prompt_version=args.answer_prompt_version,
        reward_version=args.reward_version,
        strategy_scorer_model_path=args.strategy_scorer_model_path,
        strategy_scorer_model_name=args.strategy_scorer_model_name,
        strategy_scorer_base_url=args.strategy_scorer_base_url,
        strategy_scoring_prompt_version=args.strategy_scoring_prompt_version,
        answer_model_path=args.answer_model_path,
        answer_model_base_url=args.answer_model_base_url,
        answer_model_name=args.answer_model_name,
        use_strategy_for_answer=args.use_strategy_for_answer,
        skip_strategy_generation=args.skip_strategy_generation,
        answer_temperature=args.answer_temperature,
        val_answer_temperature=args.val_answer_temperature,
        use_hard_correctness_metric=args.use_hard_correctness_metric,
        answer_no_think=args.answer_no_think,
        answer_max_tokens=args.answer_max_tokens,
        val_only=args.val_only,
    )


if __name__ == "__main__":
    main()

