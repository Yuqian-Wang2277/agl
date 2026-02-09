# Copyright (c) Microsoft. All rights reserved.

"""Training script for strategy application - Stage 2.

This script trains a model to apply problem-solving strategies to solve
problems using the Agent Lightning framework with VERL algorithm.

Example usage:

```bash
python train_strategy_application.py \\
    --stage2-mode same_domain \\
    --stage1-model-path /home/test/test16/chenlu/model/Qwen3-4B \\
    --data-base-path /home/test/test16/chenlu/projects/LLMReflection/data/ \\
    --train-subdir train_20k \\
    --model-path /home/test/test16/chenlu/model/Qwen3-4B
```
"""

import argparse
import logging
import os
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, cast

import glob
import json
import re

import agentlightning as agl

# Handle both direct execution and module import
try:
    from .config import StrategyApplicationConfig, get_verl_config
    from .data_loader import create_strategy_application_dataset
    from .similarity_matcher import SimilarityMatcher
    from .strategy_application_agent import StrategyApplicationAgent, StrategyApplicationTask
    from .strategy_extractor import StrategyExtractor
except ImportError:
    # Add parent directory to path when running directly
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from examples.strategy_application.config import StrategyApplicationConfig, get_verl_config
    from examples.strategy_application.data_loader import create_strategy_application_dataset
    from examples.strategy_application.similarity_matcher import SimilarityMatcher
    from examples.strategy_application.strategy_application_agent import (
        StrategyApplicationAgent,
        StrategyApplicationTask,
    )
    from examples.strategy_application.strategy_extractor import StrategyExtractor

logger = logging.getLogger(__name__)


def merge_remaining_validation_files(validation_output_dir: str) -> None:
    """Merge any per-worker validation files that have not yet been merged.

    Workers save per-worker files as ``validation_step{N}_worker{pid}.json``.
    The deferred merge strategy inside the agent merges the *previous* step
    when saving the *current* step, so the very last validation step is
    left un-merged.  This function scans the directory and merges any step
    whose worker files exist but whose ``validation_global_step{N}.json``
    is missing.

    Args:
        validation_output_dir: Directory containing worker validation files.
    """
    if not os.path.isdir(validation_output_dir):
        return

    # Discover all steps that have per-worker files
    worker_files = glob.glob(os.path.join(validation_output_dir, "validation_step*_worker*.json"))
    step_pattern = re.compile(r"validation_step(\d+)_worker")
    steps_with_workers: set[int] = set()
    for wf in worker_files:
        m = step_pattern.search(os.path.basename(wf))
        if m:
            steps_with_workers.add(int(m.group(1)))

    for step in sorted(steps_with_workers):
        if step == 0:
            merged_path = os.path.join(validation_output_dir, "validation_global_step0.json")
        else:
            merged_path = os.path.join(validation_output_dir, f"validation_global_step{step}.json")

        # Skip steps that already have a merged file
        if os.path.exists(merged_path):
            continue

        # Merge all worker files for this step
        pattern = os.path.join(validation_output_dir, f"validation_step{step}_worker*.json")
        step_worker_files = sorted(glob.glob(pattern))
        all_outputs: list[dict[str, object]] = []
        for swf in step_worker_files:
            try:
                with open(swf, "r", encoding="utf-8") as f:
                    outputs = json.load(f)
                    if isinstance(outputs, list):
                        all_outputs.extend(outputs)
            except Exception as e:
                logger.warning(f"Failed to load worker file {swf}: {e}")

        if all_outputs:
            try:
                with open(merged_path, "w", encoding="utf-8") as f:
                    json.dump(all_outputs, f, ensure_ascii=False, indent=2)
                logger.info(
                    f"Final merge: {len(all_outputs)} validation outputs from "
                    f"{len(step_worker_files)} workers -> {merged_path}"
                )
            except Exception as e:
                logger.error(f"Failed to write merged validation file {merged_path}: {e}")


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
    raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_attempts}")


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
    stage2_mode: str,
    stage1_model_path: str,
    embedding_model_path: str,
    consistency_threshold: float,
    numeric_tolerance: float,
    f1_threshold: float,
    batch_min_learnable_reward: float,
    batch_max_retry: int,
    oversample_ratio: float,
    # Reward weights
    format_weight: float,
    consistency_weight: float,
    correctness_weight: float,
    coverage_weight: float,
    order_weight: float,
    binding_weight: float,
    intermediate_weight: float,
) -> None:
    """Train the strategy application model."""
    original_checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    # Set up logging early (before experiment ID is determined)
    log_level = "DEBUG" if debug else "INFO"
    agl.setup_logging(log_level)
    
    # Print to console immediately (before file logging is set up)
    print("=" * 80)
    print("Strategy Application Training - Stage 2")
    print("=" * 80)
    print(f"Starting training at {datetime.now().isoformat()}")
    print(f"Checkpoint directory: {original_checkpoint_dir}")
    
    logger.info("=" * 80)
    logger.info("Strategy Application Training - Stage 2")
    logger.info("=" * 80)
    
    # Cleanup Ray processes
    try:
        import ray  # type: ignore
        if ray.is_initialized():  # type: ignore
            logger.info("Ray is already initialized, shutting down...")
            ray.shutdown()  # type: ignore
        logger.info("Ensuring clean Ray environment...")
        os.system("ray stop > /dev/null 2>&1")
    except Exception as e:
        logger.debug(f"Ray cleanup: {e}")
    
    # Auto-detect port if not using external store
    if not external_store_address:
        free_port = find_free_port()
        os.environ["AGL_SERVER_PORT"] = str(free_port)
        logger.info(f"Using auto-detected port: {free_port}")
    
    # Build data paths
    train_dir = os.path.join(data_base_path, train_subdir)
    
    # Determine validation sets
    if val_subdirs:
        validation_subdirs = val_subdirs
        logger.info(f"Using multiple validation sets: {validation_subdirs}")
    else:
        validation_subdirs = [val_subdir]
        logger.info(f"Using single validation set: {val_subdir}")
    
    # Generate experiment ID
    experiment_id = None
    if not resume_from_checkpoint and resume_from_path is None:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(original_checkpoint_dir, experiment_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"New experiment started: {experiment_id}")
        logger.info(f"Experiment checkpoint directory: {checkpoint_dir}")
    else:
        if resume_from_path:
            path_parts = os.path.normpath(os.path.abspath(resume_from_path)).split(os.sep)
            for part in path_parts:
                if len(part) == 15 and part[8] == '_' and part[:8].isdigit() and part[9:].isdigit():
                    experiment_id = part
                    checkpoint_dir = os.path.join(original_checkpoint_dir, experiment_id)
                    break
        if experiment_id is None:
            experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join(original_checkpoint_dir, experiment_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Resuming experiment: {experiment_id}")
        logger.info(f"Experiment checkpoint directory: {checkpoint_dir}")
    
    # Set up file logging
    log_dir = os.path.join(checkpoint_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{experiment_id}.log")
    
    agl.setup_logging(
        log_level,
        files={
            "agentlightning": log_file,
            "examples.strategy_application": log_file,
        }
    )
    
    logger.info(f"Logging to file: {log_file}")
    print(f"Logging to file: {log_file}")
    
    # Save training configuration
    if not resume_from_checkpoint and resume_from_path is None:
        config_save_dir = os.path.join(checkpoint_dir, "training_configs")
        os.makedirs(config_save_dir, exist_ok=True)
        config_file = os.path.join(config_save_dir, f"config_{experiment_id}.json")
        import json
        training_config = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "data_base_path": data_base_path,
            "train_subdir": train_subdir,
            "val_subdir": val_subdir,
            "val_subdirs": validation_subdirs,
            "model_path": model_path,
            "fewshot_min": fewshot_min,
            "fewshot_max": fewshot_max,
            "num_train_samples": num_train_samples,
            "num_val_samples": num_val_samples,
            "n_runners": n_runners,
            "lora": lora,
            "lora_rank": lora_rank,
            "wandb_project": wandb_project,
            "wandb_experiment": wandb_experiment,
            "checkpoint_dir": checkpoint_dir,
            "save_full_output": save_full_output,
            "resume_from_checkpoint": resume_from_checkpoint,
            "resume_from_path": resume_from_path,
            "stage2_mode": stage2_mode,
            "stage1_model_path": stage1_model_path,
            "embedding_model_path": embedding_model_path,
            "consistency_threshold": consistency_threshold,
            "numeric_tolerance": numeric_tolerance,
            "f1_threshold": f1_threshold,
            "batch_min_learnable_reward": batch_min_learnable_reward,
            "batch_max_retry": batch_max_retry,
            "oversample_ratio": oversample_ratio,
            "format_weight": format_weight,
            "consistency_weight": consistency_weight,
            "correctness_weight": correctness_weight,
            "coverage_weight": coverage_weight,
            "order_weight": order_weight,
            "binding_weight": binding_weight,
            "intermediate_weight": intermediate_weight,
        }
        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(training_config, f, ensure_ascii=False, indent=2)
            logger.info(f"Training configuration saved to: {config_file}")
        except Exception as e:
            logger.warning(f"Failed to save training configuration: {e}")
    
    logger.info(f"Training data directory: {train_dir}")
    for val_sub in validation_subdirs:
        val_dir = os.path.join(data_base_path, val_sub)
        logger.info(f"Validation data directory: {val_dir}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Stage 2 mode: {stage2_mode}")
    logger.info(f"Stage 1 model path: {stage1_model_path}")
    logger.info(f"Few-shot range: [{fewshot_min}, {fewshot_max}]")
    logger.info(f"Number of runners: {n_runners}")
    logger.info(f"WandB Project: {wandb_project}")
    logger.info(f"WandB Experiment: {wandb_experiment}")
    
    # Initialize strategy extractor (used online during rollouts)
    strategy_extractor = StrategyExtractor(
        stage1_model_path=stage1_model_path,
    )
    
    # Initialize similarity matcher for cross-domain mode
    similarity_matcher = None
    if stage2_mode == "cross_domain":
        logger.info(f"Initializing similarity matcher with model: {embedding_model_path}")
        similarity_matcher = SimilarityMatcher(embedding_model_path=embedding_model_path)
    
    # Load datasets (few-shot + problem + ground truth; strategies are extracted online)
    logger.info("Loading training dataset...")
    print(f"[{datetime.now().isoformat()}] Loading training dataset from: {train_dir}")
    try:
        train_dataset = create_strategy_application_dataset(
            train_dir,
            fewshot_min,
            fewshot_max,
            num_train_samples,
            mode=stage2_mode,
            seed=42,
            similarity_matcher=similarity_matcher,
            oversample_ratio=oversample_ratio,
            checkpoint_dir=checkpoint_dir,
        )
        print(f"[{datetime.now().isoformat()}] Training dataset loaded: {len(train_dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load training dataset: {e}", exc_info=True)
        print(f"ERROR: Failed to load training dataset: {e}")
        raise
    
    # Load validation datasets
    logger.info("Loading validation datasets...")
    print(f"[{datetime.now().isoformat()}] Loading validation datasets...")
    val_datasets = []
    for i, val_sub in enumerate(validation_subdirs):
        val_dir = os.path.join(data_base_path, val_sub)
        logger.info(f"Loading validation set {i+1}/{len(validation_subdirs)}: {val_sub}")
        print(f"[{datetime.now().isoformat()}] Loading validation set {i+1}/{len(validation_subdirs)}: {val_sub}")
        try:
            val_dataset = create_strategy_application_dataset(
                val_dir,
                fewshot_min,
                fewshot_max,
                num_val_samples,
                mode=stage2_mode,
                seed=43 + i,
                similarity_matcher=similarity_matcher,
                oversample_ratio=oversample_ratio,  # Also oversample validation for quality
                checkpoint_dir=None,  # Don't save validation dataset info
            )
            val_datasets.append(val_dataset)
            logger.info(f"  - {val_sub}: {len(val_dataset)} samples")
            print(f"[{datetime.now().isoformat()}] Validation set {val_sub}: {len(val_dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to load validation set {val_sub}: {e}", exc_info=True)
            print(f"ERROR: Failed to load validation set {val_sub}: {e}")
            raise
    
    # Merge validation datasets
    import itertools
    val_dataset = list(itertools.chain.from_iterable(val_datasets))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Combined validation dataset size: {len(val_dataset)}")
    
    # Validate datasets
    logger.info("Validating datasets...")
    for i, sample in enumerate(train_dataset):
        if not sample.get("fewshot_examples"):
            raise ValueError(f"Train sample {i} has no fewshot_examples: {sample}")
        if not sample.get("problem"):
            raise ValueError(f"Train sample {i} has no problem: {sample}")
        if not sample.get("ground_truth"):
            raise ValueError(f"Train sample {i} has no ground_truth: {sample}")
    
    logger.info("✓ Dataset validation passed")
    
    # Cast to proper type
    train_dataset = cast(agl.Dataset[StrategyApplicationTask], train_dataset)
    val_dataset = cast(agl.Dataset[StrategyApplicationTask], val_dataset)
    
    # Configure VERL algorithm
    logger.info("Configuring VERL algorithm...")
    config = get_verl_config(
        model_path,
        lora=lora,
        lora_rank=lora_rank,
        resume_from_checkpoint=resume_from_checkpoint,
        resume_from_path=resume_from_path,
        checkpoint_dir=checkpoint_dir,
        wandb_project=wandb_project,
        wandb_experiment=wandb_experiment,
    )
    
    # Update WandB configuration
    config["trainer"]["project_name"] = wandb_project
    config["trainer"]["experiment_name"] = wandb_experiment
    config["trainer"]["default_local_dir"] = checkpoint_dir
    
    algorithm = agl.VERL(config)
    
    # Configure store
    store: Optional[agl.LightningStore] = None
    if external_store_address:
        logger.info(f"Using external store: {external_store_address}")
        store = agl.LightningStoreClient(external_store_address)
    else:
        logger.info("Using in-memory store")
    
    # Create agent
    rollout_traces_dir = os.path.join(checkpoint_dir, "rollout_traces")
    validation_output_dir = os.path.join(checkpoint_dir, "validation_outputs")
    test_freq = config.get("trainer", {}).get("test_freq", 50)
    agent = StrategyApplicationAgent(
        strategy_extractor=strategy_extractor,
        save_full_output=save_full_output,
        rollout_traces_dir=rollout_traces_dir,
        validation_output_dir=validation_output_dir,
        experiment_id=experiment_id,
        test_freq=test_freq,
        consistency_threshold=consistency_threshold,
        numeric_tolerance=numeric_tolerance,
        f1_threshold=f1_threshold,
        format_weight=format_weight,
        consistency_weight=consistency_weight,
        correctness_weight=correctness_weight,
        coverage_weight=coverage_weight,
        order_weight=order_weight,
        binding_weight=binding_weight,
        intermediate_weight=intermediate_weight,
    )
    logger.info(f"Rollout traces will be saved to: {rollout_traces_dir}")
    logger.info(f"Validation outputs will be saved to: {validation_output_dir}")
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=n_runners,
        store=store,
    )
    
    # Start training
    logger.info("Starting training...")
    logger.info("=" * 80)
    print(f"[{datetime.now().isoformat()}] Starting training...")
    print("=" * 80)
    
    try:
        trainer.fit(agent, train_dataset=train_dataset, val_dataset=val_dataset)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"ERROR: Training failed: {e}")
        raise
    
    # Merge any remaining per-worker validation files that were not yet merged
    # (the deferred merge strategy leaves the last step un-merged).
    merge_remaining_validation_files(validation_output_dir)
    
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)


def main() -> None:
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train strategy application model (Stage 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Stage 2 specific arguments
    parser.add_argument(
        "--stage2-mode",
        type=str,
        choices=["same_domain", "cross_domain"],
        default=StrategyApplicationConfig.stage2_mode,
        help="Training mode: same_domain or cross_domain",
    )
    parser.add_argument(
        "--stage1-model-path",
        type=str,
        default=StrategyApplicationConfig.stage1_model_path,
        help="Path to stage 1 model for strategy extraction",
    )
    
    # Data configuration
    parser.add_argument(
        "--data-base-path",
        type=str,
        default=StrategyApplicationConfig.data_base_path,
        help="Base path for data directory",
    )
    parser.add_argument(
        "--train-subdir",
        type=str,
        default=StrategyApplicationConfig.train_subdir,
        help="Training data subdirectory name",
    )
    parser.add_argument(
        "--val-subdir",
        type=str,
        default=StrategyApplicationConfig.val_subdir,
        help="Validation data subdirectory name",
    )
    parser.add_argument(
        "--val-subdirs",
        type=str,
        nargs="+",
        default=None,
        help="List of validation data subdirectory names",
    )
    
    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        default=StrategyApplicationConfig.model_path,
        help="Path to the model",
    )
    
    # Few-shot configuration
    parser.add_argument(
        "--fewshot-min",
        type=int,
        default=StrategyApplicationConfig.fewshot_min,
        help="Minimum number of few-shot examples",
    )
    parser.add_argument(
        "--fewshot-max",
        type=int,
        default=StrategyApplicationConfig.fewshot_max,
        help="Maximum number of few-shot examples",
    )
    
    # Dataset size
    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=StrategyApplicationConfig.num_train_samples,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--num-val-samples",
        type=int,
        default=StrategyApplicationConfig.num_val_samples,
        help="Number of validation samples to generate",
    )
    
    # Training configuration
    parser.add_argument(
        "--n-runners",
        type=int,
        default=StrategyApplicationConfig.n_runners,
        help="Number of parallel runners",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Enable LoRA training",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=StrategyApplicationConfig.lora_rank,
        help="LoRA rank when LoRA is enabled",
    )
    
    # WandB configuration
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=StrategyApplicationConfig.wandb_project,
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-experiment",
        type=str,
        default=StrategyApplicationConfig.wandb_experiment,
        help="WandB experiment/run name",
    )
    
    # Checkpoint configuration
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=StrategyApplicationConfig.checkpoint_dir,
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        dest="resume_from_checkpoint",
        action="store_true",
        default=StrategyApplicationConfig.resume_from_checkpoint,
        help="Automatically resume from checkpoint",
    )
    parser.add_argument(
        "--resume-from-path",
        dest="resume_from_path",
        type=str,
        default=None,
        help="Specific checkpoint path to resume from",
    )
    
    # Cross-domain configuration
    parser.add_argument(
        "--embedding-model-path",
        type=str,
        default=StrategyApplicationConfig.embedding_model_path,
        help="Path to embedding model for cross-domain matching",
    )
    
    # Reward configuration
    parser.add_argument(
        "--consistency-threshold",
        type=float,
        default=StrategyApplicationConfig.strategy_consistency_threshold,
        help="Threshold for strategy consistency",
    )
    parser.add_argument(
        "--numeric-tolerance",
        type=float,
        default=StrategyApplicationConfig.answer_correctness_numeric_tolerance,
        help="Numeric tolerance for answer correctness",
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=StrategyApplicationConfig.answer_correctness_f1_threshold,
        help="F1 threshold for answer correctness",
    )

    # Reward weight configuration
    parser.add_argument(
        "--format-weight",
        type=float,
        default=StrategyApplicationConfig.format_weight,
        help="Weight for answer format reward component",
    )
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=StrategyApplicationConfig.consistency_weight,
        help="Weight for strategy consistency reward component",
    )
    parser.add_argument(
        "--correctness-weight",
        type=float,
        default=StrategyApplicationConfig.correctness_weight,
        help="Weight for answer correctness reward component",
    )
    parser.add_argument(
        "--consistency-coverage-weight",
        type=float,
        default=StrategyApplicationConfig.consistency_coverage_weight,
        help="Weight for strategy coverage sub-dimension",
    )
    parser.add_argument(
        "--consistency-order-weight",
        type=float,
        default=StrategyApplicationConfig.consistency_order_weight,
        help="Weight for strategy step order sub-dimension",
    )
    parser.add_argument(
        "--consistency-binding-weight",
        type=float,
        default=StrategyApplicationConfig.consistency_binding_weight,
        help="Weight for entity/variable binding sub-dimension",
    )
    parser.add_argument(
        "--consistency-intermediate-weight",
        type=float,
        default=StrategyApplicationConfig.consistency_intermediate_weight,
        help="Weight for intermediate reasoning sub-dimension",
    )
    
    # Batch sampling configuration
    parser.add_argument(
        "--batch-min-learnable-reward",
        type=float,
        default=StrategyApplicationConfig.batch_min_learnable_reward,
        help="Minimum reward value to be considered learnable",
    )
    parser.add_argument(
        "--batch-max-retry",
        type=int,
        default=StrategyApplicationConfig.batch_max_retry,
        help="Maximum retry attempts for batch sampling",
    )
    parser.add_argument(
        "--oversample-ratio",
        type=float,
        default=StrategyApplicationConfig.oversample_ratio,
        help="Oversample ratio for quality filtering (default: 1.2 = 20% more)",
    )
    
    # Infrastructure
    parser.add_argument(
        "--external-store-address",
        type=str,
        default="",
        help="External store address",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--save-full-output",
        action="store_true",
        default=StrategyApplicationConfig.save_full_output,
        help="Save complete model output",
    )
    
    args = parser.parse_args()
    
    # Validate stage1 model path
    if not args.stage1_model_path:
        raise ValueError("--stage1-model-path is required for stage 2 training")
    
    # Validate external store usage
    if args.external_store_address:
        from agentlightning.env_var import LightningEnvVar, resolve_bool_env_var
        
        if resolve_bool_env_var(LightningEnvVar.AGL_MANAGED_STORE, fallback=True):
            raise ValueError(
                "When using an external store, please set AGL_MANAGED_STORE=0."
            )
    
    # Run training
    train(
        data_base_path=args.data_base_path,
        train_subdir=args.train_subdir,
        val_subdir=args.val_subdir,
        val_subdirs=args.val_subdirs if args.val_subdirs else StrategyApplicationConfig().val_subdirs,
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
        stage2_mode=args.stage2_mode,
        stage1_model_path=args.stage1_model_path,
        embedding_model_path=args.embedding_model_path,
        consistency_threshold=args.consistency_threshold,
        numeric_tolerance=args.numeric_tolerance,
        f1_threshold=args.f1_threshold,
        batch_min_learnable_reward=args.batch_min_learnable_reward,
        batch_max_retry=args.batch_max_retry,
        oversample_ratio=args.oversample_ratio,
        format_weight=args.format_weight,
        consistency_weight=args.consistency_weight,
        correctness_weight=args.correctness_weight,
        coverage_weight=args.consistency_coverage_weight,
        order_weight=args.consistency_order_weight,
        binding_weight=args.consistency_binding_weight,
        intermediate_weight=args.consistency_intermediate_weight,
    )


if __name__ == "__main__":
    main()

