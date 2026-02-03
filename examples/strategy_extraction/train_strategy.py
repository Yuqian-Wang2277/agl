# Copyright (c) Microsoft. All rights reserved.

"""Training script for strategy extraction - Stage 1.

This script trains a model to extract problem-solving strategies from
few-shot examples using the Agent Lightning framework with VERL algorithm.

Example usage:

```bash
python train_strategy.py \\
    --data-base-path /home/test/test16/chenlu/projects/LLMReflection/data/ \\
    --train-subdir train_20k \\
    --model-path /home/test/test16/chenlu/model/Qwen3-8B \\
    --fewshot-min 3 \\
    --fewshot-max 8 \\
    --n-runners 10 \\
    --lora
```

With external store:

```bash
# Start store server
agl store --port 9999

# Run training
AGL_MANAGED_STORE=0 python train_strategy.py \\
    --external-store-address http://localhost:9999
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

import agentlightning as agl

# Handle both direct execution and module import
try:
    from .config import StrategyConfig, get_verl_config
    from .data_loader import create_strategy_dataset
    from .strategy_agent import StrategyExtractionAgent, StrategyTask
except ImportError:
    # Add parent directory to path when running directly
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from examples.strategy_extraction.config import StrategyConfig, get_verl_config
    from examples.strategy_extraction.data_loader import create_strategy_dataset
    from examples.strategy_extraction.strategy_agent import StrategyExtractionAgent, StrategyTask

logger = logging.getLogger(__name__)


def find_free_port(start_port: int = 4747, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port.
    
    Args:
        start_port: Port to start searching from.
        max_attempts: Maximum number of ports to try.
        
    Returns:
        Available port number.
        
    Raises:
        RuntimeError: If no free port found within max_attempts.
    """
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
) -> None:
    """Train the strategy extraction model.
    
    Args:
        data_base_path: Base path for data directory.
        train_subdir: Training data subdirectory name.
        val_subdir: Validation data subdirectory name (single validation set, for backward compatibility).
        val_subdirs: List of validation data subdirectory names (multiple validation sets). If provided, overrides val_subdir.
        model_path: Path to the model.
        fewshot_min: Minimum number of few-shot examples.
        fewshot_max: Maximum number of few-shot examples.
        num_train_samples: Number of training samples to generate.
        num_val_samples: Number of validation samples to generate per validation set.
        n_runners: Number of parallel runners.
        lora: Whether to use LoRA training (default: False).
        lora_rank: LoRA rank when enabled.
        external_store_address: External store address (if using external store).
        debug: Whether to enable debug logging.
        wandb_project: WandB project name.
        wandb_experiment: WandB experiment/run name.
        checkpoint_dir: Directory to save checkpoints.
        save_full_output: Whether to save complete model output to log file.
        resume_from_checkpoint: Whether to automatically resume from checkpoint if available.
        resume_from_path: Specific checkpoint path to resume from. If provided, overrides resume_from_checkpoint.
    """
    # Convert checkpoint_dir to absolute path to ensure correct saving regardless of working directory
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    # Set up logging
    log_level = "DEBUG" if debug else "INFO"
    
    # Configure logging with file output
    log_dir = os.path.join(checkpoint_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    agl.setup_logging(
        log_level,
        files={
            "agentlightning": log_file,
            "examples.strategy_extraction": log_file,
        }
    )
    
    logger.info(f"Logging to file: {log_file}")
    
    logger.info("=" * 80)
    logger.info("Strategy Extraction Training - Stage 1")
    logger.info("=" * 80)
    
    # Cleanup any existing Ray processes to avoid GPU conflicts
    try:
        import ray  # type: ignore
        if ray.is_initialized():  # type: ignore
            logger.info("Ray is already initialized, shutting down...")
            ray.shutdown()  # type: ignore
        logger.info("Ensuring clean Ray environment...")
        os.system("ray stop > /dev/null 2>&1")
    except Exception as e:
        logger.debug(f"Ray cleanup: {e}")
    
    # Auto-detect available port if not using external store
    if not external_store_address:
        free_port = find_free_port()
        os.environ["AGL_SERVER_PORT"] = str(free_port)
        logger.info(f"Using auto-detected port: {free_port}")
    
    # Build data paths
    train_dir = os.path.join(data_base_path, train_subdir)
    
    # Determine validation sets to use (do this before saving config to save actual validation sets used)
    if val_subdirs:
        validation_subdirs = val_subdirs
        logger.info(f"Using multiple validation sets: {validation_subdirs}")
    else:
        validation_subdirs = [val_subdir]
        logger.info(f"Using single validation set: {val_subdir}")
    
    # Save training configuration for reproducibility (after determining actual validation sets)
    if not resume_from_checkpoint and resume_from_path is None:
        config_save_dir = os.path.join(checkpoint_dir, "training_configs")
        os.makedirs(config_save_dir, exist_ok=True)
        config_file = os.path.join(config_save_dir, f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        training_config = {
            "timestamp": datetime.now().isoformat(),
            "data_base_path": data_base_path,
            "train_subdir": train_subdir,
            "val_subdir": val_subdir,  # Keep for backward compatibility
            "val_subdirs": validation_subdirs,  # Save actual validation sets used
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
        }
        try:
            import json
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(training_config, f, ensure_ascii=False, indent=2)
            logger.info(f"Training configuration saved to: {config_file}")
            logger.info(f"  - Validation sets: {validation_subdirs}")
        except Exception as e:
            logger.warning(f"Failed to save training configuration: {e}")
    
    logger.info(f"Training data directory: {train_dir}")
    for val_sub in validation_subdirs:
        val_dir = os.path.join(data_base_path, val_sub)
        logger.info(f"Validation data directory: {val_dir}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Few-shot range: [{fewshot_min}, {fewshot_max}]")
    logger.info(f"Number of runners: {n_runners}")
    logger.info(f"LoRA: {'Enabled' if lora else 'Disabled (default)'}" + (f" (rank={lora_rank})" if lora else ""))
    logger.info(f"WandB Project: {wandb_project}")
    logger.info(f"WandB Experiment: {wandb_experiment}")
    logger.info(f"Checkpoint Directory: {checkpoint_dir}")
    if resume_from_path:
        logger.info(f"Resume from specific checkpoint: {resume_from_path}")
    else:
        logger.info(f"Resume from checkpoint: {'Enabled (auto-detect)' if resume_from_checkpoint else 'Disabled'}")
    
    # Load datasets
    logger.info("Loading training dataset...")
    train_dataset = create_strategy_dataset(
        train_dir,
        fewshot_min,
        fewshot_max,
        num_train_samples,
        seed=42,
    )
    
    # Load validation datasets (merge multiple validation sets if provided)
    logger.info("Loading validation datasets...")
    val_datasets = []
    for i, val_sub in enumerate(validation_subdirs):
        val_dir = os.path.join(data_base_path, val_sub)
        logger.info(f"Loading validation set {i+1}/{len(validation_subdirs)}: {val_sub}")
        val_dataset = create_strategy_dataset(
            val_dir,
            fewshot_min,
            fewshot_max,
            num_val_samples,
            seed=43 + i,  # Different seed for each validation set
        )
        val_datasets.append(val_dataset)
        logger.info(f"  - {val_sub}: {len(val_dataset)} samples")
    
    # Merge all validation datasets into one
    import itertools
    val_dataset = list(itertools.chain.from_iterable(val_datasets))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Combined validation dataset size: {len(val_dataset)} (from {len(validation_subdirs)} sets)")
    
    # Validate datasets: ensure all samples have examples
    logger.info("Validating datasets...")
    for i, sample in enumerate(train_dataset):
        if not sample.get("examples"):
            raise ValueError(f"Train sample {i} has no examples: {sample}")
        if sample.get("num_shots", 0) == 0:
            raise ValueError(f"Train sample {i} has 0 shots: {sample}")
        # Check that examples have required fields
        for j, ex in enumerate(sample["examples"]):
            if "input" not in ex:
                raise ValueError(f"Train sample {i}, example {j} missing 'input'")
            if "target" not in ex:
                raise ValueError(f"Train sample {i}, example {j} missing 'target'")
    
    for i, sample in enumerate(val_dataset):
        if not sample.get("examples"):
            raise ValueError(f"Val sample {i} has no examples: {sample}")
        if sample.get("num_shots", 0) == 0:
            raise ValueError(f"Val sample {i} has 0 shots: {sample}")
        # Check that examples have required fields
        for j, ex in enumerate(sample["examples"]):
            if "input" not in ex:
                raise ValueError(f"Val sample {i}, example {j} missing 'input'")
            if "target" not in ex:
                raise ValueError(f"Val sample {i}, example {j} missing 'target'")
    
    logger.info("✓ Dataset validation passed: all samples have valid examples")
    
    # Cast to proper type
    train_dataset = cast(agl.Dataset[StrategyTask], train_dataset)
    val_dataset = cast(agl.Dataset[StrategyTask], val_dataset)
    
    # Configure VERL algorithm
    logger.info("Configuring VERL algorithm...")
    # Pass checkpoint_dir (already absolute path) to get_verl_config
    config = get_verl_config(
        model_path, 
        lora=lora, 
        lora_rank=lora_rank, 
        resume_from_checkpoint=resume_from_checkpoint, 
        resume_from_path=resume_from_path,
        checkpoint_dir=checkpoint_dir
    )
    
    # Update WandB configuration
    config["trainer"]["project_name"] = wandb_project
    config["trainer"]["experiment_name"] = wandb_experiment
    # checkpoint_dir is already set in get_verl_config, but ensure it's correct
    config["trainer"]["default_local_dir"] = checkpoint_dir
    
    # Log resume configuration for debugging
    logger.info(f"Resume configuration:")
    logger.info(f"  - resume_mode: {config['trainer'].get('resume_mode', 'not set')}")
    logger.info(f"  - resume_from_path: {config['trainer'].get('resume_from_path', 'not set')}")
    logger.info(f"  - default_local_dir: {config['trainer'].get('default_local_dir', 'not set')}")
    
    # Validate critical config parameters
    logger.info("Validating configuration...")
    assert "algorithm" in config, "Missing 'algorithm' in config"
    assert "data" in config, "Missing 'data' in config"
    assert "actor_rollout_ref" in config, "Missing 'actor_rollout_ref' in config"
    assert "trainer" in config, "Missing 'trainer' in config"
    assert config["actor_rollout_ref"]["model"]["path"] == model_path, "Model path mismatch"
    
    # Log key configurations
    logger.info("VERL Configuration:")
    logger.info(f"  - Advantage Estimator: {config['algorithm']['adv_estimator']}")
    logger.info(f"  - Batch Size: {config['data']['train_batch_size']}")
    logger.info(f"  - Learning Rate: {config['actor_rollout_ref']['actor']['optim']['lr']}")
    logger.info(f"  - Rollout Samples per Step: {config['actor_rollout_ref']['rollout']['n']}")
    logger.info(f"  - Checkpoint Format: HuggingFace (default)")
    logger.info(f"  - WandB Logging: Enabled")
    logger.info(f"  - Logger Config: {config['trainer']['logger']}")
    
    algorithm = agl.VERL(config)
    
    # Configure store
    store: Optional[agl.LightningStore] = None
    if external_store_address:
        logger.info(f"Using external store: {external_store_address}")
        store = agl.LightningStoreClient(external_store_address)
    else:
        logger.info("Using in-memory store")
        # Note: If you encounter server startup timeout issues, you can:
        # 1. Use an external store: agl store --port 9999
        # 2. Set AGL_MANAGED_STORE=0 and use --external-store-address
    
    # Create agent with rollout traces directory and validation output directory
    rollout_traces_dir = os.path.join(checkpoint_dir, "rollout_traces")
    validation_output_dir = os.path.join(checkpoint_dir, "validation_outputs")
    agent = StrategyExtractionAgent(
        save_full_output=save_full_output,
        rollout_traces_dir=rollout_traces_dir,
        validation_output_dir=validation_output_dir
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
    
    trainer.fit(agent, train_dataset=train_dataset, val_dataset=val_dataset)
    
    # Save any remaining validation outputs after training completes
    if hasattr(agent, 'validation_outputs') and agent.validation_outputs:
        saved_path = agent.save_validation_outputs(0)  # Use 0 as final step marker
        if saved_path:
            logger.info(f"Final validation outputs saved to: {saved_path}")
    
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)


def main() -> None:
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train strategy extraction model (Stage 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data configuration
    parser.add_argument(
        "--data-base-path",
        type=str,
        default=StrategyConfig.data_base_path,
        help="Base path for data directory",
    )
    parser.add_argument(
        "--train-subdir",
        type=str,
        default=StrategyConfig.train_subdir,
        help="Training data subdirectory name",
    )
    parser.add_argument(
        "--val-subdir",
        type=str,
        default=StrategyConfig.val_subdir,
        help="Validation data subdirectory name (single validation set, for backward compatibility)",
    )
    parser.add_argument(
        "--val-subdirs",
        type=str,
        nargs="+",
        default=None,
        help="List of validation data subdirectory names (multiple validation sets). If provided, overrides --val-subdir. Example: --val-subdirs test-id-subtask test-ood-task test-bbh",
    )
    
    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        default=StrategyConfig.model_path,
        help="Path to the model",
    )
    
    # Few-shot configuration
    parser.add_argument(
        "--fewshot-min",
        type=int,
        default=StrategyConfig.fewshot_min,
        help="Minimum number of few-shot examples",
    )
    parser.add_argument(
        "--fewshot-max",
        type=int,
        default=StrategyConfig.fewshot_max,
        help="Maximum number of few-shot examples",
    )
    
    # Dataset size
    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=StrategyConfig.num_train_samples,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--num-val-samples",
        type=int,
        default=StrategyConfig.num_val_samples,
        help="Number of validation samples to generate",
    )
    
    # Training configuration
    parser.add_argument(
        "--n-runners",
        type=int,
        default=StrategyConfig.n_runners,
        help="Number of parallel runners",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Enable LoRA training (default: disabled, full parameter training)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=StrategyConfig.lora_rank,
        help="LoRA rank when LoRA is enabled",
    )
    
    # WandB configuration
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=StrategyConfig.wandb_project,
        help="WandB project name for tracking",
    )
    parser.add_argument(
        "--wandb-experiment",
        type=str,
        default=StrategyConfig.wandb_experiment,
        help="WandB experiment/run name",
    )
    
    # Checkpoint configuration
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=StrategyConfig.checkpoint_dir,
        help="Directory to save checkpoints (HuggingFace safetensor format)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        dest="resume_from_checkpoint",
        action="store_true",
        default=StrategyConfig.resume_from_checkpoint,
        help="Automatically resume from checkpoint if available (default: disabled, start new training)",
    )
    parser.add_argument(
        "--no-resume-from-checkpoint",
        dest="resume_from_checkpoint",
        action="store_false",
        help="Disable automatic checkpoint resuming, start training from scratch (default behavior)",
    )
    parser.add_argument(
        "--resume-from-path",
        dest="resume_from_path",
        type=str,
        default=None,
        help="Specific checkpoint path to resume from (e.g., ./checkpoints/global_step_64). Overrides --resume-from-checkpoint.",
    )
    
    # Infrastructure
    parser.add_argument(
        "--external-store-address",
        type=str,
        default="",
        help="External store address (e.g., http://localhost:9999)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--save-full-output",
        action="store_true",
        default=StrategyConfig.save_full_output,
        help="Save complete model output for each rollout to log file (default: from config)",
    )
    parser.add_argument(
        "--no-save-full-output",
        dest="save_full_output",
        action="store_false",
        help="Disable saving complete model output to log file",
    )
    
    args = parser.parse_args()
    
    # Validate external store usage
    if args.external_store_address:
        from agentlightning.env_var import LightningEnvVar, resolve_bool_env_var
        
        if resolve_bool_env_var(LightningEnvVar.AGL_MANAGED_STORE, fallback=True):
            raise ValueError(
                "When using an external store, please set AGL_MANAGED_STORE=0. "
                "Otherwise the trainer will try to manage the store lifecycle!"
            )
    
    # Warn if LoRA is being used (not recommended for this task)
    if args.lora:
        logger.warning(
            "LoRA training is enabled. Note: Full parameter training (default) "
            "is recommended for better strategy extraction quality."
        )
    
    # Run training
    train(
        data_base_path=args.data_base_path,
        train_subdir=args.train_subdir,
        val_subdir=args.val_subdir,
        val_subdirs=args.val_subdirs if args.val_subdirs else (StrategyConfig.val_subdirs if hasattr(StrategyConfig, 'val_subdirs') else None),
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
    )


if __name__ == "__main__":
    main()

