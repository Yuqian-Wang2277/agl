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
) -> None:
    """Train the strategy extraction model.
    
    Args:
        data_base_path: Base path for data directory.
        train_subdir: Training data subdirectory name.
        val_subdir: Validation data subdirectory name.
        model_path: Path to the model.
        fewshot_min: Minimum number of few-shot examples.
        fewshot_max: Maximum number of few-shot examples.
        num_train_samples: Number of training samples to generate.
        num_val_samples: Number of validation samples to generate.
        n_runners: Number of parallel runners.
        lora: Whether to use LoRA training (default: False).
        lora_rank: LoRA rank when enabled.
        external_store_address: External store address (if using external store).
        debug: Whether to enable debug logging.
        wandb_project: WandB project name.
        wandb_experiment: WandB experiment/run name.
        checkpoint_dir: Directory to save checkpoints.
    """
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
        import ray
        if ray.is_initialized():
            logger.info("Ray is already initialized, shutting down...")
            ray.shutdown()
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
    val_dir = os.path.join(data_base_path, val_subdir)
    
    logger.info(f"Training data directory: {train_dir}")
    logger.info(f"Validation data directory: {val_dir}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Few-shot range: [{fewshot_min}, {fewshot_max}]")
    logger.info(f"Number of runners: {n_runners}")
    logger.info(f"LoRA: {'Enabled' if lora else 'Disabled (default)'}" + (f" (rank={lora_rank})" if lora else ""))
    logger.info(f"WandB Project: {wandb_project}")
    logger.info(f"WandB Experiment: {wandb_experiment}")
    logger.info(f"Checkpoint Directory: {checkpoint_dir}")
    
    # Load datasets
    logger.info("Loading training dataset...")
    train_dataset = create_strategy_dataset(
        train_dir,
        fewshot_min,
        fewshot_max,
        num_train_samples,
        seed=42,
    )
    
    logger.info("Loading validation dataset...")
    val_dataset = create_strategy_dataset(
        val_dir,
        fewshot_min,
        fewshot_max,
        num_val_samples,
        seed=43,  # Different seed for validation
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
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
    config = get_verl_config(model_path, lora=lora, lora_rank=lora_rank)
    
    # Update WandB configuration
    config["trainer"]["project_name"] = wandb_project
    config["trainer"]["experiment_name"] = wandb_experiment
    config["trainer"]["default_local_dir"] = checkpoint_dir
    
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
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=n_runners,
        store=store,
    )
    
    # Create agent
    agent = StrategyExtractionAgent()
    
    # Start training
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    trainer.fit(agent, train_dataset=train_dataset, val_dataset=val_dataset)
    
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
        help="Validation data subdirectory name",
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
    )


if __name__ == "__main__":
    main()

