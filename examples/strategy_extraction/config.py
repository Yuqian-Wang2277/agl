# Copyright (c) Microsoft. All rights reserved.

"""Configuration module for strategy extraction training."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StrategyConfig:
    """Configuration for strategy extraction training.
    
    Attributes:
        data_base_path: Base path for data directory.
        train_subdir: Subdirectory name for training data.
        val_subdir: Subdirectory name for validation data.
        model_path: Path to the model.
        fewshot_min: Minimum number of few-shot examples.
        fewshot_max: Maximum number of few-shot examples.
        num_train_samples: Number of training samples to generate.
        num_val_samples: Number of validation samples to generate.
        n_runners: Number of parallel runners.
        lora: Whether to use LoRA training.
        lora_rank: LoRA rank when LoRA is enabled.
    """
    
    data_base_path: str = "/home/test/test16/chenlu/projects/LLMReflection/data/"
    train_subdir: str = "train_20k"
    val_subdir: str = "test-bbh"  # Options: "test-id-subtask", "test-ood-task", "test-bbh"
    # Multiple validation sets configuration
    val_subdirs: List[str] = field(default_factory=lambda: ["test-id-subtask", "test-ood-task", "test-bbh"])  # Multiple validation sets for comprehensive evaluation
    model_path: str = "/home/test/test16/chenlu/model/Qwen3-4B"
    fewshot_min: int = 3
    fewshot_max: int = 5  # Reduced from 8 to avoid long prompts
    num_train_samples: int = 20000
    num_val_samples: int = 500
    n_runners: int = 10
    lora: bool = False  # Not using LoRA by default
    lora_rank: int = 32
    
    # WandB configuration
    wandb_project: str = "StrategyExtraction"
    wandb_experiment: str = "stage1"
    
    # Checkpoint configuration
    checkpoint_dir: str = "./checkpoints"
    save_freq: int = 50  # Save checkpoint every N steps
    test_freq: int = 50  # Run validation every N steps
    resume_from_checkpoint: bool = False  # Automatically resume from checkpoint if available (default: start new training)
    
    # Logging configuration
    save_full_output: bool = True  # Save complete model output for each rollout to log file


def get_verl_config(model_path: str, lora: bool = False, lora_rank: int = 32, resume_from_checkpoint: bool = False, resume_from_path: str | None = None, checkpoint_dir: str = "./checkpoints") -> Dict[str, Any]:
    """Get VERL algorithm configuration.
    
    Args:
        model_path: Path to the model.
        lora: Whether to use LoRA training.
        lora_rank: LoRA rank when LoRA is enabled.
        resume_from_checkpoint: Whether to automatically resume from checkpoint if available.
        resume_from_path: Specific checkpoint path to resume from. If provided, overrides resume_from_checkpoint.
        checkpoint_dir: Directory to save checkpoints (will be converted to absolute path in train function).
        
    Returns:
        VERL configuration dictionary.
    """
    config = {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": 32,
            # Context overflow handling:
            # - Prompts > max_prompt_length: truncated and DROPPED from training (no gradient update)
            # - Responses > max_response_length: truncated but KEPT in training
            # - Must satisfy: max_prompt_length + max_response_length < max_model_len
            "max_prompt_length": 8192,  # Conservative limit for few-shot prompts
            "max_response_length": 16384,   # Target ~500 tokens, with buffer for variation
            "filter_overlong_prompts": True,  # Enable prompt filtering
        },
        "actor_rollout_ref": {
            "model": {
                "path": model_path,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 4,
                "name": "vllm",
                "gpu_memory_utilization": 0.5,  # Further reduced to accommodate larger context window
                # Qwen3-4B supports max_model_len=32768, but we use a smaller value to:
                # 1. Save GPU memory (KV cache grows with context length)
                # 2. Avoid OOM with parallel training on 8 GPUs
                # 3. Our actual usage: ~2048 prompt + ~800 response = ~3000 tokens
                "max_model_len": 32768,  # Reasonable balance: enough for our use case, saves memory
                "enable_chunked_prefill": True,  # Better memory management for long sequences
            },
            "actor": {
                "ppo_mini_batch_size": 32,
                "ppo_micro_batch_size_per_gpu": 4,
                "optim": {"lr": 1e-6},
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "entropy_coeff": 0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.3,
                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True,
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 8,
                "fsdp_config": {"param_offload": True},
            },
        },
        "trainer": {
            "n_gpus_per_node": 8,  # Use all 8 GPUs for parallel training
            "val_before_train": False,  # Enable validation before training starts
            "critic_warmup": 0,
            # Enable WandB logging for tracking training metrics
            # Use ["console"] to disable WandB if connection issues occur
            "logger": ["console", "wandb"],
            "project_name": "StrategyExtraction",
            "experiment_name": "stage1",
            "nnodes": 1,
            "save_freq": 50,
            "test_freq": 50,  # Run validation every 50 steps
            "total_epochs": 3,
            # Ensure checkpoints are saved in HuggingFace format with safetensors
            "default_hdfs_dir": None,  # Use local filesystem
            "default_local_dir": checkpoint_dir,  # Use provided checkpoint_dir (will be absolute path)
            # If resume_from_path is specified, use it; otherwise use resume_from_checkpoint flag
            # When resume_mode is "never", VERL should skip checkpoint loading
            # Note: VERL's _load_checkpoint may still be called, but with resume_mode="never" it should handle it gracefully
            "resume_mode": "auto" if (resume_from_path is not None or resume_from_checkpoint) else "never",
            # Set resume_from_path: if specified use it, if auto-resume enabled leave None (auto-detect), if disabled set to empty string to prevent auto-detection
            "resume_from_path": resume_from_path if resume_from_path is not None else (None if resume_from_checkpoint else ""),
        },
    }
    
    if lora:
        # type: ignore - Dynamic dict structure from VERL config
        if "model" not in config["actor_rollout_ref"]:  # type: ignore
            config["actor_rollout_ref"]["model"] = {}  # type: ignore
        config["actor_rollout_ref"]["model"]["lora_rank"] = lora_rank  # type: ignore
    
    return config

