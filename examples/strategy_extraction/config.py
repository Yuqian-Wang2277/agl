# Copyright (c) Microsoft. All rights reserved.

"""Configuration module for strategy extraction training."""

from dataclasses import dataclass
from typing import Any, Dict


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
    val_subdir: str = "val_set"
    model_path: str = "/home/test/test16/chenlu/model/Qwen3-4B"
    fewshot_min: int = 2
    fewshot_max: int = 3  # Reduced from 8 to avoid long prompts
    num_train_samples: int = 1000
    num_val_samples: int = 200
    n_runners: int = 10
    lora: bool = False  # Not using LoRA by default
    lora_rank: int = 32
    
    # WandB configuration
    wandb_project: str = "StrategyExtraction"
    wandb_experiment: str = "stage1"
    
    # Checkpoint configuration
    checkpoint_dir: str = "./checkpoints"
    save_freq: int = 64  # Save checkpoint every N steps
    test_freq: int = -1  # Run validation every N steps


def get_verl_config(model_path: str, lora: bool = False, lora_rank: int = 32) -> Dict[str, Any]:
    """Get VERL algorithm configuration.
    
    Args:
        model_path: Path to the model.
        lora: Whether to use LoRA training.
        lora_rank: LoRA rank when LoRA is enabled.
        
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
            "val_before_train": False,
            "critic_warmup": 0,
            # Enable WandB logging for tracking training metrics
            "logger": ["console", "wandb"],
            "project_name": "StrategyExtraction",
            "experiment_name": "stage1",
            "nnodes": 1,
            "save_freq": 64,
            "test_freq": 32,
            "total_epochs": 2,
            # Ensure checkpoints are saved in HuggingFace format with safetensors
            "default_hdfs_dir": None,  # Use local filesystem
            "default_local_dir": "./checkpoints",
        },
    }
    
    if lora:
        # type: ignore - Dynamic dict structure from VERL config
        if "model" not in config["actor_rollout_ref"]:  # type: ignore
            config["actor_rollout_ref"]["model"] = {}  # type: ignore
        config["actor_rollout_ref"]["model"]["lora_rank"] = lora_rank  # type: ignore
    
    return config

