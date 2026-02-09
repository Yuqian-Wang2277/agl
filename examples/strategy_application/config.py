# Copyright (c) Microsoft. All rights reserved.

"""Configuration module for strategy application training."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StrategyApplicationConfig:
    """Configuration for strategy application training (Stage 2).
    
    Attributes:
        data_base_path: Base path for data directory.
        train_subdir: Subdirectory name for training data.
        val_subdir: Subdirectory name for validation data.
        val_subdirs: List of validation data subdirectory names.
        model_path: Path to the model.
        fewshot_min: Minimum number of few-shot examples.
        fewshot_max: Maximum number of few-shot examples.
        num_train_samples: Number of training samples to generate.
        num_val_samples: Number of validation samples to generate.
        n_runners: Number of parallel runners.
        lora: Whether to use LoRA training.
        lora_rank: LoRA rank when LoRA is enabled.
        wandb_project: WandB project name.
        wandb_experiment: WandB experiment/run name.
        checkpoint_dir: Directory to save checkpoints.
        save_freq: Frequency of checkpoint saving.
        test_freq: Frequency of validation.
        resume_from_checkpoint: Whether to automatically resume from checkpoint.
        save_full_output: Whether to save complete model output.
        stage2_mode: Training mode - "same_domain" or "cross_domain".
        stage1_model_path: Path to stage 1 model for strategy extraction.
        strategy_consistency_threshold: Threshold for strategy consistency check.
        answer_correctness_numeric_tolerance: Numeric tolerance for answer correctness (2%).
        answer_correctness_f1_threshold: F1 threshold for answer correctness.
        embedding_model_path: Path to embedding model for cross-domain matching.
        similarity_top_k: Number of top similar strategies to return.
        answer_format_threshold: Threshold for answer format reward (must be 1.0 to continue).
        batch_min_learnable_reward: Minimum reward value to be considered learnable.
        batch_max_retry: Maximum retry attempts for batch sampling.
    """
    
    # Base configuration
    data_base_path: str = "/home/test/test16/chenlu/projects/LLMReflection/data/"
    train_subdir: str = "train_20k"
    val_subdir: str = "test-bbh"
    val_subdirs: List[str] = field(default_factory=lambda: ["test-id-subtask", "test-ood-task", "test-bbh"])
    model_path: str = "/home/test/test16/chenlu/model/Qwen3-4B"
    fewshot_min: int = 3
    fewshot_max: int = 5
    num_train_samples: int = 20000
    num_val_samples: int = 500
    n_runners: int = 10
    lora: bool = False
    lora_rank: int = 32
    
    # WandB configuration
    wandb_project: str = "StrategyApplication"
    wandb_experiment: str = "stage2"
    
    # Checkpoint configuration
    checkpoint_dir: str = "./checkpoints_stage2"
    save_freq: int = 50
    test_freq: int = 50
    resume_from_checkpoint: bool = False
    
    # Logging configuration
    save_full_output: bool = True
    
    # Stage 2 specific configuration
    stage2_mode: str = "same_domain"  # "same_domain" | "cross_domain"
    stage1_model_path: str = "/home/test/test16/chenlu/model/Qwen3-4B"  # Path to stage 1 model for strategy extraction
    
    # Reward configuration - thresholds
    strategy_consistency_threshold: float = 0.5  # Threshold for strategy consistency
    answer_correctness_numeric_tolerance: float = 0.02  # 2% tolerance
    answer_correctness_f1_threshold: float = 0.5  # F1 threshold

    # Reward configuration - top-level weights (Stage 2A defaults)
    # R_train = format_weight * R_format
    #         + consistency_weight * R_consistency
    #         + correctness_weight * R_correctness
    format_weight: float = 0.2
    consistency_weight: float = 0.0
    correctness_weight: float = 0.8  # Stage 2A: correctness used for monitoring only

    # Strategy consistency sub-dimension weights (sum should be 1.0)
    # R_consistency = coverage_weight     * C_coverage
    #               + order_weight        * C_order
    #               + binding_weight      * C_binding
    #               + intermediate_weight * C_intermediate
    consistency_coverage_weight: float = 0.4
    consistency_order_weight: float = 0.2
    consistency_binding_weight: float = 0.2
    consistency_intermediate_weight: float = 0.2
    
    # Cross-domain matching configuration
    embedding_model_path: str = "paraphrase-multilingual-MiniLM-L12-v2"
    similarity_top_k: int = 3
    
    # Answer format and batch sampling configuration
    answer_format_threshold: float = 1.0  # Must be 1.0 to continue
    batch_min_learnable_reward: float = 0.0  # At least one reward > this value
    batch_max_retry: int = 10  # Maximum retry attempts for batch sampling
    oversample_ratio: float = 1.2  # Oversample ratio for quality filtering (1.2 = 20% more)


def get_verl_config(
    model_path: str,
    lora: bool = False,
    lora_rank: int = 32,
    resume_from_checkpoint: bool = False,
    resume_from_path: str | None = None,
    checkpoint_dir: str = "./checkpoints_stage2",
    wandb_project: str = "StrategyApplication",
    wandb_experiment: str = "stage2",
) -> Dict[str, Any]:
    """Get VERL algorithm configuration for strategy application training.
    
    Args:
        model_path: Path to the model.
        lora: Whether to use LoRA training.
        lora_rank: LoRA rank when LoRA is enabled.
        resume_from_checkpoint: Whether to automatically resume from checkpoint.
        resume_from_path: Specific checkpoint path to resume from.
        checkpoint_dir: Directory to save checkpoints.
        wandb_project: WandB project name.
        wandb_experiment: WandB experiment name.
        
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
            "max_prompt_length": 16384,
            "max_response_length": 16384,
            "filter_overlong_prompts": True,
        },
        "actor_rollout_ref": {
            "model": {
                "path": model_path,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 8,
                "log_prob_micro_batch_size_per_gpu": 4,
                "name": "vllm",
                "gpu_memory_utilization": 0.5,
                "max_model_len": 32768,
                "enable_chunked_prefill": True,
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
            "n_gpus_per_node": 8,
            "val_before_train": True,
            "critic_warmup": 0,
            "logger": ["console", "wandb"],
            "project_name": wandb_project,
            "experiment_name": wandb_experiment,
            "nnodes": 1,
            "save_freq": 50,
            "test_freq": 50,
            "total_epochs": 3,
            "default_hdfs_dir": None,
            "default_local_dir": checkpoint_dir,
            "resume_mode": "auto" if (resume_from_path is not None or resume_from_checkpoint) else "never",
            "resume_from_path": resume_from_path if resume_from_path is not None else (None if resume_from_checkpoint else ""),
        },
    }
    
    if lora:
        if "model" not in config["actor_rollout_ref"]:
            config["actor_rollout_ref"]["model"] = {}
        config["actor_rollout_ref"]["model"]["lora_rank"] = lora_rank
    
    return config

