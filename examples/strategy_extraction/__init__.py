# Copyright (c) Microsoft. All rights reserved.

"""Strategy Extraction Training - Stage 1

This example demonstrates training a model to extract problem-solving strategies
from few-shot examples using Agent Lightning framework.
"""

from .config import StrategyConfig
from .data_loader import create_strategy_dataset, load_problem_types, sample_fewshot_examples
from .prompts import format_user_prompt, get_system_prompt
from .reward import compute_format_reward
from .strategy_agent import StrategyExtractionAgent, StrategyTask

__all__ = [
    "StrategyConfig",
    "create_strategy_dataset",
    "load_problem_types",
    "sample_fewshot_examples",
    "format_user_prompt",
    "get_system_prompt",
    "compute_format_reward",
    "StrategyExtractionAgent",
    "StrategyTask",
]



