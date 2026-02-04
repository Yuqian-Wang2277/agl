# Copyright (c) Microsoft. All rights reserved.

"""Strategy Application Training - Stage 2

This example demonstrates training a model to apply problem-solving strategies
to solve problems using Agent Lightning framework.
"""

from .config import StrategyApplicationConfig, get_verl_config
from .data_loader import create_strategy_application_dataset
from .prompts import (
    format_strategy_application_prompt,
    get_strategy_application_system_prompt,
)
from .reward import (
    compute_answer_correctness,
    compute_answer_format_reward,
    compute_stage2_reward,
    extract_answer,
)
from .similarity_matcher import SimilarityMatcher
from .strategy_application_agent import (
    StrategyApplicationAgent,
    StrategyApplicationTask,
)
from .strategy_extractor import StrategyExtractor

__all__ = [
    "StrategyApplicationConfig",
    "get_verl_config",
    "create_strategy_application_dataset",
    "format_strategy_application_prompt",
    "get_strategy_application_system_prompt",
    "compute_answer_format_reward",
    "extract_answer",
    "compute_answer_correctness",
    "compute_stage2_reward",
    "SimilarityMatcher",
    "StrategyApplicationAgent",
    "StrategyApplicationTask",
    "StrategyExtractor",
]

