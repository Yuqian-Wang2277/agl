# Copyright (c) Microsoft. All rights reserved.

"""Reward utilities and version registry for strategy extraction / generation training.

This package provides:
  1. **Stage-1 format reward helpers** (migrated from the original ``reward.py``):
     ``compute_format_reward``, ``compute_detailed_format_metrics``, ``extract_strategy``
     — these are used by both stage-1 and generation training and remain importable via
     ``from .reward import compute_format_reward, extract_strategy``.

  2. **Generation reward version registry** for swapping answer-correctness logic:
     ``RewardConfig``, ``register``, ``get_reward_config``, ``list_reward_versions``

Directory layout::

    reward/
    ├── __init__.py   ← this file (format helpers + RewardConfig + registry)
    ├── v1.py         ← default generation reward version
    └── v2.py         ← your new version (copy v1 and modify)

HOW TO ADD A NEW GENERATION REWARD VERSION:
    1. Copy ``v1.py`` → ``v2.py`` inside this folder.
    2. Edit the reward helper functions as needed.
    3. Change ``name="v1"`` → ``name="v2"`` in the ``RewardConfig``.
    4. Call ``register("v2", REWARD)`` at module level.
    5. Select it at training time via  ``--reward-version v2``.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Stage-1 format reward helpers (originally in reward.py)
# ====================================================================== #

def compute_format_reward(output: str) -> float:
    """Compute format reward based on strategy tag presence and correctness.

    Requirements for full reward (1.0):
    - Must have opening tag <strategy>
    - Must have closing tag </strategy>
    - Must have non-empty content between tags
    - Must have exactly one strategy block (no multiple or nested tags)

    Args:
        output: Model output string to check.

    Returns:
        1.0 if format is correct (strategy successfully extracted), 0.0 otherwise.
    """
    if not output:
        logger.debug("Empty output, reward: 0.0")
        return 0.0

    has_opening = "<strategy>" in output
    has_closing = "</strategy>" in output

    if not has_opening or not has_closing:
        logger.debug(
            f"Missing tags - opening: {has_opening}, closing: {has_closing}, reward: 0.0"
        )
        return 0.0

    pattern = r'<strategy>(.*?)</strategy>'
    matches = re.findall(pattern, output, re.DOTALL)

    if len(matches) != 1:
        logger.debug(f"Found {len(matches)} strategy blocks, expected exactly 1, reward: 0.0")
        return 0.0

    content = matches[0].strip()
    if not content:
        logger.debug("Strategy block is empty, reward: 0.0")
        return 0.0

    logger.debug(f"Format reward: 1.0 (valid strategy with {len(content)} chars)")
    return 1.0


def compute_detailed_format_metrics(output: str) -> Dict[str, float]:
    """Compute detailed format metrics for analysis.

    Args:
        output: Model output string to check.

    Returns:
        Dictionary with detailed metrics:
        - reward: Overall format reward (0.0 or 1.0)
        - has_opening_tag: Whether opening tag is present
        - has_closing_tag: Whether closing tag is present
        - has_content: Whether non-empty content exists
        - num_blocks: Number of strategy blocks found
        - content_length: Length of strategy content (chars)
    """
    metrics: Dict[str, float] = {
        "reward": 0.0,
        "has_opening_tag": 0.0,
        "has_closing_tag": 0.0,
        "has_content": 0.0,
        "num_blocks": 0,
        "content_length": 0,
    }

    if not output:
        return metrics

    has_opening = "<strategy>" in output
    has_closing = "</strategy>" in output

    metrics["has_opening_tag"] = 1.0 if has_opening else 0.0
    metrics["has_closing_tag"] = 1.0 if has_closing else 0.0

    pattern = r'<strategy>(.*?)</strategy>'
    matches = re.findall(pattern, output, re.DOTALL)
    metrics["num_blocks"] = len(matches)

    if matches:
        content = matches[0].strip()
        metrics["content_length"] = len(content)
        metrics["has_content"] = 1.0 if content else 0.0

    metrics["reward"] = compute_format_reward(output)

    return metrics


def extract_strategy(output: str) -> Optional[str]:
    """Extract strategy content from <strategy>...</strategy> tags.

    Args:
        output: Model output string containing strategy tags.

    Returns:
        Extracted strategy content (stripped) if found, None otherwise.
    """
    if not output:
        return None

    pattern = r'<strategy>(.*?)</strategy>'
    matches = re.findall(pattern, output, re.DOTALL)

    if not matches:
        return None

    content = matches[0].strip()

    if not content:
        return None

    return content


# ====================================================================== #
#  Generation reward version registry
# ====================================================================== #

@dataclass(frozen=True)
class RewardConfig:
    """Immutable reward configuration for one training version.

    Attributes:
        name: Human-readable name of this version.
        extract_answer: ``(model_output) -> answer_str | None``.
        compute_answer_correctness:
            ``(answer, ground_truth, numeric_tolerance, f1_threshold) -> float``.
        compute_final_reward:
            ``(format_reward, correctness, format_weight, correctness_weight) -> float``.
    """

    name: str
    extract_answer: Callable[[str], Optional[str]]
    compute_answer_correctness: Callable[[str, str, float, float], float]
    compute_final_reward: Callable[[float, float, float, float], float]


REWARD_REGISTRY: Dict[str, RewardConfig] = {}


def register(version: str, config: RewardConfig) -> None:
    """Register a reward config under the given version key."""
    REWARD_REGISTRY[version] = config


def get_reward_config(version: str = "v1") -> RewardConfig:
    """Look up a registered reward config by version key.

    Raises:
        KeyError: If the version is not in the registry.
    """
    if version not in REWARD_REGISTRY:
        available = ", ".join(sorted(REWARD_REGISTRY.keys()))
        raise KeyError(
            f"Unknown reward version '{version}'. Available: {available}"
        )
    return REWARD_REGISTRY[version]


def list_reward_versions() -> list[str]:
    """Return all registered reward version keys."""
    return sorted(REWARD_REGISTRY.keys())


def _autoload_versions() -> None:
    """Auto-import all reward version modules named ``v*.py``."""
    for module_info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):  # type: ignore[name-defined]
        module_name = module_info.name
        if module_name.startswith("v"):
            importlib.import_module(f"{__name__}.{module_name}")


_autoload_versions()
