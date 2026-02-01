# Copyright (c) Microsoft. All rights reserved.

"""Reward functions for strategy extraction training."""

import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)


def compute_format_reward(output: str) -> float:
    """Compute format reward based on strategy tag presence and correctness.
    
    This function strictly checks for the presence of well-formed
    <strategy>...</strategy> tags in the output.
    
    Requirements for full reward (1.0):
    - Must have opening tag <strategy>
    - Must have closing tag </strategy>
    - Must have non-empty content between tags
    - Must have exactly one strategy block (no multiple or nested tags)
    
    Args:
        output: Model output string to check.
        
    Returns:
        1.0 if format is correct, 0.0 otherwise.
    """
    if not output:
        return 0.0
    
    # Check for opening tag
    has_opening = "<strategy>" in output
    
    # Check for closing tag
    has_closing = "</strategy>" in output
    
    # If missing either tag, return 0.0
    if not has_opening or not has_closing:
        logger.debug(
            f"Missing tags - opening: {has_opening}, closing: {has_closing}"
        )
        return 0.0
    
    # Use regex to match strategy blocks (DOTALL for multiline)
    pattern = r'<strategy>(.*?)</strategy>'
    matches = re.findall(pattern, output, re.DOTALL)
    
    # Must have exactly one match
    if len(matches) != 1:
        logger.debug(f"Found {len(matches)} strategy blocks, expected exactly 1")
        return 0.0
    
    # Check that the content is non-empty (not just whitespace)
    content = matches[0].strip()
    if not content:
        logger.debug("Strategy block is empty")
        return 0.0
    
    # All checks passed
    logger.debug(f"Format reward: 1.0 (valid strategy with {len(content)} chars)")
    return 1.0


def compute_detailed_format_metrics(output: str) -> Dict[str, float]:
    """Compute detailed format metrics for analysis.
    
    This function provides additional metrics beyond the binary reward,
    useful for debugging and analysis.
    
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
    metrics = {
        "reward": 0.0,
        "has_opening_tag": 0.0,
        "has_closing_tag": 0.0,
        "has_content": 0.0,
        "num_blocks": 0,
        "content_length": 0,
    }
    
    if not output:
        return metrics
    
    # Check tags
    has_opening = "<strategy>" in output
    has_closing = "</strategy>" in output
    
    metrics["has_opening_tag"] = 1.0 if has_opening else 0.0
    metrics["has_closing_tag"] = 1.0 if has_closing else 0.0
    
    # Find strategy blocks
    pattern = r'<strategy>(.*?)</strategy>'
    matches = re.findall(pattern, output, re.DOTALL)
    metrics["num_blocks"] = len(matches)
    
    # Check content
    if matches:
        content = matches[0].strip()
        metrics["content_length"] = len(content)
        metrics["has_content"] = 1.0 if content else 0.0
    
    # Compute overall reward
    metrics["reward"] = compute_format_reward(output)
    
    return metrics



