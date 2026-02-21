# Copyright (c) Microsoft. All rights reserved.

"""Reward v1 — default generation reward (matches original hard-coded behaviour).

- extract_answer: ``<answer>...</answer>`` tag extraction.
- compute_answer_correctness: exact → numeric → F1 matching.
- compute_final_reward: weighted sum of format reward and correctness.
"""

from __future__ import annotations

import re
from typing import Optional

from . import RewardConfig, register


# ---- Answer extraction ---- #

def extract_answer(output: str) -> Optional[str]:
    """Extract answer content from <answer>...</answer> tags."""
    if not output:
        return None
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, output, re.DOTALL)
    if not matches:
        return None
    content = matches[0].strip()
    return content if content else None


# ---- Answer correctness ---- #

def compute_answer_correctness(
    answer: str,
    ground_truth: str,
    numeric_tolerance: float = 0.02,
    f1_threshold: float = 0.5,
) -> float:
    """Compute answer correctness (exact → numeric → F1)."""
    if not answer or not ground_truth:
        return 0.0

    answer_norm = answer.strip().lower()
    gt_norm = ground_truth.strip().lower()

    # Exact match
    if answer_norm == gt_norm:
        return 1.0

    # Numeric match
    try:
        a_num = float(answer_norm)
        g_num = float(gt_norm)
        if abs(g_num) < 1e-9:
            rel_err = abs(a_num - g_num)
        else:
            rel_err = abs(a_num - g_num) / abs(g_num)
        if rel_err < numeric_tolerance:
            return 0.8
    except (ValueError, TypeError):
        pass

    # F1 match
    pred_words = set(answer_norm.split())
    gold_words = set(gt_norm.split())
    if pred_words and gold_words:
        inter = pred_words & gold_words
        if inter:
            prec = len(inter) / len(pred_words)
            rec = len(inter) / len(gold_words)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if f1 >= f1_threshold:
                return 0.5

    return 0.0


# ---- Final reward aggregation ---- #

def compute_final_reward(
    format_reward: float,
    correctness: float,
    format_weight: float,
    correctness_weight: float,
) -> float:
    """Weighted sum of format reward and correctness."""
    return format_weight * format_reward + correctness_weight * correctness


# ---- Config object ---- #

REWARD = RewardConfig(
    name="v1",
    extract_answer=extract_answer,
    compute_answer_correctness=compute_answer_correctness,
    compute_final_reward=compute_final_reward,
)

register("v1", REWARD)

