"""Answer evaluation environment for Actor-Judge Phase II.

evaluate() returns:
  -1  format completely broken (no </strategy> or no <answer> tag)
   0  format OK but answer incorrect
   1  format OK and answer correct

y = -1 samples are stored in the buffer but never enter Judge pairwise training,
preventing corrupted outputs from "poisoning" the Judge's evaluation standard.

Correctness logic is ported from strategy_extraction/reward/v1.py
(exact → numeric → F1 cascade).
"""

from __future__ import annotations

import re
from typing import Optional

from prompts import STRATEGY_CLOSE, STRATEGY_OPEN, ANSWER_OPEN, ANSWER_CLOSE


# ---------------------------------------------------------------------------
# Format validation
# ---------------------------------------------------------------------------

def _strategy_format_valid(strategy_text: str) -> bool:
    """Check that the strategy output contains a properly closed <strategy> tag."""
    return STRATEGY_OPEN in strategy_text and STRATEGY_CLOSE in strategy_text


def _answer_format_valid(answer_text: str) -> bool:
    """Check that the answer output contains a properly closed <answer> tag."""
    return ANSWER_OPEN in answer_text and ANSWER_CLOSE in answer_text


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(output: str) -> Optional[str]:
    """Extract text inside the first <answer>…</answer> block."""
    if not output:
        return None
    matches = re.findall(r"<answer>(.*?)</answer>", output, re.DOTALL)
    if not matches:
        return None
    content = matches[0].strip()
    return content if content else None


# ---------------------------------------------------------------------------
# Correctness cascade (exact → numeric → F1)
# ---------------------------------------------------------------------------

def _compute_correctness(
    answer: str,
    ground_truth: str,
    numeric_tolerance: float = 0.02,
    f1_threshold: float = 0.5,
) -> int:
    """Return 1 if answer is correct, 0 otherwise."""
    if not answer or not ground_truth:
        return 0

    ans_norm = answer.strip().lower()
    gt_norm  = ground_truth.strip().lower()

    # Exact match
    if ans_norm == gt_norm:
        return 1

    # Numeric match
    try:
        a_num = float(ans_norm)
        g_num = float(gt_norm)
        denom = abs(g_num) if abs(g_num) > 1e-9 else 1.0
        if abs(a_num - g_num) / denom < numeric_tolerance:
            return 1
    except (ValueError, TypeError):
        pass

    # Token-level F1 match
    pred_words = set(ans_norm.split())
    gold_words = set(gt_norm.split())
    if pred_words and gold_words:
        inter = pred_words & gold_words
        if inter:
            prec = len(inter) / len(pred_words)
            rec  = len(inter) / len(gold_words)
            f1   = 2 * prec * rec / (prec + rec)
            if f1 >= f1_threshold:
                return 1

    return 0


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def evaluate(strategy_text: str, answer_text: str, answer_gold: str) -> int:
    """Evaluate a (strategy, answer) pair against the gold answer.

    Args:
        strategy_text: Full text output from Stage-1 (should contain
                       <strategy>…</strategy>).
        answer_text:   Full text output from Stage-2 (should contain
                       <answer>…</answer>).
        answer_gold:   Ground-truth answer string.

    Returns:
        -1 if format is broken, 0 if wrong, 1 if correct.
    """
    # Format_Check — prevents broken outputs from poisoning Judge training
    if not _strategy_format_valid(strategy_text):
        return -1
    if not _answer_format_valid(answer_text):
        return -1

    answer = extract_answer(answer_text)
    if answer is None:
        return 0

    return _compute_correctness(answer, answer_gold)


def compute_length_penalty(strategy_text: str, coeff: float, threshold: int) -> float:
    """Optional length penalty to combat Length Hacking.

    Returns a non-positive float that is subtracted from the sparse reward.
    Only active when coeff > 0.
    """
    if coeff <= 0.0:
        return 0.0
    # Approximate token count by word-splitting (fast, no tokenizer needed here)
    approx_tokens = len(strategy_text.split())
    excess = max(0, approx_tokens - threshold)
    return -coeff * excess
