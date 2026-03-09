# Copyright (c) Microsoft. All rights reserved.

"""Reward v2 — strategy-scorer reward (replaces indirect answer-correctness signal).

Key difference from v1:
    The primary reward comes from a **separately trained strategy-scorer LLM**
    that directly evaluates strategy quality, rather than using answer
    correctness as an indirect proxy.

    A fixed (frozen) answer-generation model can optionally contribute a
    secondary correctness signal, but its weight defaults to 0.

Components:
    - extract_answer / compute_answer_correctness: reused from v1 for
      optional answer-correctness evaluation.
    - extract_score: parse a 0–1 quality score from the scorer LLM output.
    - compute_final_reward(fmt, scorer, corr, fw, sw, cw): weighted sum.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from . import RewardConfig, register
from .hybrid_grounded_reward import parse_oc_scorer_response

logger = logging.getLogger(__name__)

# ---- Reuse v1 answer helpers ---- #

from .v1 import compute_answer_correctness, extract_answer  # noqa: E402


# ---- Score extraction ---- #

def extract_score(output: str) -> float:
    """Parse a strategy-quality score from the scorer LLM output.

    Handles the rubric JSON format produced by the Qwen3-8B scorer
    (``dimension_scores``, ``weighted_total_0_100``, ``score``, etc.) as well
    as simpler outputs (plain number or JSON ``{"score": 0.75}``).

    Tries, in order:
        1. JSON object (possibly wrapped in code fences) with a ``"score"`` key.
        2. A standalone decimal number on its own line.
        3. The first decimal number found anywhere in the text.

    Returns 0.0 if nothing can be parsed.  Values are clamped to [0, 1].
    """
    if not output:
        return 0.0

    text = _strip_code_fence(output.strip())

    # 1a. Direct JSON parse
    obj = _try_parse_json(text)
    if obj is not None:
        if "score" in obj:
            return _clamp01(float(obj["score"]))
        if "dimension_scores" in obj:
            return _clamp01(float(parse_oc_scorer_response(text)["final_score_01"]))

    # 1b. Regex fallback — find the outermost { … } block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        obj = _try_parse_json(m.group(0))
        if obj is not None:
            if "score" in obj:
                return _clamp01(float(obj["score"]))
            if "dimension_scores" in obj:
                return _clamp01(float(parse_oc_scorer_response(m.group(0))["final_score_01"]))

    # 2. Standalone number on a line (most common for simple scorers)
    for line in text.splitlines():
        line = line.strip()
        try:
            return _clamp01(float(line))
        except ValueError:
            continue

    # 3. First decimal / integer anywhere
    m = re.search(r"(\d+\.?\d*)", text)
    if m:
        try:
            return _clamp01(float(m.group(1)))
        except ValueError:
            pass

    logger.warning("extract_score: could not parse a score from: %s", text[:200])
    return 0.0


# ---- Helpers ---- #

def _strip_code_fence(text: str) -> str:
    """Remove optional markdown code fences wrapping model output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _try_parse_json(text: str) -> dict | None:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


def _clamp01(value: float) -> float:
    """Clamp *value* into [0, 1].  Scores in (1, 100] are auto-scaled."""
    if value > 1.0 and value <= 100.0:
        value = value / 100.0
    elif value > 100.0:
        value = 1.0
    return max(0.0, min(1.0, value))


# ---- Final reward aggregation ---- #

def compute_final_reward(
    format_reward: float,
    scorer_reward: float,
    correctness: float,
    format_weight: float,
    scorer_weight: float,
    correctness_weight: float,
) -> float:
    """Weighted sum of format, scorer, and (optional) correctness rewards."""
    return (
        format_weight * format_reward
        + scorer_weight * scorer_reward
        + correctness_weight * correctness
    )


# ---- Config object ---- #

REWARD = RewardConfig(
    name="v2",
    extract_answer=extract_answer,
    compute_answer_correctness=compute_answer_correctness,
    compute_final_reward=compute_final_reward,
    extract_score=extract_score,
)

register("v2", REWARD)
