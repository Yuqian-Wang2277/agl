"""Answer evaluation environment for Actor-Judge Phase II.

evaluate() / evaluate_detailed() return:
  outcome -1  format completely broken (no </strategy> or no <answer> tag)
  outcome  0  format OK but answer incorrect (v3 hard_correct == 0)
  outcome  1  format OK and answer correct (v3 hard_correct == 1)

y = -1 samples are stored in the buffer but never enter Judge pairwise training,
preventing corrupted outputs from "poisoning" the Judge's evaluation standard.

After strict XML format checks, correctness uses ``strategy_extraction.reward.v3``
(``compute_answer_judgement``).  The v3 **soft_score** in [0, 1] is returned for
WandB / JSON logging only — RL still uses the hard outcome {-1,0,1}.
"""

from __future__ import annotations

import importlib
import logging
import re
import sys
import types
from pathlib import Path
from typing import Optional, Tuple

from prompts import STRATEGY_CLOSE, STRATEGY_OPEN, ANSWER_OPEN, ANSWER_CLOSE

logger = logging.getLogger(__name__)

_EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_ROOT))

# Load v3 without executing ``strategy_extraction/__init__.py`` (that pulls agentlightning).
if "strategy_extraction" not in sys.modules:
    _pkg = types.ModuleType("strategy_extraction")
    _pkg.__path__ = [str(_EXAMPLES_ROOT / "strategy_extraction")]
    sys.modules["strategy_extraction"] = _pkg

compute_answer_judgement = importlib.import_module(
    "strategy_extraction.reward.v3"
).compute_answer_judgement


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
# Answer extraction (first <answer> block — same boundary v3 expects inside tags)
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
# Public interface
# ---------------------------------------------------------------------------


def evaluate_detailed(
    strategy_text: str,
    answer_text: str,
    answer_gold: str,
    task_meta: Optional[dict] = None,
) -> Tuple[int, float]:
    """Return (hard_outcome, v3_soft_score).  soft is 0.0 when outcome == -1."""
    soft = 0.0
    if not _strategy_format_valid(strategy_text):
        return -1, soft
    if not _answer_format_valid(answer_text):
        return -1, soft

    span = extract_answer(answer_text)
    if span is None:
        return 0, soft

    try:
        detail = compute_answer_judgement(
            answer=span,
            ground_truth=answer_gold,
            task_meta=task_meta,
        )
        soft = float(detail.get("soft_score", 0.0))
        hard = int(detail.get("hard_correct", 0))
        return (1 if hard else 0), soft
    except Exception as exc:
        logger.warning("compute_answer_judgement failed: %s", exc)
        return 0, 0.0


def evaluate(
    strategy_text: str,
    answer_text: str,
    answer_gold: str,
    task_meta: Optional[dict] = None,
) -> int:
    """Hard label only (backward compatible)."""
    outcome, _ = evaluate_detailed(strategy_text, answer_text, answer_gold, task_meta)
    return outcome


def compute_length_penalty(strategy_text: str, coeff: float, threshold: int) -> float:
    """Optional length penalty to combat Length Hacking.

    Returns a non-positive float that is subtracted from the sparse reward.
    Only active when coeff > 0.
    """
    if coeff <= 0.0:
        return 0.0
    approx_tokens = len(strategy_text.split())
    excess = max(0, approx_tokens - threshold)
    return -coeff * excess
