import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_ROOT / "banchmark" / "HARDMath2"))
from eval_hardmath import compare_math_answers  # noqa: E402


def score(model_output: str, ground_truth: str) -> tuple[bool, float]:
    """Score a HARDMath2 answer against ground truth.

    Returns (hard_correct, soft_score).
    """
    return compare_math_answers(model_output, ground_truth)
