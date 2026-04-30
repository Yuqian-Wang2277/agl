import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_ROOT / "banchmark" / "linguini"))
from run_linguini_passk import parse_numbered_answers, score_answers  # noqa: E402


def score(model_output: str, ground_truth: Any, eval_type: str = "single") -> tuple[bool, float]:
    """Score a Linguini answer against ground truth.

    ground_truth may be a JSON string (list) or already a list.
    Returns (hard_correct, soft_score) where hard_correct = score == 1.0.
    """
    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
        except json.JSONDecodeError:
            gt = [ground_truth]
    else:
        gt = ground_truth
    predicted = parse_numbered_answers(model_output)
    s = score_answers(predicted, gt, eval_type)
    return s >= 1.0, float(s)
