import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_ROOT / "banchmark" / "BBH-ID-OOD"))
from eval_mist_inline import compute_answer_judgement, extract_answer  # noqa: E402


def score(model_output: str, ground_truth: str) -> tuple[bool, float]:
    """Score a BBH/ID/OOD answer against ground truth.

    Extracts <answer> tag first, then routes through compute_answer_judgement
    (yes_no / option_letter / numeric / short_phrase / medium_phrase / long_sentence).
    Returns (hard_correct, soft_score).
    """
    ans = extract_answer(model_output) or ""
    j = compute_answer_judgement(ans, ground_truth)
    return bool(j["hard_correct"]), float(j["soft_score"])
