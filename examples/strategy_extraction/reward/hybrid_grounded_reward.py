"""Hybrid grounded reward helpers for Phase-2 training.

This module separates three signals:
- grounded_proxy: exact success ratio over K answer samples
- soft_correctness_mean: mean router-like correctness score over K samples
- oc_scorer: outcome-conditioned strategy scorer parsed from JSON rubric
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Sequence


AsyncAnswerFn = Callable[[str], Awaitable[str]]
ExtractAnswerFn = Callable[[str], str | None]
CorrectnessFn = Callable[[str, str], float]


_CAP_FLAG_MAP = {
    "correct_but_unrelated": 55.0,
    "wrong_but_strategy_ok": 75.0,
    "core_wrong_step": 35.0,
    "generic_template": 50.0,
    "leaked_answer": 60.0,
}
_PENALTY_MAP = {
    "generic_not_operational": 15.0,
    "memorization_leakage": 15.0,
    "contradiction": 20.0,
    "missing_check": 10.0,
}
_DIMENSION_KEYS = [
    "A_outcome_support",
    "B_executability",
    "C_example_grounding",
    "D_problem_coverage",
    "E_transfer_robustness",
    "F_clarity_economy",
]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


async def evaluate_strategy_k_samples(
    strategy: str,
    targets: Sequence[str],
    task_meta: Dict[str, Any],
    answer_fn: AsyncAnswerFn,
    extract_answer_fn: ExtractAnswerFn,
    correctness_fn: CorrectnessFn,
    k: int = 4,
) -> Dict[str, Any]:
    """Evaluate strategy with K sampled answers.

    Args:
        strategy: Strategy text.
        targets: Candidate references; first non-empty one is used for scoring.
        task_meta: Reserved metadata for future router logic.
        answer_fn: Async callable that generates answer from strategy.
        extract_answer_fn: Extractor from raw answer output.
        correctness_fn: Score function returning [0, 1].
        k: Number of samples for grounded proxy.
    """
    del task_meta  # reserved for future router-specific branches
    if k <= 0:
        k = 1

    ground_truth = ""
    for target in targets:
        if isinstance(target, str) and target.strip():
            ground_truth = target
            break

    raw_answers = await asyncio.gather(*[answer_fn(strategy) for _ in range(k)])
    extracted_answers: List[str] = []
    router_scores: List[float] = []
    exact_success = 0
    for raw in raw_answers:
        extracted = extract_answer_fn(raw) or ""
        extracted_answers.append(extracted)
        score = correctness_fn(extracted, ground_truth) if ground_truth else 0.0
        router_scores.append(score)
        if score == 1.0:
            exact_success += 1

    grounded_proxy = exact_success / k if k > 0 else 0.0
    soft_correctness_mean = sum(router_scores) / len(router_scores) if router_scores else 0.0
    return {
        "answer_raw_list": raw_answers,
        "answer_extracted_list": extracted_answers,
        "grounded_proxy": grounded_proxy,
        "soft_correctness_mean": soft_correctness_mean,
        "single_sample_correctness": router_scores[0] if router_scores else 0.0,
        "router_scores": router_scores,
    }


def select_representative_answer(answer_raw_list: Sequence[str], router_scores: Sequence[float]) -> Dict[str, Any]:
    """Select a representative answer for OC scorer prompting."""
    if not answer_raw_list:
        return {
            "representative_answer_raw": "",
            "representative_index": -1,
            "representative_label": "incorrect",
        }

    best_idx = 0
    if router_scores:
        best_idx = max(range(len(answer_raw_list)), key=lambda i: router_scores[i] if i < len(router_scores) else -1)
        for i, score in enumerate(router_scores):
            if i < len(answer_raw_list) and score == 1.0:
                best_idx = i
                break
    label = "correct" if best_idx < len(router_scores) and router_scores[best_idx] == 1.0 else "incorrect"
    return {
        "representative_answer_raw": answer_raw_list[best_idx],
        "representative_index": best_idx,
        "representative_label": label,
    }


def _extract_json_obj(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    text = raw.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    block = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, re.IGNORECASE)
    if block:
        try:
            parsed = json.loads(block.group(1))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

    loose = re.search(r"(\{[\s\S]*\})", text)
    if loose:
        try:
            parsed = json.loads(loose.group(1))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
    return {}


def _compute_cap_ceiling(cap_flags: Sequence[str]) -> float:
    ceilings = [_CAP_FLAG_MAP.get(flag) for flag in cap_flags if isinstance(flag, str)]
    valid = [value for value in ceilings if value is not None]
    return min(valid) if valid else 100.0


def _compute_penalty_total(penalties: Sequence[str]) -> float:
    total = 0.0
    for penalty in penalties:
        if isinstance(penalty, str):
            total += _PENALTY_MAP.get(penalty, 0.0)
    return total


def build_outcome_conditioned_scorer_prompt(
    examples_text: str,
    strategy: str,
    problem: str,
    answer: str,
    correctness_label: str,
) -> List[Dict[str, str]]:
    """Build OC-scorer prompt messages when not using external TOML."""
    system = (
        "You are a strict evaluator of STRATEGY QUALITY, not writing style.\n"
        "Score six dimensions from 0 to 5 with decimal values when needed.\n"
        "Apply hard cap rules and penalties, and output JSON only."
    )
    user = (
        f"Few-shot examples:\n{examples_text}\n\n"
        f"Strategy:\n{strategy}\n\n"
        f"Problem:\n{problem}\n\n"
        f"Generated answer:\n{answer}\n\n"
        f"Answer was: {correctness_label}\n\n"
        "Return only valid JSON with dimension_scores, cap_flags, penalties, and verdict."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_oc_scorer_response(raw_json: str) -> Dict[str, Any]:
    """Parse OC-scorer JSON and compute deterministic final score."""
    obj = _extract_json_obj(raw_json)
    dims = obj.get("dimension_scores", {}) if isinstance(obj, dict) else {}
    if not isinstance(dims, dict):
        dims = {}

    score_values = {k: _to_float(dims.get(k), 0.0) for k in _DIMENSION_KEYS}
    weighted_raw_100 = 20.0 * (
        0.30 * score_values["A_outcome_support"]
        + 0.20 * score_values["B_executability"]
        + 0.15 * score_values["C_example_grounding"]
        + 0.15 * score_values["D_problem_coverage"]
        + 0.15 * score_values["E_transfer_robustness"]
        + 0.05 * score_values["F_clarity_economy"]
    )

    cap_flags = obj.get("cap_flags", []) if isinstance(obj.get("cap_flags", []), list) else []
    penalties = obj.get("penalties", []) if isinstance(obj.get("penalties", []), list) else []
    cap_ceiling = _compute_cap_ceiling(cap_flags)
    penalty_total = _compute_penalty_total(penalties)
    final_100 = min(max(weighted_raw_100 - penalty_total, 0.0), cap_ceiling)
    final_01 = round(final_100 / 100.0, 4)

    return {
        "final_score_01": final_01,
        "final_score_100": round(final_100, 4),
        "dimension_scores": score_values,
        "cap_flags": cap_flags,
        "penalties": penalties,
        "raw_obj": obj,
    }


def compute_hybrid_reward(
    format_r: float,
    oc_scorer_r: float,
    grounded_proxy: float,
    format_weight: float = 0.1,
    scorer_weight: float = 0.4,
    proxy_weight: float = 0.5,
) -> float:
    """Compute final hybrid reward."""
    total = format_weight + scorer_weight + proxy_weight
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Reward weights must sum to 1.0, got {total}")
    return round(format_weight * format_r + scorer_weight * oc_scorer_r + proxy_weight * grounded_proxy, 4)


def select_effective_proxy(
    grounded_proxy: float,
    soft_correctness_mean: float,
    group_gp_values: Sequence[float],
    soft_fallback_scale: float = 0.3,
) -> float:
    """Fallback to soft correctness when all group grounded proxies are zero."""
    if group_gp_values and max(group_gp_values) == 0.0:
        return round(soft_correctness_mean * soft_fallback_scale, 4)
    return grounded_proxy
