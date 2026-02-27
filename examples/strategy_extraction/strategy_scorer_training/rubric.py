# Copyright (c) Microsoft. All rights reserved.

"""Rubric schema and score utilities for single-round strategy scoring."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

RUBRIC_VERSION = "single_round_self_evolution_v1"

# All dimensions are scored on a 0-5 scale.
DIMENSION_WEIGHTS: Dict[str, float] = {
    # A. Strategy quality (70)
    "goal_alignment_constraints": 15.0,
    "causal_effectiveness": 20.0,
    "executability": 15.0,
    "risk_and_fallback": 10.0,
    "evaluation_and_verification": 10.0,
    # B. Single-round self-evolution proxy (30)
    "self_check": 10.0,
    "proactive_optimization": 10.0,
    "learning_loop_design": 10.0,
}


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to [low, high]."""
    return max(low, min(high, value))


def normalize_dimension_scores(raw: Dict[str, Any] | None) -> Dict[str, float]:
    """Normalize raw dimension scores to full schema with [0, 5] range."""
    normalized: Dict[str, float] = {k: 0.0 for k in DIMENSION_WEIGHTS}
    if not raw:
        return normalized

    for key in DIMENSION_WEIGHTS:
        val = raw.get(key, 0.0)
        try:
            normalized[key] = clamp(float(val), 0.0, 5.0)
        except (TypeError, ValueError):
            normalized[key] = 0.0
    return normalized


def normalize_deductions(raw: Iterable[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    """Normalize deduction items, forcing points into [-40, 0]."""
    normalized: List[Dict[str, Any]] = []
    if not raw:
        return normalized

    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            points = float(item.get("points", 0.0))
        except (TypeError, ValueError):
            points = 0.0
        # Deductions are non-positive; cap a single deduction to rubric bounds.
        points = clamp(points, -40.0, 0.0)
        normalized.append(
            {
                "type": str(item.get("type", "unspecified")),
                "points": points,
                "reason": str(item.get("reason", "")),
            }
        )
    return normalized


def compute_weighted_total(
    dimension_scores: Dict[str, float],
    deductions: Iterable[Dict[str, Any]] | None = None,
) -> float:
    """Compute rubric total score in [0, 100]."""
    base = 0.0
    for key, weight in DIMENSION_WEIGHTS.items():
        base += (dimension_scores.get(key, 0.0) / 5.0) * weight

    penalty = 0.0
    for item in deductions or []:
        try:
            penalty += min(0.0, float(item.get("points", 0.0)))
        except (TypeError, ValueError):
            continue

    return clamp(base + penalty, 0.0, 100.0)


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize parsed scorer payload and fill required fields."""
    dims = normalize_dimension_scores(payload.get("dimension_scores"))
    deductions = normalize_deductions(payload.get("deductions"))

    total_from_dims = compute_weighted_total(dims, deductions)
    total_raw = payload.get("weighted_total_0_100", total_from_dims)
    try:
        total = clamp(float(total_raw), 0.0, 100.0)
    except (TypeError, ValueError):
        total = total_from_dims

    score_raw = payload.get("score", total / 100.0)
    try:
        score = clamp(float(score_raw), 0.0, 1.0)
    except (TypeError, ValueError):
        score = total / 100.0

    confidence_raw = payload.get("confidence", 0.5)
    try:
        confidence = clamp(float(confidence_raw), 0.0, 1.0)
    except (TypeError, ValueError):
        confidence = 0.5

    return {
        "rubric_version": RUBRIC_VERSION,
        "dimension_scores": dims,
        "deductions": deductions,
        "weighted_total_0_100": total,
        "score": score,
        "confidence": confidence,
    }

