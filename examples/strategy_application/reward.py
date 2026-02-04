# Copyright (c) Microsoft. All rights reserved.

"""Reward functions for strategy application training."""

import logging
import re
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_answer_format_reward(output: str) -> float:
    """Compute format reward based on answer tag presence and correctness.
    
    This function strictly checks for the presence of well-formed
    <answer>...</answer> tags in the output.
    
    Requirements for full reward (1.0):
    - Must have opening tag <answer>
    - Must have closing tag </answer>
    - Must have non-empty content between tags
    - Must have exactly one answer block (no multiple or nested tags)
    
    Args:
        output: Model output string to check.
        
    Returns:
        1.0 if format is correct (answer successfully extracted), 0.0 otherwise.
    """
    if not output:
        logger.debug("Empty output, reward: 0.0")
        return 0.0
    
    # Check for opening tag
    has_opening = "<answer>" in output
    
    # Check for closing tag
    has_closing = "</answer>" in output
    
    # If missing either tag, return 0.0
    if not has_opening or not has_closing:
        logger.debug(
            f"Missing tags - opening: {has_opening}, closing: {has_closing}, reward: 0.0"
        )
        return 0.0
    
    # Use regex to match answer blocks (DOTALL for multiline)
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, output, re.DOTALL)
    
    # Must have exactly one match
    if len(matches) != 1:
        logger.debug(f"Found {len(matches)} answer blocks, expected exactly 1, reward: 0.0")
        return 0.0
    
    # Check that the content is non-empty (not just whitespace)
    content = matches[0].strip()
    if not content:
        logger.debug("Answer block is empty, reward: 0.0")
        return 0.0
    
    # All checks passed
    logger.debug(f"Format reward: 1.0 (valid answer with {len(content)} chars)")
    return 1.0


def extract_answer(output: str) -> Optional[str]:
    """Extract answer content from <answer>...</answer> tags.
    
    This function extracts the content between <answer> and </answer> tags
    from the model output. If multiple answer blocks are found, returns the
    first one. If no valid answer block is found, returns None.
    
    Args:
        output: Model output string containing answer tags.
        
    Returns:
        Extracted answer content (stripped) if found, None otherwise.
    """
    if not output:
        return None
    
    # Use regex to match answer blocks (DOTALL for multiline)
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, output, re.DOTALL)
    
    if not matches:
        return None
    
    # Return the first match, stripped of whitespace
    content = matches[0].strip()
    
    # Return None if content is empty (only whitespace)
    if not content:
        return None
    
    return content


def _extract_strategy_steps(strategy: str) -> Optional[list[str]]:
    """Extract step texts from a strategy string using multiple patterns."""
    # Extract steps from strategy using various patterns
    step_patterns = [
        r"\d+\.\s+(.+?)(?=\n\d+\.|\Z)",  # "1. step", "2. step"
        r"Step\s+\d+[:\-]?\s*(.+?)(?=Step\s+\d+|\Z)",  # "Step 1:", "Step 2:"
        r"步骤\d+[：:\-]?\s*(.+?)(?=步骤\d+|\Z)",  # Chinese "步骤1："
    ]

    steps: list[str] = []
    for pattern in step_patterns:
        matches = re.findall(pattern, strategy, re.IGNORECASE | re.DOTALL)
        if matches:
            steps = [m.strip() for m in matches if m.strip()]
            if steps:
                break

    # If no structured steps found, try to extract numbered lines as fallback
    if not steps:
        lines = strategy.split("\n")
        for line in lines:
            line = line.strip()
            if re.match(r"^\d+[\.\)]\s+", line):
                step_content = re.sub(r"^\d+[\.\)]\s+", "", line)
                if step_content:
                    steps.append(step_content)

    if not steps:
        return None
    return steps


def compute_step_coverage(strategy: str, answer: str) -> float:
    """Compute step coverage score based on how many strategy steps appear in the answer.

    Returns:
        1.0 for full coverage,
        0.7 for >70%,
        0.4 for >40%,
        0.0 otherwise.
    """
    if not strategy or not answer:
        return 0.0

    steps = _extract_strategy_steps(strategy)
    if not steps:
        # If still no steps, assume strategy is followed if answer is non-empty
        logger.debug("No steps found in strategy, using fallback coverage check")
        return 1.0 if answer.strip() else 0.0

    matched_steps = 0
    answer_lower = answer.lower()
    for step in steps:
        step_lower = step.lower()
        step_words = [w for w in step_lower.split() if len(w) > 3]
        if not step_words:
            continue
        if any(word in answer_lower for word in step_words[:3]):
            matched_steps += 1

    match_ratio = matched_steps / len(steps) if steps else 0.0

    if match_ratio >= 1.0:
        return 1.0
    if match_ratio >= 0.7:
        return 0.7
    if match_ratio >= 0.4:
        return 0.4
    return 0.0


def compute_step_order_consistency(strategy: str, answer: str) -> float:
    """Compute step order consistency score.

    Checks whether the order of step-related content in the answer generally
    matches the order of steps in the strategy.

    Returns:
        1.0 if order is strictly preserved,
        0.5 if mildly violated,
        0.0 otherwise.
    """
    if not strategy or not answer:
        return 0.0

    steps = _extract_strategy_steps(strategy)
    if not steps:
        return 0.0

    answer_lower = answer.lower()
    positions: list[int] = []
    for step in steps:
        step_lower = step.lower()
        step_words = [w for w in step_lower.split() if len(w) > 3]
        pos_candidates = []
        for w in step_words[:3]:
            idx = answer_lower.find(w)
            if idx != -1:
                pos_candidates.append(idx)
        if pos_candidates:
            positions.append(min(pos_candidates))

    if len(positions) < 2:
        # Not enough information to judge order
        return 0.0

    violations = 0
    for i in range(1, len(positions)):
        if positions[i] < positions[i - 1]:
            violations += 1

    if violations == 0:
        return 1.0
    if violations <= 1:
        return 0.5
    return 0.0


def compute_entity_binding_consistency(strategy: str, problem: str, answer: str) -> float:
    """Compute a coarse entity/variable binding consistency score.

    This is a lightweight heuristic focusing mainly on numeric problems:
    - Checks whether numbers from the problem appear in the answer.
    - Approximates that numbers are combined in some way (not fully semantic).

    Returns:
        1.0: Most key entities appear and are reused.
        0.5: Some entities appear.
        0.0: Almost no entities from the problem appear in the answer.
    """
    if not problem or not answer:
        return 0.0

    # Extract numeric entities from problem
    problem_numbers = re.findall(r"-?\d+\.?\d*", problem)
    if not problem_numbers:
        # For non-numeric tasks we currently don't implement deep binding;
        # return a neutral 0.5 when answer is non-empty.
        return 0.5 if answer.strip() else 0.0

    answer_numbers = re.findall(r"-?\d+\.?\d*", answer)
    if not answer_numbers:
        return 0.0

    problem_set = set(problem_numbers)
    answer_set = set(answer_numbers)

    intersection = problem_set & answer_set
    if not intersection:
        return 0.0

    coverage_ratio = len(intersection) / len(problem_set)
    if coverage_ratio >= 0.8:
        return 1.0
    if coverage_ratio >= 0.4:
        return 0.5
    return 0.0


def compute_intermediate_reasoning_consistency(strategy: str, answer: str) -> float:
    """Compute whether intermediate reasoning steps are reflected in the answer.

    Heuristic:
    - If strategy mentions words like 'first', 'then', 'finally', or multiple
      numbered steps, we expect several clauses/sentences in the answer.
    - We use the number of sentences/segments as a proxy.

    Returns:
        1.0: Answer has rich, multi-step structure.
        0.5: Some structure present.
        0.0: Almost no structure (single short sentence).
    """
    if not strategy or not answer:
        return 0.0

    # Very simple segmentation by punctuation
    segments = re.split(r"[。\.\n]", answer)
    non_empty_segments = [s for s in segments if s.strip()]

    if len(non_empty_segments) >= 4:
        return 1.0
    if len(non_empty_segments) >= 2:
        return 0.5
    return 0.0


def compute_answer_correctness(
    answer: str,
    ground_truth: str,
    numeric_tolerance: float = 0.02,
    f1_threshold: float = 0.5,
) -> float:
    """Compute answer correctness reward using flexible matching.
    
    Tries multiple matching strategies:
    1. Exact match (normalized) = 1.0
    2. Numeric tolerance (if numeric, error < tolerance) = 0.8
    3. F1 score (if F1 > threshold) = 0.5
    4. Otherwise = 0.0
    
    Args:
        answer: Model's answer.
        ground_truth: Ground truth answer.
        numeric_tolerance: Relative tolerance for numeric comparison (default 2%).
        f1_threshold: F1 threshold for partial match (default 0.5).
        
    Returns:
        Correctness score: 1.0, 0.8, 0.5, or 0.0.
    """
    if not answer or not ground_truth:
        return 0.0
    
    # Normalize strings
    answer_norm = answer.strip().lower()
    ground_truth_norm = ground_truth.strip().lower()
    
    # 1. Try exact match (normalized)
    if answer_norm == ground_truth_norm:
        return 1.0
    
    # 2. Try numeric comparison
    try:
        answer_num = float(answer_norm)
        ground_truth_num = float(ground_truth_norm)
        
        # Calculate relative error
        if abs(ground_truth_num) < 1e-9:
            # Avoid division by zero
            relative_error = abs(answer_num - ground_truth_num)
        else:
            relative_error = abs(answer_num - ground_truth_num) / abs(ground_truth_num)
        
        if relative_error < numeric_tolerance:
            return 0.8
    except (ValueError, TypeError):
        # Not numeric, continue to other checks
        pass
    
    # 3. Try F1 score (simple word-level F1)
    def simple_f1(pred: str, gold: str) -> float:
        pred_words = set(pred.split())
        gold_words = set(gold.split())
        
        if not pred_words or not gold_words:
            return 0.0
        
        intersection = pred_words & gold_words
        if not intersection:
            return 0.0
        
        precision = len(intersection) / len(pred_words)
        recall = len(intersection) / len(gold_words)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    f1_score = simple_f1(answer_norm, ground_truth_norm)
    if f1_score >= f1_threshold:
        return 0.5
    
    # 4. No match
    return 0.0


def compute_stage2_reward(
    strategy: str,
    problem: str,
    answer: str,
    ground_truth: str,
    consistency_threshold: float,
    numeric_tolerance: float,
    f1_threshold: float,
    *,
    format_weight: float,
    consistency_weight: float,
    correctness_weight: float,
    coverage_weight: float,
    order_weight: float,
    binding_weight: float,
    intermediate_weight: float,
) -> Tuple[float, Dict[str, float]]:
    """Compute final reward for stage 2 training with decomposed, weighted signals.

    Reward components:
    - R_format      ∈ {0,1}
    - R_consistency ∈ [0,1] (from multiple sub-dimensions)
    - R_correctness ∈ [0,1]

    Final reward:
        R_train = format_weight      * R_format
                + consistency_weight * R_consistency
                + correctness_weight * R_correctness
    """
    # Step 1: Check answer format
    R_format = compute_answer_format_reward(answer)

    # Step 2: Extract answer content (even if format is 0, for logging/analysis)
    extracted_answer = extract_answer(answer) or ""

    # Step 3: Compute strategy consistency (multi-dimensional)
    if not strategy or not strategy.strip():
        C_coverage = 0.0
        C_order = 0.0
        C_binding = 0.0
        C_intermediate = 0.0
    else:
        C_coverage = compute_step_coverage(strategy, extracted_answer)
        C_order = compute_step_order_consistency(strategy, extracted_answer)
        C_binding = compute_entity_binding_consistency(strategy, problem, extracted_answer)
        C_intermediate = compute_intermediate_reasoning_consistency(strategy, extracted_answer)

    # Weighted total consistency
    weight_sum = max(coverage_weight + order_weight + binding_weight + intermediate_weight, 1e-8)
    R_consistency_raw = (
        coverage_weight * C_coverage
        + order_weight * C_order
        + binding_weight * C_binding
        + intermediate_weight * C_intermediate
    ) / weight_sum

    # Optional thresholding for logging (not hard gating for training)
    if R_consistency_raw < consistency_threshold:
        logger.debug(
            "Consistency below threshold: "
            f"total={R_consistency_raw:.3f}, threshold={consistency_threshold}"
        )

    # Step 4: Compute correctness
    correctness = compute_answer_correctness(
        extracted_answer,
        ground_truth,
        numeric_tolerance,
        f1_threshold,
    )

    # Step 5: Combine into final training reward
    final_reward = (
        format_weight * R_format
        + consistency_weight * R_consistency_raw
        + correctness_weight * correctness
    )

    details: Dict[str, float] = {
        "format": R_format,
        "coverage": C_coverage,
        "order": C_order,
        "binding": C_binding,
        "intermediate": C_intermediate,
        "consistency_total": R_consistency_raw,
        "correctness": correctness,
        "final": final_reward,
    }

    return final_reward, details

