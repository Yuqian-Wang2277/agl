# Copyright (c) Microsoft. All rights reserved.

"""Data loading and sampling module for strategy application training."""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from examples.strategy_extraction.data_loader import (
    load_problem_types,
    sample_fewshot_examples,
)

logger = logging.getLogger(__name__)


def create_strategy_application_dataset(
    data_dir: str,
    fewshot_min: int,
    fewshot_max: int,
    num_samples: int,
    mode: str = "same_domain",
    seed: int = 42,
    similarity_matcher: Optional[Any] = None,
    oversample_ratio: float = 1.2,
    checkpoint_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Create a dataset for strategy application training.
    
    Each sample contains:
    - fewshot_examples: Few-shot examples used to extract strategy online
    - problem: Problem to solve
    - ground_truth: Correct answer
    - problem_type: Target problem type
    - source_problem_type: Source problem type for few-shot (for cross-domain mode)
    
    Args:
        data_dir: Path to the data directory.
        fewshot_min: Minimum number of few-shot examples per sample.
        fewshot_max: Maximum number of few-shot examples per sample.
        num_samples: Total number of samples to generate.
        mode: "same_domain" or "cross_domain".
        seed: Random seed for reproducibility.
        oversample_ratio: Ratio to oversample (e.g., 1.2 means sample 20% more, then filter).
        similarity_matcher: SimilarityMatcher instance (required for advanced cross-domain mode).
        checkpoint_dir: Directory to save dataset info (optional).
        
    Returns:
        List of dataset samples, each as a dictionary.
        
    Raises:
        ValueError: If mode is invalid or required components are missing.
    """
    if mode not in ["same_domain", "cross_domain"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'same_domain' or 'cross_domain'")
    
    random.seed(seed)
    
    # Load all problem types
    logger.info(f"Loading problem types from: {data_dir}")
    try:
        problem_types_data = load_problem_types(data_dir)
    except Exception as e:
        logger.error(f"Failed to load problem types from {data_dir}: {e}", exc_info=True)
        raise
    
    if not problem_types_data:
        raise ValueError(f"No problem types found in {data_dir}")
    
    logger.info(f"Successfully loaded {len(problem_types_data)} problem types")
    
    problem_type_names = list(problem_types_data.keys())
    
    # Calculate target samples with oversampling
    target_samples = int(num_samples * oversample_ratio)
    logger.info(
        f"Creating strategy application dataset with {num_samples} samples "
        f"(oversampling to {target_samples} for quality filtering), "
        f"mode: {mode}, few-shot range: [{fewshot_min}, {fewshot_max}]"
    )
    
    dataset: List[Dict[str, Any]] = []
    failed_extractions = 0
    failed_matches = 0
    skipped_invalid_samples = 0
    
    # Initialize exclude_indices for each problem type
    exclude_indices_dict: Dict[str, Set[int]] = {
        problem_type: set() for problem_type in problem_type_names
    }
    
    # Track sampling counts per target problem type
    sample_counts: Dict[str, int] = {pt: 0 for pt in problem_type_names}
    
    # Protection against infinite loops: track consecutive failures
    consecutive_failures = 0
    max_consecutive_failures = 1000
    
    # Pre-compute flattened source examples for cross-domain mode
    # Mapping from text -> list of (problem_type, example_dict) tuples
    cross_domain_source_index: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    all_source_texts: List[str] = []
    if mode == "cross_domain":
        for pt, examples in problem_types_data.items():
            for ex in examples:
                text = str(ex.get("input", ""))
                if not text:
                    continue
                all_source_texts.append(text)
                cross_domain_source_index.setdefault(text, []).append((pt, ex))
    
    while len(dataset) < target_samples:
        if mode == "same_domain":
            # Same domain: few-shot and problem from the same problem_type
            problem_type = random.choice(problem_type_names)

            # Sample few-shot examples for this problem type
            n_shots = random.randint(fewshot_min, fewshot_max)
            problem_type_examples = problem_types_data[problem_type]

            if n_shots > len(problem_type_examples):
                logger.warning(
                    f"Skipping {problem_type}: requested {n_shots} shots "
                    f"but only {len(problem_type_examples)} available"
                )
                continue

            try:
                fewshot_examples, updated_indices = sample_fewshot_examples(
                    problem_type_examples,
                    n_shots,
                    exclude_indices_dict[problem_type],
                )
                exclude_indices_dict[problem_type] = updated_indices
                
                # Validate few-shot examples have valid input/target
                valid_fewshot = []
                for ex in fewshot_examples:
                    ex_input = ex.get("input", "")
                    ex_target = ex.get("target", [])
                    if ex_input and ex_input.strip():
                        if isinstance(ex_target, list):
                            if ex_target and ex_target[0]:
                                valid_fewshot.append(ex)
                        elif ex_target:
                            valid_fewshot.append(ex)
                
                if not valid_fewshot:
                    skipped_invalid_samples += 1
                    consecutive_failures += 1
                    logger.debug(f"Skipping {problem_type}: no valid few-shot examples after filtering")
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Too many consecutive failures ({consecutive_failures}), stopping dataset creation")
                        break
                    continue
                
                fewshot_examples = valid_fewshot

                # Sample a problem to solve (different from few-shot examples)
                remaining_examples = [
                    ex for i, ex in enumerate(problem_type_examples)
                    if i not in exclude_indices_dict[problem_type]
                ]

                if not remaining_examples:
                    # Reset if exhausted
                    exclude_indices_dict[problem_type] = set()
                    remaining_examples = problem_type_examples

                problem_example = random.choice(remaining_examples)
                problem = problem_example.get("input", "")
                target_value: Any = problem_example.get("target", [])
                if isinstance(target_value, list):
                    ground_truth_final: str = target_value[0] if target_value else ""
                else:
                    ground_truth_final = str(target_value)
                
                # Skip samples with empty problem or ground_truth
                if not problem or not problem.strip():
                    skipped_invalid_samples += 1
                    consecutive_failures += 1
                    logger.debug(f"Skipping sample with empty problem in {problem_type}")
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Too many consecutive failures ({consecutive_failures}), stopping dataset creation")
                        break
                    continue
                if not ground_truth_final or not ground_truth_final.strip():
                    skipped_invalid_samples += 1
                    consecutive_failures += 1
                    logger.debug(f"Skipping sample with empty ground_truth in {problem_type}")
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Too many consecutive failures ({consecutive_failures}), stopping dataset creation")
                        break
                    continue

                dataset.append(
                    {
                        "fewshot_examples": fewshot_examples,
                        "problem": problem,
                        "ground_truth": ground_truth_final,
                        "problem_type": problem_type,
                        "source_problem_type": None,  # Same domain
                    }
                )

                sample_counts[problem_type] += 1
                consecutive_failures = 0  # Reset on success

            except Exception as e:
                logger.error(f"Error creating sample for {problem_type}: {e}")
                continue

        else:  # cross_domain
            # Cross-domain: few-shot from (possibly different) source domain, problem from target domain
            target_problem_type = random.choice(problem_type_names)

            # Sample a problem from target domain
            target_examples = problem_types_data[target_problem_type]
            if not target_examples:
                continue

            problem_example = random.choice(target_examples)
            problem = problem_example.get("input", "")
            target_value_cross: Any = problem_example.get("target", [])
            if isinstance(target_value_cross, list):
                ground_truth_final_cross: str = target_value_cross[0] if target_value_cross else ""
            else:
                ground_truth_final_cross = str(target_value_cross)
            
            # Skip samples with empty problem or ground_truth
            if not problem or not problem.strip():
                skipped_invalid_samples += 1
                consecutive_failures += 1
                logger.debug(f"Skipping cross-domain sample with empty problem in {target_problem_type}")
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(f"Too many consecutive failures ({consecutive_failures}), stopping dataset creation")
                    break
                continue
            if not ground_truth_final_cross or not ground_truth_final_cross.strip():
                skipped_invalid_samples += 1
                consecutive_failures += 1
                logger.debug(f"Skipping cross-domain sample with empty ground_truth in {target_problem_type}")
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(f"Too many consecutive failures ({consecutive_failures}), stopping dataset creation")
                    break
                continue

            # If we don't have a similarity matcher or precomputed texts, fall back to random source few-shot
            fewshot_examples: List[Dict[str, Any]] = []
            source_problem_type: Optional[str] = None

            if similarity_matcher is not None and all_source_texts:
                # Use embedding-based matching: problem text vs. all source example inputs
                try:
                    top_k = max(fewshot_min, min(fewshot_max, 5))
                    best_matches = similarity_matcher.find_best_strategy(
                        problem,
                        all_source_texts,
                        top_k=top_k,
                    )

                    for text, _score in best_matches:
                        for pt, ex in cross_domain_source_index.get(text, []):
                            fewshot_examples.append(ex)
                            # Use the first problem type as the nominal source
                            if source_problem_type is None:
                                source_problem_type = pt
                            if len(fewshot_examples) >= fewshot_max:
                                break
                        if len(fewshot_examples) >= fewshot_max:
                            break
                except Exception as e:
                    failed_matches += 1
                    logger.warning(f"Similarity-based few-shot selection failed: {e}")
                    fewshot_examples = []

            # Fallback: random few-shot from a random source problem type
            if not fewshot_examples:
                source_problem_type = random.choice(problem_type_names)
                source_examples = problem_types_data[source_problem_type]
                if len(source_examples) < fewshot_min:
                    logger.warning(
                        f"Skipping cross-domain sample: source {source_problem_type} has "
                        f"only {len(source_examples)} examples (< fewshot_min={fewshot_min})"
                    )
                    continue
                n_shots = random.randint(fewshot_min, min(fewshot_max, len(source_examples)))
                fewshot_examples = random.sample(source_examples, n_shots)
            
            # Validate few-shot examples are not empty
            if not fewshot_examples:
                logger.debug(f"Skipping cross-domain sample: no few-shot examples available")
                continue
            
            # Validate that few-shot examples have valid input/target
            valid_fewshot = []
            for ex in fewshot_examples:
                ex_input = ex.get("input", "")
                ex_target = ex.get("target", [])
                if ex_input and ex_input.strip():
                    if isinstance(ex_target, list):
                        if ex_target and ex_target[0]:
                            valid_fewshot.append(ex)
                    elif ex_target:
                        valid_fewshot.append(ex)
            
            if not valid_fewshot:
                skipped_invalid_samples += 1
                consecutive_failures += 1
                logger.debug(f"Skipping cross-domain sample: no valid few-shot examples")
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(f"Too many consecutive failures ({consecutive_failures}), stopping dataset creation")
                    break
                continue
            
            fewshot_examples = valid_fewshot

            dataset.append(
                {
                    "fewshot_examples": fewshot_examples,
                    "problem": problem,
                    "ground_truth": ground_truth_final_cross,
                    "problem_type": target_problem_type,
                    "source_problem_type": source_problem_type,
                }
            )

            sample_counts[target_problem_type] += 1
            consecutive_failures = 0  # Reset on success
    
    # Filter to target size if oversampled
    if len(dataset) > num_samples:
        logger.info(f"Oversampled to {len(dataset)} samples, filtering to {num_samples} best samples")
        # Simple filtering: keep first num_samples (they're already randomized)
        dataset = dataset[:num_samples]
    
    # Log sampling statistics
    logger.info(f"Dataset created with {len(dataset)} samples")
    logger.info(f"Sampling distribution: {sample_counts}")
    logger.info(f"Quality statistics: {failed_extractions} failed extractions, {failed_matches} failed matches, {skipped_invalid_samples} skipped invalid samples")
    
    if len(dataset) < num_samples:
        logger.warning(
            f"Dataset size ({len(dataset)}) is less than requested ({num_samples}). "
            f"This may be due to invalid samples in the data. Consider checking data quality."
        )
    
    # Save dataset info to checkpoint directory if provided
    if checkpoint_dir:
        dataset_info = {
            "mode": mode,
            "num_samples": len(dataset),
            "target_samples": num_samples,
            "oversample_ratio": oversample_ratio,
            "fewshot_min": fewshot_min,
            "fewshot_max": fewshot_max,
            "seed": seed,
            "sample_counts": sample_counts,
            "quality_stats": {
                "failed_extractions": failed_extractions,
                "failed_matches": failed_matches,
            },
        }
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        info_file = os.path.join(checkpoint_dir, "dataset_info.json")
        try:
            with open(info_file, "w", encoding="utf-8") as f:
                json.dump(dataset_info, f, ensure_ascii=False, indent=2)
            logger.info(f"Dataset info saved to: {info_file}")
        except Exception as e:
            logger.warning(f"Failed to save dataset info: {e}")
    
    return dataset

