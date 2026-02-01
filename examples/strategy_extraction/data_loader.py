# Copyright (c) Microsoft. All rights reserved.

"""Data loading and sampling module for strategy extraction training."""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


def load_problem_types(data_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load all problem types from the data directory.
    
    Scans the directory for problem type folders, loads JSON files,
    and returns examples organized by problem type.
    
    Args:
        data_dir: Path to the data directory containing problem type folders.
        
    Returns:
        Dictionary mapping problem type names to lists of examples.
        Each example contains 'input' and 'target' fields.
        
    Raises:
        FileNotFoundError: If the data directory does not exist.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    problem_types: Dict[str, List[Dict[str, Any]]] = {}
    
    # Iterate through subdirectories (problem types)
    for problem_type_dir in data_path.iterdir():
        if not problem_type_dir.is_dir():
            continue
        
        problem_type = problem_type_dir.name
        examples: List[Dict[str, Any]] = []
        
        # Load all JSON files in this problem type directory
        for json_file in problem_type_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                # Extract examples from the JSON structure
                if "examples" in data:
                    examples.extend(data["examples"])
                    
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue
        
        if examples:
            problem_types[problem_type] = examples
            logger.info(f"Loaded {len(examples)} examples for problem type: {problem_type}")
    
    logger.info(f"Total problem types loaded: {len(problem_types)}")
    return problem_types


def sample_fewshot_examples(
    problem_type_examples: List[Dict[str, Any]],
    n: int,
    exclude_indices: Set[int],
) -> Tuple[List[Dict[str, Any]], Set[int]]:
    """Sample n examples from a problem type, avoiding previously used indices.
    
    This function maintains sampling coverage by tracking used indices and
    resetting when necessary.
    
    Args:
        problem_type_examples: List of all examples for this problem type.
        n: Number of examples to sample.
        exclude_indices: Set of indices to exclude (already used).
        
    Returns:
        Tuple of (sampled_examples, updated_exclude_indices).
        
    Raises:
        ValueError: If n is larger than the total number of examples.
    """
    total_examples = len(problem_type_examples)
    
    if n > total_examples:
        raise ValueError(
            f"Requested {n} examples but only {total_examples} available"
        )
    
    # Calculate available indices
    all_indices = set(range(total_examples))
    available_indices = all_indices - exclude_indices
    
    # Reset exclude_indices if not enough available samples
    if len(available_indices) < n:
        logger.debug(
            f"Resetting exclude_indices for this problem type "
            f"(available: {len(available_indices)}, needed: {n})"
        )
        exclude_indices = set()
        available_indices = all_indices
    
    # Sample n indices randomly
    sampled_indices = random.sample(list(available_indices), n)
    
    # Get the actual examples
    sampled_examples = [problem_type_examples[i] for i in sampled_indices]
    
    # Update exclude_indices
    updated_exclude_indices = exclude_indices | set(sampled_indices)
    
    return sampled_examples, updated_exclude_indices


def create_strategy_dataset(
    data_dir: str,
    fewshot_min: int,
    fewshot_max: int,
    num_samples: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Create a dataset for strategy extraction training.
    
    Each sample contains:
    - problem_type: The type of problem
    - examples: Few-shot examples sampled from that type
    - num_shots: Number of examples included
    
    Args:
        data_dir: Path to the data directory.
        fewshot_min: Minimum number of few-shot examples per sample.
        fewshot_max: Maximum number of few-shot examples per sample.
        num_samples: Total number of samples to generate.
        seed: Random seed for reproducibility.
        
    Returns:
        List of dataset samples, each as a dictionary.
        
    Raises:
        ValueError: If data_dir contains no valid problem types or if
                   fewshot_max exceeds available examples.
    """
    random.seed(seed)
    
    # Load all problem types
    problem_types_data = load_problem_types(data_dir)
    
    if not problem_types_data:
        raise ValueError(f"No problem types found in {data_dir}")
    
    # Verify that all problem types have enough examples
    min_examples = min(len(examples) for examples in problem_types_data.values())
    if fewshot_max > min_examples:
        logger.warning(
            f"fewshot_max ({fewshot_max}) exceeds minimum available examples "
            f"({min_examples}). Some problem types may have limited examples."
        )
    
    # Initialize exclude_indices for each problem type
    exclude_indices_dict: Dict[str, Set[int]] = {
        problem_type: set() for problem_type in problem_types_data.keys()
    }
    
    dataset: List[Dict[str, Any]] = []
    problem_type_names = list(problem_types_data.keys())
    
    # Track sampling counts for balancing
    sample_counts: Dict[str, int] = {pt: 0 for pt in problem_type_names}
    
    logger.info(
        f"Creating dataset with {num_samples} samples, "
        f"few-shot range: [{fewshot_min}, {fewshot_max}]"
    )
    
    while len(dataset) < num_samples:
        # Choose problem type with lowest count (for balance)
        min_count = min(sample_counts.values())
        candidates = [pt for pt, count in sample_counts.items() if count == min_count]
        problem_type = random.choice(candidates)
        
        # Randomly choose number of shots
        n_shots = random.randint(fewshot_min, fewshot_max)
        
        # Ensure this problem type has enough examples
        problem_type_examples = problem_types_data[problem_type]
        if n_shots > len(problem_type_examples):
            logger.warning(
                f"Skipping {problem_type}: requested {n_shots} shots "
                f"but only {len(problem_type_examples)} available"
            )
            continue
        
        # Sample examples
        try:
            examples, updated_indices = sample_fewshot_examples(
                problem_type_examples,
                n_shots,
                exclude_indices_dict[problem_type],
            )
            
            exclude_indices_dict[problem_type] = updated_indices
            
            # Create dataset sample
            dataset.append({
                "problem_type": problem_type,
                "examples": examples,
                "num_shots": n_shots,
            })
            
            sample_counts[problem_type] += 1
            
        except ValueError as e:
            logger.error(f"Error sampling from {problem_type}: {e}")
            continue
    
    # Log sampling statistics
    logger.info(f"Dataset created with {len(dataset)} samples")
    logger.info(f"Sampling distribution: {sample_counts}")
    
    return dataset



