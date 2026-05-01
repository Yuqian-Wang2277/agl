"""
Convert BIG-bench {input, target} training data to MetaICL k-shot format.

MetaICL training sample (full text, answer included):
    Input: {ex1.input}
    Output: {ex1.target}

    Input: {ex2.input}
    Output: {ex2.target}

    ...

    Input: {test.input}
    Output: {test.target}   <- loss computed only here (DataCollatorForCompletionOnlyLM)

CoT variant adds a Think: field before the final Output:
    ...
    Input: {test.input}
    Think: {reasoning}
    Output: {test.target}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "train_all_project_suitable"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_target(target) -> str:
    if isinstance(target, list):
        return str(target[0]) if target else ""
    return str(target)


def _is_valid(ex: dict) -> bool:
    return (
        isinstance(ex, dict)
        and str(ex.get("input", "")).strip() != ""
        and _norm_target(ex.get("target", "")).strip() != ""
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_task_examples(data_dir: Path = DEFAULT_DATA_DIR) -> Dict[str, List[dict]]:
    """Load all examples from all task-type directories.

    Returns:
        {task_name: [example_dict, ...]}
    """
    tasks: Dict[str, List[dict]] = {}
    for task_dir in sorted(data_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for json_file in sorted(task_dir.glob("*.json")):
            try:
                raw = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            examples = [ex for ex in raw.get("examples", []) if _is_valid(ex)]
            if examples:
                tasks.setdefault(task_dir.name, []).extend(examples)
    return tasks


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------

def make_metaicl_text(
    shot_examples: List[dict],
    test_example: dict,
    k: int = 4,
) -> str:
    """Full training text for MetaICL (answer included after last Output:)."""
    parts: List[str] = []
    for ex in shot_examples[:k]:
        parts.append(f"Input: {ex['input']}\nOutput: {_norm_target(ex['target'])}")
    parts.append(
        f"Input: {test_example['input']}\nOutput: {_norm_target(test_example['target'])}"
    )
    return "\n\n".join(parts)


def make_metaicl_cot_text(
    shot_examples: List[dict],
    test_example: dict,
    reasoning: str,
    k: int = 4,
) -> str:
    """Full training text for MetaICL-CoT (Think: field before last Output:)."""
    parts: List[str] = []
    for ex in shot_examples[:k]:
        parts.append(f"Input: {ex['input']}\nOutput: {_norm_target(ex['target'])}")
    parts.append(
        f"Input: {test_example['input']}\n"
        f"Think: {reasoning}\n"
        f"Output: {_norm_target(test_example['target'])}"
    )
    return "\n\n".join(parts)


def make_metaicl_prompt(
    shot_examples: List[dict],
    test_input: str,
    k: int = 4,
    cot: bool = False,
) -> str:
    """Inference prompt — ends just before the model's answer token(s).

    For MetaICL:   ...\\nOutput:
    For CoT:        ...\\nThink:   (model generates Think then Output itself)
    """
    parts: List[str] = []
    for ex in shot_examples[:k]:
        parts.append(f"Input: {ex['input']}\nOutput: {_norm_target(ex['target'])}")
    if cot:
        parts.append(f"Input: {test_input}\nThink:")
    else:
        parts.append(f"Input: {test_input}\nOutput:")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_metaicl_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    k: int = 4,
    samples_per_task: int = 2000,
    val_ratio: float = 0.05,
    seed: int = 42,
    cot_reasoning_fn: Optional[Callable[[List[dict], dict], str]] = None,
) -> Tuple[List[dict], List[dict]]:
    """Build balanced train / val splits for MetaICL SFT.

    Each task type contributes equally (samples_per_task train samples).
    Val split is stratified from a held-out pool within each task type.

    Args:
        data_dir:          Root of train_all_project_suitable/ (44 task dirs).
        k:                 Few-shot count per training sample.
        samples_per_task:  Number of training samples generated per task type.
        val_ratio:         Fraction of each task's examples held out for val.
        seed:              Random seed for reproducibility.
        cot_reasoning_fn:  If provided, generates CoT text for each sample.
                           Signature: fn(shot_examples, test_example) -> str

    Returns:
        (train_samples, val_samples) — each is List[{"text": str}]
    """
    rng = random.Random(seed)
    tasks = load_task_examples(data_dir)

    train_samples: List[dict] = []
    val_samples: List[dict] = []

    for task_name, examples in sorted(tasks.items()):
        if len(examples) < k + 1:
            continue

        rng.shuffle(examples)
        n_val_pool = max(k + 1, int(len(examples) * val_ratio))
        val_pool = examples[:n_val_pool]
        train_pool = examples[n_val_pool:]

        if len(train_pool) < k + 1:
            train_pool = examples  # too few examples — use all for train too

        # ── Train samples ─────────────────────────────────────────────────
        for _ in range(samples_per_task):
            if len(train_pool) < k + 1:
                break
            picked = rng.sample(train_pool, k + 1)
            shots, test_ex = picked[:k], picked[-1]

            if cot_reasoning_fn is not None:
                reasoning = cot_reasoning_fn(shots, test_ex)
                text = make_metaicl_cot_text(shots, test_ex, reasoning, k)
            else:
                text = make_metaicl_text(shots, test_ex, k)

            train_samples.append({"text": text})

        # ── Val samples (smaller, fixed) ──────────────────────────────────
        n_val_samples = max(10, samples_per_task // 20)
        pool = val_pool if len(val_pool) >= k + 1 else train_pool
        for _ in range(n_val_samples):
            picked = rng.sample(pool, k + 1)
            shots, test_ex = picked[:k], picked[-1]
            val_samples.append({"text": make_metaicl_text(shots, test_ex, k)})

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train, val = build_metaicl_dataset()
    print(f"Train: {len(train):,}  Val: {len(val):,}")
    print("\n--- Sample training example ---")
    print(train[0]["text"])
    print("\n--- Sample inference prompt ---")
    tasks = load_task_examples()
    task_examples = next(iter(tasks.values()))
    prompt = make_metaicl_prompt(task_examples[:4], task_examples[4]["input"])
    print(prompt)
