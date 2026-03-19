# Copyright (c) Microsoft. All rights reserved.

"""Prompt management for Actor-Judge Phase II training.

Prompts are stored as TOML files under category subdirectories, making it
trivial to iterate on wording without touching any Python code:

    prompt/
    ├── __init__.py                        ← this file (loader utilities)
    ├── strategy_generation/               ← Actor Stage-1: few-shot → <strategy>
    │   ├── fewshot_extract_v1.toml        ← baseline: induce strategy from examples
    │   └── fewshot_extract_v2.toml        ← copy v1 and modify
    ├── answer_generation/                 ← Actor Stage-2: strategy + Q → <answer>
    │   └── strategy_guided_v1.toml        ← baseline: strategy-conditioned answer
    └── judge_evaluation/                  ← Judge: context + Q + S → scalar score
        └── quality_scalar_v1.toml         ← baseline: strategy quality scoring

NAMING CONVENTION
    <purpose>_v<N>.toml
    - purpose: describes what the prompt does (not just "v1")
    - version: integer, increments for iterations on the same purpose

HOW TO ADD A NEW VERSION
    1. Copy  ``<category>/fewshot_extract_v1.toml``
          →  ``<category>/fewshot_extract_v2.toml``
    2. Edit  the ``system`` / ``user`` template strings.
    3. Select it via the ``version`` parameter:
         build_strategy_prompt(examples, version="fewshot_extract_v2")

TOML CONTRACT
    Every TOML file MUST define:
        system  (str)  — system message content
        user    (str)  — user message template; may contain {placeholders}

    Placeholder names used by each category:
        strategy_generation   {examples_text}
        answer_generation     {strategy}, {problem}
        judge_evaluation      {examples_text}, {question}, {strategy}
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Dict, List

_prompt_dir = Path(__file__).parent.absolute()


def load_prompt(category: str, version: str) -> Dict[str, str]:
    """Load a TOML prompt file and return its contents as a dict.

    Args:
        category: Subdirectory name, one of
                  ``"strategy_generation"``, ``"answer_generation"``,
                  ``"judge_evaluation"``.
        version:  File stem, e.g. ``"fewshot_extract_v1"``.

    Returns:
        Dict with at least ``"system"`` and ``"user"`` keys (raw strings).

    Raises:
        FileNotFoundError: If the TOML file does not exist.
    """
    path = _prompt_dir / category / f"{version}.toml"
    if not path.exists():
        available = list_versions(category)
        raise FileNotFoundError(
            f"Prompt '{category}/{version}' not found at {path}. "
            f"Available versions: {available}"
        )
    with open(path, "rb") as f:
        return tomllib.load(f)


def list_versions(category: str) -> list[str]:
    """Return sorted list of available prompt version stems in *category*."""
    cat_dir = _prompt_dir / category
    if not cat_dir.is_dir():
        return []
    return sorted(p.stem for p in cat_dir.glob("*.toml"))


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def format_examples(examples: List[Dict[str, Any]]) -> str:
    """Format few-shot examples into a numbered text block.

    Output::

        Example 1:
        Problem: <input>
        Solution: <target>

        Example 2:
        ...

    This string is substituted into the ``{examples_text}`` placeholder.
    """
    parts: List[str] = []
    for i, ex in enumerate(examples, 1):
        inp = ex.get("input", "")
        tgt = ex.get("target", "")
        if isinstance(tgt, list):
            tgt = tgt[0] if tgt else ""
        parts.append(f"Example {i}:")
        parts.append(f"Problem: {inp}")
        parts.append(f"Solution: {tgt}")
        parts.append("")
    return "\n".join(parts)
