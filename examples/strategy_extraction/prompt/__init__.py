# Copyright (c) Microsoft. All rights reserved.

"""Prompt management for strategy generation training.

This package organises prompts as TOML files under category subdirectories,
following the same pattern used by LLMReflection::

    prompt/
    ├── __init__.py               ← this file (loader utilities)
    ├── strategy_generation/      ← prompts for the TRACED strategy-gen call
    │   ├── v1.toml
    │   └── v2.toml               ← copy v1.toml and modify
    └── answer_generation/        ← prompts for the UN-TRACED answer-gen call
        ├── v1.toml
        └── v2.toml

HOW TO ADD A NEW VERSION:
    1. Copy ``<category>/v1.toml`` → ``<category>/v2.toml``.
    2. Edit the ``system`` / ``user`` template strings as needed.
    3. Select it at training time via CLI:
       ``--strategy-prompt-version v2`` or ``--answer-prompt-version v2``.

TOML files must define at least ``system`` and ``user`` keys.
The ``user`` template may contain ``{placeholders}`` that the agent fills
at runtime (e.g. ``{examples_text}``, ``{strategy}``, ``{problem}``).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Dict, List

prompt_dir = Path(__file__).parent.absolute()


def load_prompt(category: str, version: str) -> Dict[str, str]:
    """Load a TOML prompt file and return its contents as a dict.

    Args:
        category: Subdirectory name (e.g. ``"strategy_generation"``).
        version: File stem (e.g. ``"v1"``).

    Returns:
        Dict with at least ``system`` and ``user`` keys.

    Raises:
        FileNotFoundError: If the TOML file does not exist.
    """
    path = prompt_dir / category / f"{version}.toml"
    if not path.exists():
        available = list_versions(category)
        raise FileNotFoundError(
            f"Prompt '{category}/{version}' not found at {path}. "
            f"Available versions: {available}"
        )
    with open(path, "rb") as f:
        return tomllib.load(f)


def list_versions(category: str) -> list[str]:
    """List available prompt versions (TOML file stems) in *category*."""
    cat_dir = prompt_dir / category
    if not cat_dir.is_dir():
        return []
    return sorted(p.stem for p in cat_dir.glob("*.toml"))


def format_examples(examples: List[Dict[str, Any]]) -> str:
    """Serialize a list of few-shot examples into a text block.

    Each example is rendered as::

        Example 1:
        Problem: <input>
        Solution: <target>

    This output is intended to be substituted into the ``{examples_text}``
    placeholder of a strategy-generation user prompt.
    """
    parts: list[str] = []
    for i, ex in enumerate(examples, 1):
        input_text = ex.get("input", "")
        target = ex.get("target", [])
        if isinstance(target, list):
            target_text = target[0] if target else ""
        else:
            target_text = str(target)
        parts.append(f"Example {i}:")
        parts.append(f"Problem: {input_text}")
        parts.append(f"Solution: {target_text}")
        parts.append("")
    return "\n".join(parts)
