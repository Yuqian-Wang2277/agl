"""Structured strategy parser and format reward helper.

This module implements the Phase-1 structured strategy schema parser.
It converts `<strategy>...</strategy>` text into a typed intermediate
representation and exposes a three-level format reward:

- 1.0: fully structured (TYPE + STEPS>=3 + CHECK>=1)
- 0.3: has strategy tags but fails structured parsing
- 0.0: no strategy block
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class StrategyIR:
    """Structured strategy representation."""

    type: str
    steps: list[str]
    checks: list[str]
    fallback: list[str]
    fmt: str = "structured"


_TYPE_RE = re.compile(r"(?:TYPE|类型)\s*[:：]\s*(.+)", re.IGNORECASE)
_STEPS_SEC = re.compile(
    r"(?:STEPS|步骤)\s*[:：](.*?)(?=(?:CHECK|验证|FALLBACK|备选)\s*[:：]|</strategy>|$)",
    re.DOTALL | re.IGNORECASE,
)
_CHECK_SEC = re.compile(
    r"(?:CHECK|验证)\s*[:：](.*?)(?=(?:FALLBACK|备选)\s*[:：]|</strategy>|$)",
    re.DOTALL | re.IGNORECASE,
)
_FALLBACK_SEC = re.compile(
    r"(?:FALLBACK|备选)\s*[:：](.*?)(?=</strategy>|$)",
    re.DOTALL | re.IGNORECASE,
)
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.+)", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*]\s+(.+)", re.MULTILINE)


def extract_strategy_block(output: str) -> Optional[str]:
    """Extract raw content inside `<strategy>...</strategy>`."""
    if not output:
        return None
    match = re.search(r"<strategy>(.*?)</strategy>", output, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).strip()
    return raw or None


def parse_strategy_ir(text: str) -> Optional[StrategyIR]:
    """Parse structured strategy text into `StrategyIR`.

    Returns `None` if required sections are missing or malformed.
    """
    if not text or not text.strip():
        return None

    type_match = _TYPE_RE.search(text)
    steps_match = _STEPS_SEC.search(text)
    check_match = _CHECK_SEC.search(text)
    fallback_match = _FALLBACK_SEC.search(text)

    if not type_match or not steps_match:
        return None

    steps = [step.strip() for step in _NUMBERED.findall(steps_match.group(1)) if step.strip()]
    checks = [item.strip() for item in _BULLET.findall(check_match.group(1)) if item.strip()] if check_match else []
    fallback = (
        [item.strip() for item in _BULLET.findall(fallback_match.group(1)) if item.strip()]
        if fallback_match
        else []
    )

    if len(steps) < 3 or len(checks) < 1:
        return None

    return StrategyIR(
        type=type_match.group(1).strip(),
        steps=steps,
        checks=checks,
        fallback=fallback,
    )


def compute_format_reward_structured(strategy_output: str) -> float:
    """Compute three-level format reward for structured strategy outputs."""
    raw = extract_strategy_block(strategy_output)
    if not raw:
        return 0.0
    return 1.0 if parse_strategy_ir(raw) is not None else 0.3
