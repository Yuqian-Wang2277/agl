"""Prompt-building API for Actor-Judge Phase II.

This module is a thin Python layer on top of the TOML-based prompt package
(``prompt/``).  Prompt text lives in TOML files so it can be iterated without
touching any Python code.  See ``prompt/__init__.py`` for the full guide.

Three prompt families, one function each:

    build_strategy_prompt(fewshot_examples, version="fewshot_extract_v1")
        → List[{role, content}]  for tokenizer.apply_chat_template
        → Actor Stage-1: few-shot context → <strategy>…</strategy>

    build_answer_prompt(strategy, question, version="strategy_guided_v1")
        → List[{role, content}]
        → Actor Stage-2: strategy + Q' → <answer>…</answer>

    build_judge_prompt(fewshot_examples, question, strategy,
                       version="quality_scalar_v1", context_text_raw="")
        → str  (raw text, NOT chat-template messages)
        → Judge input ending with the <|judge|> anchor token
        → For parity with rollout / ODVA, pass ``context_text_raw=`` output of
          ``judge_rollout_context_text(fewshot_examples)`` (same as
          ``Experience.context_text``).  Omitting it falls back to
          ``format_examples()``, which is a different layout (train–eval skew).

Sentinel tokens (used by rollout_engine and env):

    STRATEGY_OPEN / STRATEGY_CLOSE   "<strategy>" / "</strategy>"
    ANSWER_OPEN   / ANSWER_CLOSE     "<answer>"   / "</answer>"
    JUDGE_TOKEN                      "<|judge|>"

Prompt file naming convention:
    prompt/<family>/<purpose>_v<N>.toml
    e.g. prompt/strategy_generation/fewshot_extract_v1.toml

Quick iteration guide:
    1. cp prompt/strategy_generation/fewshot_extract_v1.toml \
          prompt/strategy_generation/fewshot_extract_v2.toml
    2. Edit v2.toml
    3. Pass version="fewshot_extract_v2" to build_strategy_prompt(), or set
       cfg.strategy_prompt_version = "fewshot_extract_v2" in ActorJudgeConfig.
"""

from __future__ import annotations

from typing import Any, Dict, List

from prompt import format_examples, load_prompt


# ---------------------------------------------------------------------------
# Sentinel tokens — imported throughout the codebase
# ---------------------------------------------------------------------------

STRATEGY_OPEN  = "<strategy>"
STRATEGY_CLOSE = "</strategy>"
ANSWER_OPEN    = "<answer>"
ANSWER_CLOSE   = "</answer>"
JUDGE_TOKEN    = "<|judge|>"     # registered in tokenizer before first use


def judge_rollout_context_text(fewshot_examples: List[Dict[str, Any]]) -> str:
    """Build the same few-shot prefix string stored in ``Experience.context_text``.

    Must stay byte-for-byte aligned with ``rollout_engine`` so Judge prompts match
    ODVA / dense-reward training (avoids train–eval skew vs ``format_examples``).
    """
    lines: List[str] = []
    for ex in fewshot_examples or []:
        inp = ex.get("input", "")
        tgt = ex.get("target", "")
        if isinstance(tgt, list):
            tgt = tgt[0] if tgt else ""
        lines.append(f"Q: {inp}  A: {tgt}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage-1: Strategy generation
# ---------------------------------------------------------------------------

def build_strategy_prompt(
    fewshot_examples: List[Dict[str, Any]],
    version: str = "fewshot_extract_v1",
) -> List[Dict[str, str]]:
    """Build chat messages for Actor Stage-1 (strategy generation).

    Args:
        fewshot_examples: List of {"input": str, "target": str|list} dicts.
        version:          Prompt version to load from
                          ``prompt/strategy_generation/<version>.toml``.

    Returns:
        List of {"role": ..., "content": ...} dicts suitable for
        ``tokenizer.apply_chat_template``.
    """
    tmpl = load_prompt("strategy_generation", version)
    examples_text = format_examples(fewshot_examples)
    user_content  = tmpl["user"].format(examples_text=examples_text)
    return [
        {"role": "system", "content": tmpl["system"].strip()},
        {"role": "user",   "content": user_content.strip()},
    ]


# ---------------------------------------------------------------------------
# Stage-2: Answer generation
# ---------------------------------------------------------------------------

def build_answer_prompt(
    strategy: str,
    question: str,
    version: str = "strategy_guided_v1",
) -> List[Dict[str, str]]:
    """Build chat messages for Actor Stage-2 (answer generation).

    Args:
        strategy: Strategy text from Stage-1. Outer <strategy> tags are
                  stripped automatically so the model receives clean content.
        question: New question Q' the actor must answer.
        version:  Prompt version to load from
                  ``prompt/answer_generation/<version>.toml``.

    Returns:
        List of {"role": ..., "content": ...} dicts.
    """
    # Strip outer tags so the model sees clean strategy body
    body = strategy
    if STRATEGY_OPEN in body:
        body = body.split(STRATEGY_OPEN, 1)[-1]
    if STRATEGY_CLOSE in body:
        body = body.rsplit(STRATEGY_CLOSE, 1)[0]
    body = body.strip()

    tmpl = load_prompt("answer_generation", version)
    user_content = tmpl["user"].format(strategy=body, problem=question)
    return [
        {"role": "system", "content": tmpl["system"].strip()},
        {"role": "user",   "content": user_content.strip()},
    ]


# ---------------------------------------------------------------------------
# Judge input
# ---------------------------------------------------------------------------

def build_judge_prompt(
    fewshot_examples: List[Dict[str, Any]],
    question: str,
    strategy: str,
    version: str = "quality_scalar_v1",
    context_text_raw: str = "",
) -> str:
    """Build the raw-text Judge input ending with the <|judge|> anchor token.

    The Judge does NOT use chat-template messages because we need the
    <|judge|> token to be literally the last token in the sequence so that
    ``attention_mask.sum(dim=1) - 1`` always points to it.

    Args:
        fewshot_examples: Task context examples.  Ignored when
                          ``context_text_raw`` is non-empty.
        question:         New question Q'.
        strategy:         Candidate strategy S to be evaluated.
        version:          Prompt version to load from
                          ``prompt/judge_evaluation/<version>.toml``.
        context_text_raw: Pre-formatted context string (e.g. from
                          ``Experience.context_text``).  When provided,
                          ``fewshot_examples`` and ``format_examples()`` are
                          bypassed entirely, avoiding a list-of-dicts round-trip.

    Returns:
        A single string.  The caller must tokenise this directly
        (NOT via apply_chat_template).
    """
    tmpl = load_prompt("judge_evaluation", version)
    if context_text_raw:
        examples_text = context_text_raw
    elif fewshot_examples:
        examples_text = format_examples(fewshot_examples)
    else:
        examples_text = ""
    user_content = tmpl["user"].format(
        examples_text=examples_text,
        question=question,
        strategy=strategy,
    )
    # Combine system + user as plain text, then append <|judge|> anchor
    prompt = (
        f"{tmpl['system'].strip()}\n\n"
        f"{user_content.strip()}\n"
        f"{JUDGE_TOKEN}"
    )
    return prompt


# ---------------------------------------------------------------------------
# Utility used by rollout_engine and train.py
# ---------------------------------------------------------------------------

def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Render chat messages to a single string via the tokenizer."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
