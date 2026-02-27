# Copyright (c) Microsoft. All rights reserved.

"""Batch score generated strategies with Qwen3-8B using rubric prompts."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from openai import OpenAI

try:
    from ..prompt import format_examples, load_prompt
    from .rubric import RUBRIC_VERSION, clamp, normalize_payload
except ImportError:
    # Support direct script execution.
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from examples.strategy_extraction.prompt import format_examples, load_prompt
    from examples.strategy_extraction.strategy_scorer_training.rubric import (
        RUBRIC_VERSION,
        clamp,
        normalize_payload,
    )

logger = logging.getLogger(__name__)


class SafeDict(dict[str, Any]):
    """Dict that returns empty string for missing format keys."""

    def __missing__(self, key: str) -> str:
        return ""


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {idx}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {idx}: expected object, got {type(obj).__name__}")
            rows.append(obj)
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    """Write rows to JSONL."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    """Best-effort parse of a JSON object from model output."""
    cleaned = _strip_code_fence(text)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def _extract_numeric_score(text: str) -> float | None:
    """Fallback parse when JSON output is not available."""
    cleaned = _strip_code_fence(text)
    # Prefer standalone number lines.
    for line in cleaned.splitlines():
        line = line.strip()
        try:
            val = float(line)
            if val > 1.0 and val <= 100.0:
                val = val / 100.0
            return clamp(val, 0.0, 1.0)
        except ValueError:
            continue

    m = re.search(r"[-+]?\d*\.?\d+", cleaned)
    if not m:
        return None
    try:
        val = float(m.group(0))
    except ValueError:
        return None
    if val > 1.0 and val <= 100.0:
        val = val / 100.0
    return clamp(val, 0.0, 1.0)


def _format_examples_text(examples: Any) -> str:
    if isinstance(examples, list):
        return format_examples(examples) if examples else "(none)"
    return "(none)"


def build_messages(prompt_cfg: Dict[str, str], sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build chat messages for one scoring sample."""
    task = str(sample.get("task", "")).strip()
    strategy = str(sample.get("strategy", "")).strip()
    examples_text = _format_examples_text(sample.get("examples", []))

    format_values = SafeDict(
        task=task or "(none)",
        strategy=strategy,
        examples_text=examples_text,
    )

    system_prompt = prompt_cfg["system"].strip()
    user_prompt = prompt_cfg["user"].format_map(format_values).strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def score_one(
    client: OpenAI,
    prompt_cfg: Dict[str, str],
    model: str,
    sample: Dict[str, Any],
    *,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Run one scoring call and normalize output."""
    strategy = str(sample.get("strategy", "")).strip()
    if not strategy:
        return {
            "status": "invalid_input",
            "error": "Missing or empty `strategy` field.",
            "parsed": normalize_payload({}),
            "raw_output": "",
        }

    messages = build_messages(prompt_cfg, sample)
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw_output = response.choices[0].message.content or ""

    parsed_obj = _extract_json_object(raw_output)
    if parsed_obj is not None:
        normalized = normalize_payload(parsed_obj)
        return {
            "status": "ok",
            "parsed": normalized,
            "raw_output": raw_output,
        }

    numeric = _extract_numeric_score(raw_output)
    if numeric is None:
        return {
            "status": "parse_error",
            "error": "Could not parse JSON or numeric score from model output.",
            "parsed": normalize_payload({}),
            "raw_output": raw_output,
        }

    fallback_payload = {
        "score": numeric,
        "weighted_total_0_100": numeric * 100.0,
    }
    return {
        "status": "ok_numeric_fallback",
        "parsed": normalize_payload(fallback_payload),
        "raw_output": raw_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score strategies with Qwen3-8B using rubric prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-jsonl", type=str, required=True, help="Input JSONL path.")
    parser.add_argument("--output-jsonl", type=str, required=True, help="Output JSONL path.")
    parser.add_argument("--base-url", type=str, default="http://localhost:8100/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, default="qwen3-8b-scorer")
    parser.add_argument("--prompt-version", type=str, default="v2")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_path = Path(args.output_jsonl)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {output_path}. Use --overwrite to replace it."
        )

    prompt_cfg = load_prompt("strategy_scoring", args.prompt_version)
    rows = read_jsonl(args.input_jsonl)
    if args.limit > 0:
        rows = rows[: args.limit]

    logger.info("Rubric version: %s", RUBRIC_VERSION)
    logger.info("Loaded %d rows from %s", len(rows), args.input_jsonl)
    logger.info("Scoring model: %s (%s)", args.model, args.base_url)

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
    )

    outputs: List[Dict[str, Any]] = []
    for idx, sample in enumerate(rows, start=1):
        sample_id = sample.get("id", idx)
        try:
            result = score_one(
                client,
                prompt_cfg,
                args.model,
                sample,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:
            result = {
                "status": "api_error",
                "error": str(exc),
                "parsed": normalize_payload({}),
                "raw_output": "",
            }

        output_row = {
            "id": sample_id,
            "task": sample.get("task", ""),
            "strategy": sample.get("strategy", ""),
            "rubric_version": RUBRIC_VERSION,
            "result": result,
        }
        outputs.append(output_row)

        if idx % 20 == 0 or idx == len(rows):
            logger.info("Processed %d/%d", idx, len(rows))

    write_jsonl(args.output_jsonl, outputs)
    logger.info("Saved %d scored rows to %s", len(outputs), args.output_jsonl)


if __name__ == "__main__":
    main()

