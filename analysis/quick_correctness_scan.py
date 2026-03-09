#!/usr/bin/env python3
"""Quick correctness scan across all problem types.

Example:
    python3 analysis/quick_correctness_scan.py \
        --base-dataset-path data/strategy_gen_train.jsonl \
        --n-per-type 50 \
        --answer-base-url http://localhost:8200/v1 \
        --output-path analysis/quick_scan_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import httpx


def _extract_answer(output: str) -> str:
    if not output:
        return ""
    m = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return output.strip()


def _to_str_target(raw: Any) -> str:
    if isinstance(raw, list):
        if raw:
            return str(raw[0])
        return ""
    return str(raw) if raw is not None else ""


def _compute_correctness(answer: str, ground_truth: str, numeric_tolerance: float = 0.02) -> float:
    if not answer or not ground_truth:
        return 0.0
    a = answer.strip().lower()
    g = ground_truth.strip().lower()
    if a == g:
        return 1.0
    try:
        af = float(a)
        gf = float(g)
        if abs(gf) < 1e-9:
            err = abs(af - gf)
        else:
            err = abs(af - gf) / abs(gf)
        if err < numeric_tolerance:
            return 0.8
    except Exception:
        pass
    return 0.0


def _pick_model_name(models_payload: Dict[str, Any]) -> str:
    data = models_payload.get("data", [])
    if isinstance(data, list) and data:
        model_id = data[0].get("id")
        if isinstance(model_id, str) and model_id:
            return model_id
    raise RuntimeError("Cannot infer model name from /models response")


def _load_dataset(path: str) -> List[Dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(path)
    items: List[Dict[str, Any]] = []
    if dataset_path.suffix.lower() == ".jsonl":
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    items.append(obj)
    else:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            items = [x for x in payload if isinstance(x, dict)]
        else:
            raise ValueError("Dataset JSON must be a list.")
    return items


def _extract_problem_fields(item: Dict[str, Any]) -> Dict[str, str]:
    problem_type = str(item.get("problem_type", "unknown"))
    problem = item.get("problem", item.get("input", ""))
    target = item.get("ground_truth", item.get("target", ""))
    return {
        "problem_type": problem_type,
        "problem": str(problem),
        "ground_truth": _to_str_target(target),
    }


async def run_scan(args: argparse.Namespace) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    rows = [_extract_problem_fields(x) for x in _load_dataset(args.base_dataset_path)]
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["problem"] and row["ground_truth"]:
            grouped[row["problem_type"]].append(row)

    sampled_rows: List[Dict[str, str]] = []
    for ptype, items in grouped.items():
        local = list(items)
        rng.shuffle(local)
        sampled_rows.extend(local[: min(args.n_per_type, len(local))])

    sem = asyncio.Semaphore(args.concurrency)
    results_by_type: Dict[str, List[float]] = defaultdict(list)

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        models_resp = await client.get(args.answer_base_url.rstrip("/") + "/models")
        models_resp.raise_for_status()
        answer_model = args.answer_model or _pick_model_name(models_resp.json())

        async def one_sample(sample: Dict[str, str]) -> None:
            async with sem:
                payload = {
                    "model": answer_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Solve the problem and output your final answer in "
                                "<answer>...</answer>."
                            ),
                        },
                        {
                            "role": "user",
                            "content": sample["problem"],
                        },
                    ],
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                }
                url = args.answer_base_url.rstrip("/") + "/chat/completions"
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                output = choices[0].get("message", {}).get("content", "") if choices else ""
                extracted = _extract_answer(output)
                score = _compute_correctness(extracted, sample["ground_truth"], numeric_tolerance=args.numeric_tolerance)
                results_by_type[sample["problem_type"]].append(score)

        await asyncio.gather(*[one_sample(sample) for sample in sampled_rows])

    per_type = []
    excluded = []
    for ptype in sorted(results_by_type.keys()):
        scores = results_by_type[ptype]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        row = {
            "problem_type": ptype,
            "n_samples": len(scores),
            "mean_correctness": round(mean_score, 4),
            "excluded": mean_score < args.exclude_threshold,
        }
        per_type.append(row)
        if row["excluded"]:
            excluded.append(ptype)

    report = {
        "base_dataset_path": str(Path(args.base_dataset_path).resolve()),
        "n_per_type": args.n_per_type,
        "sampled_total": sum(x["n_samples"] for x in per_type),
        "problem_type_count": len(per_type),
        "exclude_threshold": args.exclude_threshold,
        "excluded_problem_types": excluded,
        "per_type": per_type,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick scan of per-type correctness for exclusion list.")
    parser.add_argument("--base-dataset-path", type=str, required=True)
    parser.add_argument("--n-per-type", type=int, default=50)
    parser.add_argument("--answer-base-url", type=str, required=True)
    parser.add_argument("--answer-model", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--numeric-tolerance", type=float, default=0.02)
    parser.add_argument("--exclude-threshold", type=float, default=0.02)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=str, default="analysis/quick_scan_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(run_scan(args))
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

