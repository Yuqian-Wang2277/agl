#!/usr/bin/env python3
"""Throughput benchmark for Phase-2 grounded proxy rollout cost.

Example:
    python3 analysis/throughput_benchmark.py \
        --n-prompts 100 \
        --grounded-proxy-k 4 \
        --answer-base-url http://localhost:8200/v1 \
        --scorer-base-url http://localhost:8100/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Any, Dict, List, Sequence

import httpx


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


async def _resolve_model(base_url: str, fallback: str) -> str:
    if fallback:
        return fallback
    url = base_url.rstrip("/") + "/models"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
    model_list = data.get("data", [])
    if not isinstance(model_list, list) or not model_list:
        raise RuntimeError(f"No models found at {url}")
    model_id = model_list[0].get("id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"Invalid model payload at {url}")
    return model_id


async def _chat_once(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = await client.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "") or ""


def _answer_messages(i: int) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Solve the problem and output only <answer>...</answer>.",
        },
        {
            "role": "user",
            "content": f"Problem: What is {i} + {i+1}? Return only <answer>...</answer>.",
        },
    ]


def _scorer_messages(i: int) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Evaluate strategy quality using dimensions A_outcome_support, "
                "B_executability, C_example_grounding, D_problem_coverage, "
                "E_transfer_robustness, F_clarity_economy (0-5). "
                "Output JSON with dimension_scores/cap_flags/penalties/verdict only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Few-shot examples: arithmetic examples.\n"
                "Strategy: identify operands, compute sum, verify.\n"
                f"Problem: What is {i} + {i+1}?\n"
                f"Generated answer: <answer>{2*i+1}</answer>\n"
                "Answer was: correct"
            ),
        },
    ]


async def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    answer_model = await _resolve_model(args.answer_base_url, args.answer_model)
    scorer_model = await _resolve_model(args.scorer_base_url, args.scorer_model)

    sem = asyncio.Semaphore(args.concurrency)
    answer_latencies: List[float] = []
    scorer_latencies: List[float] = []
    hybrid_rollout_latencies: List[float] = []
    baseline_rollout_latencies: List[float] = []

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        async def one_prompt(i: int) -> None:
            async with sem:
                # Baseline scorer-only rollout (for slowdown estimate)
                t0 = time.perf_counter()
                s0 = time.perf_counter()
                await _chat_once(
                    client,
                    args.scorer_base_url,
                    scorer_model,
                    _scorer_messages(i),
                    max_tokens=args.scorer_max_tokens,
                    temperature=0.0,
                )
                scorer_latencies.append(time.perf_counter() - s0)
                baseline_rollout_latencies.append(time.perf_counter() - t0)

                # Hybrid rollout: K answer calls + 1 scorer call
                t1 = time.perf_counter()

                async def one_answer() -> None:
                    t_a = time.perf_counter()
                    await _chat_once(
                        client,
                        args.answer_base_url,
                        answer_model,
                        _answer_messages(i),
                        max_tokens=args.answer_max_tokens,
                        temperature=args.answer_temperature,
                    )
                    answer_latencies.append(time.perf_counter() - t_a)

                await asyncio.gather(*[one_answer() for _ in range(args.grounded_proxy_k)])

                s1 = time.perf_counter()
                await _chat_once(
                    client,
                    args.scorer_base_url,
                    scorer_model,
                    _scorer_messages(i),
                    max_tokens=args.scorer_max_tokens,
                    temperature=0.0,
                )
                scorer_latencies.append(time.perf_counter() - s1)

                hybrid_rollout_latencies.append(time.perf_counter() - t1)

        start = time.perf_counter()
        await asyncio.gather(*[one_prompt(i) for i in range(args.n_prompts)])
        elapsed = time.perf_counter() - start

    answer_req = args.n_prompts * args.grounded_proxy_k
    scorer_req = args.n_prompts * 2  # one baseline + one hybrid per prompt

    hybrid_p50 = _percentile(hybrid_rollout_latencies, 0.50)
    hybrid_p95 = _percentile(hybrid_rollout_latencies, 0.95)
    baseline_p50 = _percentile(baseline_rollout_latencies, 0.50)
    baseline_p95 = _percentile(baseline_rollout_latencies, 0.95)
    slowdown_ratio = (hybrid_p50 / baseline_p50) if baseline_p50 > 0 else 0.0

    return {
        "n_prompts": args.n_prompts,
        "grounded_proxy_k": args.grounded_proxy_k,
        "answer_model": answer_model,
        "scorer_model": scorer_model,
        "single_rollout_latency_p50": hybrid_p50,
        "single_rollout_latency_p95": hybrid_p95,
        "baseline_scorer_only_latency_p50": baseline_p50,
        "baseline_scorer_only_latency_p95": baseline_p95,
        "answer_model_qps_utilized": answer_req / elapsed if elapsed > 0 else 0.0,
        "scorer_model_qps_utilized": scorer_req / elapsed if elapsed > 0 else 0.0,
        "estimated_training_slowdown_ratio": slowdown_ratio,
        "answer_latency_mean": statistics.mean(answer_latencies) if answer_latencies else 0.0,
        "scorer_latency_mean": statistics.mean(scorer_latencies) if scorer_latencies else 0.0,
        "elapsed_seconds": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark throughput impact of K-sample grounded proxy.")
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--grounded-proxy-k", type=int, default=4)
    parser.add_argument("--answer-base-url", type=str, required=True)
    parser.add_argument("--scorer-base-url", type=str, required=True)
    parser.add_argument("--answer-model", type=str, default="")
    parser.add_argument("--scorer-model", type=str, default="")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--answer-max-tokens", type=int, default=128)
    parser.add_argument("--scorer-max-tokens", type=int, default=256)
    parser.add_argument("--answer-temperature", type=float, default=0.7)
    parser.add_argument("--output-path", type=str, default="analysis/throughput_benchmark_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(run_benchmark(args))
    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

