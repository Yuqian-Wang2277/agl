#!/usr/bin/env python3
"""
Unified evaluation for MATH-500, StrategyQA, and ReClor.

Modes
-----
FS     Few-shot ICL: answer model sees k same-category solved examples and
       answers the target problem directly.
MIST   Two-stage: strategy model extracts a transferable strategy from k
       examples (semi_structured prompt), then answer model applies it.

Few-shot pool construction
--------------------------
MATH-500   : k examples drawn from the same ``subject`` (7 subjects),
             leave-one-out exclusion.
StrategyQA : k examples drawn from the full pool (single category) with
             balanced Yes/No sampling.
ReClor     : k examples drawn from the full pool (single category),
             leave-one-out exclusion.

Draws are fully deterministic: shot_seed XOR md5(problem_id) XOR sample_idx.

Pass@k
------
Each problem is run --num-samples times with independent random seeds.
pass@k = fraction of problems where ≥1 of the first k samples is correct.

Scoring
-------
MATH-500   : normalized LaTeX exact match → SymPy symbolic/numerical → char-F1
StrategyQA : exact yes/no match (hard binary)
ReClor     : exact A/B/C/D match (hard binary)

Usage
-----
  # FS mode (all three benchmarks)
  python eval_unified.py --mode FS \\
      --answer-model Qwen3-8B \\
      --answer-api-base http://localhost:8200/v1

  # MIST mode
  python eval_unified.py --mode MIST \\
      --strategy-model Qwen3-4B \\
      --strategy-api-base http://localhost:8100/v1 \\
      --answer-model Qwen3-8B \\
      --answer-api-base http://localhost:8200/v1

  # Subset of benchmarks, quick smoke test
  python eval_unified.py --mode FS --benchmarks math500 strategyqa \\
      --max-problems 10 --num-samples 1
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

try:
    from openai import AsyncOpenAI
except ImportError:
    print("[ERROR] 'openai' package not found. Install with: pip install openai", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_STRATEGY_PROMPT_DIR = _REPO_ROOT / "examples" / "strategy_extraction" / "prompt"
_DEFAULT_PROMPTS_DIR = _SCRIPT_DIR / "prompts"

_LABELS = ["A", "B", "C", "D"]


# ═══════════════════════════════════════════════════════════════════════════════
# TOML loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_toml(path: Path) -> Dict[str, str]:
    if tomllib is None:
        raise ImportError(
            "TOML loading requires Python ≥ 3.11 or the 'tomli' package.\n"
            "Install with: pip install tomli"
        )
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)  # type: ignore[attr-defined]


# ═══════════════════════════════════════════════════════════════════════════════
# Problem dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Problem:
    id: str
    category: str         # used for few-shot pool grouping
    question_text: str    # formatted prompt text (no answer)
    ground_truth: Any     # type depends on benchmark


# ═══════════════════════════════════════════════════════════════════════════════
# Math answer utilities  (shared for MATH-500 scoring)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_answer_tag(text: str) -> Optional[str]:
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else None


def _extract_boxed(text: str) -> Optional[str]:
    result: Optional[str] = None
    pos = 0
    while pos < len(text):
        idx = text.find(r"\boxed", pos)
        if idx == -1:
            break
        j = idx + 6
        while j < len(text) and text[j] in (" ", "\t"):
            j += 1
        if j < len(text) and text[j] == "{":
            depth, start, i = 1, j + 1, j + 1
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            if depth == 0:
                result = text[start : i - 1]
        pos = idx + 1
    return result


def _normalize_latex(s: str) -> str:
    s = re.sub(r"\$", "", s)
    s = re.sub(r"\\left\s*[\(\[|]", "(", s)
    s = re.sub(r"\\right\s*[\)\]|]", ")", s)
    for token in (r"\cdot", r"\times"):
        s = s.replace(token, "*")
    for token in (r"\,", r"\;", r"\:", r"\!", r"\ ", r"\approx", r"\sim"):
        s = s.replace(token, "")
    s = re.sub(r"\\operatorname\{(\w+)\}", r"\\\1", s)
    s = re.sub(r"\s+", "", s)
    return s.lower().strip()


def _rhs_only(expr: str) -> str:
    for sep in (r"\approx", r"\sim", "="):
        if sep in expr:
            parts = expr.split(sep, 1)
            rhs = parts[1].strip() if len(parts) == 2 else ""
            if rhs:
                return rhs
    return expr


def _try_sympy_equal(pred: str, gt: str) -> Optional[bool]:
    try:
        import sympy  # noqa: PLC0415
        from sympy.parsing.latex import parse_latex  # noqa: PLC0415
        import warnings  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_sym = parse_latex(pred)
            g_sym = parse_latex(gt)

        try:
            if sympy.simplify(p_sym - g_sym) == 0:
                return True
        except Exception:
            pass

        try:
            free = (p_sym - g_sym).free_symbols
            rng = random.Random(20240426)
            all_close = True
            for _ in range(4):
                subs = {sym: complex(rng.uniform(0.5, 2.0), 0.0) for sym in free}
                p_val = complex(sympy.N(p_sym.subs(subs)))
                g_val = complex(sympy.N(g_sym.subs(subs)))
                tol = 1e-3 * max(1.0, abs(g_val))
                if abs(p_val - g_val) > tol:
                    all_close = False
                    break
            if all_close:
                return True
        except Exception:
            pass

    except Exception:
        pass
    return None


def compare_math(model_output: str, gt: str) -> Tuple[bool, float]:
    """Compare model output against a LaTeX ground-truth expression.

    Returns (hard_correct, soft_score).
    """
    pred_text = _extract_answer_tag(model_output) or model_output
    # Accept either boxed or raw expression from model
    pred_inner = _extract_boxed(pred_text) or pred_text
    pred_expr = re.sub(r"\$", "", pred_inner).strip()
    # Ground truth may or may not be boxed
    gt_inner = _extract_boxed(gt) or gt
    gt_expr = re.sub(r"\$", "", gt_inner).strip()

    pred_norm = _normalize_latex(pred_expr)
    gt_norm = _normalize_latex(gt_expr)
    if pred_norm == gt_norm:
        return True, 1.0

    pred_rhs = _normalize_latex(_rhs_only(pred_expr))
    gt_rhs = _normalize_latex(_rhs_only(gt_expr))
    if pred_rhs == gt_rhs and pred_rhs:
        return True, 1.0

    for p_cand, g_cand in [(pred_expr, gt_expr), (_rhs_only(pred_expr), _rhs_only(gt_expr))]:
        if not p_cand or not g_cand:
            continue
        if _try_sympy_equal(p_cand, g_cand) is True:
            return True, 1.0

    ratio = SequenceMatcher(None, pred_norm, gt_norm).ratio()
    return False, float(ratio)


# ═══════════════════════════════════════════════════════════════════════════════
# String extraction utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_yes_no(text: str) -> Optional[str]:
    content = _extract_answer_tag(text) or text
    content_lower = content.strip().lower()
    # Check first word / full content
    if re.search(r"\byes\b", content_lower):
        return "yes"
    if re.search(r"\bno\b", content_lower):
        return "no"
    return None


def _extract_abcd(text: str) -> Optional[str]:
    content = _extract_answer_tag(text) or text
    # Prefer a bare letter at the start of the answer block
    m = re.search(r"\b([ABCD])\b", content.strip())
    if m:
        return m.group(1)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark base class and concrete implementations
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkBase(ABC):
    name: str
    balanced_sampling: bool = False  # balanced yes/no shots for boolean tasks

    @abstractmethod
    def load(self, data_root: Path) -> List[Problem]: ...

    @abstractmethod
    def format_example(self, p: Problem) -> str:
        """Return a single formatted example (question + answer) for few-shot text."""

    @abstractmethod
    def extract_answer(self, response: str) -> Optional[str]: ...

    @abstractmethod
    def score(self, predicted: Optional[str], truth: Any) -> Tuple[bool, float]:
        """Return (hard_correct: bool, soft_score: float ∈ [0,1])."""


class Math500Benchmark(BenchmarkBase):
    name = "math500"
    balanced_sampling = False

    def load(self, data_root: Path) -> List[Problem]:
        path = data_root / "MATH-500" / "data" / "test.json"
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        counters: Dict[str, int] = defaultdict(int)
        result: List[Problem] = []
        for item in items:
            subj = item["subject"]
            idx = counters[subj]
            counters[subj] += 1
            result.append(Problem(
                id=f"math500_{subj}_{idx}",
                category=subj,
                question_text=item["problem"],
                ground_truth=item["answer"],
            ))
        return result

    def format_example(self, p: Problem) -> str:
        return f"Problem: {p.question_text}\nAnswer: {p.ground_truth}"

    def extract_answer(self, response: str) -> Optional[str]:
        content = _extract_answer_tag(response) or response
        boxed = _extract_boxed(content)
        return (boxed or content).strip() or None

    def score(self, predicted: Optional[str], truth: Any) -> Tuple[bool, float]:
        if not predicted:
            return False, 0.0
        return compare_math(predicted, str(truth))


class StrategyQABenchmark(BenchmarkBase):
    name = "strategyqa"
    balanced_sampling = True  # draw ~k/2 Yes and ~k/2 No examples

    def load(self, data_root: Path) -> List[Problem]:
        path = data_root / "StrategyQA" / "data" / "test.json"
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        return [
            Problem(
                id=item["qid"],
                category="all",
                question_text=item["question"],
                ground_truth=item["answer"],  # bool
            )
            for item in items
        ]

    def format_example(self, p: Problem) -> str:
        ans_str = "Yes" if p.ground_truth else "No"
        return f"Question: {p.question_text}\nAnswer: {ans_str}"

    def extract_answer(self, response: str) -> Optional[str]:
        return _extract_yes_no(response)

    def score(self, predicted: Optional[str], truth: Any) -> Tuple[bool, float]:
        expected = "yes" if truth else "no"
        hard = predicted == expected
        return hard, float(hard)


class ReClorBenchmark(BenchmarkBase):
    name = "reclor"
    balanced_sampling = False

    def load(self, data_root: Path) -> List[Problem]:
        path = data_root / "ReClor" / "data" / "test.json"
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        result: List[Problem] = []
        for item in items:
            choices = "\n".join(
                f"({_LABELS[i]}) {ans}" for i, ans in enumerate(item["answers"])
            )
            question_text = (
                f"Context: {item['context']}\n\n"
                f"Question: {item['question']}\n\n"
                f"{choices}"
            )
            result.append(Problem(
                id=item["id_string"],
                category="all",
                question_text=question_text,
                ground_truth=item["label"],  # int 0-3
            ))
        return result

    def format_example(self, p: Problem) -> str:
        return f"{p.question_text}\nAnswer: {_LABELS[p.ground_truth]}"

    def extract_answer(self, response: str) -> Optional[str]:
        return _extract_abcd(response)

    def score(self, predicted: Optional[str], truth: Any) -> Tuple[bool, float]:
        hard = predicted == _LABELS[truth]
        return hard, float(hard)


_BENCHMARK_REGISTRY: Dict[str, type] = {
    "math500": Math500Benchmark,
    "strategyqa": StrategyQABenchmark,
    "reclor": ReClorBenchmark,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Few-shot sampling
# ═══════════════════════════════════════════════════════════════════════════════

def _pid_hash(pid: str) -> int:
    return int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)


def make_shot_seed(pid: str, sample_idx: int, global_seed: int) -> int:
    """Diverse seed: varies per (pid, sample_idx) so each pass@k sample sees
    a different few-shot subset."""
    return (global_seed ^ _pid_hash(pid) ^ sample_idx) & 0xFFFFFFFF


def select_few_shot(
    pool: List[Problem],
    target: Problem,
    k: int,
    rng: random.Random,
    balanced: bool = False,
) -> List[Problem]:
    """Return k examples from pool excluding target.

    When balanced=True and pool contains boolean ground_truth, samples ~k/2
    from each class to avoid showing only one answer polarity.
    """
    candidates = [p for p in pool if p.id != target.id]
    if not candidates:
        return []
    k = min(k, len(candidates))

    if balanced and k >= 2 and isinstance(candidates[0].ground_truth, bool):
        pos = [p for p in candidates if p.ground_truth]
        neg = [p for p in candidates if not p.ground_truth]
        n_pos = k // 2
        n_neg = k - n_pos
        sampled: List[Problem] = (
            rng.sample(pos, min(n_pos, len(pos)))
            + rng.sample(neg, min(n_neg, len(neg)))
        )
        # Pad with remaining candidates if a class was too small
        if len(sampled) < k:
            remaining = [p for p in candidates if p not in sampled]
            sampled += rng.sample(remaining, min(k - len(sampled), len(remaining)))
        rng.shuffle(sampled)
        return sampled[:k]

    return rng.sample(candidates, k)


def format_examples_text(bench: BenchmarkBase, examples: List[Problem]) -> str:
    parts = [f"Example {i}:\n{bench.format_example(ex)}" for i, ex in enumerate(examples, 1)]
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM client
# ═══════════════════════════════════════════════════════════════════════════════

async def call_llm(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    seed: Optional[int] = None,
    no_think: bool = False,
    extra_body: Optional[Dict[str, Any]] = None,
    retries: int = 3,
) -> str:
    if no_think:
        msgs = list(messages)
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i]["role"] == "user":
                msgs[i] = {**msgs[i], "content": msgs[i]["content"] + " /no_think"}
                break
        messages = msgs

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        kwargs["seed"] = seed
    if extra_body:
        kwargs["extra_body"] = extra_body

    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {retries} attempts") from last_exc


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline: FS and MIST
# ═══════════════════════════════════════════════════════════════════════════════

async def run_fs(
    bench: BenchmarkBase,
    problem: Problem,
    examples: List[Problem],
    client: AsyncOpenAI,
    model: str,
    prompt: Dict[str, str],
    *,
    temperature: float,
    seed: Optional[int],
    no_think: bool,
    max_tokens: int,
) -> str:
    messages = [
        {"role": "system", "content": prompt["system"].strip()},
        {
            "role": "user",
            "content": prompt["user"].format(
                examples_text=format_examples_text(bench, examples),
                problem=problem.question_text,
            ),
        },
    ]
    return await call_llm(
        client, model, messages,
        temperature=temperature, seed=seed, no_think=no_think, max_tokens=max_tokens,
    )


async def run_mist(
    bench: BenchmarkBase,
    problem: Problem,
    examples: List[Problem],
    strategy_client: AsyncOpenAI,
    strategy_model: str,
    answer_client: AsyncOpenAI,
    answer_model: str,
    strategy_prompt: Dict[str, str],
    answer_prompt: Dict[str, str],
    *,
    temperature: float,
    seed: Optional[int],
    strategy_no_think: bool,
    answer_no_think: bool,
    strategy_rep_penalty: Optional[float],
    max_tokens: int,
) -> Tuple[str, str]:
    examples_text = format_examples_text(bench, examples)
    strat_extra = {"repetition_penalty": strategy_rep_penalty} if strategy_rep_penalty else None

    # Stage 1 — extract strategy from examples
    strategy = await call_llm(
        strategy_client, strategy_model,
        [
            {"role": "system", "content": strategy_prompt["system"].strip()},
            {"role": "user", "content": strategy_prompt["user"].format(examples_text=examples_text)},
        ],
        temperature=temperature, seed=seed,
        no_think=strategy_no_think, extra_body=strat_extra, max_tokens=max_tokens,
    )

    # Stage 2 — apply strategy to answer the problem
    answer = await call_llm(
        answer_client, answer_model,
        [
            {"role": "system", "content": answer_prompt["system"].strip()},
            {
                "role": "user",
                "content": answer_prompt["user"].format(
                    strategy=strategy,
                    problem=problem.question_text,
                ),
            },
        ],
        temperature=temperature, seed=seed,
        no_think=answer_no_think, max_tokens=max_tokens,
    )
    return strategy, answer


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pass_at_k(
    per_prob: Dict[str, List[float]], k: int, thresh: float = 0.5
) -> Optional[float]:
    if not per_prob:
        return None
    results = [
        any(v >= thresh for v in scores[:k])
        for scores in per_prob.values()
        if scores
    ]
    return sum(results) / len(results) if results else None


# ═══════════════════════════════════════════════════════════════════════════════
# Per-benchmark state container
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchState:
    bench: BenchmarkBase
    problems: List[Problem]
    shot_pool: Dict[str, List[Problem]]
    per_prob_hard: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    per_prob_soft: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    per_cat_hard: Dict[str, Dict[str, List[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    per_cat_soft: Dict[str, Dict[str, List[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    detail_rows: List[Dict[str, Any]] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary printing and result saving
# ═══════════════════════════════════════════════════════════════════════════════

def _print_bench_summary(
    state: BenchState,
    args: argparse.Namespace,
    elapsed: float,
) -> None:
    bench = state.bench
    all_hard = [v for sc in state.per_prob_hard.values() for v in sc]
    all_soft = [v for sc in state.per_prob_soft.values() for v in sc]
    k_vals = [k for k in (1, 2, 3) if k <= args.num_samples]

    print("\n" + "=" * 80)
    print(f"Benchmark: {bench.name.upper()}  |  Mode: {args.mode}  |  Model: {args.answer_model}")
    if args.mode == "MIST":
        print(f"  Strategy model: {args.strategy_model}")
    print(f"  shot_k={args.shot_k}  num_samples={args.num_samples}  seed={args.shot_seed}")
    print("=" * 80)
    print(f"Problems:  {len(state.problems)}")
    print(f"Elapsed:   {elapsed:.1f}s")
    print(f"Acc_hard:  {_mean(all_hard):.4f}")
    if bench.name == "math500":
        print(f"Acc_soft:  {_mean(all_soft):.4f}")

    if k_vals:
        print()
        for k in k_vals:
            ph = _pass_at_k(state.per_prob_hard, k)
            h_s = f"{ph:.4f}" if ph is not None else "NA"
            line = f"pass@{k}(hard): {h_s}"
            if bench.name == "math500":
                ps = _pass_at_k(state.per_prob_soft, k)
                s_s = f"{ps:.4f}" if ps is not None else "NA"
                line += f"   pass@{k}(soft): {s_s}"
            print(line)

    print("\nBy category:")
    for cat in sorted(state.per_cat_hard):
        t_hard = [v for sc in state.per_cat_hard[cat].values() for v in sc]
        t_soft = [v for sc in state.per_cat_soft[cat].values() for v in sc]
        n = len(state.per_cat_hard[cat])
        pass_parts: List[str] = []
        for k in k_vals:
            ph = _pass_at_k(state.per_cat_hard[cat], k)
            if ph is not None:
                part = f"p@{k}(h)={ph:.4f}"
                if bench.name == "math500":
                    ps = _pass_at_k(state.per_cat_soft[cat], k)
                    part += f" p@{k}(s)={ps:.4f}" if ps is not None else ""
                pass_parts.append(part)
        print(
            f"  {cat:30s}  n={n:4d}"
            f"  acc_h={_mean(t_hard):.4f}"
            + (f"  acc_s={_mean(t_soft):.4f}" if bench.name == "math500" else "")
            + ("  " + "  ".join(pass_parts) if pass_parts else "")
        )
    print("=" * 80)


def _save_bench_results(
    state: BenchState,
    args: argparse.Namespace,
    elapsed: float,
    ts: str,
) -> None:
    bench = state.bench
    k_vals = [k for k in (1, 2, 3) if k <= args.num_samples]
    all_hard = [v for sc in state.per_prob_hard.values() for v in sc]
    all_soft = [v for sc in state.per_prob_soft.values() for v in sc]

    safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", args.answer_model)
    out_path = Path(args.output_dir) / f"{bench.name}_{args.mode}_{safe_model}_{ts}.json"

    summary: Dict[str, Any] = {
        "benchmark": bench.name,
        "mode": args.mode,
        "timestamp": ts,
        "answer_model": args.answer_model,
        "strategy_model": args.strategy_model if args.mode == "MIST" else None,
        "strategy_prompt_version": args.strategy_prompt_version if args.mode == "MIST" else None,
        "shot_k": args.shot_k,
        "shot_seed": args.shot_seed,
        "num_problems": len(state.problems),
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "elapsed_sec": round(elapsed, 1),
        "acc_hard": round(_mean(all_hard), 4),
        "acc_soft": round(_mean(all_soft), 4),
        "pass_at_k_hard": {
            str(k): round(_pass_at_k(state.per_prob_hard, k) or 0.0, 4) for k in k_vals
        },
        "pass_at_k_soft": {
            str(k): round(_pass_at_k(state.per_prob_soft, k) or 0.0, 4) for k in k_vals
        },
        "by_category": {
            cat: {
                "n_problems": len(state.per_cat_hard[cat]),
                "acc_hard": round(
                    _mean([v for sc in state.per_cat_hard[cat].values() for v in sc]), 4
                ),
                "acc_soft": round(
                    _mean([v for sc in state.per_cat_soft[cat].values() for v in sc]), 4
                ),
                "pass_at_k_hard": {
                    str(k): round(_pass_at_k(state.per_cat_hard[cat], k) or 0.0, 4)
                    for k in k_vals
                },
                "pass_at_k_soft": {
                    str(k): round(_pass_at_k(state.per_cat_soft[cat], k) or 0.0, 4)
                    for k in k_vals
                },
            }
            for cat in sorted(state.per_cat_hard)
        },
    }
    if args.save_details:
        summary["rows"] = state.detail_rows

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("[%s] Results saved to %s", bench.name, out_path)
    print(f"[{bench.name}] Results saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

async def run_eval(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )

    data_root = Path(args.data_root)
    prompts_dir = Path(args.prompts_dir)
    strategy_prompt_dir = Path(args.strategy_prompt_dir)

    # ── Select and instantiate benchmarks ─────────────────────────────────────
    bench_names: List[str] = args.benchmarks
    unknown = [n for n in bench_names if n not in _BENCHMARK_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {unknown}. Choices: {list(_BENCHMARK_REGISTRY)}")
    benchmarks: List[BenchmarkBase] = [_BENCHMARK_REGISTRY[n]() for n in bench_names]

    # ── Load strategy prompt (shared across benchmarks in MIST mode) ──────────
    strategy_prompt: Optional[Dict[str, str]] = None
    if args.mode == "MIST":
        strategy_prompt = load_toml(
            strategy_prompt_dir / "strategy_generation" / f"{args.strategy_prompt_version}.toml"
        )
        logger.info("Strategy prompt: %s", args.strategy_prompt_version)

    # ── Load data, build shot pools, load answer prompts ─────────────────────
    states: Dict[str, BenchState] = {}
    fs_prompts: Dict[str, Dict[str, str]] = {}
    mist_answer_prompts: Dict[str, Dict[str, str]] = {}

    for bench in benchmarks:
        all_problems = bench.load(data_root)
        shot_pool: Dict[str, List[Problem]] = defaultdict(list)
        for p in all_problems:
            shot_pool[p.category].append(p)

        eval_problems = all_problems
        if args.max_problems is not None:
            eval_problems = all_problems[: args.max_problems]

        states[bench.name] = BenchState(
            bench=bench,
            problems=eval_problems,
            shot_pool=dict(shot_pool),  # always full pool for shots
        )
        cat_summary = {cat: len(probs) for cat, probs in sorted(shot_pool.items())}
        logger.info(
            "[%s] %d problems loaded (%d to eval), categories: %s",
            bench.name, len(all_problems), len(eval_problems), cat_summary,
        )

        if args.mode == "FS":
            fs_prompts[bench.name] = load_toml(prompts_dir / f"{bench.name}_fs.toml")
        else:
            mist_answer_prompts[bench.name] = load_toml(
                prompts_dir / f"{bench.name}_mist_answer.toml"
            )

    # ── LLM clients ───────────────────────────────────────────────────────────
    api_key = args.api_key or "dummy"
    answer_client = AsyncOpenAI(api_key=api_key, base_url=args.answer_api_base)
    strategy_client: Optional[AsyncOpenAI] = None
    if args.mode == "MIST":
        if not args.strategy_api_base:
            raise ValueError("--strategy-api-base is required for MIST mode")
        strategy_client = AsyncOpenAI(api_key=api_key, base_url=args.strategy_api_base)

    temperature = args.temperature
    if args.num_samples > 1 and temperature == 0.0:
        logger.warning(
            "num_samples=%d with temperature=0; auto-switching to 0.7 "
            "(pass --temperature explicitly to override)",
            args.num_samples,
        )
        temperature = 0.7

    # ── Shared concurrency controls ────────────────────────────────────────────
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    done_count = 0
    total_calls = sum(len(s.problems) for s in states.values()) * args.num_samples

    # ── Per-sample coroutine ───────────────────────────────────────────────────
    async def process_one(
        state: BenchState,
        problem: Problem,
        sample_idx: int,
    ) -> None:
        nonlocal done_count
        bench = state.bench

        shot_seed = make_shot_seed(problem.id, sample_idx, args.shot_seed)
        rng = random.Random(shot_seed)
        examples = select_few_shot(
            state.shot_pool[problem.category],
            problem,
            args.shot_k,
            rng,
            balanced=bench.balanced_sampling,
        )

        llm_seed: Optional[int] = args.seed + sample_idx if args.seed is not None else None
        strategy_text = ""
        output = ""

        try:
            async with semaphore:
                if args.mode == "FS":
                    output = await run_fs(
                        bench, problem, examples,
                        answer_client, args.answer_model,
                        fs_prompts[bench.name],
                        temperature=temperature,
                        seed=llm_seed,
                        no_think=args.answer_no_think,
                        max_tokens=args.max_tokens,
                    )
                else:  # MIST
                    rep_pen = (
                        args.strategy_rep_penalty if args.strategy_rep_penalty > 0 else None
                    )
                    assert strategy_client is not None
                    assert strategy_prompt is not None
                    strategy_text, output = await run_mist(
                        bench, problem, examples,
                        strategy_client, args.strategy_model,
                        answer_client, args.answer_model,
                        strategy_prompt, mist_answer_prompts[bench.name],
                        temperature=temperature,
                        seed=llm_seed,
                        strategy_no_think=args.strategy_no_think,
                        answer_no_think=args.answer_no_think,
                        strategy_rep_penalty=rep_pen,
                        max_tokens=args.max_tokens,
                    )
        except Exception as exc:
            logger.warning(
                "[%s] %s sample %d failed: %s", bench.name, problem.id, sample_idx, exc
            )

        predicted = bench.extract_answer(output)
        hard, soft = bench.score(predicted, problem.ground_truth)

        async with lock:
            state.per_prob_hard[problem.id].append(float(hard))
            state.per_prob_soft[problem.id].append(float(soft))
            state.per_cat_hard[problem.category][problem.id].append(float(hard))
            state.per_cat_soft[problem.category][problem.id].append(float(soft))
            state.detail_rows.append({
                "problem_id": problem.id,
                "category": problem.category,
                "sample_idx": sample_idx,
                "hard": int(hard),
                "soft": round(soft, 4),
                "ground_truth": str(problem.ground_truth),
                "predicted": predicted,
                "response": output,
                "strategy": strategy_text if args.mode == "MIST" else None,
                "few_shot_ids": [e.id for e in examples],
            })
            done_count += 1
            if done_count % 100 == 0 or done_count == total_calls:
                logger.info("Progress: %d / %d calls done", done_count, total_calls)

        logger.debug(
            "[%s] %s sample %d: hard=%s soft=%.3f", bench.name, problem.id, sample_idx, hard, soft
        )

    # ── Launch all tasks concurrently across all benchmarks ───────────────────
    all_tasks = [
        process_one(state, problem, sample_idx)
        for state in states.values()
        for problem in state.problems
        for sample_idx in range(args.num_samples)
    ]
    logger.info(
        "Starting evaluation: %d benchmark(s) × up to %d problems × %d samples = %d calls"
        "  (concurrency=%d)",
        len(benchmarks), max(len(s.problems) for s in states.values()),
        args.num_samples, len(all_tasks), args.concurrency,
    )
    start_time = datetime.now()
    await asyncio.gather(*all_tasks)
    elapsed = (datetime.now() - start_time).total_seconds()

    # ── Results ───────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)

    for bench in benchmarks:
        state = states[bench.name]
        _print_bench_summary(state, args, elapsed)
        _save_bench_results(state, args, elapsed, ts)


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ═══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--mode", choices=["FS", "MIST"], default="FS",
        help=(
            "Evaluation mode.\n"
            "  FS   — Few-shot ICL: answer model uses in-context examples directly.\n"
            "  MIST — Two-stage: strategy model extracts a strategy from examples,\n"
            "         then answer model applies it to the new problem."
        ),
    )
    p.add_argument(
        "--benchmarks", nargs="+",
        default=["math500", "strategyqa", "reclor"],
        choices=list(_BENCHMARK_REGISTRY),
        metavar="BENCH",
        help="Benchmarks to evaluate (default: all three).",
    )

    # Models
    p.add_argument("--answer-model", default="Qwen3-8B",
                   help="Answer model name served via OpenAI-compatible API.")
    p.add_argument("--answer-api-base", default="http://localhost:8200/v1",
                   help="Answer model API base URL.")
    p.add_argument("--strategy-model", default="Qwen3-4B",
                   help="Strategy model name (MIST mode only).")
    p.add_argument("--strategy-api-base", default="",
                   help="Strategy model API base URL (MIST mode only).")
    p.add_argument("--api-key", default="",
                   help="API key (any non-empty value works with local vLLM).")

    # Prompt directories
    p.add_argument(
        "--prompts-dir", default=str(_DEFAULT_PROMPTS_DIR),
        help="Directory containing benchmark-specific TOML prompt files.",
    )
    p.add_argument(
        "--strategy-prompt-dir", default=str(_DEFAULT_STRATEGY_PROMPT_DIR),
        help="Root directory of strategy_generation/ TOML prompts (MIST mode).",
    )
    p.add_argument(
        "--strategy-prompt-version", default="semi_structured",
        help="Strategy prompt TOML name under strategy_generation/ (MIST mode).",
    )

    # Few-shot
    p.add_argument("--shot-k", type=int, default=3,
                   help="Number of few-shot examples per problem.")
    p.add_argument("--shot-seed", type=int, default=42,
                   help="Global seed for deterministic few-shot selection.")

    # Sampling / pass@k
    p.add_argument("--num-samples", type=int, default=3,
                   help="Answer samples per problem (for pass@1/2/3). Default: 3.")
    p.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (auto-switched to 0.7 when num-samples > 1).",
    )
    p.add_argument("--seed", type=int, default=42,
                   help="Base LLM seed; sample i uses seed+i.")

    # Generation
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="Max tokens per generation call.")
    p.add_argument("--answer-no-think", action="store_true",
                   help="Append /no_think to answer model messages (Qwen3).")
    p.add_argument("--strategy-no-think", action="store_true",
                   help="Append /no_think to strategy model messages (Qwen3, MIST).")
    p.add_argument(
        "--strategy-rep-penalty", type=float, default=1.1,
        help="Repetition penalty for strategy model (MIST; ≤0 to disable).",
    )

    # Execution
    p.add_argument("--concurrency", type=int, default=32,
                   help="Max concurrent async LLM calls across all benchmarks.")
    p.add_argument("--max-problems", type=int, default=None,
                   help="Cap on problems per benchmark (smoke-test shortcut).")

    # Data and output
    p.add_argument(
        "--data-root", default=str(_SCRIPT_DIR),
        help="Directory containing MATH-500/, StrategyQA/, ReClor/ subdirs.",
    )
    p.add_argument("--output-dir", default="./results",
                   help="Directory to write JSON result files.")
    p.add_argument("--save-details", action="store_true",
                   help="Include per-sample detail rows in saved JSON.")

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
