#!/usr/bin/env python3
"""
HARDMath2 benchmark evaluation — few-shot and MIST modes.

Modes
-----
few-shot  ICL: answer model learns from same-type solved examples and answers
          the target problem directly (no explicit strategy step).
MIST      Two-stage: strategy model first extracts a transferable strategy from
          the few-shot examples, then answer model applies it to the new problem.

Data construction
-----------------
For each problem to evaluate, k few-shot examples are drawn from the SAME
problem type (integrals, nonlinear_pde, wkb, …), excluding the target problem.
The draw uses a deterministic per-problem seed derived from --data-seed, so
results are fully reproducible across runs.

Pass@k
------
Each problem is evaluated --num-samples times with different LLM seeds.
pass@k = fraction of problems where at least 1 of the first k attempts is
correct (hard threshold ≥ 0.5, i.e. the boolean hard_correct flag).

Evaluation (answer comparison)
-------------------------------
1. Extract model answer from <answer>…</answer> tags.
2. Extract \\boxed{…} content from both model answer and ground truth.
3. Normalize LaTeX (remove whitespace, standardize operators).
4. Exact string match → hard correct if passes.
5. Sympy symbolic / numerical check (best-effort, skipped if sympy absent).
6. Character-level F1 similarity as soft score.

Usage
-----
  # few-shot (only answer model needed)
  python eval_hardmath.py --mode few-shot \\
      --answer-model-name Qwen3-8B \\
      --answer-model-base-url http://localhost:8200/v1

  # MIST (strategy model + answer model)
  python eval_hardmath.py --mode MIST \\
      --strategy-model-name Qwen3-4B \\
      --strategy-model-base-url http://localhost:8100/v1 \\
      --answer-model-name Qwen3-8B \\
      --answer-model-base-url http://localhost:8200/v1

  # Quick smoke test (5 problems, 1 sample)
  python eval_hardmath.py --mode few-shot --max-samples 5 --num-samples 1
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── TOML loading ─────────────────────────────────────────────────────────────
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

# ── OpenAI client ─────────────────────────────────────────────────────────────
try:
    from openai import AsyncOpenAI
except ImportError:
    print("[ERROR] 'openai' package not found. Install with: pip install openai", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
DATA_DIR = _SCRIPT_DIR / "data"
_AGENT_LIGHTNING = _SCRIPT_DIR.parent.parent / "agent-lightning"
DEFAULT_PROMPT_DIR = _AGENT_LIGHTNING / "examples" / "strategy_extraction" / "prompt"


# ═══════════════════════════════════════════════════════════════════════════════
# TOML prompt loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_toml_prompt(prompt_dir: Path, subdir: str, name: str) -> Dict[str, str]:
    """Load a prompt TOML file and return its key→value mapping."""
    if tomllib is None:
        raise ImportError(
            "TOML loading requires Python ≥ 3.11 or the 'tomli' package.\n"
            "Install with: pip install tomli"
        )
    path = prompt_dir / subdir / f"{name}.toml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_hardmath_data(data_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load all HARDMath2 .jsonl files, grouped by problem type."""
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                problem = json.loads(raw)
                ptype = problem.get("type", jsonl_file.stem)
                by_type[ptype].append(problem)
    return dict(by_type)


# ═══════════════════════════════════════════════════════════════════════════════
# Few-shot selection
# ═══════════════════════════════════════════════════════════════════════════════

def select_fewshot(
    type_problems: List[Dict[str, Any]],
    target_local_idx: int,
    k: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Return k random problems from the same type, excluding the target."""
    candidates = [p for i, p in enumerate(type_problems) if i != target_local_idx]
    k = min(k, len(candidates))
    return rng.sample(candidates, k) if k > 0 else []


def format_examples_text(examples: List[Dict[str, Any]]) -> str:
    """Format few-shot examples into a prompt-ready string."""
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(f"Example {i}:\nProblem: {ex['prompt']}\nSolution: {ex['solution']}")
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Answer comparison  (official HARDMath evaluation)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_answer_tag(text: str) -> Optional[str]:
    """Extract content from <answer>…</answer> (last occurrence)."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else None


def _extract_boxed(text: str) -> Optional[str]:
    """Extract content of the last \\boxed{…} with balanced-brace matching."""
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
    """Normalize a LaTeX expression for surface-level string comparison."""
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
    """If expression contains '=' or '≈', return only the right-hand side."""
    for sep in (r"\approx", r"\sim", "="):
        if sep in expr:
            parts = expr.split(sep, 1)
            rhs = parts[1].strip() if len(parts) == 2 else ""
            if rhs:
                return rhs
    return expr


def _try_sympy_equal(pred: str, gt: str) -> Optional[bool]:
    """Attempt sympy symbolic / numerical equality check.

    Returns True (equal), False (definitively not equal), or None (undecided).
    """
    try:
        import sympy  # noqa: PLC0415
        from sympy.parsing.latex import parse_latex  # noqa: PLC0415
        import warnings  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_sym = parse_latex(pred)
            g_sym = parse_latex(gt)

        # Symbolic simplification
        try:
            diff = sympy.simplify(p_sym - g_sym)
            if diff == 0:
                return True
        except Exception:
            pass

        # Numerical sampling at multiple random points
        try:
            free = (p_sym - g_sym).free_symbols
            rng = random.Random(20240426)
            n_trials = 4
            all_close = True
            for _ in range(n_trials):
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


def compare_math_answers(model_output: str, gt_solution: str) -> Tuple[bool, float]:
    """Compare model output against HARDMath ground truth.

    Returns
    -------
    hard_correct : bool
        True if the answer is mathematically equivalent (exact or sympy check).
    soft_score : float in [0, 1]
        1.0 if hard_correct, else character-level similarity ratio.
    """
    # ── Extract model answer ──────────────────────────────────────────────────
    pred_text = _extract_answer_tag(model_output) or model_output
    pred_expr = re.sub(r"\$", "", _extract_boxed(pred_text) or pred_text).strip()
    gt_expr = re.sub(r"\$", "", _extract_boxed(gt_solution) or gt_solution).strip()

    # ── 1. Normalized exact match ─────────────────────────────────────────────
    pred_norm = _normalize_latex(pred_expr)
    gt_norm = _normalize_latex(gt_expr)
    if pred_norm == gt_norm:
        return True, 1.0

    # Also try matching only the RHS when expression contains '='
    pred_rhs_norm = _normalize_latex(_rhs_only(pred_expr))
    gt_rhs_norm = _normalize_latex(_rhs_only(gt_expr))
    if pred_rhs_norm == gt_rhs_norm and pred_rhs_norm:
        return True, 1.0

    # ── 2. Sympy symbolic / numerical comparison ──────────────────────────────
    for p_cand, g_cand in [
        (pred_expr, gt_expr),
        (_rhs_only(pred_expr), _rhs_only(gt_expr)),
    ]:
        if not p_cand or not g_cand:
            continue
        result = _try_sympy_equal(p_cand, g_cand)
        if result is True:
            return True, 1.0

    # ── 3. Character F1 soft score ────────────────────────────────────────────
    ratio = SequenceMatcher(None, pred_norm, gt_norm).ratio()
    return False, float(ratio)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM calls
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
    retry_delay: float = 1.0,
) -> str:
    """Call an OpenAI-compatible chat endpoint with automatic retry."""
    # Qwen3 no-think: append /no_think to the last user message
    if no_think:
        msgs: List[Dict[str, str]] = list(messages)
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
                await asyncio.sleep(retry_delay * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {retries} attempts") from last_exc


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation modes
# ═══════════════════════════════════════════════════════════════════════════════

async def run_fewshot(
    problem: Dict[str, Any],
    fewshot_examples: List[Dict[str, Any]],
    client: AsyncOpenAI,
    model: str,
    prompt: Dict[str, str],
    *,
    temperature: float,
    seed: Optional[int],
    no_think: bool,
    max_tokens: int,
) -> str:
    """Few-shot mode: answer model uses in-context examples directly."""
    messages = [
        {"role": "system", "content": prompt["system"].strip()},
        {
            "role": "user",
            "content": prompt["user"].format(
                examples_text=format_examples_text(fewshot_examples),
                problem=problem["prompt"],
            ),
        },
    ]
    return await call_llm(
        client, model, messages,
        temperature=temperature, seed=seed,
        no_think=no_think, max_tokens=max_tokens,
    )


async def run_mist(
    problem: Dict[str, Any],
    fewshot_examples: List[Dict[str, Any]],
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
    """MIST mode: extract strategy from examples, then apply to new problem."""
    examples_text = format_examples_text(fewshot_examples)

    # Stage 1 — strategy generation
    strat_extra: Optional[Dict[str, Any]] = None
    if strategy_rep_penalty and strategy_rep_penalty > 0:
        strat_extra = {"repetition_penalty": strategy_rep_penalty}

    strategy = await call_llm(
        strategy_client, strategy_model,
        [
            {"role": "system", "content": strategy_prompt["system"].strip()},
            {"role": "user", "content": strategy_prompt["user"].format(examples_text=examples_text)},
        ],
        temperature=temperature, seed=seed,
        no_think=strategy_no_think,
        extra_body=strat_extra,
        max_tokens=max_tokens,
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
                    problem=problem["prompt"],
                ),
            },
        ],
        temperature=temperature, seed=seed,
        no_think=answer_no_think,
        max_tokens=max_tokens,
    )
    return strategy, answer


async def run_mist_inline(
    problem: Dict[str, Any],
    fewshot_examples: List[Dict[str, Any]],
    client: AsyncOpenAI,
    model: str,
    inline_strategy_prompt: Dict[str, str],
    answer_prompt: Dict[str, str],
    *,
    temperature: float,
    seed: Optional[int],
    no_think: bool,
    max_tokens: int,
) -> Tuple[str, str]:
    """Inline MIST (train-free, single model): generate strategy then answer.

    The same client/model handles both the strategy extraction step and the
    answer generation step.  Designed for closed-source APIs (GPT-4o, Claude,
    etc.) where no separate fine-tuned strategy model is available.

    Step 1 — use ``inline_strategy_prompt`` to derive a two-layer strategy
              (FIRST_ORDER meta pattern + SECOND_ORDER cognitive steps) from
              the few-shot examples.
    Step 2 — use ``answer_prompt`` to apply that strategy to the new problem.
    """
    examples_text = format_examples_text(fewshot_examples)

    # Stage 1 — inline strategy extraction
    strategy = await call_llm(
        client, model,
        [
            {"role": "system", "content": inline_strategy_prompt["system"].strip()},
            {
                "role": "user",
                "content": inline_strategy_prompt["user"].format(examples_text=examples_text),
            },
        ],
        temperature=temperature, seed=seed,
        no_think=no_think, max_tokens=max_tokens,
    )

    # Stage 2 — strategy-guided answer generation
    answer = await call_llm(
        client, model,
        [
            {"role": "system", "content": answer_prompt["system"].strip()},
            {
                "role": "user",
                "content": answer_prompt["user"].format(
                    strategy=strategy,
                    problem=problem["prompt"],
                ),
            },
        ],
        temperature=temperature, seed=seed,
        no_think=no_think, max_tokens=max_tokens,
    )
    return strategy, answer


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

async def run_eval(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    logger.info("Loading HARDMath2 data from %s", data_dir)
    problems_by_type = load_hardmath_data(data_dir)
    type_summary = {k: len(v) for k, v in sorted(problems_by_type.items())}
    logger.info("Loaded %d problems: %s", sum(type_summary.values()), type_summary)

    # Flat list: (ptype, local_idx, global_idx, problem)
    eval_items: List[Tuple[str, int, int, Dict[str, Any]]] = []
    g_idx = 0
    for ptype in sorted(problems_by_type):
        for local_idx, problem in enumerate(problems_by_type[ptype]):
            eval_items.append((ptype, local_idx, g_idx, problem))
            g_idx += 1

    if args.max_samples is not None:
        eval_items = eval_items[: args.max_samples]

    # ── Load prompts ──────────────────────────────────────────────────────────
    prompt_dir = Path(args.prompt_dir)
    inline_strategy_prompt: Optional[Dict[str, str]] = None
    strategy_prompt_data: Optional[Dict[str, str]] = None

    if args.mode == "few-shot":
        answer_prompt = load_toml_prompt(prompt_dir, "answer_generation", args.answer_prompt_version)
    elif args.mode == "MIST":
        strategy_prompt_data = load_toml_prompt(
            prompt_dir, "strategy_generation", args.strategy_prompt_version
        )
        answer_prompt = load_toml_prompt(prompt_dir, "answer_generation", args.answer_prompt_version)
    else:  # mist-inline
        inline_strategy_prompt = load_toml_prompt(
            prompt_dir, "answer_generation", args.inline_strategy_prompt_version
        )
        answer_prompt = load_toml_prompt(prompt_dir, "answer_generation", args.answer_prompt_version)

    logger.info(
        "Prompts loaded — mode=%s answer_prompt=%s%s",
        args.mode,
        args.answer_prompt_version,
        f" strategy_prompt={args.strategy_prompt_version}" if args.mode == "MIST"
        else (f" inline_strategy_prompt={args.inline_strategy_prompt_version}"
              if args.mode == "mist-inline" else ""),
    )

    # ── LLM clients ───────────────────────────────────────────────────────────
    api_key = args.api_key or "dummy"
    answer_client = AsyncOpenAI(api_key=api_key, base_url=args.answer_model_base_url)
    strategy_client = (
        AsyncOpenAI(api_key=api_key, base_url=args.strategy_model_base_url)
        if args.mode == "MIST"
        else None
    )

    # Auto-switch temperature for multi-sample diversity
    temperature = args.temperature
    if args.num_samples > 1 and temperature == 0.0:
        logger.warning(
            "num_samples=%d with temperature=0; auto-switching to 0.7 "
            "(pass --temperature explicitly to override)",
            args.num_samples,
        )
        temperature = 0.7

    base_seed = args.seed

    # ── Per-problem result tracking for pass@k ────────────────────────────────
    per_problem_hard: Dict[str, List[float]] = defaultdict(list)
    per_problem_soft: Dict[str, List[float]] = defaultdict(list)
    per_type_hard: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_type_soft: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    detail_rows: List[Dict[str, Any]] = []
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)
    done_count = 0
    total_calls = len(eval_items) * args.num_samples

    async def process(
        ptype: str,
        local_idx: int,
        global_problem_idx: int,
        problem: Dict[str, Any],
        sample_idx: int,
    ) -> None:
        nonlocal done_count
        problem_key = f"{ptype}/{problem.get('index', local_idx)}"

        # Deterministic, per-problem few-shot selection
        fewshot_rng = random.Random(args.data_seed * 100_000 + global_problem_idx)
        fewshot = select_fewshot(problems_by_type[ptype], local_idx, args.fewshot_k, fewshot_rng)

        seed = base_seed + sample_idx if base_seed is not None else None
        output = ""
        strategy_text = ""

        try:
            async with semaphore:
                if args.mode == "few-shot":
                    output = await run_fewshot(
                        problem, fewshot,
                        answer_client, args.answer_model_name, answer_prompt,
                        temperature=temperature, seed=seed,
                        no_think=args.answer_no_think, max_tokens=args.max_tokens,
                    )
                elif args.mode == "MIST":
                    rep_pen = args.strategy_repetition_penalty if args.strategy_repetition_penalty > 0 else None
                    strategy_text, output = await run_mist(
                        problem, fewshot,
                        strategy_client, args.strategy_model_name,
                        answer_client, args.answer_model_name,
                        strategy_prompt_data, answer_prompt,
                        temperature=temperature, seed=seed,
                        strategy_no_think=args.strategy_no_think,
                        answer_no_think=args.answer_no_think,
                        strategy_rep_penalty=rep_pen,
                        max_tokens=args.max_tokens,
                    )
                else:  # mist-inline
                    strategy_text, output = await run_mist_inline(
                        problem, fewshot,
                        answer_client, args.answer_model_name,
                        inline_strategy_prompt, answer_prompt,
                        temperature=temperature, seed=seed,
                        no_think=args.answer_no_think,
                        max_tokens=args.max_tokens,
                    )
        except Exception as exc:
            logger.warning("Problem %s sample %d failed: %s", problem_key, sample_idx, exc)

        hard, soft = compare_math_answers(output, problem["solution"])

        async with lock:
            per_problem_hard[problem_key].append(float(hard))
            per_problem_soft[problem_key].append(float(soft))
            per_type_hard[ptype][problem_key].append(float(hard))
            per_type_soft[ptype][problem_key].append(float(soft))
            detail_rows.append(
                {
                    "problem_key": problem_key,
                    "ptype": ptype,
                    "sample_idx": sample_idx,
                    "hard": int(hard),
                    "soft": round(soft, 4),
                    "solution": problem["solution"],
                    "output": output,
                    "strategy": strategy_text if args.mode in ("MIST", "mist-inline") else None,
                }
            )
            done_count += 1
            if done_count % 50 == 0 or done_count == total_calls:
                logger.info("Progress: %d / %d calls completed", done_count, total_calls)

        logger.debug("Problem %s sample %d: hard=%s soft=%.3f", problem_key, sample_idx, hard, soft)

    # ── Launch all tasks ──────────────────────────────────────────────────────
    tasks = [
        process(ptype, local_idx, gidx, problem, sample_idx)
        for ptype, local_idx, gidx, problem in eval_items
        for sample_idx in range(args.num_samples)
    ]
    logger.info(
        "Starting evaluation: %d problems × %d samples = %d calls  (concurrency=%d)",
        len(eval_items), args.num_samples, len(tasks), args.concurrency,
    )
    start_time = datetime.now()
    await asyncio.gather(*tasks)
    elapsed = (datetime.now() - start_time).total_seconds()

    # ═══════════════════════════════════════════════════════════════════════════
    # Metrics
    # ═══════════════════════════════════════════════════════════════════════════

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

    all_hard = [v for sc in per_problem_hard.values() for v in sc]
    all_soft = [v for sc in per_problem_soft.values() for v in sc]
    k_vals = [k for k in (1, 2, 3) if k <= args.num_samples]

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"HARDMath2 Evaluation  |  Mode: {args.mode}  |  Answer: {args.answer_model_name}")
    if args.mode == "MIST":
        print(
            f"  Strategy model: {args.strategy_model_name}"
            f"  |  Strategy prompt: {args.strategy_prompt_version}"
        )
    elif args.mode == "mist-inline":
        print(f"  Inline strategy prompt: {args.inline_strategy_prompt_version}")
    print(
        f"  Answer prompt: {args.answer_prompt_version}"
        f"  |  Few-shot k: {args.fewshot_k}"
        f"  |  data_seed: {args.data_seed}"
    )
    print("=" * 90)
    print(f"Problems:       {len(eval_items)}")
    print(f"Samples/prob:   {args.num_samples}  (temperature={temperature:.2f}, base_seed={base_seed})")
    print(f"Elapsed:        {elapsed:.1f}s")
    print(f"Acc_hard(all):  {_mean(all_hard):.4f}")
    print(f"Acc_soft(all):  {_mean(all_soft):.4f}")

    if k_vals:
        print()
        for k in k_vals:
            ph = _pass_at_k(per_problem_hard, k)
            ps = _pass_at_k(per_problem_soft, k)
            h_s = f"{ph:.4f}" if ph is not None else "NA"
            s_s = f"{ps:.4f}" if ps is not None else "NA"
            print(f"pass@{k}(hard): {h_s}   pass@{k}(soft): {s_s}")

    print("\nBy problem type:")
    for ptype in sorted(per_type_hard):
        t_hard = [v for sc in per_type_hard[ptype].values() for v in sc]
        t_soft = [v for sc in per_type_soft[ptype].values() for v in sc]
        n = len(per_type_hard[ptype])
        pass_parts = []
        for k in k_vals:
            ph = _pass_at_k(per_type_hard[ptype], k)
            ps = _pass_at_k(per_type_soft[ptype], k)
            pass_parts.append(
                f"p@{k}(h)={ph:.4f} p@{k}(s)={ps:.4f}"
                if ph is not None else f"p@{k}=NA"
            )
        print(
            f"  - {ptype:25s}  n={n:3d}"
            f"  acc_h={_mean(t_hard):.4f}  acc_s={_mean(t_soft):.4f}"
            "  " + "  ".join(pass_parts)
        )
    print("=" * 90 + "\n")

    # ── Save results ──────────────────────────────────────────────────────────
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", args.answer_model_name)
        out_path = os.path.join(
            args.output_dir,
            f"hardmath_{args.mode}_{safe_model}_{ts}.json",
        )
        summary: Dict[str, Any] = {
            "mode": args.mode,
            "timestamp": ts,
            "answer_model": args.answer_model_name,
            "answer_prompt": args.answer_prompt_version,
            "strategy_model": args.strategy_model_name if args.mode == "MIST" else None,
            "strategy_prompt": args.strategy_prompt_version if args.mode == "MIST" else None,
            "inline_strategy_prompt": (
                args.inline_strategy_prompt_version if args.mode == "mist-inline" else None
            ),
            "fewshot_k": args.fewshot_k,
            "data_seed": args.data_seed,
            "num_problems": len(eval_items),
            "num_samples": args.num_samples,
            "temperature": temperature,
            "elapsed_sec": round(elapsed, 1),
            "acc_hard": round(_mean(all_hard), 4),
            "acc_soft": round(_mean(all_soft), 4),
            "pass_at_k_hard": {
                str(k): round(_pass_at_k(per_problem_hard, k) or 0, 4) for k in k_vals
            },
            "pass_at_k_soft": {
                str(k): round(_pass_at_k(per_problem_soft, k) or 0, 4) for k in k_vals
            },
            "by_type": {
                ptype: {
                    "n_problems": len(per_type_hard[ptype]),
                    "acc_hard": round(
                        _mean([v for sc in per_type_hard[ptype].values() for v in sc]), 4
                    ),
                    "acc_soft": round(
                        _mean([v for sc in per_type_soft[ptype].values() for v in sc]), 4
                    ),
                    "pass_at_k_hard": {
                        str(k): round(_pass_at_k(per_type_hard[ptype], k) or 0, 4)
                        for k in k_vals
                    },
                    "pass_at_k_soft": {
                        str(k): round(_pass_at_k(per_type_soft[ptype], k) or 0, 4)
                        for k in k_vals
                    },
                }
                for ptype in sorted(per_type_hard)
            },
        }
        if args.save_details:
            summary["rows"] = detail_rows

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("Results saved to %s", out_path)
        print(f"Results saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ═══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    p.add_argument(
        "--mode", choices=["few-shot", "MIST", "mist-inline"], default="MIST",
        help=(
            "Evaluation mode.\n"
            "  few-shot    — ICL: answer model uses in-context examples directly.\n"
            "  MIST        — two-model: fine-tuned strategy model + answer model.\n"
            "  mist-inline — train-free single-model MIST: the answer model first\n"
            "                extracts a two-layer strategy (FIRST_ORDER + SECOND_ORDER)\n"
            "                from the few-shot examples, then applies it to the new problem.\n"
            "                Designed for closed-source APIs (no separate strategy model)."
        ),
    )

    # Models
    p.add_argument("--answer-model-name", default="Qwen3-8B",
                   help="Answer model name served by vLLM.")
    p.add_argument("--answer-model-base-url", default="http://localhost:8200/v1",
                   help="Answer model OpenAI-compatible base URL.")
    p.add_argument("--strategy-model-name", default="Qwen3-4B",
                   help="Strategy model name (MIST mode only).")
    p.add_argument("--strategy-model-base-url", default="http://localhost:8100/v1",
                   help="Strategy model base URL (MIST mode only).")
    p.add_argument("--api-key", default="",
                   help="API key for vLLM (any non-empty value works with local vLLM).")

    # Prompts
    p.add_argument("--prompt-dir", default=str(DEFAULT_PROMPT_DIR),
                   help="Root directory containing TOML prompt subdirs.")
    p.add_argument(
        "--answer-prompt-version", default=None,
        help=(
            "Answer prompt TOML name (without .toml). "
            "Defaults: 'hardmath_few_shot' in few-shot mode, 'v1' in MIST mode."
        ),
    )
    p.add_argument(
        "--strategy-prompt-version", default="repetition_controls_2026-04-01",
        help="Strategy generation prompt TOML name under strategy_generation/ (MIST mode only).",
    )
    p.add_argument(
        "--inline-strategy-prompt-version", default="mist_inline_strategy",
        help=(
            "Strategy extraction prompt TOML name under answer_generation/ (mist-inline mode).\n"
            "Default: 'mist_inline_strategy'  (semi-structured two-layer prompt)."
        ),
    )

    # Few-shot
    p.add_argument("--fewshot-k", type=int, default=3,
                   help="Number of few-shot examples per problem (same type). Default: 3.")
    p.add_argument("--data-seed", type=int, default=42,
                   help="Seed for deterministic few-shot example selection.")

    # Sampling / pass@k
    p.add_argument("--num-samples", type=int, default=3,
                   help="Answer samples per problem for pass@1/2/3 computation. Default: 3.")
    p.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (0=greedy; auto-switched to 0.7 when num-samples > 1).",
    )
    p.add_argument("--seed", type=int, default=42,
                   help="Base LLM seed; sample i uses seed+i. Default: 42.")

    # Generation
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="Max tokens per LLM generation call.")
    p.add_argument("--answer-no-think", action="store_true",
                   help="Append /no_think to answer model user messages (Qwen3).")
    p.add_argument("--strategy-no-think", action="store_true",
                   help="Append /no_think to strategy model user messages (Qwen3, MIST).")
    p.add_argument("--strategy-repetition-penalty", type=float, default=1.1,
                   help="Repetition penalty for strategy model extra_body (MIST, ≤0 to disable).")

    # Execution
    p.add_argument("--concurrency", type=int, default=32,
                   help="Max concurrent async LLM calls. Default: 32.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap on number of problems evaluated (for quick smoke tests).")

    # Data path
    p.add_argument("--data-dir", type=str, default=None,
                   help="Path to HARDMath2 data directory. Defaults to ./data relative to this script.")

    # Output
    p.add_argument("--output-dir", default="./results",
                   help="Directory to write JSON result files. Default: ./results")
    p.add_argument("--save-details", action="store_true",
                   help="Include per-problem detail rows in saved JSON.")

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Apply mode-specific prompt defaults
    if args.answer_prompt_version is None:
        args.answer_prompt_version = {
            "few-shot": "hardmath_few_shot",
            "MIST": "v1",
            "mist-inline": "mist_inline_answer",
        }[args.mode]

    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
