#!/usr/bin/env python3
"""
Standalone evaluation for BBH / ID / OOD benchmarks — two modes.

Decoupled from the agent-lightning training pipeline.
Requires: Python >= 3.11, openai >= 1.0.0  (only stdlib + openai)

Modes (--mode / MODE env var):
  mist-inline  (default) — two sequential API calls per problem:
      Call 1: strategy extraction → <strategy>FIRST_ORDER + SECOND_ORDER</strategy>
      Call 2: apply strategy to answer new problem → <answer>...</answer>
      Prompts: mist_inline_strategy.toml + mist_inline_answer.toml

  few-shot — single API call per problem (ICL direct answer):
      The model sees 3 solved examples and directly solves the new problem.
      Prompt: ICL(few-shot).toml  ({examples_text} + {problem})

Usage (see also eval_mist_inline.sh for the bash wrapper):
  # MIST-inline (default):
  ANSWER_MODEL_NAME=gpt-4o OPENAI_API_KEY=sk-... python eval_mist_inline.py

  # Few-shot:
  MODE=few-shot ANSWER_MODEL_NAME=gpt-4o OPENAI_API_KEY=sk-... python eval_mist_inline.py

Environment variables:
  MODE                     Evaluation mode: mist-inline | few-shot  (default: mist-inline)
  ANSWER_MODEL_BASE_URL    OpenAI-compatible base URL (default: https://api.openai.com/v1)
  ANSWER_MODEL_NAME        Model identifier          (default: gpt-4o)
  OPENAI_API_KEY           API key
  EVAL_CONCURRENCY         Async workers            (default: 4)
  VAL_SAMPLES_PER_SUBTASK  Samples per subtask file  (default: 20)
  VAL_SAMPLING_SEED        Data sampling seed        (default: 42)
  TEMPERATURE              LLM temperature           (default: 0.0)
  LLM_SEED                 OpenAI request seed       (default: 42)
  FEWSHOT_K                Few-shot examples/problem (default: 3)
  OUTPUT_DIR               Results directory         (default: ./results)
  MAX_RETRIES              API retries per call      (default: 3)
  RETRY_DELAY_SEC          Retry delay in seconds    (default: 1.0)
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import math
import os
import random
import re
import sys
import unicodedata
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
        sys.exit("[ERROR] tomllib not found — use Python >= 3.11 or: pip install tomli")

try:
    from openai import AsyncOpenAI
except ImportError:
    sys.exit("[ERROR] openai not found — install with: pip install openai")

# ── Paths (all relative to this script) ───────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
TEST_DATA_BASE = REPO_ROOT / "examples" / "strategy_extraction" / "test"
PROMPT_DIR = REPO_ROOT / "examples" / "strategy_extraction" / "prompt" / "answer_generation"

TEST_SUBDIRS = ["test-bbh", "test-id-subtask", "test-ood-task"]

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Answer extraction & scoring
#  Adapted from examples/strategy_extraction/reward/v3.py — stdlib only,
#  no agentlightning / VERL / training dependencies.
# ═══════════════════════════════════════════════════════════════════════════

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|(?:\{[^{}]*\}))*)\}", re.DOTALL)
_FINAL_CUE_RE = re.compile(
    r"(?:final\s*answer|answer\s*is|答案是|最终答案)\s*[:：]\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$")
_STRATEGY_TAG_RE = re.compile(r"<strategy>(.*?)</strategy>", re.DOTALL)

_YES_TOKENS = {"yes", "y", "true", "t", "correct", "right", "是", "对", "正确", "有"}
_NO_TOKENS = {"no", "n", "false", "f", "incorrect", "wrong", "否", "不", "不是", "错误", "无"}

# Numeric tolerances
_NUM_ABS_TOL = 1e-3
_NUM_REL_TOL = 0.02
_NUM_REL_EPS = 0.1
_NUM_PARSE_FAIL_SOFT_CAP = 0.25
_NUM_EXACT_EPS = 1e-12
_NUM_PERCENT_PP_TOL = 0.5
_SHORT_SOFT_EDIT_CAP = 0.5


def _normalize_text(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "")
    s = s.strip().strip("`\"'“”‘’")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[\s\.,;:!?\-_=+~|/\\]+", "", s)
    s = re.sub(r"[\s\.,;:!?\-_=+~|/\\]+$", "", s)
    return s.lower().strip()


def _normalize_exact(text: str) -> str:
    return re.sub(r"\s+", " ", _normalize_text(text))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[一-鿿]|[a-z0-9]+", _normalize_exact(text))


def _char_f1(pred: str, gt: str) -> float:
    p = _normalize_exact(pred).replace(" ", "")
    g = _normalize_exact(gt).replace(" ", "")
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    return SequenceMatcher(None, p, g).ratio()


def _token_f1(pred: str, gt: str) -> float:
    p_toks, g_toks = _tokenize(pred), _tokenize(gt)
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    p_c: Dict[str, int] = {}
    g_c: Dict[str, int] = {}
    for t in p_toks:
        p_c[t] = p_c.get(t, 0) + 1
    for t in g_toks:
        g_c[t] = g_c.get(t, 0) + 1
    inter = sum(min(c, g_c.get(k, 0)) for k, c in p_c.items())
    if inter == 0:
        return 0.0
    precision = inter / len(p_toks)
    recall = inter / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def _exact_match(a: str, b: str) -> bool:
    return _normalize_exact(a) == _normalize_exact(b)


def _parse_ground_truths(gt: str) -> List[str]:
    gt = (gt or "").strip()
    if not gt:
        return []
    try:
        obj = ast.literal_eval(gt)
        if isinstance(obj, tuple) and obj and all(isinstance(x, (int, float)) for x in obj):
            return [gt]
        if isinstance(obj, (list, tuple)):
            vals = [str(x).strip() for x in obj if str(x).strip()]
            if vals:
                return vals
    except Exception:
        pass
    return [gt]


def _strip_fence(text: str) -> str:
    return _FENCE_RE.sub("", (text or "").strip()).strip()


def _extract_final_span(output: str) -> str:
    text = (output or "").strip()
    if not text:
        return ""
    tag_matches = _ANSWER_TAG_RE.findall(text)
    if tag_matches:
        cand = (tag_matches[-1] or "").strip()
        if cand:
            return cand
    cue_matches = list(_FINAL_CUE_RE.finditer(text))
    if cue_matches:
        cand = (cue_matches[-1].group(1) or "").strip()
        if cand:
            return _strip_fence(cand)
    boxed = _BOXED_RE.findall(text)
    if boxed:
        cand = (boxed[-1] or "").strip()
        if cand:
            return cand
    lines = [ln.strip() for ln in _strip_fence(text).splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return text


def extract_answer(output: str) -> Optional[str]:
    span = _extract_final_span(output).strip()
    return span if span else None


def extract_strategy(output: str) -> Optional[str]:
    matches = _STRATEGY_TAG_RE.findall(output or "")
    if not matches:
        return None
    content = matches[0].strip()
    return content if content else None


def _split_candidates(span: str) -> List[str]:
    if not span:
        return []
    base = span.strip()
    pieces = [base]
    for sep in [r"\|", r";", r"；", r"\bor\b", r"或者"]:
        new_pieces: List[str] = []
        for p in pieces:
            parts = [x.strip() for x in re.split(sep, p, flags=re.IGNORECASE) if x.strip()]
            new_pieces.extend(parts if parts else [p])
        pieces = new_pieces
    dedup: List[str] = []
    seen: set = set()
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        k = _normalize_exact(p)
        if k in seen:
            continue
        seen.add(k)
        dedup.append(p)
    return dedup or [base]


def _parse_numeric(text: str) -> Optional[float]:
    s = _normalize_text(text)
    if not s:
        return None
    s = s.replace(",", "")
    if s.endswith("%"):
        try:
            return float(s[:-1].strip()) / 100.0
        except Exception:
            return None
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?/[+-]?\d+(?:\.\d+)?", s):
        try:
            a, b = s.split("/")
            den = float(b)
            if abs(den) < 1e-12:
                return None
            return float(a) / den
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def _is_integer_literal(text: str) -> bool:
    s = unicodedata.normalize("NFKC", (text or "").strip()).replace(",", "")
    return bool(re.fullmatch(r"[+-]?\d+", s))


def _decimal_places(text: str) -> int:
    s = unicodedata.normalize("NFKC", (text or "").strip()).replace(",", "")
    m = re.fullmatch(r"[+-]?\d+\.(\d+)", s)
    return len(m.group(1)) if m else 0


def _is_percent_literal(text: str) -> bool:
    s = unicodedata.normalize("NFKC", text or "").lower()
    return "%" in s or "percent" in s or "percentage" in s or "百分点" in s


def _numeric_subtype(gt_raw: str) -> str:
    if _is_percent_literal(gt_raw):
        return "percent_probability"
    if _is_integer_literal(gt_raw):
        return "discrete_exact"
    if _decimal_places(gt_raw) > 0:
        return "fixed_decimal"
    return "discrete_exact"


def _normalize_yn(text: str) -> str:
    s = _normalize_exact(text)
    if re.search(r"\bnot\s+(false|wrong|incorrect)\b", s):
        return "yes"
    if re.search(r"\bnot\s+(true|correct|right)\b", s):
        return "no"
    toks = set(_tokenize(s))
    if toks & _YES_TOKENS and not (toks & _NO_TOKENS):
        return "yes"
    if toks & _NO_TOKENS and not (toks & _YES_TOKENS):
        return "no"
    if s in _YES_TOKENS:
        return "yes"
    if s in _NO_TOKENS:
        return "no"
    return "unknown"


def _route_type(gt_list: List[str]) -> str:
    if not gt_list:
        return "short_phrase"
    if len(gt_list) > 1:
        nums = [_parse_numeric(x) for x in gt_list]
        return "multi_numeric_set" if all(v is not None for v in nums) else "multi_text_set"
    gt = gt_list[0]
    if _normalize_yn(gt) != "unknown":
        return "yes_no"
    gt_norm = _normalize_exact(gt)
    if re.fullmatch(r"[a-h]", gt_norm) or re.fullmatch(r"\(\s*[a-z]\s*\)", gt_norm):
        return "option_letter"
    if _parse_numeric(gt) is not None:
        return "numeric"
    tok_n = len(_tokenize(gt_norm))
    char_n = len(gt_norm)
    if char_n <= 12 and tok_n <= 3:
        return "short_phrase"
    if char_n <= 40:
        return "medium_phrase"
    return "long_sentence"


def _score_yn(pred: str, gt: str) -> Tuple[float, int]:
    p, g = _normalize_yn(pred), _normalize_yn(gt)
    ok = int(p != "unknown" and g != "unknown" and p == g)
    return float(ok), ok


def _score_option_letter(pred: str, gt: str) -> Tuple[float, int]:
    p, g = _normalize_exact(pred), _normalize_exact(gt)
    if _exact_match(pred, gt):
        return 1.0, 1
    m_gt = re.fullmatch(r"\(\s*([a-z])\s*\)", g)
    if m_gt and re.fullmatch(r"[a-z]", p) and p == m_gt.group(1):
        return 0.5, 0
    p_c = re.sub(r"^[\(\[]?([a-z])[\)\]\.\s]*$", r"\1", p)
    g_c = re.sub(r"^[\(\[]?([a-z])[\)\]\.\s]*$", r"\1", g)
    if p_c == g_c:
        return 1.0, 1
    m = re.match(r"^[\(\[]?([a-z])[\)\]\.\s:]+(.+)$", p)
    if m and m.group(1) == g_c:
        return 1.0, 1
    return 0.0, 0


def _score_numeric(pred: str, gt: str) -> Tuple[float, int]:
    p, g = _parse_numeric(pred), _parse_numeric(gt)
    if p is None or g is None:
        return min(_NUM_PARSE_FAIL_SOFT_CAP, _char_f1(pred, gt)), 0
    sub = _numeric_subtype(gt)
    abs_err = abs(p - g)
    rel_err = abs_err / max(abs(g), _NUM_REL_EPS)
    if sub == "discrete_exact":
        h = int(abs_err <= _NUM_EXACT_EPS)
        return float(h), h
    if sub == "fixed_decimal":
        d = _decimal_places(gt)
        tol = 0.5 * (10 ** (-d)) if d > 0 else _NUM_EXACT_EPS
        h = int(abs_err <= tol)
        s = max(0.0, min(1.0, 1.0 - abs_err / max(tol, 1e-12)))
        return s, h
    if sub == "percent_probability":
        pp = abs_err * 100.0
        tol = max(1e-9, _NUM_PERCENT_PP_TOL)
        h = int(pp <= tol)
        s = max(0.0, min(1.0, 1.0 - pp / tol))
        return s, h
    h = int(abs_err <= _NUM_ABS_TOL or rel_err <= _NUM_REL_TOL)
    decay = min(
        abs_err / max(_NUM_ABS_TOL, 1e-12),
        rel_err / max(_NUM_REL_TOL, 1e-12),
    )
    return max(0.0, min(1.0, 1.0 - decay)), h


def _sym_only(text: str) -> bool:
    n = _normalize_exact(text or "")
    return bool(n) and len(_tokenize(n)) == 0


def _ws_eq(pred: str, gt: str) -> bool:
    p, g = _normalize_exact(pred), _normalize_exact(gt)
    if not p or not g:
        return False
    return re.sub(r"\s+", "", p) == re.sub(r"\s+", "", g)


def _score_short(pred: str, gt: str) -> Tuple[float, int]:
    if _exact_match(pred, gt):
        return 1.0, 1
    if _sym_only(pred) and _sym_only(gt) and _ws_eq(pred, gt):
        return 0.5, 0
    sim = SequenceMatcher(None, _normalize_exact(pred), _normalize_exact(gt)).ratio()
    if sim >= 0.92:
        return min(_SHORT_SOFT_EDIT_CAP, 0.5), 0
    if sim >= 0.85:
        return min(_SHORT_SOFT_EDIT_CAP, 0.35), 0
    return 0.0, 0


def _neg_conflict(pred: str, gt: str) -> bool:
    pat = r"\b(no|not|never|none|n't)\b|不|没|无|非|不是"
    return bool(re.search(pat, _normalize_exact(pred))) != bool(re.search(pat, _normalize_exact(gt)))


def _num_conflict_penalty(pred: str, gt: str) -> float:
    def _nums(t: str) -> List[float]:
        out: List[float] = []
        for n in re.findall(r"[+-]?\d+(?:\.\d+)?", unicodedata.normalize("NFKC", t or "")):
            try:
                out.append(float(n))
            except Exception:
                pass
        return out

    g_nums = _nums(gt)
    if not g_nums:
        return 0.0
    p_nums = _nums(pred)
    matched = sum(
        1 for g in g_nums
        if any(abs(g - p) <= max(1e-6, 0.01 * max(abs(g), 1.0)) for p in p_nums)
    )
    return min(0.3, 0.15 * max(0, len(g_nums) - matched))


def _score_medium_long(pred: str, gt: str) -> Tuple[float, int]:
    if _exact_match(pred, gt):
        return 1.0, 1
    if _sym_only(pred) and _sym_only(gt) and _ws_eq(pred, gt):
        return 0.5, 0
    base = 0.7 * _char_f1(pred, gt) + 0.3 * _token_f1(pred, gt)
    penalty = (0.25 if _neg_conflict(pred, gt) else 0.0) + _num_conflict_penalty(pred, gt)
    return max(0.0, min(1.0, base - penalty)), 0


def _multiset_f1(pred_items: List[str], gt_items: List[str]) -> float:
    p_c: Dict[str, int] = {}
    g_c: Dict[str, int] = {}
    for x in pred_items:
        p_c[x] = p_c.get(x, 0) + 1
    for x in gt_items:
        g_c[x] = g_c.get(x, 0) + 1
    inter = sum(min(c, g_c.get(k, 0)) for k, c in p_c.items())
    if not pred_items and not gt_items:
        return 1.0
    if not pred_items or not gt_items:
        return 0.0
    p = inter / len(pred_items)
    r = inter / len(gt_items)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _score_multi_set(
    pred_items: List[str], gt_items: List[str], numeric: bool = False
) -> Tuple[float, int]:
    def _norm(vals: List[str]) -> List[str]:
        if numeric:
            ns = [x for x in (_parse_numeric(v) for v in vals) if x is not None]
            return [f"{x:.12g}" for x in ns]
        return [_normalize_exact(v) for v in vals if _normalize_exact(v)]

    p_n, g_n = _norm(pred_items), _norm(gt_items)
    h = int(sorted(p_n) == sorted(g_n))
    return _multiset_f1(p_n, g_n), h


def _clip01(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return max(0.0, min(1.0, x))


def compute_answer_judgement(answer: str, ground_truth: str) -> Dict[str, Any]:
    """Compute soft_score and hard_correct, matching reward/v3.py logic."""
    gt_list = _parse_ground_truths(ground_truth)
    ans_span = answer or ""
    pred_cands = _split_candidates(ans_span) or [ans_span]
    ans_type = _route_type(gt_list)
    soft_best, hard_best = 0.0, 0

    if not gt_list:
        return {
            "soft_score": 0.0, "hard_correct": 0,
            "answer_type": ans_type, "extracted_answer": ans_span,
        }

    if ans_type in {"multi_text_set", "multi_numeric_set"}:
        soft_best, hard_best = _score_multi_set(pred_cands, gt_list, numeric=ans_type == "multi_numeric_set")
    else:
        gt = gt_list[0]
        for cand in pred_cands:
            if ans_type == "yes_no":
                s, h = _score_yn(cand, gt)
            elif ans_type == "option_letter":
                s, h = _score_option_letter(cand, gt)
            elif ans_type == "numeric":
                s, h = _score_numeric(cand, gt)
            elif ans_type == "short_phrase":
                s, h = _score_short(cand, gt)
            else:
                s, h = _score_medium_long(cand, gt)
            if s > soft_best:
                soft_best = s
            if h > hard_best:
                hard_best = h
        n = max(1, len(pred_cands))
        soft_best /= float(n)
        hard_best = int(hard_best == 1 and n == 1)

    return {
        "soft_score": _clip01(soft_best),
        "hard_correct": int(hard_best),
        "answer_type": ans_type,
        "extracted_answer": ans_span,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading and stratified sampling
#  Adapted from examples/strategy_extraction/train_strategy_generation.py
#  _create_strategy_generation_dataset_per_subtask()
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(
    test_data_base: Path,
    subdirs: List[str],
    fewshot_k: int = 3,
    samples_per_subtask: int = 20,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Stratified sampling: exactly samples_per_subtask items per subtask JSON.

    Within each subtask file, picks (fewshot_k + 1) examples at random:
    the first fewshot_k become the few-shot context; the last one is the
    problem to solve. This mirrors _create_strategy_generation_dataset_per_subtask
    in train_strategy_generation.py with fewshot_min == fewshot_max == fewshot_k.
    """
    rng = random.Random(seed)
    dataset: List[Dict[str, Any]] = []

    for subdir in subdirs:
        data_dir = test_data_base / subdir
        if not data_dir.exists():
            logger.warning("Test subdir not found: %s", data_dir)
            continue

        split_name = subdir
        for problem_type_dir in sorted(
            [p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name
        ):
            problem_type = problem_type_dir.name
            for subtask_file in sorted(problem_type_dir.glob("*.json"), key=lambda p: p.name):
                try:
                    raw = json.loads(subtask_file.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning("Skip invalid JSON %s: %s", subtask_file, e)
                    continue

                examples_raw = raw.get("examples", [])
                if not isinstance(examples_raw, list):
                    logger.warning("Skip malformed examples field: %s", subtask_file)
                    continue

                # Filter valid examples
                pool: List[Dict[str, Any]] = []
                for ex in examples_raw:
                    if not isinstance(ex, dict):
                        continue
                    ex_input = ex.get("input", "")
                    ex_target = ex.get("target", [])
                    if not ex_input or not str(ex_input).strip():
                        continue
                    if isinstance(ex_target, list):
                        if not ex_target or not str(ex_target[0]).strip():
                            continue
                    elif not str(ex_target).strip():
                        continue
                    pool.append(ex)

                if len(pool) < 2:
                    logger.warning("Skip tiny subtask (<2 examples): %s", subtask_file)
                    continue

                subtask_name = subtask_file.stem
                subtask_id = f"{problem_type}/{subtask_name}"

                # Need at least fewshot_k+1 examples for one sample
                effective_k = min(fewshot_k, len(pool) - 1)
                n_pick = effective_k + 1  # few-shot + 1 problem

                for _ in range(samples_per_subtask):
                    picked = rng.sample(pool, n_pick)
                    fewshot = picked[:effective_k]
                    problem_ex = picked[-1]

                    problem_text = str(problem_ex.get("input", "") or "")
                    target_val = problem_ex.get("target", [])
                    if isinstance(target_val, list):
                        ground_truth = str(target_val[0]) if target_val else ""
                    else:
                        ground_truth = str(target_val)

                    if not problem_text.strip() or not ground_truth.strip():
                        continue

                    dataset.append({
                        "split": split_name,
                        "problem_type": problem_type,
                        "subtask": subtask_name,
                        "subtask_id": subtask_id,
                        "examples": fewshot,
                        "num_shots": len(fewshot),
                        "problem": problem_text,
                        "ground_truth": ground_truth,
                    })

    logger.info("Loaded %d evaluation samples across %d splits.", len(dataset), len(subdirs))
    return dataset


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt loading and formatting
# ═══════════════════════════════════════════════════════════════════════════

def load_toml_prompt(prompt_dir: Path, name: str) -> Dict[str, str]:
    path = prompt_dir / f"{name}.toml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def format_examples(examples: List[Dict[str, Any]]) -> str:
    """Serialize few-shot examples matching prompt/__init__.py format_examples()."""
    parts: List[str] = []
    for i, ex in enumerate(examples, 1):
        target = ex.get("target", [])
        if isinstance(target, list):
            target_text = target[0] if target else ""
        else:
            target_text = str(target)
        parts.append(f"Example {i}:")
        parts.append(f"Problem: {ex.get('input', '')}")
        parts.append(f"Solution: {target_text}")
        parts.append("")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
#  API call helpers
# ═══════════════════════════════════════════════════════════════════════════

async def _chat(
    client: AsyncOpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    seed: int,
    max_tokens: int = 4096,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> str:
    """Single chat completion call with retry logic."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_exc = e
            logger.warning("API call failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
    raise RuntimeError(f"API call failed after {max_retries} attempts") from last_exc


async def mist_inline_solve(
    client: AsyncOpenAI,
    task: Dict[str, Any],
    strategy_prompt: Dict[str, str],
    answer_prompt: Dict[str, str],
    model: str,
    temperature: float,
    seed: int,
    max_tokens: int = 4096,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Tuple[str, str, str]:
    """Two-call MIST-inline pipeline.

    Returns:
        (strategy_raw, strategy_text, answer_raw)
        strategy_text is the content extracted from <strategy>...</strategy>,
        falling back to strategy_raw if tags are absent.
    """
    examples_text = format_examples(task["examples"])

    # ── Call 1: strategy extraction ──────────────────────────────────────
    strategy_user = strategy_prompt["user"].format(examples_text=examples_text)
    strategy_raw = await _chat(
        client, model,
        system=strategy_prompt["system"],
        user=strategy_user,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    strategy_text = extract_strategy(strategy_raw) or strategy_raw.strip()

    # ── Call 2: answer generation ─────────────────────────────────────────
    answer_user = answer_prompt["user"].format(
        strategy=strategy_text,
        problem=task["problem"],
    )
    answer_raw = await _chat(
        client, model,
        system=answer_prompt["system"],
        user=answer_user,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    return strategy_raw, strategy_text, answer_raw


async def few_shot_solve(
    client: AsyncOpenAI,
    task: Dict[str, Any],
    icl_prompt: Dict[str, str],
    model: str,
    temperature: float,
    seed: int,
    max_tokens: int = 4096,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> str:
    """Single-call few-shot (ICL) pipeline.

    The model receives few-shot examples and the new problem in one message
    and answers directly — no intermediate strategy extraction.

    Returns:
        answer_raw  (full model response)
    """
    examples_text = format_examples(task["examples"])
    user = icl_prompt["user"].format(
        examples_text=examples_text,
        problem=task["problem"],
    )
    return await _chat(
        client, model,
        system=icl_prompt["system"],
        user=user,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation loop
# ═══════════════════════════════════════════════════════════════════════════

async def run_eval(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    mode = args.mode
    if mode not in ("mist-inline", "few-shot"):
        logger.error("Unknown mode '%s'. Choose: mist-inline | few-shot", mode)
        sys.exit(1)
    logger.info("Mode: %s", mode)

    # Load prompts
    if mode == "mist-inline":
        strategy_prompt = load_toml_prompt(args.prompt_dir, args.inline_strategy_prompt_version)
        answer_prompt = load_toml_prompt(args.prompt_dir, args.answer_prompt_version)
        logger.info("Prompts: strategy=%s, answer=%s",
                    args.inline_strategy_prompt_version, args.answer_prompt_version)
    else:  # few-shot
        icl_prompt = load_toml_prompt(args.prompt_dir, args.icl_prompt_version)
        strategy_prompt = answer_prompt = {}  # unused in few-shot mode
        logger.info("Prompt: icl=%s", args.icl_prompt_version)

    # Load dataset
    dataset = load_dataset(
        test_data_base=args.test_data_base,
        subdirs=args.test_subdirs,
        fewshot_k=args.fewshot_k,
        samples_per_subtask=args.samples_per_subtask,
        seed=args.val_sampling_seed,
    )
    if not dataset:
        logger.error("Dataset is empty — check test data paths.")
        sys.exit(1)
    if args.max_samples is not None:
        dataset = dataset[: args.max_samples]
        logger.info("max-samples cap applied: using %d problems.", len(dataset))

    logger.info(
        "Evaluation: %d problems, model=%s, concurrency=%d, temperature=%.2f",
        len(dataset), args.answer_model_name, args.concurrency, args.temperature,
    )

    # OpenAI client
    client = AsyncOpenAI(
        base_url=args.answer_model_base_url,
        api_key=args.api_key,
        timeout=120.0,
    )

    # Per-problem results
    results: List[Dict[str, Any]] = [{}] * len(dataset)
    queue: asyncio.Queue[Tuple[int, Dict[str, Any]]] = asyncio.Queue()
    for idx, task in enumerate(dataset):
        queue.put_nowait((idx, task))

    completed = 0
    total = len(dataset)
    lock = asyncio.Lock()

    async def _worker() -> None:
        nonlocal completed
        while True:
            try:
                idx, task = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                if mode == "mist-inline":
                    strategy_raw, strategy_text, answer_raw = await mist_inline_solve(
                        client=client,
                        task=task,
                        strategy_prompt=strategy_prompt,
                        answer_prompt=answer_prompt,
                        model=args.answer_model_name,
                        temperature=args.temperature,
                        seed=args.llm_seed,
                        max_tokens=args.max_tokens,
                        max_retries=args.max_retries,
                        retry_delay=args.retry_delay_sec,
                    )
                else:  # few-shot
                    answer_raw = await few_shot_solve(
                        client=client,
                        task=task,
                        icl_prompt=icl_prompt,
                        model=args.answer_model_name,
                        temperature=args.temperature,
                        seed=args.llm_seed,
                        max_tokens=args.max_tokens,
                        max_retries=args.max_retries,
                        retry_delay=args.retry_delay_sec,
                    )
                    strategy_raw = strategy_text = ""
                extracted = extract_answer(answer_raw)
                judgement = compute_answer_judgement(
                    answer=extracted or "",
                    ground_truth=task["ground_truth"],
                )
            except Exception as e:
                logger.warning("Problem %d failed: %s", idx, e)
                strategy_raw = strategy_text = answer_raw = ""
                extracted = None
                judgement = {"soft_score": 0.0, "hard_correct": 0, "answer_type": "unknown", "extracted_answer": ""}

            results[idx] = {
                "idx": idx,
                "split": task["split"],
                "problem_type": task["problem_type"],
                "subtask": task["subtask"],
                "subtask_id": task["subtask_id"],
                "problem": task["problem"],
                "ground_truth": task["ground_truth"],
                "strategy_raw": strategy_raw,
                "strategy_extracted": strategy_text,
                "answer_raw": answer_raw,
                "extracted_answer": extracted or "",
                "soft_score": judgement["soft_score"],
                "hard_correct": judgement["hard_correct"],
                "answer_type": judgement.get("answer_type", ""),
            }
            async with lock:
                completed += 1
                if completed % 50 == 0 or completed == total:
                    logger.info("Progress: %d/%d", completed, total)
            queue.task_done()

    workers = [asyncio.create_task(_worker()) for _ in range(args.concurrency)]
    await asyncio.gather(*workers)
    await client.close()

    # ── Metrics ────────────────────────────────────────────────────────────
    valid = [r for r in results if r]

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    overall_soft = [r["soft_score"] for r in valid]
    overall_hard = [float(r["hard_correct"]) for r in valid]

    by_split_soft: Dict[str, List[float]] = {}
    by_split_hard: Dict[str, List[float]] = {}
    for r in valid:
        sp = r["split"]
        by_split_soft.setdefault(sp, []).append(r["soft_score"])
        by_split_hard.setdefault(sp, []).append(float(r["hard_correct"]))

    by_subtask_hard: Dict[str, List[float]] = {}
    for r in valid:
        by_subtask_hard.setdefault(r["subtask_id"], []).append(float(r["hard_correct"]))

    width = 90
    print("\n" + "=" * width)
    print(f"{mode} eval — BBH / ID / OOD   pass@1 summary")
    print("=" * width)
    print(f"Model:       {args.answer_model_name}")
    print(f"Endpoint:    {args.answer_model_base_url}")
    print(f"Temperature: {args.temperature:.2f}   seed={args.llm_seed}")
    print(f"Samples/subtask: {args.samples_per_subtask}   fewshot_k={args.fewshot_k}")
    print(f"Problems:    {len(valid)}/{len(dataset)}")
    print("-" * width)
    if overall_hard:
        print(f"pass@1 (hard, all):  {_mean(overall_hard):.4f}   ({sum(int(v) for v in overall_hard)}/{len(overall_hard)})")
    if overall_soft:
        print(f"pass@1 (soft, all):  {_mean(overall_soft):.4f}")
    print()
    print("By split:")
    for split in sorted(by_split_hard.keys()):
        hard_vals = by_split_hard.get(split, [])
        soft_vals = by_split_soft.get(split, [])
        n = len(hard_vals)
        hard_acc = _mean(hard_vals)
        soft_acc = _mean(soft_vals)
        correct = sum(int(v) for v in hard_vals)
        print(f"  {split:25s}  n={n:4d}  pass@1(hard)={hard_acc:.4f} ({correct}/{n})  pass@1(soft)={soft_acc:.4f}")
    print("=" * width + "\n")

    # ── Save results ────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_slug = re.sub(r"[^a-zA-Z0-9_.-]", "_", args.answer_model_name)
    mode_slug = mode.replace("-", "_")
    result_file = out_dir / f"eval_{mode_slug}_{model_slug}_{timestamp}.json"

    summary = {
        "timestamp": timestamp,
        "mode": mode,
        "model": args.answer_model_name,
        "base_url": args.answer_model_base_url,
        "temperature": args.temperature,
        "llm_seed": args.llm_seed,
        "fewshot_k": args.fewshot_k,
        "samples_per_subtask": args.samples_per_subtask,
        "val_sampling_seed": args.val_sampling_seed,
        "inline_strategy_prompt_version": args.inline_strategy_prompt_version if mode == "mist-inline" else "",
        "answer_prompt_version": args.answer_prompt_version if mode == "mist-inline" else "",
        "icl_prompt_version": args.icl_prompt_version if mode == "few-shot" else "",
        "total_problems": len(dataset),
        "completed_problems": len(valid),
        "pass1_hard_all": _mean(overall_hard),
        "pass1_soft_all": _mean(overall_soft),
        "by_split": {
            sp: {
                "n": len(by_split_hard.get(sp, [])),
                "pass1_hard": _mean(by_split_hard.get(sp, [])),
                "pass1_soft": _mean(by_split_soft.get(sp, [])),
            }
            for sp in sorted(set(by_split_hard) | set(by_split_soft))
        },
        "by_subtask": {
            sid: {
                "n": len(v),
                "pass1_hard": _mean(v),
            }
            for sid, v in sorted(by_subtask_hard.items())
        },
        "results": valid,
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Results saved: %s", result_file)


# ═══════════════════════════════════════════════════════════════════════════
#  Argument parsing and entry point
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone MIST-inline evaluation for BBH / ID / OOD benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model / API
    p.add_argument("--answer-model-base-url", default=os.environ.get("ANSWER_MODEL_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--answer-model-name", default=os.environ.get("ANSWER_MODEL_NAME", "gpt-4o"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy-key"),
                   help="API key (also reads OPENAI_API_KEY env var).")
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.0")))
    p.add_argument("--llm-seed", type=int, default=int(os.environ.get("LLM_SEED", "42")))
    p.add_argument("--max-tokens", type=int, default=8192)
    # Data
    p.add_argument("--test-data-base", type=Path,
                   default=TEST_DATA_BASE,
                   help="Base directory containing test-bbh / test-id-subtask / test-ood-task.")
    p.add_argument("--test-subdirs", nargs="+", default=TEST_SUBDIRS)
    p.add_argument("--fewshot-k", type=int, default=int(os.environ.get("FEWSHOT_K", "3")),
                   help="Number of few-shot examples per problem.")
    p.add_argument("--samples-per-subtask", type=int,
                   default=int(os.environ.get("VAL_SAMPLES_PER_SUBTASK", "20")))
    p.add_argument("--val-sampling-seed", type=int,
                   default=int(os.environ.get("VAL_SAMPLING_SEED", "42")))
    p.add_argument("--max-samples", type=int, default=None,
                   help="Hard cap on total problems (useful for quick smoke tests).")
    # Mode
    p.add_argument(
        "--mode",
        choices=["mist-inline", "few-shot"],
        default=os.environ.get("MODE", "mist-inline"),
        help="mist-inline: 2-call strategy+answer pipeline. few-shot: 1-call ICL direct answer.",
    )
    # Prompts
    p.add_argument("--prompt-dir", type=Path, default=PROMPT_DIR)
    p.add_argument("--inline-strategy-prompt-version", default="mist_inline_strategy")
    p.add_argument("--answer-prompt-version", default="mist_inline_answer")
    p.add_argument("--icl-prompt-version", default="ICL(few-shot)",
                   help="Prompt file stem for few-shot mode (default: 'ICL(few-shot)').")
    # Execution
    p.add_argument("--concurrency", type=int,
                   default=int(os.environ.get("EVAL_CONCURRENCY", "4")))
    p.add_argument("--max-retries", type=int,
                   default=int(os.environ.get("MAX_RETRIES", "3")))
    p.add_argument("--retry-delay-sec", type=float,
                   default=float(os.environ.get("RETRY_DELAY_SEC", "1.0")))
    # Output
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", str(SCRIPT_DIR / "results")))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
