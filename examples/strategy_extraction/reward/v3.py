# Copyright (c) Microsoft. All rights reserved.

"""Reward v3 — type-routed answer judging with continuous soft score.

Design goals:
1) Keep high precision on dominant short-label samples.
2) Reduce misjudgment on yes/no and numeric answers.

Notes:
- This module keeps the existing RewardConfig interface:
  `compute_answer_correctness(answer, ground_truth, numeric_tolerance, f1_threshold) -> float`
  and returns a soft score in [0, 1].
- A richer helper `compute_answer_judgement(...)` is also provided and returns
  both `soft_score` and `hard_correct` for logging/analysis.
"""

from __future__ import annotations

import ast
import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import RewardConfig, register


@dataclass(frozen=True)
class JudgeConfig:
    # Numeric thresholds
    numeric_abs_tol: float = 1e-3
    numeric_rel_tol: float = 0.02
    numeric_rel_eps: float = 0.1
    numeric_parse_fail_soft_cap: float = 0.25
    numeric_exact_eps: float = 1e-12
    numeric_percent_pp_tol: float = 0.5  # percentage-point tolerance, e.g. 0.5pp

    # Short phrase fuzzy cap (non-exact branch)
    short_soft_edit_cap: float = 0.5

    # Candidate split
    enable_candidate_split: bool = True
    split_on_comma_for_short_candidates: bool = False

    # Set mode: "set" ignores duplicates; "multiset" counts duplicates
    set_mode: str = "multiset"


CFG = JudgeConfig()

# --- Extraction patterns --- #
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
BOXED_PATTERN = re.compile(
    r"\\boxed\{((?:[^{}]|(?:\{[^{}]*\}))*)\}",
    re.DOTALL,
)
FINAL_CUE_PATTERN = re.compile(
    r"(?:final\s*answer|answer\s*is|答案是|最终答案)\s*[:：]\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
FENCE_PATTERN = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$")


# --- Normalization / tokenization --- #
def _normalize_text(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "")
    s = s.strip().strip("`\"'“”‘’")
    s = re.sub(r"\s+", " ", s)
    # Trim leading/trailing punctuation noise.
    s = re.sub(r"^[\s\.,;:!?\-_=+~|/\\]+", "", s)
    s = re.sub(r"[\s\.,;:!?\-_=+~|/\\]+$", "", s)
    return s.lower().strip()


def _normalize_for_exact(text: str) -> str:
    s = _normalize_text(text)
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize(text: str) -> List[str]:
    s = _normalize_for_exact(text)
    # Keep CJK chars and latin/alnum chunks.
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", s)


def _char_f1(pred: str, gt: str) -> float:
    p = _normalize_for_exact(pred).replace(" ", "")
    g = _normalize_for_exact(gt).replace(" ", "")
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    # SequenceMatcher is robust on mixed CJK/Latin strings.
    return SequenceMatcher(None, p, g).ratio()


def _token_f1(pred: str, gt: str) -> float:
    p_toks = _tokenize(pred)
    g_toks = _tokenize(gt)
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    p_count: Dict[str, int] = {}
    g_count: Dict[str, int] = {}
    for t in p_toks:
        p_count[t] = p_count.get(t, 0) + 1
    for t in g_toks:
        g_count[t] = g_count.get(t, 0) + 1
    inter = 0
    for t, c in p_count.items():
        inter += min(c, g_count.get(t, 0))
    if inter == 0:
        return 0.0
    precision = inter / len(p_toks)
    recall = inter / len(g_toks)
    return (2 * precision * recall) / (precision + recall)


def _exact_match(a: str, b: str) -> bool:
    return _normalize_for_exact(a) == _normalize_for_exact(b)


# --- Parsing helpers --- #
def _parse_ground_truths(ground_truth: str) -> List[str]:
    gt = (ground_truth or "").strip()
    if not gt:
        return []
    # Try Python-literal list/tuple: "['a', 'b']", "(2, 2)", etc.
    try:
        obj = ast.literal_eval(gt)
        # Tuple of only int/float (e.g. tensor/matrix *shape* "(2,2)") is one answer,
        # not a multiset of separate numeric labels. Expanding it makes preds like
        # "(2,2)" fail _parse_numeric and score 0 against ['2','2'].
        if isinstance(obj, tuple) and len(obj) > 0:
            if all(isinstance(x, (int, float)) for x in obj):
                return [gt]
        if isinstance(obj, (list, tuple)):
            vals = [str(x).strip() for x in obj if str(x).strip()]
            if vals:
                return vals
    except Exception:
        pass
    return [gt]


def _strip_code_fence(text: str) -> str:
    return FENCE_PATTERN.sub("", (text or "").strip()).strip()


def _extract_final_span(output: str) -> str:
    text = (output or "").strip()
    if not text:
        return ""

    # 1) Last non-empty <answer>...</answer>
    tag_matches = ANSWER_TAG_PATTERN.findall(text)
    if tag_matches:
        candidate = (tag_matches[-1] or "").strip()
        if candidate:
            return candidate

    # 2) Last explicit final cue
    cue_matches = list(FINAL_CUE_PATTERN.finditer(text))
    if cue_matches:
        candidate = (cue_matches[-1].group(1) or "").strip()
        if candidate:
            return _strip_code_fence(candidate)

    # 3) Last boxed expression
    boxed_matches = BOXED_PATTERN.findall(text)
    if boxed_matches:
        candidate = (boxed_matches[-1] or "").strip()
        if candidate:
            return candidate

    # 4) Last non-empty line
    lines = [ln.strip() for ln in _strip_code_fence(text).splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return text


def extract_answer(output: str) -> Optional[str]:
    """Extract final answer span with robust fallback."""
    span = _extract_final_span(output)
    span = span.strip()
    return span if span else None


def _is_short_candidate_piece(piece: str) -> bool:
    s = _normalize_for_exact(piece)
    if not s:
        return False
    tok = _tokenize(s)
    return len(tok) <= 6 and len(s) <= 40


def _split_candidates(final_span: str, cfg: JudgeConfig) -> List[str]:
    if not final_span:
        return []
    base = final_span.strip()
    if not cfg.enable_candidate_split:
        return [base]

    # First pass: strong delimiters.
    pieces = [base]
    for sep_pat in [r"\|", r";", r"；", r"\bor\b", r"或者"]:
        new_pieces: List[str] = []
        for p in pieces:
            split_parts = [x.strip() for x in re.split(sep_pat, p, flags=re.IGNORECASE) if x.strip()]
            new_pieces.extend(split_parts if split_parts else [p])
        pieces = new_pieces

    # Optional comma split only when all parts look like short candidates.
    if cfg.split_on_comma_for_short_candidates and len(pieces) == 1:
        maybe = [x.strip() for x in re.split(r",|，", pieces[0]) if x.strip()]
        if len(maybe) > 1 and all(_is_short_candidate_piece(x) for x in maybe):
            pieces = maybe

    dedup: List[str] = []
    seen = set()
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        k = _normalize_for_exact(p)
        if k in seen:
            continue
        seen.add(k)
        dedup.append(p)
    return dedup or [base]


# --- Type routing --- #
YES_TOKENS = {
    "yes", "y", "true", "t", "correct", "right",
    "是", "对", "正确", "有",
}
NO_TOKENS = {
    "no", "n", "false", "f", "incorrect", "wrong",
    "否", "不", "不是", "错误", "无",
}


def _extract_numbers(text: str) -> List[float]:
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", unicodedata.normalize("NFKC", text or ""))
    out: List[float] = []
    for n in nums:
        try:
            out.append(float(n))
        except Exception:
            continue
    return out


def _parse_numeric(text: str) -> Optional[float]:
    s = _normalize_text(text)
    if not s:
        return None
    s = s.replace(",", "")
    # Percent
    if s.endswith("%"):
        try:
            return float(s[:-1].strip()) / 100.0
        except Exception:
            return None
    # Fraction
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?/[+-]?\d+(?:\.\d+)?", s):
        try:
            a, b = s.split("/")
            den = float(b)
            if abs(den) < 1e-12:
                return None
            return float(a) / den
        except Exception:
            return None
    # Plain float
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
    if not m:
        return 0
    return len(m.group(1))


def _is_percent_literal(text: str) -> bool:
    s = unicodedata.normalize("NFKC", text or "").lower()
    return "%" in s or "percent" in s or "percentage" in s or "百分点" in s


def _is_continuous_numeric_task(task_meta: Optional[Dict[str, Any]]) -> bool:
    """Only enable relative-error branch for explicit approximate/continuous tasks."""
    if not isinstance(task_meta, dict):
        return False
    mode = str(task_meta.get("numeric_mode", "")).lower()
    if mode in {"continuous", "approx", "approximate", "measurement"}:
        return True
    if bool(task_meta.get("allow_relative_tolerance", False)):
        return True
    return False


def _numeric_subtype(
    gt_raw: str,
    task_meta: Optional[Dict[str, Any]],
) -> str:
    # Prefer explicit task metadata if provided.
    if _is_continuous_numeric_task(task_meta):
        return "continuous_measurement"
    if _is_percent_literal(gt_raw):
        return "percent_probability"
    if _is_integer_literal(gt_raw):
        return "discrete_exact"
    if _decimal_places(gt_raw) > 0:
        return "fixed_decimal"
    # Conservative fallback when no metadata is available.
    return "discrete_exact"


def _normalize_yes_no_label(text: str) -> str:
    s = _normalize_for_exact(text)
    # Simple double-negation guards first.
    if re.search(r"\bnot\s+(false|wrong|incorrect)\b", s):
        return "yes"
    if re.search(r"\bnot\s+(true|correct|right)\b", s):
        return "no"

    toks = set(_tokenize(s))
    if toks & YES_TOKENS and not (toks & NO_TOKENS):
        return "yes"
    if toks & NO_TOKENS and not (toks & YES_TOKENS):
        return "no"

    # Exact Chinese/English phrase fallback.
    if s in YES_TOKENS:
        return "yes"
    if s in NO_TOKENS:
        return "no"
    return "unknown"


def _route_type(gt_list: Sequence[str]) -> str:
    if not gt_list:
        return "short_phrase"
    if len(gt_list) > 1:
        nums = [_parse_numeric(x) for x in gt_list]
        if all(v is not None for v in nums):
            return "multi_numeric_set"
        return "multi_text_set"

    gt = gt_list[0]
    yn = _normalize_yes_no_label(gt)
    if yn != "unknown":
        return "yes_no"

    gt_norm = _normalize_for_exact(gt)
    if re.fullmatch(r"[a-h]", gt_norm):
        return "option_letter"
    # "(C)" / "(a)" style labels: treat like option_letter, not short_phrase fuzzy.
    if re.fullmatch(r"\(\s*[a-z]\s*\)", gt_norm):
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


# --- Scorers --- #
def _score_yes_no(pred: str, gt: str) -> Tuple[float, int]:
    p = _normalize_yes_no_label(pred)
    g = _normalize_yes_no_label(gt)
    ok = int(p != "unknown" and g != "unknown" and p == g)
    return float(ok), ok


def _score_option_letter(pred: str, gt: str) -> Tuple[float, int]:
    p = _normalize_for_exact(pred)
    g = _normalize_for_exact(gt)

    if _exact_match(pred, gt):
        return 1.0, 1

    # Gold is "(C)" but model outputs bare "C": letter correct, format incomplete → partial soft.
    m_gt_paren = re.fullmatch(r"\(\s*([a-z])\s*\)", g)
    if m_gt_paren:
        inner_g = m_gt_paren.group(1)
        if re.fullmatch(r"[a-z]", p) and p == inner_g:
            return 0.5, 0

    # 1. 基础清理：解决纯选项的符号包裹问题（原版已实现）
    # 容忍 "(a)" / "a." / "a)"
    p_clean = re.sub(r"^[\(\[]?([a-z])[\)\]\.\s]*$", r"\1", p)
    g_clean = re.sub(r"^[\(\[]?([a-z])[\)\]\.\s]*$", r"\1", g)
    if p_clean == g_clean:
        return 1.0, 1

    # 2. 进阶优化：解决“选项+文字”共存的问题（新增方案）
    # 场景：模型输出 "a. 苹果" 或 "a) apple" 或 "a: 苹果"
    # 正则释义：匹配开头的一个字母，紧接着必须是标点(.) (:) ()) 或空格，然后跟着任意长度的其他文字
    match = re.match(r"^[\(\[]?([a-z])[\)\]\.\s:]+(.+)$", p)
    if match:
        extracted_letter = match.group(1)  # 只把最前面的选项字母抠出来
        if extracted_letter == g_clean:
            # 只要抬头阵的选项字母对了，我们就认为答对了！
            return 1.0, 1

    # 如果以上都不满足，说明选项真的选错了
    return 0.0, 0


def _score_numeric(
    pred: str,
    gt: str,
    cfg: JudgeConfig,
    *,
    task_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[float, int]:
    p = _parse_numeric(pred)
    g = _parse_numeric(gt)
    if p is None or g is None:
        soft = min(cfg.numeric_parse_fail_soft_cap, _char_f1(pred, gt))
        return soft, 0

    subtype = _numeric_subtype(gt, task_meta)
    abs_err = abs(p - g)
    rel_err = abs_err / max(abs(g), cfg.numeric_rel_eps)

    if subtype == "discrete_exact":
        hard = int(abs_err <= cfg.numeric_exact_eps)
        soft = float(hard)
        return soft, hard

    if subtype == "fixed_decimal":
        d = _decimal_places(gt)
        abs_tol = 0.5 * (10 ** (-d)) if d > 0 else cfg.numeric_exact_eps
        hard = int(abs_err <= abs_tol)
        soft = max(0.0, min(1.0, 1.0 - (abs_err / max(abs_tol, 1e-12))))
        return soft, hard

    if subtype == "percent_probability":
        # parse_numeric converts "12%" -> 0.12, so convert error to percentage points.
        pp_err = abs_err * 100.0
        pp_tol = max(1e-9, cfg.numeric_percent_pp_tol)
        hard = int(pp_err <= pp_tol)
        soft = max(0.0, min(1.0, 1.0 - (pp_err / pp_tol)))
        return soft, hard

    # continuous_measurement: keep abs+rel combined rule
    hard = int(abs_err <= cfg.numeric_abs_tol or rel_err <= cfg.numeric_rel_tol)
    abs_part = abs_err / max(cfg.numeric_abs_tol, 1e-12)
    rel_part = rel_err / max(cfg.numeric_rel_tol, 1e-12)
    decay = min(abs_part, rel_part)
    soft = max(0.0, min(1.0, 1.0 - decay))
    return soft, hard


def _is_symbol_only_span(text: str) -> bool:
    """True if normalized text has no alphanumeric/CJK tokens (e.g. Dyck ') )' / '[]')."""
    n = _normalize_for_exact(text or "")
    return bool(n) and len(_tokenize(n)) == 0


def _ws_stripped_equal(pred: str, gt: str) -> bool:
    """Compare strings after normalization and removing all whitespace."""
    p = _normalize_for_exact(pred)
    g = _normalize_for_exact(gt)
    if not p or not g:
        return False
    return re.sub(r"\s+", "", p) == re.sub(r"\s+", "", g)


def _score_short_phrase(pred: str, gt: str, cfg: JudgeConfig) -> Tuple[float, int]:
    if _exact_match(pred, gt):
        return 1.0, 1
    # Dyck / delimiter-only: ') )' vs '))' — SequenceMatcher ~0.8, below fuzzy thresholds.
    if (
        _is_symbol_only_span(pred)
        and _is_symbol_only_span(gt)
        and _ws_stripped_equal(pred, gt)
    ):
        return 0.5, 0
    sim = SequenceMatcher(None, _normalize_for_exact(pred), _normalize_for_exact(gt)).ratio()
    if sim >= 0.92:
        return min(cfg.short_soft_edit_cap, 0.5), 0
    if sim >= 0.85:
        return min(cfg.short_soft_edit_cap, 0.35), 0
    return 0.0, 0


def _negation_conflict(pred: str, gt: str) -> bool:
    p = _normalize_for_exact(pred)
    g = _normalize_for_exact(gt)
    neg_pat = r"\b(no|not|never|none|n't)\b|不|没|无|非|不是"
    p_neg = bool(re.search(neg_pat, p))
    g_neg = bool(re.search(neg_pat, g))
    return p_neg != g_neg


def _numeric_conflict_penalty(pred: str, gt: str) -> float:
    p_nums = _extract_numbers(pred)
    g_nums = _extract_numbers(gt)
    if not g_nums:
        return 0.0
    matched = 0
    for g in g_nums:
        # Numeric symbol equality with small tolerance.
        if any(abs(g - p) <= max(1e-6, 0.01 * max(abs(g), 1.0)) for p in p_nums):
            matched += 1
    mismatches = max(0, len(g_nums) - matched)
    return min(0.3, 0.15 * mismatches)


def _score_medium_long(pred: str, gt: str) -> Tuple[float, int]:
    if _exact_match(pred, gt):
        return 1.0, 1
    if (
        _is_symbol_only_span(pred)
        and _is_symbol_only_span(gt)
        and _ws_stripped_equal(pred, gt)
    ):
        return 0.5, 0
    hard = 0
    base = 0.7 * _char_f1(pred, gt) + 0.3 * _token_f1(pred, gt)
    penalty = 0.0
    if _negation_conflict(pred, gt):
        penalty += 0.25
    penalty += _numeric_conflict_penalty(pred, gt)
    soft = max(0.0, min(1.0, base - penalty))
    return soft, hard


def _norm_list(vals: Sequence[str]) -> List[str]:
    return [_normalize_for_exact(v) for v in vals if _normalize_for_exact(v)]


def _multiset_f1(pred_items: Sequence[str], gt_items: Sequence[str]) -> float:
    p_count: Dict[str, int] = {}
    g_count: Dict[str, int] = {}
    for x in pred_items:
        p_count[x] = p_count.get(x, 0) + 1
    for x in gt_items:
        g_count[x] = g_count.get(x, 0) + 1
    inter = 0
    for k, c in p_count.items():
        inter += min(c, g_count.get(k, 0))
    if not pred_items and not gt_items:
        return 1.0
    if not pred_items or not gt_items:
        return 0.0
    p = inter / len(pred_items)
    r = inter / len(gt_items)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def _score_multi_set(
    pred_items: Sequence[str],
    gt_items: Sequence[str],
    cfg: JudgeConfig,
    numeric: bool = False,
) -> Tuple[float, int]:
    if numeric:
        p_nums = [x for x in (_parse_numeric(v) for v in pred_items) if x is not None]
        g_nums = [x for x in (_parse_numeric(v) for v in gt_items) if x is not None]
        p_norm = [f"{x:.12g}" for x in p_nums]
        g_norm = [f"{x:.12g}" for x in g_nums]
    else:
        p_norm = _norm_list(pred_items)
        g_norm = _norm_list(gt_items)

    if cfg.set_mode == "set":
        p_set = sorted(set(p_norm))
        g_set = sorted(set(g_norm))
        hard = int(p_set == g_set)
        soft = _multiset_f1(p_set, g_set)
        return soft, hard

    # multiset mode
    hard = int(sorted(p_norm) == sorted(g_norm))
    soft = _multiset_f1(p_norm, g_norm)
    return soft, hard


def _clip01(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return max(0.0, min(1.0, x))


def compute_answer_judgement(
    answer: str,
    ground_truth: str,
    *,
    numeric_tolerance: float = 0.02,
    f1_threshold: float = 0.5,  # kept for signature compatibility; not used directly
    task_meta: Optional[Dict[str, Any]] = None,
    cfg: JudgeConfig = CFG,
) -> Dict[str, Any]:
    """Return detailed judgement with both soft and hard signals."""
    _ = f1_threshold

    cfg_run = JudgeConfig(
        numeric_abs_tol=cfg.numeric_abs_tol,
        numeric_rel_tol=max(1e-9, numeric_tolerance),
        numeric_rel_eps=cfg.numeric_rel_eps,
        numeric_parse_fail_soft_cap=cfg.numeric_parse_fail_soft_cap,
        numeric_exact_eps=cfg.numeric_exact_eps,
        numeric_percent_pp_tol=cfg.numeric_percent_pp_tol,
        short_soft_edit_cap=cfg.short_soft_edit_cap,
        enable_candidate_split=cfg.enable_candidate_split,
        split_on_comma_for_short_candidates=cfg.split_on_comma_for_short_candidates,
        set_mode=cfg.set_mode,
    )

    gt_list = _parse_ground_truths(ground_truth)
    ans_span = answer or ""
    pred_candidates = _split_candidates(ans_span, cfg_run)
    if not pred_candidates:
        pred_candidates = [ans_span]

    ans_type = _route_type(gt_list)
    soft_best = 0.0
    hard_best = 0

    if not gt_list:
        return {
            "soft_score": 0.0,
            "hard_correct": 0,
            "answer_type": ans_type,
            "extracted_answer": ans_span,
            "candidates": pred_candidates,
        }

    if ans_type in {"multi_text_set", "multi_numeric_set"}:
        numeric = ans_type == "multi_numeric_set"
        soft, hard = _score_multi_set(pred_candidates, gt_list, cfg_run, numeric=numeric)
        soft_best, hard_best = soft, hard
    else:
        gt = gt_list[0]
        for cand in pred_candidates:
            if ans_type == "yes_no":
                soft, hard = _score_yes_no(cand, gt)
            elif ans_type == "option_letter":
                soft, hard = _score_option_letter(cand, gt)
            elif ans_type == "numeric":
                soft, hard = _score_numeric(cand, gt, cfg_run, task_meta=task_meta)
            elif ans_type == "short_phrase":
                soft, hard = _score_short_phrase(cand, gt, cfg_run)
            else:
                soft, hard = _score_medium_long(cand, gt)
            if soft > soft_best:
                soft_best = soft
            if hard > hard_best:
                hard_best = hard

        # Candidate ambiguity penalty for single-element classes.
        n = max(1, len(pred_candidates))
        soft_best = soft_best / float(n)
        # Hard requires unique deterministic answer and strict match.
        hard_best = int(hard_best == 1 and n == 1)

    return {
        "soft_score": _clip01(soft_best),
        "hard_correct": int(hard_best),
        "answer_type": ans_type,
        "extracted_answer": ans_span,
        "candidates": pred_candidates,
    }


def compute_answer_correctness(
    answer: str,
    ground_truth: str,
    numeric_tolerance: float = 0.02,
    f1_threshold: float = 0.5,
) -> float:
    """Compatibility hook required by RewardConfig.

    Returns soft score in [0, 1]. Use `compute_answer_judgement` if you need
    hard correctness and diagnostics.
    """
    detail = compute_answer_judgement(
        answer=answer,
        ground_truth=ground_truth,
        numeric_tolerance=numeric_tolerance,
        f1_threshold=f1_threshold,
    )
    return float(detail["soft_score"])


def compute_final_reward(
    format_reward: float,
    correctness: float,
    format_weight: float,
    correctness_weight: float,
) -> float:
    return format_weight * format_reward + correctness_weight * correctness


REWARD = RewardConfig(
    name="v3",
    extract_answer=extract_answer,
    compute_answer_correctness=compute_answer_correctness,
    compute_final_reward=compute_final_reward,
)

register("v3", REWARD)

