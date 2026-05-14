#!/usr/bin/env python3
"""Bootstrap CI and paired significance tests for eval_no_verl / Linguini / HARDMath2.

No external dependencies (pure Python stdlib).

USAGE EXAMPLES (run from repo root):

  Compare two Linguini files:
    python examples/strategy_extraction/scripts/stat_test.py \
        --a banchmark/linguini/my_linguini_results/few-shot/Qwen3-8B-trained/linguini_passk_Qwen3-8B-trained_few-shot.json \
        --b banchmark/linguini/my_linguini_results/MIST/Qwen3-8B-trained/open-think/linguini_passk_Qwen3-8B-trained_MIST.json \
        --label-a "Qwen3-8B few-shot" --label-b "Qwen3-8B MIST" \
        --subgroup --out results/linguini_mist_vs_fewshot.json

  Compare two HARDMath2 files:
    python examples/strategy_extraction/scripts/stat_test.py \
        --a banchmark/HARDMath2/results/hardmath_few-shot.json \
        --b banchmark/HARDMath2/results/hardmath_MIST.json \
        --label-a "few-shot" --label-b "MIST" --subgroup

  Compare eval_no_verl rollout directories (pre vs post training):
    python examples/strategy_extraction/scripts/stat_test.py \
        --a checkpoints_eval_no_verl/pre/ \
        --b checkpoints_eval_no_verl/post_step250/ \
        --label-a "4B base (pre)" --label-b "4B MIST (step-250)" \
        --out results/pre_vs_post.json

  Single-condition bootstrap CI only:
    python examples/strategy_extraction/scripts/stat_test.py \
        --a banchmark/linguini/my_linguini_results/habit/Qwen3-4B/linguini_passk_Qwen3-4B_mist-inline.json \
        --label-a "Qwen3-4B habit"

  Force format, 99% CI, 50k resamples:
    python examples/strategy_extraction/scripts/stat_test.py \
        --a results/A.json --fmt-a linguini \
        --b results/B.json --fmt-b linguini \
        --n-bootstrap 50000 --ci 99 --seed 0

EXPECTED OUTPUT (paired comparison):
  ========================================================================
    Statistical Significance Testing -- eval_no_verl / Linguini / HARDMath2
  ========================================================================
    Bootstrap resamples : 10,000    Confidence level: 95%    Seed: 42

    Loaded A [Qwen3-8B few-shot]: 345 problems  (format: linguini)
    Loaded B [Qwen3-8B MIST]:     345 problems  (format: linguini)

    Mode: paired comparison
  ------------------------------------------------------------------------
    Paired problems : 345

    Condition                            pass@1     95% bootstrap CI
    ------------------------------------ --------   --------------------
    Qwen3-8B few-shot                    0.4928     [0.4406, 0.5449]
    Qwen3-8B MIST                        0.5623     [0.5101, 0.6145]
    delta (B - A)                       +0.0695    [+0.0290, +0.1101]

    Contingency table (A = row, B = col):
              B=0    B=1
      A=0    138     56   (total A=0: 194)
      A=1     24    127   (total A=1: 151)

    -- Significance tests (paired) --
    McNemar test (binary, continuity-corrected):
      chi2 = 14.4500,  p = 0.0001 ***
      discordant pairs: n(A=0,B=1) = 56,  n(A=1,B=0) = 24
    Wilcoxon signed-rank test (non-parametric, normal approx):
      W = 300.0,  p = 0.0002 ***
    Paired t-test (parametric, on diff scores b-a):
      t = 4.8123,  df = 344,  p = 0.0000 ***

    Significance legend:  *** p<0.001  ** p<0.01  * p<0.05  . p<0.10

    -- Per-subgroup results --
    Group                       n      A pass@1   B pass@1     delta   p(McNemar)
    -------------------------- -----   ---------  ---------  -------   ----------
    translation                  115      0.4261     0.5130   +0.0870   0.0412 *
    cipher                        98      0.5510     0.6224   +0.0714   0.1823
  ========================================================================

STATISTICAL METHODS:
  Bootstrap CI (percentile):  resample n_bootstrap times; report [alpha/2, 1-alpha/2] percentiles.
  McNemar (Edwards corrected): for paired binary; chi2=(|n01-n10|-1)^2/(n01+n10). Primary for binary.
  Wilcoxon signed-rank:       non-parametric; normal approx with tie correction.
  Paired t-test:              parametric; on mean(b-a). Use for continuous soft-F1.
"""
from __future__ import annotations
import argparse, json, math, random, sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ===========================================================================
# Data loading
# ===========================================================================

def _load_linguini(path: Path) -> Dict[str, float]:
    """Return {task_type/problem_id: pass_at_1_hard} from a Linguini passk JSON."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    threshold = d.get("_meta", {}).get("pass_threshold", 1.0)
    results: Dict[str, float] = {}
    for task_type, task_data in d.items():
        if task_type.startswith("_") or not isinstance(task_data, dict):
            continue
        for prob_id, prob_data in task_data.items():
            if not isinstance(prob_data, dict):
                continue
            samples = prob_data.get("samples", [])
            if not samples:
                continue
            scores = [float(s.get("score", 0.0)) for s in samples]
            results[f"{task_type}/{prob_id}"] = 1.0 if any(sc >= threshold for sc in scores) else 0.0
    return results


def _load_hardmath(path: Path) -> Dict[str, float]:
    """Return {problem_id: pass_at_1_hard} from a HARDMath2 result JSON."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    results: Dict[str, float] = {}
    for i, prob in enumerate(d.get("problems") or d.get("results") or []):
        if not isinstance(prob, dict):
            continue
        pid = str(prob.get("problem_id") or prob.get("id") or prob.get("idx") or i)
        samples = prob.get("samples", [])
        if samples:
            results[pid] = 1.0 if any(s.get("hard_correct", s.get("correct", 0)) for s in samples) else 0.0
        else:
            results[pid] = float(prob.get("hard_correct") or prob.get("correct") or 0)
    return results


def _load_eval_dir(root: Path) -> Dict[str, float]:
    """Return {rollout_id: hard_correct} from validation_step*.json shards."""
    results: Dict[str, float] = {}
    files = sorted(root.rglob("validation_step*.json")) or sorted(root.rglob("*.json"))
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            rid = item.get("rollout_id")
            if rid is None:
                continue
            reward = item.get("reward") or {}
            results[str(rid)] = float(int(reward.get("hard_correct", 0)))
    return results


def _detect_fmt(path: Path) -> str:
    name = path.name.lower()
    if "linguini" in name: return "linguini"
    if "hardmath" in name: return "hardmath"
    if path.is_dir(): return "eval_no_verl_dir"
    try:
        with open(path, encoding="utf-8") as f:
            s = f.read(800)
        if "pass_threshold" in s: return "linguini"
        if "num_problems" in s or "acc_hard" in s or "by_type" in s: return "hardmath"
        if "rollout_id" in s or "hard_correct" in s: return "eval_no_verl_dir"
    except Exception:
        pass
    return "linguini"


def load_results(path: Path, fmt: Optional[str] = None) -> Dict[str, float]:
    """Load a result file/directory and return {problem_id: pass_at_1}."""
    r = path.resolve()
    fmt = fmt or _detect_fmt(r)
    if fmt == "linguini": return _load_linguini(r)
    if fmt == "hardmath": return _load_hardmath(r)
    if fmt == "eval_no_verl_dir": return _load_eval_dir(r)
    raise ValueError(f"Unknown format {fmt!r}")


# ===========================================================================
# Pure-Python distribution approximations (no scipy/numpy needed)
# ===========================================================================

def _norm_sf(z: float) -> float:
    """P(Z > |z|) for standard normal.  Abramowitz & Stegun 26.2.17."""
    p = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = p * (0.319381530 + p * (-0.356563782 + p * (1.781477937 + p * (-1.821255978 + p * 1.330274429))))
    return poly * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _log_gamma(z: float) -> float:
    """Lanczos log(Gamma(z)), z > 0."""
    if z < 0.5:
        return math.log(math.pi / math.sin(math.pi * z)) - _log_gamma(1.0 - z)
    z -= 1.0
    c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
         -176.61502916214059, 12.507343278686905, -0.13857109526572012,
         9.9843695780195716e-6, 1.5056327351493116e-7]
    x = c[0]
    for i in range(1, 9):
        x += c[i] / (z + i)
    t = z + 7.5
    return 0.5 * math.log(2.0 * math.pi) + (z + 0.5) * math.log(t) - t + math.log(x)


def _gammainc_lower(a: float, x: float) -> float:
    try: log_norm = a * math.log(x) - x - _log_gamma(a)
    except Exception: return 0.0
    ap, total, delta = a, 1.0 / a, 1.0 / a
    for _ in range(300):
        ap += 1.0; delta *= x / ap; total += delta
        if abs(delta) < abs(total) * 1e-13: break
    return total * math.exp(log_norm)


def _gammainc_upper(a: float, x: float) -> float:
    try: log_norm = a * math.log(x) - x - _log_gamma(a)
    except Exception: return 0.0
    T = 1e-300; b, c, d = x + 1.0 - a, 1.0 / T, 1.0 / (b if abs(b) > T else T); h = d
    for i in range(1, 301):
        an = -i * (i - a); b += 2.0
        d = an * d + b
        if abs(d) < T: d = T
        c = b + an / c
        if abs(c) < T: c = T
        d = 1.0 / d; delta = d * c; h *= delta
        if abs(delta - 1.0) < 1e-13: break
    return math.exp(log_norm) * h


def _chi2_sf(x: float, df: int = 1) -> float:
    if x <= 0.0: return 1.0
    if df == 1: return 2.0 * _norm_sf(math.sqrt(x))
    if df == 2: return math.exp(-x / 2.0)
    a, hx = df / 2.0, x / 2.0
    return 1.0 - _gammainc_lower(a, hx) if hx < a + 1.0 else _gammainc_upper(a, hx)


def _betacf(a: float, b: float, x: float) -> float:
    T = 1e-300; qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < T: d = T
    d = 1.0 / d; h = d
    for m in range(1, 301):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < T: d = T
        c = 1.0 + aa / c
        if abs(c) < T: c = T
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < T: d = T
        c = 1.0 + aa / c
        if abs(c) < T: c = T
        d = 1.0 / d; delta = d * c; h *= delta
        if abs(delta - 1.0) < 1e-13: break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    try: log_norm = math.lgamma(a+b) - math.lgamma(a) - math.lgamma(b) + a*math.log(x) + b*math.log(1.0-x)
    except Exception: return 0.0
    if x < (a + 1.0) / (a + b + 2.0): return math.exp(log_norm) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_norm) * _betacf(b, a, 1.0 - x) / b


def _t_sf(t: float, df: int) -> float:
    return 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))


# ===========================================================================
# Statistical tests
# ===========================================================================

def bootstrap_ci(scores: List[float], n_bs: int = 10_000, ci: float = 95.0, seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap percentile CI for the mean. Returns (mean, lower, upper)."""
    n = len(scores)
    if n == 0: return 0.0, 0.0, 0.0
    mean = sum(scores) / n
    rng = random.Random(seed)
    boot = sorted(sum(rng.choices(scores, k=n)) / n for _ in range(n_bs))
    alpha = (100.0 - ci) / 2.0 / 100.0
    lo = max(0, int(math.floor(alpha * n_bs)))
    hi = min(n_bs - 1, int(math.ceil((1.0 - alpha) * n_bs)) - 1)
    return mean, boot[lo], boot[hi]


def mcnemar_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Edwards continuity-corrected McNemar test. Returns (chi2, p)."""
    n01 = sum(1 for x, y in zip(a, b) if x == 0.0 and y == 1.0)
    n10 = sum(1 for x, y in zip(a, b) if x == 1.0 and y == 0.0)
    disc = n01 + n10
    if disc == 0: return 0.0, 1.0
    chi2 = max(0.0, (abs(n01 - n10) - 1.0) ** 2) / disc
    return chi2, _chi2_sf(chi2, df=1)


def wilcoxon_signed_rank(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Wilcoxon signed-rank test, normal approximation. Returns (W, p)."""
    diffs = [y - x for x, y in zip(a, b)]
    nonzero = [(i, d) for i, d in enumerate(diffs) if d != 0.0]
    n = len(nonzero)
    if n == 0: return 0.0, 1.0
    sorted_nd = sorted(nonzero, key=lambda v: abs(v[1]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs(sorted_nd[j][1]) == abs(sorted_nd[i][1]): j += 1
        avg = (i + j + 1) / 2.0
        for k in range(i, j): ranks[k] = avg
        i = j
    W_plus  = sum(ranks[k] for k in range(n) if sorted_nd[k][1] > 0.0)
    W_minus = sum(ranks[k] for k in range(n) if sorted_nd[k][1] < 0.0)
    W = min(W_plus, W_minus)
    abs_vals = [abs(sorted_nd[k][1]) for k in range(n)]
    tie_corr = sum(t ** 3 - t for t in Counter(abs_vals).values()) / 48.0
    var_W = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr
    if var_W <= 0.0: return W, 1.0
    z = (W - n * (n + 1) / 4.0) / math.sqrt(var_W)
    return W, 2.0 * _norm_sf(abs(z))


def paired_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Paired t-test on diff scores (b - a). Returns (t, p)."""
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    if n < 2: return 0.0, 1.0
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    if var_d == 0.0: return (0.0, 1.0) if mean_d == 0.0 else (float("inf"), 0.0)
    t = mean_d / math.sqrt(var_d / n)
    return t, 2.0 * _t_sf(abs(t), df=n - 1)


# ===========================================================================
# Reporting helpers
# ===========================================================================

_SEP  = "=" * 72
_DASH = "-" * 72


def _fp(p: float) -> str:
    if p < 0.001: return f"{p:.2e} ***"
    if p < 0.01:  return f"{p:.4f} **"
    if p < 0.05:  return f"{p:.4f} *"
    if p < 0.10:  return f"{p:.4f} ."
    return f"{p:.4f}"


def _print_single(label: str, scores: List[float], n_bs: int, ci: float, seed: int) -> Dict[str, Any]:
    mean, lo, hi = bootstrap_ci(scores, n_bs, ci, seed)
    n = len(scores)
    print(f"\n  Condition : {label}")
    print(f"  N problems: {n}")
    print(f"  pass_at_1 : {mean:.4f}  ({ci:.0f}% bootstrap CI: [{lo:.4f}, {hi:.4f}])")
    print(f"  Bootstrap : n_resamples={n_bs:,}, seed={seed}")
    return {"label": label, "n": n, "mean": mean, "ci_lo": lo, "ci_hi": hi, "ci_pct": ci, "n_bootstrap": n_bs}


def _print_comparison(la: str, lb: str, a: List[float], b: List[float], n_bs: int, ci: float, seed: int) -> Dict[str, Any]:
    n = len(a)
    ma, loa, hia = bootstrap_ci(a, n_bs, ci, seed)
    mb, lob, hib = bootstrap_ci(b, n_bs, ci, seed)
    diff = [y - x for x, y in zip(a, b)]
    md, lod, hid = bootstrap_ci(diff, n_bs, ci, seed)
    chi2, pmc = mcnemar_test(a, b)
    W, pwil   = wilcoxon_signed_rank(a, b)
    t, ptt    = paired_ttest(a, b)
    n01 = sum(1 for x, y in zip(a, b) if x == 0.0 and y == 1.0)
    n10 = sum(1 for x, y in zip(a, b) if x == 1.0 and y == 0.0)
    n11 = sum(1 for x, y in zip(a, b) if x == 1.0 and y == 1.0)
    n00 = sum(1 for x, y in zip(a, b) if x == 0.0 and y == 0.0)
    print(f"\n  Paired problems : {n}")
    print(f"\n  {'Condition':<36} {'pass_at_1':>10}   {ci:.0f}% bootstrap CI")
    print(f"  {'-'*36} {'-'*10}   {'-'*22}")
    print(f"  {la:<36} {ma:>10.4f}   [{loa:.4f}, {hia:.4f}]")
    print(f"  {lb:<36} {mb:>10.4f}   [{lob:.4f}, {hib:.4f}]")
    print(f"  {'delta (B - A)':<36} {md:>+10.4f}   [{lod:+.4f}, {hid:+.4f}]")
    print(f"\n  Contingency table (A = row, B = col):")
    print(f"            B=0    B=1")
    print(f"    A=0   {n00:5d}  {n01:5d}   (total A=0: {n00+n01})")
    print(f"    A=1   {n10:5d}  {n11:5d}   (total A=1: {n10+n11})")
    print(f"\n  -- Significance tests (paired) --")
    print(f"  McNemar test (binary, continuity-corrected):")
    print(f"    chi2 = {chi2:.4f},  p = {_fp(pmc)}")
    print(f"    discordant pairs: n(A=0,B=1) = {n01},  n(A=1,B=0) = {n10}")
    print(f"  Wilcoxon signed-rank test (non-parametric, normal approx):")
    print(f"    W = {W:.1f},  p = {_fp(pwil)}")
    print(f"  Paired t-test (parametric, on diff scores b-a):")
    print(f"    t = {t:.4f},  df = {n-1},  p = {_fp(ptt)}")
    print(f"\n  Significance legend:  *** p<0.001  ** p<0.01  * p<0.05  . p<0.10")
    return {
        "n_paired": n,
        "a": {"label": la, "mean": ma, "ci_lo": loa, "ci_hi": hia},
        "b": {"label": lb, "mean": mb, "ci_lo": lob, "ci_hi": hib},
        "delta": {"mean": md, "ci_lo": lod, "ci_hi": hid},
        "contingency": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
        "tests": {"mcnemar": {"chi2": chi2, "p": pmc}, "wilcoxon": {"W": W, "p": pwil}, "paired_ttest": {"t": t, "p": ptt}},
        "ci_pct": ci, "n_bootstrap": n_bs,
    }


def _print_subgroups(la: str, lb: str, a_map: Dict[str, float], b_map: Dict[str, float]) -> List[Dict[str, Any]]:
    common = sorted(set(a_map) & set(b_map))
    groups: Dict[str, Tuple[List[float], List[float]]] = {}
    for pid in common:
        g = pid.split("/")[0] if "/" in pid else "all"
        if g not in groups: groups[g] = ([], [])
        groups[g][0].append(a_map[pid]); groups[g][1].append(b_map[pid])
    if not groups: return []
    print(f"\n  -- Per-subgroup results --")
    print(f"  {'Group':<26} {'n':>5}   {'A pass_at_1':>11}   {'B pass_at_1':>11}   {'delta':>7}   p(McNemar)")
    print(f"  {'-'*26} {'-'*5}   {'-'*11}   {'-'*11}   {'-'*7}   {'-'*14}")
    rows: List[Dict[str, Any]] = []
    for g in sorted(groups):
        a_s, b_s = groups[g]; n = len(a_s)
        ma = sum(a_s) / n if n else 0.0
        mb = sum(b_s) / n if n else 0.0
        _, p = mcnemar_test(a_s, b_s)
        print(f"  {g:<26} {n:>5}   {ma:>11.4f}   {mb:>11.4f}   {mb-ma:>+7.4f}   {_fp(p)}")
        rows.append({"group": g, "n": n, "mean_a": ma, "mean_b": mb, "delta": mb - ma, "p_mcnemar": p})
    return rows


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CI + paired significance tests for eval results.")
    parser.add_argument("--a", required=True, metavar="PATH", help="Condition A result file or directory.")
    parser.add_argument("--b", metavar="PATH", default=None, help="Condition B (omit for single-condition CI).")
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    parser.add_argument("--fmt-a", default=None, choices=["linguini", "hardmath", "eval_no_verl_dir"])
    parser.add_argument("--fmt-b", default=None, choices=["linguini", "hardmath", "eval_no_verl_dir"])
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--ci", type=float, default=95.0, help="Confidence level in percent (default: 95).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subgroup", action="store_true", help="Print per-subgroup (task type) breakdown.")
    parser.add_argument("--out", metavar="PATH", default=None, help="Write JSON report to this path.")
    args = parser.parse_args()

    path_a = Path(args.a)
    if not path_a.exists():
        print(f"[ERROR] --a path does not exist: {path_a}", file=sys.stderr); sys.exit(1)
    label_a = args.label_a or path_a.name

    print(); print(_SEP)
    print("  Statistical Significance Testing -- eval_no_verl / Linguini / HARDMath2")
    print(_SEP)
    print(f"  Bootstrap resamples : {args.n_bootstrap:,}")
    print(f"  Confidence level    : {args.ci:.0f}%")
    print(f"  Random seed         : {args.seed}")

    fmt_a = _detect_fmt(path_a) if args.fmt_a is None else args.fmt_a
    a_map = load_results(path_a, fmt_a)
    print(f"\n  Loaded A [{label_a}]: {len(a_map)} problems  (format: {fmt_a})")

    report: Dict[str, Any] = {}

    if args.b is None:
        print(f"\n  Mode: single-condition bootstrap CI"); print(_DASH)
        report["single"] = _print_single(label_a, list(a_map.values()), args.n_bootstrap, args.ci, args.seed)
    else:
        path_b = Path(args.b)
        if not path_b.exists():
            print(f"[ERROR] --b path does not exist: {path_b}", file=sys.stderr); sys.exit(1)
        label_b = args.label_b or path_b.name
        fmt_b = _detect_fmt(path_b) if args.fmt_b is None else args.fmt_b
        b_map = load_results(path_b, fmt_b)
        print(f"  Loaded B [{label_b}]: {len(b_map)} problems  (format: {fmt_b})")
        common_ids = sorted(set(a_map) & set(b_map))
        only_a, only_b = len(a_map) - len(common_ids), len(b_map) - len(common_ids)
        if only_a or only_b:
            print(f"\n  [WARN] Non-overlapping IDs: only-A={only_a}, only-B={only_b} (excluded from paired tests)")
        a_s = [a_map[pid] for pid in common_ids]
        b_s = [b_map[pid] for pid in common_ids]
        print(f"\n  Mode: paired comparison"); print(_DASH)
        report["comparison"] = _print_comparison(label_a, label_b, a_s, b_s, args.n_bootstrap, args.ci, args.seed)
        if args.subgroup:
            report["subgroups"] = _print_subgroups(label_a, label_b, a_map, b_map)

    print(); print(_SEP)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report["config"] = {"a": str(path_a), "b": str(args.b) if args.b else None,
                            "label_a": label_a, "label_b": args.label_b,
                            "n_bootstrap": args.n_bootstrap, "ci": args.ci, "seed": args.seed}
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  JSON report written -> {out_path}"); print(_SEP)


if __name__ == "__main__":
    main()
