#!/usr/bin/env python3
"""
Linguini benchmark evaluation with pass@1 / pass@2 / pass@3.

Supports three prompting modes selectable via --mode:

  zero-shot  — answer directly using zero-shot.toml (no examples needed)
  few-shot   — in-context learning using linguini-few-shot.toml; few-shot
               examples from the same task_type are shown to the model,
               drawn from the full dataset with leave-one-out exclusion
  MIST       — two-step: (1) extract a problem-solving strategy from few-shot
               examples using strategy_generation/repetition_controls_2026-04-01.toml,
               (2) apply the strategy to answer the new problem using
               answer_generation/v1.toml

Scoring logic (official Linguini eval_type semantics):
  single / simple : exact string match per sub-answer (after strip+lower)
  multi           : each sub-answer matches any candidate in answer[i] list

Few-shot examples are drawn from dataset.jsonl itself using leave-one-out:
for each test problem P of task_type T, shot examples are sampled from
all other problems of task_type T (P itself is excluded).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import openai
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Default path to agent-lightning TOML prompt directory
# ---------------------------------------------------------------------------
_DEFAULT_PROMPT_DIR = Path(
    "/home/test/test16/chenlu/projects/agent-lightning/examples/strategy_extraction/prompt"
)

_DATASET_FILE = Path(__file__).parent / "dataset.jsonl"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load dataset.jsonl; return list of problem dicts."""
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


# ---------------------------------------------------------------------------
# Shot pool: per-task_type indexed collection, with leave-one-out support
# ---------------------------------------------------------------------------

def build_shot_pool(
    problems: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Build a {task_type: [problem, ...]} mapping from all problems."""
    pool: Dict[str, List[Dict[str, Any]]] = {}
    for p in problems:
        tt = p["task_type"]
        pool.setdefault(tt, []).append(p)
    return pool


def _pid_hash(pid: str) -> int:
    """Stable numeric hash of a problem id (MD5 first 8 hex chars)."""
    return int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)


def make_shot_seed(pid: str, sample_idx: int, global_seed: int, diverse: bool) -> int:
    """Deterministic seed for shot sampling.

    diverse=True  → seed varies per (pid, sample_idx) → each pass@k sample
                    sees a different random subset of few-shot examples.
    diverse=False → seed varies per pid only → all k samples share the same
                    subset (only LLM temperature produces diversity).
    """
    h = _pid_hash(pid)
    if diverse:
        return (global_seed ^ h ^ sample_idx) & 0xFFFFFFFF
    return (global_seed ^ h) & 0xFFFFFFFF


def format_answers_for_example(problem: Dict[str, Any]) -> str:
    """Format ground-truth answers of a shot example into a numbered list string."""
    answer = problem["answer"]
    eval_type = problem.get("eval_type", "single")
    lines = []
    for i, ans in enumerate(answer, 1):
        if eval_type == "multi":
            # ans is a list of acceptable forms; show the first one
            display = ans[0] if isinstance(ans, list) and ans else str(ans)
        else:
            display = str(ans)
        lines.append(f"{i}. {display}")
    return "\n".join(lines)


def format_problem_text(problem: Dict[str, Any]) -> str:
    """Format a Linguini problem as context + task block (no answer)."""
    return f"Context:\n{problem['context']}\n\nTask:\n{problem['query']}"


def draw_shot_examples(
    pool: Dict[str, List[Dict[str, Any]]],
    task_type: str,
    exclude_id: str,
    shot_num: int,
    seed: int,
) -> Tuple[List[str], str]:
    """Sample ``shot_num`` examples of ``task_type`` excluding ``exclude_id``.

    Returns:
        shot_ids      — list of example IDs used (for audit)
        examples_text — formatted text ready for the ``{examples_text}`` placeholder
    """
    candidates = [p for p in pool.get(task_type, []) if p["id"] != exclude_id]
    if not candidates or shot_num <= 0:
        return [], ""
    rng = random.Random(seed)
    selected = rng.sample(candidates, min(shot_num, len(candidates)))
    shot_ids = [p["id"] for p in selected]
    parts: List[str] = []
    for i, ex in enumerate(selected, 1):
        answers_text = format_answers_for_example(ex)
        parts.append(f"=== Example {i} ===")
        parts.append(f"Context:\n{ex['context']}")
        parts.append(f"\nTask:\n{ex['query']}")
        parts.append(f"\nAnswers:\n{answers_text}")
        parts.append("")
    return shot_ids, "\n".join(parts)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Scoring utilities (official Linguini eval_type semantics)
# ---------------------------------------------------------------------------

def _normalize(s: Any) -> str:
    """Normalize an answer string for comparison.

    - strip whitespace, lowercase
    - unify Unicode quotation marks to ASCII equivalents so that a model
      outputting a straight apostrophe (U+0027) is not penalised for a
      ground-truth that uses a right single quotation mark (U+2019), etc.
    - strip trailing sentence punctuation so that a model adding a period
      to a translation ("She remembered the woman.") is not penalised when
      the ground truth omits it (or vice versa).
    """
    text = str(s).strip().lower()
    # Curly / typographic quotes → ASCII equivalents
    text = (text
            .replace("\u2019", "'").replace("\u2018", "'")   # right/left single quote
            .replace("\u201c", '"').replace("\u201d", '"')   # right/left double quote
            .replace("\u2032", "'").replace("\u2033", '"'))  # prime / double prime
    # Strip trailing sentence punctuation (applied to both prediction and ground truth)
    text = text.rstrip(".,;:!?")
    return text


def score_answers(
    predicted: List[str],
    ground_truth: List[Any],
    eval_type: str,
) -> float:
    """Compute fractional accuracy for one problem.

    Returns a float in [0.0, 1.0]: proportion of sub-answers correct.

    eval_type semantics:
      single / simple : ground_truth[i] is a string; exact match after normalize
      multi           : ground_truth[i] is a list of acceptable strings;
                        match if prediction equals any candidate.
                        If ground_truth[i] happens to be a plain string
                        (data inconsistency), treat it as a single-candidate list.
    """
    if not ground_truth:
        return 0.0
    n = len(ground_truth)
    # Align predictions: pad with empty strings if model gave fewer answers
    preds = list(predicted) + [""] * max(0, n - len(predicted))
    correct = 0
    for i in range(n):
        pred_norm = _normalize(preds[i])
        if eval_type == "multi":
            gt_item = ground_truth[i]
            # Guard: gt_item should be a list, but dataset may contain a plain string
            candidates = gt_item if isinstance(gt_item, list) else [gt_item]
            if any(pred_norm == _normalize(c) for c in candidates):
                correct += 1
        else:  # single or simple
            if pred_norm == _normalize(ground_truth[i]):
                correct += 1
    return correct / n


def _score_to_float(x: Any) -> float:
    if x is None:
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _is_pass(score: Any, threshold: float) -> bool:
    return _score_to_float(score) + 1e-9 >= threshold


def verify_response(response: Any) -> bool:
    if isinstance(response, str):
        response = response.strip()
    return bool(response)


# ---------------------------------------------------------------------------
# Answer extraction from model output
# ---------------------------------------------------------------------------

def parse_numbered_answers(text: str) -> List[str]:
    """Extract a numbered list from inside <answer>...</answer> tags.

    Falls back to scanning the entire text if no tags are found.

    Example model output:
        <answer>
        1. ɨnnetakʼa
        2. ɨŋɡɨrʼɨ
        </answer>

    Returns:
        ['ɨnnetakʼa', 'ɨŋɡɨrʼɨ']
    """
    # Try to extract the content inside <answer>...</answer>
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    block = match.group(1) if match else text

    # Parse numbered lines: "1. answer", "1) answer", "(1) answer"
    answers: List[str] = []
    for line in block.splitlines():
        line = line.strip()
        m = re.match(r"^\(?(\d+)[.)]\s*(.+)$", line)
        if m:
            idx = int(m.group(1))
            val = m.group(2).strip()
            # Extend list to fit the index (1-based)
            while len(answers) < idx:
                answers.append("")
            answers[idx - 1] = val

    return answers


# ---------------------------------------------------------------------------
# Pass@k statistics
# ---------------------------------------------------------------------------

@dataclass
class PassKStats:
    n_problems: int
    pass_at: Dict[int, float]
    mean_score_sample: List[float]


def compute_pass_at_k(
    per_problem_scores: List[List[float]], ks: List[int], threshold: float
) -> Dict[int, float]:
    """per_problem_scores[i] = list of sample scores for problem i."""
    out: Dict[int, float] = {}
    if not per_problem_scores:
        for k in ks:
            out[k] = 0.0
        return out
    n_s = len(per_problem_scores[0])
    for k in ks:
        kk = min(k, n_s)
        hits = 0
        for row in per_problem_scores:
            passes = [_is_pass(row[j], threshold) for j in range(kk)]
            hits += 1 if any(passes) else 0
        out[k] = hits / len(per_problem_scores)
    return out


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

class OpenAICompatibleChat:
    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        system_prompt: str = "",
        sleep_time: float = 0.05,
        no_think: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.sleep_time = sleep_time
        self.no_think = no_think
        kwargs: Dict[str, Any] = {"api_key": api_key or "EMPTY"}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**kwargs)

    def get_response(
        self, user_prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        sp = system_prompt if system_prompt is not None else self.system_prompt
        messages = [
            {"role": "system", "content": sp},
            {"role": "user", "content": user_prompt},
        ]
        patience = 64
        max_tokens = self.max_tokens
        extra_body: Dict[str, Any] = {}
        if self.no_think:
            # vLLM (local): disable thinking via chat_template_kwargs
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            # DeepSeek API: disable thinking via the 'thinking' field
            extra_body["thinking"] = {"type": "disabled"}
        while patience > 0:
            patience -= 1
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    n=1,
                    **({"extra_body": extra_body} if extra_body else {}),
                )
                text = resp.choices[0].message.content
                if text:
                    return text
            except Exception as e:
                err = str(e)
                if "limit" not in err.lower():
                    print(err)
                if "reduce the length" in err.lower() or "max_tokens" in err.lower():
                    max_tokens = max(8, int(max_tokens * 0.9))
                if max_tokens <= 8:
                    return ""
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)
        return ""


class OllamaChat:
    def __init__(self, host: str, model: str, temperature: float = 0.7):
        host = host.rstrip("/")
        if not host.startswith("http"):
            host = f"http://{host}"
        self.url = f"{host}/api/generate"
        self.model = model
        self.temperature = temperature

    def get_response(
        self, user_prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        try:
            r = requests.post(self.url, json=payload, timeout=600)
        except Exception as e:
            print(f"Ollama request error: {e}")
            return ""
        if r.status_code != 200:
            print(f"Ollama HTTP {r.status_code}: {r.text[:500]}")
            return ""
        return r.json().get("response") or ""


# ---------------------------------------------------------------------------
# Model factory helpers
# ---------------------------------------------------------------------------

def _make_openai_client(
    model: str,
    api_key: str,
    api_base: str,
    backend: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    sleep_time: float,
    no_think: bool = False,
) -> OpenAICompatibleChat:
    key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    base_url: Optional[str] = api_base or None
    if backend == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
        key = api_key or os.getenv("OPENROUTER_API_KEY") or key
    return OpenAICompatibleChat(
        model, key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        sleep_time=sleep_time,
        no_think=no_think,
    )


def load_solver(args: argparse.Namespace) -> Any:
    if args.backend == "ollama":
        return OllamaChat(args.ollama_host, args.model, temperature=args.temperature)
    return _make_openai_client(
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        backend=args.backend,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        system_prompt="",
        sleep_time=args.sleep_time,
        no_think=getattr(args, "no_think", False),
    )


def load_strategy_solver(args: argparse.Namespace) -> Any:
    """Strategy-generation model for MIST mode (may be same as the answer solver)."""
    s_model = args.strategy_model or args.model
    s_key = args.strategy_api_key or args.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    s_base = args.strategy_api_base or args.api_base
    if args.backend == "ollama":
        return OllamaChat(
            args.ollama_host, s_model, temperature=args.strategy_temperature
        )
    return _make_openai_client(
        model=s_model,
        api_key=s_key,
        api_base=s_base,
        backend=args.backend,
        temperature=args.strategy_temperature,
        max_tokens=args.strategy_max_tokens,
        system_prompt="",
        sleep_time=args.sleep_time,
        no_think=getattr(args, "strategy_no_think", False),
    )


# ---------------------------------------------------------------------------
# TOML prompt loader
# ---------------------------------------------------------------------------

def load_toml_prompt(path: Path) -> Dict[str, str]:
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Solve-function factory
# ---------------------------------------------------------------------------

def make_solve_fn(
    mode: str,
    solver: Any,
    strategy_solver: Optional[Any],
    shot_pool: Dict[str, List[Dict[str, Any]]],
    shot_num: int,
    shot_seed: int,
    diverse_context: bool,
    prompt_dir: Path,
) -> Callable[[Dict[str, Any], int], Tuple[str, Dict[str, Any]]]:
    """Return a ``(problem_dict, sample_idx) -> (response_str, extra_info_dict)`` callable."""

    if mode == "zero-shot":
        tmpl = load_toml_prompt(prompt_dir / "answer_generation" / "zero-shot.toml")
        sys_p = tmpl["system"].strip()
        usr_tpl = tmpl["user"]

        def _zero_shot(
            prob: Dict[str, Any], si: int
        ) -> Tuple[str, Dict[str, Any]]:
            problem_text = format_problem_text(prob)
            user_prompt = usr_tpl.format(problem=problem_text)
            resp = solver.get_response(user_prompt, system_prompt=sys_p)
            return resp, {"prompt": user_prompt, "few_shot_ids": []}

        return _zero_shot

    elif mode == "few-shot":
        tmpl = load_toml_prompt(
            prompt_dir / "answer_generation" / "linguini-few-shot.toml"
        )
        sys_p = tmpl["system"].strip()
        usr_tpl = tmpl["user"]

        def _few_shot(
            prob: Dict[str, Any], si: int
        ) -> Tuple[str, Dict[str, Any]]:
            seed = make_shot_seed(prob["id"], si, shot_seed, diverse_context)
            shot_ids, examples_text = draw_shot_examples(
                shot_pool, prob["task_type"], prob["id"], shot_num, seed
            )
            problem_text = format_problem_text(prob)
            user_prompt = usr_tpl.format(
                examples_text=examples_text, problem=problem_text
            )
            resp = solver.get_response(user_prompt, system_prompt=sys_p)
            return resp, {"prompt": user_prompt, "few_shot_ids": shot_ids}

        return _few_shot

    elif mode == "MIST":
        strat_tmpl = load_toml_prompt(
            prompt_dir / "strategy_generation" / "repetition_controls_2026-04-01.toml"
        )
        strat_sys_p = strat_tmpl["system"].strip()
        strat_usr_tpl = strat_tmpl["user"]

        ans_tmpl = load_toml_prompt(prompt_dir / "answer_generation" / "v1.toml")
        ans_sys_p = ans_tmpl["system"].strip()
        ans_usr_tpl = ans_tmpl["user"]

        s_solver = strategy_solver or solver

        def _mist(
            prob: Dict[str, Any], si: int
        ) -> Tuple[str, Dict[str, Any]]:
            # Step 1: sample few-shot examples
            seed = make_shot_seed(prob["id"], si, shot_seed, diverse_context)
            shot_ids, examples_text = draw_shot_examples(
                shot_pool, prob["task_type"], prob["id"], shot_num, seed
            )
            # Step 2: extract strategy from few-shot examples
            strategy_user_prompt = strat_usr_tpl.format(examples_text=examples_text)
            strategy = s_solver.get_response(
                strategy_user_prompt, system_prompt=strat_sys_p
            )
            # Step 3: apply strategy to answer the problem
            problem_text = format_problem_text(prob)
            answer_user_prompt = ans_usr_tpl.format(
                strategy=strategy, problem=problem_text
            )
            resp = solver.get_response(answer_user_prompt, system_prompt=ans_sys_p)
            return resp, {
                "few_shot_ids": shot_ids,
                "strategy_prompt": strategy_user_prompt,
                "strategy": strategy,
                "answer_prompt": answer_user_prompt,
            }

        return _mist

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: zero-shot, few-shot, MIST")


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_bucket(
    bucket_label: str,
    problems: List[Dict[str, Any]],
    solve_fn: Callable[[Dict[str, Any], int], Tuple[str, Dict[str, Any]]],
    args: argparse.Namespace,
    results_root: Dict[str, Any],
) -> Tuple[PassKStats, List[List[float]]]:
    """Evaluate one bucket (task_type) of problems."""
    per_problem_scores: List[List[float]] = []
    bucket_results = results_root.setdefault(bucket_label, {})

    for prob in tqdm(problems, desc=bucket_label):
        pid = prob["id"]
        entry = bucket_results.get(pid)
        if entry and entry.get("samples"):
            existing = entry["samples"]
            if len(existing) >= args.num_samples and all(
                verify_response(s.get("response")) for s in existing[: args.num_samples]
            ):
                scores = [
                    _score_to_float(s.get("score"))
                    for s in existing[: args.num_samples]
                ]
                per_problem_scores.append(scores)
                continue

        row = bucket_results.setdefault(pid, {"samples": [], "task_type": prob["task_type"]})
        # Always refresh ground-truth fields so resumed results also contain them
        row.update({
            "eval_type": prob.get("eval_type", "single"),
            "ground_truth": prob["answer"],
        })
        samples: List[Dict[str, Any]] = row["samples"]

        for si in range(args.num_samples):
            if si < len(samples) and verify_response(samples[si].get("response")):
                continue
            while len(samples) <= si:
                samples.append({})
            print(f"[{bucket_label}][{pid}] sample {si + 1}/{args.num_samples}")
            try:
                resp, extra = solve_fn(prob, si)
                predicted = parse_numbered_answers(resp)
                sc = score_answers(predicted, prob["answer"], prob.get("eval_type", "single"))
                samples[si].update(extra)
                samples[si]["response"] = resp
                samples[si]["predicted_answers"] = predicted
                samples[si]["score"] = sc
            except Exception as e:
                print(f"  ERROR: {e}")
                samples[si]["error"] = repr(e)
                samples[si]["score"] = 0.0

        scores = [_score_to_float(samples[i].get("score")) for i in range(args.num_samples)]
        per_problem_scores.append(scores)

    ks = [k for k in (1, 2, 3) if k <= args.num_samples]
    pass_at = compute_pass_at_k(per_problem_scores, ks, args.pass_threshold)
    means = []
    for j in range(args.num_samples):
        col = [
            _score_to_float(row[j])
            for row in per_problem_scores
            if j < len(row)
        ]
        means.append(sum(col) / len(col) if col else 0.0)
    stats = PassKStats(
        n_problems=len(per_problem_scores),
        pass_at=pass_at,
        mean_score_sample=means,
    )
    return stats, per_problem_scores


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    parser = argparse.ArgumentParser(description="Linguini benchmark pass@k evaluation")

    # ── Core model ──────────────────────────────────────────────────────────
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Model name (OpenAI id, vLLM served name, Ollama tag, etc.)")
    parser.add_argument(
        "--backend", type=str, default="openai_compatible",
        choices=["openai", "openrouter", "openai_compatible", "ollama"],
        help="openai_compatible: use --api_base (e.g. vLLM). ollama: use --ollama_host.",
    )
    parser.add_argument("--api_key", type=str, default="",
                        help="API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--api_base", type=str, default="",
                        help="Base URL for openai_compatible backend (e.g. http://localhost:8000/v1)")
    parser.add_argument("--ollama_host", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--sleep_time", type=float, default=0.05)

    # ── Prompting mode ──────────────────────────────────────────────────────
    parser.add_argument(
        "--mode", type=str, default="zero-shot",
        choices=["zero-shot", "few-shot", "MIST"],
        help=(
            "Prompting strategy:\n"
            "  zero-shot  — answer directly (no examples)\n"
            "  few-shot   — in-context learning from same-type examples (leave-one-out)\n"
            "  MIST       — extract strategy from shot examples, then answer"
        ),
    )
    parser.add_argument("--shot_num", type=int, default=3,
                        help="Number of few-shot examples (few-shot / MIST modes)")
    parser.add_argument("--shot_seed", type=int, default=42,
                        help="Global seed for few-shot example sampling")
    parser.add_argument(
        "--context_diversity_mode", type=str, default="diverse",
        choices=["diverse", "reproducible"],
        help=(
            "diverse (default): each pass@k sample draws a different random subset "
            "of few-shot examples. "
            "reproducible: all k samples share the same subset."
        ),
    )
    parser.add_argument(
        "--prompt_dir", type=str, default=str(_DEFAULT_PROMPT_DIR),
        help="Directory containing TOML prompt files",
    )

    # ── MIST: strategy generation model ─────────────────────────────────────
    parser.add_argument("--strategy_model", type=str, default="",
                        help="Model for strategy generation in MIST mode (default: same as --model)")
    parser.add_argument("--strategy_api_base", type=str, default="",
                        help="API base URL for strategy model (default: same as --api_base)")
    parser.add_argument("--strategy_api_key", type=str, default="",
                        help="API key for strategy model (default: same as --api_key)")
    parser.add_argument("--strategy_temperature", type=float, default=0.7)
    parser.add_argument("--strategy_max_tokens", type=int, default=4096)

    # ── Think mode control ───────────────────────────────────────────────────
    parser.add_argument(
        "--no_think", action="store_true", default=False,
        help="Disable thinking mode for the answer model (Qwen3 / vLLM: sets enable_thinking=False)",
    )
    parser.add_argument(
        "--strategy_no_think", action="store_true", default=False,
        help="Disable thinking mode for the strategy model (MIST mode only)",
    )

    # ── Data / output ────────────────────────────────────────────────────────
    parser.add_argument("--dataset", type=str, default=str(_DATASET_FILE),
                        help="Path to dataset.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/passk",
                        help="Directory for result JSON files")
    parser.add_argument("--output_file", type=str, default="",
                        help="Output filename (default: auto-generated)")

    # ── Sampling / generation ────────────────────────────────────────────────
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Independent completions per problem (for pass@k)")
    parser.add_argument("--pass_threshold", type=float, default=1.0,
                        help="Score threshold for a 'pass' (default: 1.0 = all sub-answers correct)")

    # ── Task type filtering ──────────────────────────────────────────────────
    parser.add_argument(
        "--task_type", type=str, default="ALL",
        help="ALL | translation | fill_blanks | match_letters | text_to_num | num_to_text",
    )

    args = parser.parse_args()

    # ── Validation ───────────────────────────────────────────────────────────
    if args.backend == "openai_compatible" and not args.api_base:
        parser.error("--backend openai_compatible requires --api_base")
    if args.mode in ("few-shot", "MIST") and args.shot_num <= 0:
        parser.error(f"--shot_num must be > 0 for --mode {args.mode}")

    # ── Output paths ─────────────────────────────────────────────────────────
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", args.model)[:80]
    if not args.output_file:
        args.output_file = f"linguini_passk_{safe_name}_{args.mode}.json"
    os.makedirs(os.path.join(here, args.output_dir), exist_ok=True)
    out_path = os.path.join(here, args.output_dir, args.output_file)

    # ── Load dataset ─────────────────────────────────────────────────────────
    print(f"Loading {args.dataset} ...")
    problems = load_dataset(Path(args.dataset))
    print(f"  Loaded {len(problems)} problems.")

    # ── Resume ───────────────────────────────────────────────────────────────
    results: Dict[str, Any] = {}
    if os.path.exists(out_path):
        print(f"Resuming from {out_path}")
        results = read_json(out_path)

    meta = results.setdefault("_meta", {})
    meta.update({
        "model": args.model,
        "mode": args.mode,
        "shot_num": args.shot_num,
        "shot_seed": args.shot_seed,
        "context_diversity_mode": args.context_diversity_mode,
        "backend": args.backend,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "pass_threshold": args.pass_threshold,
        "dataset": args.dataset,
    })
    if args.mode == "MIST":
        meta["strategy_model"] = args.strategy_model or args.model
        meta["strategy_api_base"] = args.strategy_api_base or args.api_base

    # ── Build solvers ────────────────────────────────────────────────────────
    solver = load_solver(args)
    strategy_solver = load_strategy_solver(args) if args.mode == "MIST" else None
    prompt_dir = Path(args.prompt_dir)

    # ── Build shot pool ───────────────────────────────────────────────────────
    shot_pool = build_shot_pool(problems)

    # ── Print configuration summary ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Linguini pass@k evaluation")
    print(f"  Mode        : {args.mode}")
    print(f"  Answer model: {args.model}  (backend={args.backend}, T={args.temperature})")
    if args.mode == "MIST":
        sm = args.strategy_model or args.model
        print(f"  Strategy mdl: {sm}  (T={args.strategy_temperature})")
    if args.mode != "zero-shot":
        ctx_label = "diverse" if args.context_diversity_mode == "diverse" else "reproducible"
        print(f"  Shot num    : {args.shot_num}  seed={args.shot_seed}  context={ctx_label}")
        print(f"  Shot pool   : { {tt: len(v) for tt, v in sorted(shot_pool.items())} }")
    print(f"  Num samples : {args.num_samples}  (pass@1/2/3)")
    print(f"  Task type   : {args.task_type}")
    print(f"  Output      : {out_path}")
    print(f"{'='*60}\n")

    # ── Solve function ────────────────────────────────────────────────────────
    solve_fn = make_solve_fn(
        mode=args.mode,
        solver=solver,
        strategy_solver=strategy_solver,
        shot_pool=shot_pool,
        shot_num=args.shot_num,
        shot_seed=args.shot_seed,
        diverse_context=(args.context_diversity_mode == "diverse"),
        prompt_dir=prompt_dir,
    )

    # ── Buckets (task types) ──────────────────────────────────────────────────
    all_task_types = ["translation", "fill_blanks", "match_letters", "text_to_num", "num_to_text"]
    if args.task_type == "ALL":
        bucket_types = all_task_types
    else:
        bucket_types = [args.task_type]

    summary: Dict[str, Any] = {"buckets": {}}
    all_rows: List[List[float]] = []

    for tt in bucket_types:
        bucket_problems = [p for p in problems if p["task_type"] == tt]
        if not bucket_problems:
            print(f"Skip empty bucket '{tt}'")
            continue
        n_pool = len(shot_pool.get(tt, []))
        print(
            f"Evaluating bucket '{tt}' ({len(bucket_problems)} problems, "
            f"{n_pool} in shot pool) ..."
        )
        stats, rows = run_bucket(tt, bucket_problems, solve_fn, args, results)
        summary["buckets"][tt] = {
            "n": stats.n_problems,
            "pass@k": {f"pass@{k}": stats.pass_at[k] for k in sorted(stats.pass_at)},
            "mean_score_per_sample_index": stats.mean_score_sample,
        }
        all_rows.extend(rows)
        save_json(results, out_path)
        print(json.dumps(summary["buckets"][tt], indent=2))

    ks = [k for k in (1, 2, 3) if k <= args.num_samples]
    summary["overall"] = {
        "n_problems": len(all_rows),
        **{
            f"pass@{k}": v
            for k, v in compute_pass_at_k(all_rows, ks, args.pass_threshold).items()
        },
    }
    results["_summary"] = summary
    save_json(results, out_path)

    print("\n=== Overall (micro-average over all completed problems) ===")
    print(json.dumps(summary["overall"], indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
