#!/usr/bin/env python3
"""score.py - Six-mode induction scoring for Pre-/Post-GIST strategy samples."""

import argparse
import json
import os
import re
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from tqdm import tqdm

PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "six_mode_scoring.txt")
SIX_MODES = [
    "compositional",
    "analogical",
    "pattern_extrapolation",
    "procedural",
    "constraint_based",
    "schema_induction",
]
MAX_TASK_CHARS = 800
MAX_STRATEGY_CHARS = 3000


def call_gemini(prompt: str, model: str, api_key: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model)
    response = client.generate_content(
        prompt,
        generation_config={"temperature": 0.0},
    )
    return response.text


def call_openai(prompt: str, model: str, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content


def load_prompt_template(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_prompt(template: str, task: str, strategy: str) -> str:
    task_text = task[:MAX_TASK_CHARS] + ("..." if len(task) > MAX_TASK_CHARS else "")
    strategy_text = strategy[:MAX_STRATEGY_CHARS] + ("..." if len(strategy) > MAX_STRATEGY_CHARS else "")
    return template.replace("{task}", task_text).replace("{strategy}", strategy_text)


def extract_json_from_response(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def score_one(
    record: dict,
    template: str,
    api: str,
    model: str,
    api_key: str,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> dict:
    prompt = build_prompt(template, record.get("task", ""), record.get("strategy", ""))
    call_fn = call_gemini if api == "gemini" else call_openai
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            raw_response = call_fn(prompt, model, api_key)
            parsed = extract_json_from_response(raw_response)
            if parsed is None:
                raise ValueError("Could not parse JSON from response: " + raw_response[:300])
            return {**record, "scores": parsed, "raw_response": raw_response, "error": None}
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            is_rate_limit = any(kw in err_str for kw in ("rate", "quota", "429", "resource exhausted"))
            wait = retry_delay * (2 ** (attempt - 1)) if is_rate_limit else retry_delay
            if attempt < max_retries:
                print(f"  [retry {attempt}/{max_retries}] {record.get('rollout_id')} - {e} - wait {wait:.0f}s")
                time.sleep(wait)
    return {**record, "scores": None, "raw_response": None, "error": str(last_err)}


def load_existing_results(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {r["rollout_id"]: r for r in data if r.get("rollout_id")}


def save_results(results: list, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def compute_radar_summary(score_file: str) -> dict:
    with open(score_file, encoding="utf-8") as f:
        results = json.load(f)
    mode_values = {m: [] for m in SIX_MODES}
    for r in results:
        scores = r.get("scores") or {}
        for mode in SIX_MODES:
            entry = scores.get(mode, {})
            if isinstance(entry, dict):
                val = entry.get("score")
            else:
                val = entry
            if val is not None and val != "N/A":
                try:
                    mode_values[mode].append(float(val))
                except (TypeError, ValueError):
                    pass
    summary = {}
    for mode in SIX_MODES:
        vals = mode_values[mode]
        summary[mode] = round(sum(vals) / len(vals), 3) if vals else None
    valid = [v for v in summary.values() if v is not None]
    summary["composite_mean"] = round(sum(valid) / len(valid), 3) if valid else None
    summary["n_scored"] = len(results)
    summary["n_valid_responses"] = sum(1 for r in results if r.get("scores") is not None)
    return summary


def print_radar_table(label: str, summary: dict):
    print("\n" + "=" * 52)
    print("  " + label)
    print("=" * 52)
    for mode in SIX_MODES:
        val = summary.get(mode)
        bar = ("*" * int(round((val or 0) * 4))) if val is not None else ""
        val_str = f"{val:.3f}" if val is not None else "N/A "
        print(f"  {mode:<27} {val_str:>5}  {bar}")
    print(f"  {'composite_mean':<27} {summary.get('composite_mean', 'N/A'):>5}")
    print(f"  (n_scored={summary.get('n_scored')}, valid={summary.get('n_valid_responses')})")


def run_scoring(args):
    api_key = args.api_key
    if not api_key:
        env_key = "GEMINI_API_KEY" if args.api == "gemini" else "OPENAI_API_KEY"
        api_key = os.environ.get(env_key, "")
    if not api_key:
        env_key = "GEMINI_API_KEY" if args.api == "gemini" else "OPENAI_API_KEY"
        sys.exit(f"Error: no API key provided. Use --api-key or export {env_key}=...")

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} samples from {args.input}")

    if not os.path.exists(args.prompt_file):
        sys.exit(f"Prompt file not found: {args.prompt_file}")
    template = load_prompt_template(args.prompt_file)

    existing = load_existing_results(args.output)
    if existing:
        print(f"Resuming: {len(existing)} already scored, {len(records) - len(existing)} remaining")
    to_score = [r for r in records if r.get("rollout_id") not in existing]

    all_results = list(existing.values())
    failed = []

    def _score_task(record):
        return score_one(record, template, args.api, args.model, api_key, args.max_retries)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_score_task, r): r for r in to_score}
        with tqdm(total=len(to_score), desc="Scoring", unit="sample") as pbar:
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                if result.get("error"):
                    failed.append(result.get("rollout_id", "?"))
                    pbar.set_postfix(failed=len(failed))
                pbar.update(1)
                if len(all_results) % 10 == 0:
                    save_results(all_results, args.output)

    save_results(all_results, args.output)
    print(f"\nDone. {len(all_results)} results saved to {args.output}")
    if failed:
        print(f"  Warning: {len(failed)} samples failed: {failed[:10]}")
    summary = compute_radar_summary(args.output)
    print_radar_table(os.path.basename(args.input), summary)


def run_summarize(args):
    if not args.pre_scores or not args.post_scores:
        sys.exit("--summarize requires --pre-scores and --post-scores")
    pre_summary = compute_radar_summary(args.pre_scores)
    post_summary = compute_radar_summary(args.post_scores)
    radar = {"pre_gist": pre_summary, "post_gist": post_summary}
    out_path = args.summary_out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(radar, f, ensure_ascii=False, indent=2)
    print_radar_table("Pre-GIST", pre_summary)
    print_radar_table("Post-GIST", post_summary)
    print(f"\nRadar summary saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Score strategy samples using the six-mode induction rubric.",
    )
    parser.add_argument("--input", help="Input JSON (output of select_samples.py)")
    parser.add_argument("--api", choices=["gemini", "openai"], help="API backend")
    parser.add_argument("--model", help="Model name, e.g. gemini-2.5-flash or gpt-4o")
    parser.add_argument("--api-key", dest="api_key", help="API key (or set env var)")
    parser.add_argument("--output", help="Output path for scored results JSON")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent API threads (default: 8)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries per sample (default: 3)")
    parser.add_argument("--prompt-file", default=PROMPT_FILE,
                        help="Path to scoring prompt template")
    parser.add_argument("--summarize", action="store_true",
                        help="Summarize two scored files into radar_summary.json")
    parser.add_argument("--pre-scores",
                        help="Pre-GIST scored results file (for --summarize)")
    parser.add_argument("--post-scores",
                        help="Post-GIST scored results file (for --summarize)")
    parser.add_argument("--summary-out", default="results/radar_summary.json",
                        help="Output for radar summary (default: results/radar_summary.json)")
    args = parser.parse_args()

    if args.summarize:
        run_summarize(args)
    else:
        if not args.input or not args.api or not args.model or not args.output:
            parser.error("Scoring mode requires --input, --api, --model, and --output")
        run_scoring(args)


if __name__ == "__main__":
    main()
