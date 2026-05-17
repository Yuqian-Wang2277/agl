#!/usr/bin/env python3
# 4_llm_judge.py - Four-dimension LLM-as-Judge scoring (3 models independently).
#
# Usage:
#   python 4_llm_judge.py --split pre \
#     --tasks data/pre_gist_tasks.json \
#     --rules generated_rules/pre_gist_rules.json \
#     --apis gemini,openai,anthropic \
#     --gemini-key $GEMINI_API_KEY \
#     --openai-key $OPENAI_API_KEY \
#     --anthropic-key $ANTHROPIC_API_KEY \
#     --output judge_results/pre_gist_judge.json

import json
import os
import re
import time
import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

SYS_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "prompts", "llm_judge_system.txt")
USER_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "prompts", "llm_judge_user.txt")

DIMS = ["abstraction_score", "coverage_score", "copying_score", "transferability_score"]
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
}


def load_prompt(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def parse_response(text):
    clean = re.sub(r"```(?:json)?|```", "", text).strip()
    scores = json.loads(clean)
    scores["copying_score_inv"] = 6 - int(scores.get("copying_score", 3))
    scores["induction_quality"] = statistics.mean([
        int(scores.get("abstraction_score", 3)),
        int(scores.get("coverage_score", 3)),
        scores["copying_score_inv"],
        int(scores.get("transferability_score", 3)),
    ])
    return scores


def call_gemini(sys_p, user_p, model, api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model, system_instruction=sys_p)
    r = client.generate_content(user_p, generation_config={"temperature": 0.0})
    return r.text


def call_openai(sys_p, user_p, model, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    r = client.chat.completions.create(
        model=model, temperature=0.0,
        messages=[{"role": "system", "content": sys_p},
                   {"role": "user", "content": user_p}])
    return r.choices[0].message.content


def call_anthropic(sys_p, user_p, model, api_key):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    r = client.messages.create(
        model=model, max_tokens=1024, temperature=0.0,
        system=sys_p,
        messages=[{"role": "user", "content": user_p}])
    return r.content[0].text


API_CALLERS = {"gemini": call_gemini, "openai": call_openai, "anthropic": call_anthropic}


def score_rule(rule_rec, task, sys_tmpl, user_tmpl, api_configs, retries=3):
    inductive_rule = rule_rec.get("inductive_rule", "")
    examples = task.get("examples", [])
    exs = (examples + [{}, {}, {}])[:3]

    def get_inp(ex): return ex.get("input", "").strip()[:400]
    def get_out(ex):
        t = ex.get("target", [])
        return str(t[0] if t else "")[:200]

    user_prompt = (user_tmpl
        .replace("{task_description}", task.get("problem", "").strip()[:500])
        .replace("{ex1_input}", get_inp(exs[0])).replace("{ex1_output}", get_out(exs[0]))
        .replace("{ex2_input}", get_inp(exs[1])).replace("{ex2_output}", get_out(exs[1]))
        .replace("{ex3_input}", get_inp(exs[2])).replace("{ex3_output}", get_out(exs[2]))
        .replace("{inductive_rule}", inductive_rule[:2000])
    )

    per_api = {}
    for api_name, cfg in api_configs.items():
        caller = API_CALLERS[api_name]
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                raw = caller(sys_tmpl, user_prompt, cfg["model"], cfg["key"])
                per_api[api_name] = parse_response(raw)
                per_api[api_name]["raw"] = raw
                break
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(2 ** attempt)
        if api_name not in per_api:
            per_api[api_name] = {"error": str(last_err)}

    ensemble = {}
    for dim in DIMS + ["copying_score_inv", "induction_quality"]:
        vals = [per_api[a][dim] for a in per_api
                if isinstance(per_api[a].get(dim), (int, float))]
        ensemble[dim] = round(statistics.mean(vals), 3) if vals else None

    return {
        "rule_id": rule_rec.get("rule_id"),
        "inductive_rule": inductive_rule,
        "per_api": per_api,
        "ensemble": ensemble,
    }


def cohen_kappa(scores1, scores2):
    from collections import Counter
    n = min(len(scores1), len(scores2))
    if n == 0:
        return 0.0
    po = sum(1 for a, b in zip(scores1[:n], scores2[:n]) if round(a) == round(b)) / n
    c1 = Counter(round(s) for s in scores1[:n])
    c2 = Counter(round(s) for s in scores2[:n])
    pe = sum(c1.get(k, 0) * c2.get(k, 0) for k in range(1, 7)) / (n * n)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0


def load_ckpt(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {(r["rollout_id"], r.get("rule_id", 0)): r for r in d if r.get("rollout_id")}


def save(res, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


def main():
    pa = argparse.ArgumentParser(description="LLM-as-Judge 4-dimension scoring.")
    pa.add_argument("--split", choices=["pre", "post"], required=True)
    pa.add_argument("--tasks", required=True)
    pa.add_argument("--rules", required=True)
    pa.add_argument("--apis", default="gemini,openai",
                    help="Comma-separated: gemini,openai,anthropic")
    pa.add_argument("--gemini-key", default=os.environ.get("GEMINI_API_KEY", ""))
    pa.add_argument("--openai-key", default=os.environ.get("OPENAI_API_KEY", ""))
    pa.add_argument("--anthropic-key", default=os.environ.get("ANTHROPIC_API_KEY", ""))
    pa.add_argument("--gemini-model", default=DEFAULT_MODELS["gemini"])
    pa.add_argument("--openai-model", default=DEFAULT_MODELS["openai"])
    pa.add_argument("--anthropic-model", default=DEFAULT_MODELS["anthropic"])
    pa.add_argument("--output", required=True)
    pa.add_argument("--workers", type=int, default=8)
    args = pa.parse_args()

    api_names = [a.strip() for a in args.apis.split(",")]
    key_map = {"gemini": args.gemini_key, "openai": args.openai_key, "anthropic": args.anthropic_key}
    model_map = {"gemini": args.gemini_model, "openai": args.openai_model, "anthropic": args.anthropic_model}
    api_configs = {n: {"key": key_map[n], "model": model_map[n]}
                   for n in api_names if key_map.get(n)}
    if not api_configs:
        raise SystemExit("No valid API keys provided. Use --gemini-key / --openai-key / --anthropic-key.")

    with open(args.tasks) as f:
        tasks = json.load(f)
    with open(args.rules) as f:
        rules_list = json.load(f)
    tasks_map = {t["rollout_id"]: t for t in tasks}

    sys_tmpl = load_prompt(SYS_PROMPT_FILE)
    user_tmpl = load_prompt(USER_PROMPT_FILE)

    existing = load_ckpt(args.output)
    work_items = []
    for rec in rules_list:
        task = tasks_map.get(rec["rollout_id"])
        if not task:
            continue
        for rr in rec.get("rules", []):
            if not rr.get("inductive_rule") or rr.get("error"):
                continue
            key = (rec["rollout_id"], rr.get("rule_id", 0))
            if key not in existing:
                work_items.append((rec["rollout_id"], task, rr))

    print(f"Scoring {len(work_items)} rules across {len(api_configs)} APIs")
    all_results = list(existing.values())
    failed = []

    def _score(item):
        rid, task, rr = item
        res = score_rule(rr, task, sys_tmpl, user_tmpl, api_configs)
        res["rollout_id"] = rid
        res["problem_type"] = task["problem_type"]
        return res

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_score, it): it for it in work_items}
        with tqdm(total=len(work_items), desc=f"Judge[{args.split}]", unit="rule") as bar:
            for fut in as_completed(futs):
                r = fut.result()
                all_results.append(r)
                if any("error" in r["per_api"].get(a, {}) for a in api_configs):
                    failed.append(r["rollout_id"])
                bar.update(1)
                if len(all_results) % 20 == 0:
                    save(all_results, args.output)

    save(all_results, args.output)
    print(f"\nSaved {len(all_results)} scored rules to {args.output}")

    # Print score summary
    valid = [r for r in all_results if r.get("ensemble")]
    print("\n=== Score Summary ===")
    for dim in ["abstraction_score", "coverage_score", "copying_score",
                "transferability_score", "induction_quality"]:
        vals = [r["ensemble"][dim] for r in valid if r["ensemble"].get(dim) is not None]
        if vals:
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {dim:<30} mean={statistics.mean(vals):.3f}  std={std:.3f}")

    # Cohen kappa between first two APIs
    apis = list(api_configs.keys())
    if len(apis) >= 2:
        print("\n=== Cohen's kappa (inter-rater agreement) ===")
        for dim in DIMS:
            s1, s2 = [], []
            for r in all_results:
                v1 = r.get("per_api", {}).get(apis[0], {}).get(dim)
                v2 = r.get("per_api", {}).get(apis[1], {}).get(dim)
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    s1.append(v1)
                    s2.append(v2)
            if s1 and s2:
                kappa = cohen_kappa(s1, s2)
                print(f"  {apis[0]} vs {apis[1]} [{dim}]: kappa={kappa:.4f}")

    if failed:
        print(f"\nWarning: {len(failed)} rules had API errors")


if __name__ == "__main__":
    main()
