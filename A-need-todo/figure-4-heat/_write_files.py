#!/usr/bin/env python3
"""One-shot file generator: writes all remaining pipeline scripts."""
import os

BASE = os.path.dirname(os.path.abspath(__file__))


def w(name, content):
    path = os.path.join(BASE, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  wrote: {name}")


# ============================================================
# 3_loo_attribution.py
# ============================================================
LOO = r'''#!/usr/bin/env python3
"""3_loo_attribution.py - LOO attribution (bge-m3 + local vLLM).

Steps per rule R_i (M rules per task):
  1. Baseline embedding:  e_i = bge_m3.encode(R_i)
  2. LOO regen:           remove Ex_j, call same-condition vLLM
  3. Sensitivity:         s_ij = 1 - cosine_sim(e_i, e_i^(-j))
  4. Attribution:         a_ij = softmax([s_i1,s_i2,s_i3])_j
  5. Average over M:      a_bar_j = mean_i(a_ij)
  6. Metrics:             H_ex / D_max / S_total

Usage:
  python 3_loo_attribution.py --split pre \
    --tasks data/pre_gist_tasks.json \
    --rules generated_rules/pre_gist_rules.json \
    --api-base http://localhost:8000/v1 --model Qwen3-4B \
    --output loo_results/pre_gist_loo.json
"""
import json, math, os, re, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

K, EMBED_MODEL, TEMP, MAX_TOK = 3, "BAAI/bge-m3", 0.7, 4096


def extract_fl(text):
    m = re.search(r"FEWSHOT_LEARNING:\s*(.*?)(?=\n[A-Z_]{3,}:|$)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def build_up(examples, problem):
    lines = ["Here are example problems and their solutions:\n"]
    for i, ex in enumerate(examples, 1):
        t = ex.get("target", [])
        lines.append(
            f"Example {i}:\nProblem: {ex.get('input','').strip()}\n"
            f"Solution: {t[0] if t else ''}\n"
        )
    lines += [
        "\nExtract the two-layer problem-solving strategy following the exact format above.",
        f"\nCurrent problem:\n{problem.strip()}",
    ]
    return "\n".join(lines)


def call_vllm(client, model, sp, up, retries=3):
    for n in range(1, retries + 1):
        try:
            r = client.chat.completions.create(
                model=model, temperature=TEMP, max_tokens=MAX_TOK,
                messages=[{"role": "system", "content": sp},
                           {"role": "user", "content": up}],
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            if n < retries:
                time.sleep(2 ** n)
            else:
                raise RuntimeError(str(e))


def cossim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-10 and nb > 1e-10 else 0.0


def sm(vals):
    a = np.array(vals, dtype=float)
    a -= a.max()
    e = np.exp(a)
    return (e / e.sum()).tolist()


def load_ckpt(p):
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        d = json.load(f)
    return {r["rollout_id"]: r for r in d if r.get("rollout_id")}


def save(res, p):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


def proc(task, rules_rec, emb, client, model):
    rules = [r for r in rules_rec.get("rules", [])
             if r.get("inductive_rule") and not r.get("error")]
    if not rules:
        return {"rollout_id": task["rollout_id"], "problem_type": task["problem_type"],
                "error": "no valid rules"}
    per_rule = []
    all_s = []
    sp = task["system_prompt"]
    for rr in rules:
        ei = emb.encode(rr["inductive_rule"], normalize_embeddings=True)
        sl = []
        ld = []
        for j in range(K):
            loo_ex = [ex for k, ex in enumerate(task["examples"]) if k != j]
            try:
                lr = extract_fl(call_vllm(client, model, sp, build_up(loo_ex, task["problem"])))
            except Exception:
                lr = ""
            sij = 1.0 - cossim(ei, emb.encode(lr, normalize_embeddings=True)) if lr else 0.0
            sl.append(sij)
            all_s.append(sij)
            ld.append({"example_removed": j, "loo_rule": lr, "s_ij": sij})
        per_rule.append({
            "rule_id": rr["rule_id"],
            "inductive_rule": rr["inductive_rule"],
            "s": sl, "a": sm(sl), "loo_details": ld,
        })
    M = len(per_rule)
    ab = [sum(pr["a"][j] for pr in per_rule) / M for j in range(K)]
    Hex = -sum(v * math.log(v) for v in ab if v > 1e-12) / math.log(K)
    return {
        "rollout_id": task["rollout_id"], "problem_type": task["problem_type"],
        "reward_correctness": task.get("reward_correctness", 0.0), "M_valid": M,
        "a_bar": ab, "H_ex": Hex, "D_max": max(ab),
        "S_total_task": float(np.mean(all_s)) if all_s else 0.0,
        "per_rule": per_rule, "error": None,
    }


def global_metrics(results):
    v = [r for r in results if not r.get("error")]
    if not v:
        return {}
    ag = [sum(r["a_bar"][j] for r in v) / len(v) for j in range(K)]
    Hg = -sum(x * math.log(x) for x in ag if x > 1e-12) / math.log(K)
    return {
        "n_tasks": len(v),
        "H_ex_mean": float(np.mean([r["H_ex"] for r in v])), "H_ex_global": Hg,
        "D_max_mean": float(np.mean([r["D_max"] for r in v])), "D_max_global": max(ag),
        "S_total_mean": float(np.mean([r["S_total_task"] for r in v])),
        "a_global": ag, "attribution_matrix": [r["a_bar"] for r in v],
        "task_ids": [r["rollout_id"] for r in v],
        "problem_types": [r["problem_type"] for r in v],
    }


def main():
    pa = argparse.ArgumentParser(description="LOO attribution computation.")
    pa.add_argument("--split", choices=["pre", "post"], required=True)
    pa.add_argument("--tasks", required=True)
    pa.add_argument("--rules", required=True)
    pa.add_argument("--api-base", required=True)
    pa.add_argument("--model", required=True)
    pa.add_argument("--output", required=True)
    pa.add_argument("--embed-model", default=EMBED_MODEL)
    pa.add_argument("--workers", type=int, default=2)
    args = pa.parse_args()

    with open(args.tasks) as f:
        tasks = json.load(f)
    with open(args.rules) as f:
        rl = json.load(f)
    rm = {r["rollout_id"]: r for r in rl}
    print(f"Tasks: {len(tasks)}, rule entries: {len(rl)}")
    print(f"Loading {args.embed_model}...")
    emb = SentenceTransformer(args.embed_model)
    client = OpenAI(api_key="dummy", base_url=args.api_base)

    existing = load_ckpt(args.output)
    if existing:
        print(f"Resume: {len(existing)} done")
    todo = [t for t in tasks if t["rollout_id"] not in existing]
    res = list(existing.values())
    failed = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(proc, t, rm.get(t["rollout_id"], {"rules": []}),
                             emb, client, args.model): t for t in todo}
        with tqdm(total=len(todo), desc=f"LOO[{args.split}]", unit="task") as bar:
            for fut in as_completed(futs):
                r = fut.result()
                res.append(r)
                if r.get("error"):
                    failed.append(r.get("rollout_id"))
                    bar.set_postfix(fail=len(failed))
                bar.update(1)
                if len(res) % 5 == 0:
                    save(res, args.output)

    save(res, args.output)
    m = global_metrics(res)
    print(f"\n[{args.split.upper()}] H_ex={m.get('H_ex_mean',0):.4f}  "
          f"D_max={m.get('D_max_mean',0):.4f}  S={m.get('S_total_mean',0):.4f}")
    print(f"  a_global={[f'{x:.3f}' for x in m.get('a_global',[])]}")
    mp = args.output.replace(".json", "_metrics.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.output}  metrics: {mp}")
    if failed:
        print(f"Failed {len(failed)}: {failed[:5]}")


if __name__ == "__main__":
    main()
'''

# ============================================================
# 4_llm_judge.py
# ============================================================
JUDGE = r'''#!/usr/bin/env python3
"""4_llm_judge.py - Four-dimension LLM-as-Judge scoring (3 models independently).

Usage:
  python 4_llm_judge.py --split pre \
    --tasks data/pre_gist_tasks.json \
    --rules generated_rules/pre_gist_rules.json \
    --apis gemini,openai,anthropic \
    --gemini-key $GEMINI_API_KEY \
    --openai-key $OPENAI_API_KEY \
    --anthropic-key $ANTHROPIC_API_KEY \
    --output judge_results/pre_gist_judge.json
"""
import json, os, re, time, argparse, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SYS_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "llm_judge_system.txt")
USER_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "llm_judge_user.txt")
DIMS = ["abstraction_score", "coverage_score", "copying_score", "transferability_score"]


def load_prompt(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def parse_response(text):
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
    scores = json.loads(text)
    scores["copying_score_inv"] = 6 - int(scores.get("copying_score", 3))
    scores["induction_quality"] = statistics.mean([
        int(scores.get("abstraction_score", 3)),
        int(scores.get("coverage_score", 3)),
        scores["copying_score_inv"],
        int(scores.get("transferability_score", 3)),
    ])
    return scores


def call_gemini(prompt_sys, prompt_user, model, api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model, system_instruction=prompt_sys)
    r = client.generate_content(prompt_user, generation_config={"temperature": 0.0})
    return r.text


def call_openai(prompt_sys, prompt_user, model, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    r = client.chat.completions.create(
        model=model, temperature=0.0,
        messages=[{"role": "system", "content": prompt_sys},
                   {"role": "user", "content": prompt_user}],
    )
    return r.choices[0].message.content


def call_anthropic(prompt_sys, prompt_user, model, api_key):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    r = client.messages.create(
        model=model, max_tokens=1024, temperature=0.0,
        system=prompt_sys,
        messages=[{"role": "user", "content": prompt_user}],
    )
    return r.content[0].text


API_CALLERS = {"gemini": call_gemini, "openai": call_openai, "anthropic": call_anthropic}
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
}


def format_examples(examples):
    lines = []
    for i, ex in enumerate(examples, 1):
        t = ex.get("target", [])
        lines.append(f"Example {i}: Input={ex.get('input','').strip()[:200]}  Output={t[0] if t else ''}")
    return "\n".join(lines)


def score_rule(rule_rec, task, sys_tmpl, user_tmpl, api_configs, retries=3):
    inductive_rule = rule_rec.get("inductive_rule", "")
    examples = task.get("examples", [])
    exs = examples if len(examples) >= 3 else examples + [{}] * (3 - len(examples))

    def get_inp(ex): return ex.get("input", "").strip()[:300]
    def get_out(ex):
        t = ex.get("target", [])
        return str(t[0] if t else "")[:200]

    user_prompt = (user_tmpl
        .replace("{task_description}", task.get("problem", "").strip()[:400])
        .replace("{ex1_input}", get_inp(exs[0])).replace("{ex1_output}", get_out(exs[0]))
        .replace("{ex2_input}", get_inp(exs[1])).replace("{ex2_output}", get_out(exs[1]))
        .replace("{ex3_input}", get_inp(exs[2])).replace("{ex3_output}", get_out(exs[2]))
        .replace("{inductive_rule}", inductive_rule[:1500])
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

    # Ensemble: average valid scores across APIs
    ensemble = {}
    for dim in DIMS + ["copying_score_inv", "induction_quality"]:
        vals = [per_api[a][dim] for a in per_api if isinstance(per_api[a].get(dim), (int, float))]
        ensemble[dim] = round(statistics.mean(vals), 3) if vals else None

    return {
        "rule_id": rule_rec.get("rule_id"),
        "inductive_rule": inductive_rule,
        "per_api": per_api,
        "ensemble": ensemble,
    }


def cohen_kappa(scores1, scores2, dim):
    from collections import Counter
    n = min(len(scores1), len(scores2))
    agreed = sum(1 for a, b in zip(scores1[:n], scores2[:n]) if round(a) == round(b))
    po = agreed / n if n else 0
    c1 = Counter(round(s) for s in scores1[:n])
    c2 = Counter(round(s) for s in scores2[:n])
    pe = sum(c1.get(k, 0) * c2.get(k, 0) for k in range(1, 7)) / (n * n) if n else 0
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return round(kappa, 4)


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
    pa = argparse.ArgumentParser()
    pa.add_argument("--split", choices=["pre", "post"], required=True)
    pa.add_argument("--tasks", required=True)
    pa.add_argument("--rules", required=True)
    pa.add_argument("--apis", default="gemini,openai",
                    help="Comma-separated list of APIs to use: gemini,openai,anthropic")
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
    api_configs = {}
    key_map = {"gemini": args.gemini_key, "openai": args.openai_key, "anthropic": args.anthropic_key}
    model_map = {"gemini": args.gemini_model, "openai": args.openai_model, "anthropic": args.anthropic_model}
    for name in api_names:
        if not key_map.get(name):
            print(f"Warning: no API key for {name}, skipping")
            continue
        api_configs[name] = {"key": key_map[name], "model": model_map[name]}

    if not api_configs:
        raise SystemExit("No valid API configs. Provide at least one API key.")

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

    print(f"Scoring {len(work_items)} rules ({len(api_configs)} APIs each)")
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

    # Summary stats per API and ensemble
    print("\n=== Score Summary ===")
    valid = [r for r in all_results if r.get("ensemble")]
    for dim in ["abstraction_score", "coverage_score", "copying_score", "transferability_score", "induction_quality"]:
        vals = [r["ensemble"][dim] for r in valid if r["ensemble"].get(dim) is not None]
        if vals:
            print(f"  {dim}: mean={statistics.mean(vals):.3f} std={statistics.stdev(vals) if len(vals)>1 else 0:.3f}")

    # Cohen kappa between first two APIs
    apis = list(api_configs.keys())
    if len(apis) >= 2:
        print("\n=== Cohen's kappa (inter-rater) ===")
        for dim in DIMS:
            s1 = [r["per_api"][apis[0]].get(dim) for r in all_results
                  if isinstance(r.get("per_api", {}).get(apis[0], {}).get(dim), (int, float))]
            s2 = [r["per_api"][apis[1]].get(dim) for r in all_results
                  if isinstance(r.get("per_api", {}).get(apis[1], {}).get(dim), (int, float))]
            if len(s1) > 1 and len(s1) == len(s2):
                print(f"  {apis[0]} vs {apis[1]} [{dim}]: kappa={cohen_kappa(s1, s2, dim)}")

    if failed:
        print(f"\nWarning: {len(failed)} rules had API errors")


if __name__ == "__main__":
    main()
'''

# ============================================================
# 5_analyze.py
# ============================================================
ANALYZE = r'''#!/usr/bin/env python3
"""5_analyze.py - Representation space analysis + final summary.json.

Computes:
  - Silhouette Score (cosine) per split
  - Nearest-Centroid Accuracy (LOO)
  - Confusion matrix (judge predicts task family from rule text)
  - Aggregates all metrics into results/summary.json

Usage:
  python 5_analyze.py \
    --pre-tasks data/pre_gist_tasks.json \
    --post-tasks data/post_gist_tasks.json \
    --pre-rules generated_rules/pre_gist_rules.json \
    --post-rules generated_rules/post_gist_rules.json \
    --pre-loo loo_results/pre_gist_loo_metrics.json \
    --post-loo loo_results/post_gist_loo_metrics.json \
    --pre-judge judge_results/pre_gist_judge.json \
    --post-judge judge_results/post_gist_judge.json \
    --judge-api gemini --judge-key $GEMINI_API_KEY \
    --output results/summary.json
'''
import json, os, re, argparse, statistics
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

EMBED_MODEL = "BAAI/bge-m3"
DIMS = ["abstraction_score", "coverage_score", "copying_score_inv", "transferability_score", "induction_quality"]


def embed_rules(rules_list, emb):
    texts = []
    labels = []
    for rec in rules_list:
        pt = rec.get("problem_type", "unknown")
        for rr in rec.get("rules", []):
            txt = rr.get("inductive_rule", "").strip()
            if txt:
                texts.append(txt)
                labels.append(pt)
    if not texts:
        return np.array([]), []
    vecs = emb.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return vecs, labels


def nearest_centroid_loo(vecs, labels):
    le = LabelEncoder().fit(labels)
    y = le.transform(labels)
    correct = 0
    for i in range(len(vecs)):
        mask = np.ones(len(vecs), dtype=bool)
        mask[i] = False
        y_hat = -1
        best_sim = -1.0
        for cls in np.unique(y):
            idxs = np.where(mask & (y == cls))[0]
            if len(idxs) == 0:
                continue
            centroid = vecs[idxs].mean(axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-12)
            sim = float(np.dot(vecs[i], centroid))
            if sim > best_sim:
                best_sim = sim
                y_hat = cls
        if y_hat == y[i]:
            correct += 1
    return correct / len(vecs) if len(vecs) > 0 else 0.0


def judge_predict_family(rules_list, tasks_map, api_name, api_key, model, sample=50):
    """Ask LLM to predict task family from rule text alone. Returns (true_labels, pred_labels)."""
    import random
    items = []
    for rec in rules_list:
        task = tasks_map.get(rec["rollout_id"], {})
        for rr in rec.get("rules", []):
            txt = rr.get("inductive_rule", "").strip()
            if txt:
                items.append((task.get("problem_type", "unknown"), txt))
    all_families = sorted(set(x[0] for x in items))
    items = random.sample(items, min(sample, len(items)))

    def _predict(inductive_rule, families):
        prompt = (
            f"Given the following inductive rule extracted from a few-shot learning strategy, "
            f"predict which task family it belongs to.\n\n"
            f"Rule:\n{inductive_rule[:800]}\n\n"
            f"Choose exactly one from: {', '.join(families)}\n"
            f"Output only the task family name, nothing else."
        )
        try:
            if api_name == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                r = genai.GenerativeModel(model).generate_content(prompt)
                return r.text.strip()
            elif api_name == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                r = client.chat.completions.create(
                    model=model, temperature=0.0,
                    messages=[{"role": "user", "content": prompt}])
                return r.choices[0].message.content.strip()
            elif api_name == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                r = client.messages.create(model=model, max_tokens=50, temperature=0.0,
                    messages=[{"role": "user", "content": prompt}])
                return r.content[0].text.strip()
        except Exception:
            return ""

    true_labels, pred_labels = [], []
    for true_fam, rule_text in tqdm(items, desc="Family prediction"):
        pred = _predict(rule_text, all_families)
        matched = next((f for f in all_families if f.lower() in pred.lower()), "unknown")
        true_labels.append(true_fam)
        pred_labels.append(matched)
    return true_labels, pred_labels, all_families


def load_judge_summary(judge_path):
    if not os.path.exists(judge_path):
        return {}
    with open(judge_path) as f:
        data = json.load(f)
    summary = {}
    for dim in DIMS:
        vals = [r.get("ensemble", {}).get(dim) for r in data
                if isinstance(r.get("ensemble", {}).get(dim), (int, float))]
        summary[dim] = round(statistics.mean(vals), 3) if vals else None
    return summary


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--pre-tasks", required=True)
    pa.add_argument("--post-tasks", required=True)
    pa.add_argument("--pre-rules", required=True)
    pa.add_argument("--post-rules", required=True)
    pa.add_argument("--pre-loo", required=True, help="pre_gist_loo_metrics.json")
    pa.add_argument("--post-loo", required=True, help="post_gist_loo_metrics.json")
    pa.add_argument("--pre-judge", default="")
    pa.add_argument("--post-judge", default="")
    pa.add_argument("--judge-api", choices=["gemini", "openai", "anthropic"], default="gemini")
    pa.add_argument("--judge-key", default=os.environ.get("GEMINI_API_KEY", ""))
    pa.add_argument("--judge-model", default="gemini-2.5-flash")
    pa.add_argument("--embed-model", default=EMBED_MODEL)
    pa.add_argument("--output", default="results/summary.json")
    pa.add_argument("--skip-confusion", action="store_true",
                    help="Skip LLM confusion matrix (saves API calls)")
    args = pa.parse_args()

    print(f"Loading embedding model: {args.embed_model}...")
    emb = SentenceTransformer(args.embed_model)

    with open(args.pre_tasks) as f: pre_tasks = json.load(f)
    with open(args.post_tasks) as f: post_tasks = json.load(f)
    with open(args.pre_rules) as f: pre_rules = json.load(f)
    with open(args.post_rules) as f: post_rules = json.load(f)

    tasks_map_pre = {t["rollout_id"]: t for t in pre_tasks}
    tasks_map_post = {t["rollout_id"]: t for t in post_tasks}

    with open(args.pre_loo) as f: pre_loo = json.load(f)
    with open(args.post_loo) as f: post_loo = json.load(f)

    summary = {"pre_gist": {}, "post_gist": {}}

    for split, rules_list, loo_m in [("pre_gist", pre_rules, pre_loo),
                                      ("post_gist", post_rules, post_loo)]:
        print(f"\n=== {split} ===")
        summary[split]["loo"] = {k: loo_m.get(k) for k in
            ["H_ex_mean", "H_ex_global", "D_max_mean", "D_max_global",
             "S_total_mean", "a_global", "n_tasks"]}

        print("  Embedding rules...")
        vecs, labels = embed_rules(rules_list, emb)
        np.save(os.path.join(os.path.dirname(args.output),
                             f"../embeddings/{split}_emb.npy"), vecs)
        with open(os.path.join(os.path.dirname(args.output),
                               "../embeddings/labels.json"), "w") as f:
            json.dump({"pre_gist": labels if split == "pre_gist" else [],
                        "post_gist": labels if split == "post_gist" else []}, f)

        if len(vecs) >= 2 and len(set(labels)) >= 2:
            le = LabelEncoder().fit(labels)
            y = le.transform(labels)
            sil = silhouette_score(vecs, y, metric="cosine")
            print(f"  Silhouette Score (cosine): {sil:.4f}")
            summary[split]["silhouette"] = round(float(sil), 4)

            print("  Nearest-centroid accuracy (LOO)...")
            nc_acc = nearest_centroid_loo(vecs, labels)
            print(f"  NC Accuracy: {nc_acc:.4f}")
            summary[split]["nc_accuracy"] = round(float(nc_acc), 4)
        else:
            summary[split]["silhouette"] = None
            summary[split]["nc_accuracy"] = None

    # LLM-as-Judge scores summary
    if args.pre_judge and os.path.exists(args.pre_judge):
        summary["pre_gist"]["judge"] = load_judge_summary(args.pre_judge)
    if args.post_judge and os.path.exists(args.post_judge):
        summary["post_gist"]["judge"] = load_judge_summary(args.post_judge)

    # Confusion matrix
    if not args.skip_confusion and args.judge_key:
        print("\nRunning LLM confusion matrix prediction...")
        for split, rules_list, tm in [("pre_gist", pre_rules, tasks_map_pre),
                                        ("post_gist", post_rules, tasks_map_post)]:
            tl, pl, families = judge_predict_family(
                rules_list, tm, args.judge_api, args.judge_key, args.judge_model, sample=50)
            le = LabelEncoder().fit(families)
            try:
                cm = confusion_matrix(le.transform(tl), le.transform(pl),
                                      labels=le.transform(families))
                summary[split]["confusion_matrix"] = cm.tolist()
                summary[split]["confusion_labels"] = families
                diag_acc = cm.diagonal().sum() / cm.sum() if cm.sum() > 0 else 0
                summary[split]["confusion_diag_acc"] = round(float(diag_acc), 4)
                print(f"  [{split}] confusion diag accuracy: {diag_acc:.4f}")
            except Exception as e:
                print(f"  [{split}] confusion matrix error: {e}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON: Pre-GIST vs Post-GIST")
    print("=" * 60)
    metrics_to_compare = [
        ("H_ex (LOO, mean)", "loo.H_ex_mean"),
        ("D_max (LOO, mean)", "loo.D_max_mean"),
        ("S_total (LOO, mean)", "loo.S_total_mean"),
        ("Silhouette Score", "silhouette"),
        ("NC Accuracy", "nc_accuracy"),
    ]
    for label, key in metrics_to_compare:
        def get_val(split, key):
            parts = key.split(".")
            d = summary[split]
            for p in parts:
                if isinstance(d, dict):
                    d = d.get(p)
                else:
                    return None
            return d
        pre_v = get_val("pre_gist", key)
        post_v = get_val("post_gist", key)
        pv = f"{pre_v:.4f}" if isinstance(pre_v, float) else str(pre_v)
        pov = f"{post_v:.4f}" if isinstance(post_v, float) else str(post_v)
        print(f"  {label:<30} Pre={pv:>8}  Post={pov:>8}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nFull summary saved to: {args.output}")


if __name__ == "__main__":
    main()
'''

# ============================================================
# prompts
# ============================================================
SYS_PROMPT = r"""You are an expert evaluator for few-shot learning strategy quality.
You will be given:
  (1) A task description
  (2) K support examples (input-output pairs) used to generate the rule
  (3) A generated inductive rule [INDUCTIVE_RULE]

Your job is to evaluate the quality of the inductive rule on four dimensions.
Be objective, critical, and consistent.
Output ONLY a valid JSON object. Do not include any explanation outside the JSON.
"""

USER_PROMPT = r"""## Calibration Examples (for reference only — do not score these)

[CALIBRATION EXAMPLE A — High abstraction, Low copying]
Task: Sentiment classification of product reviews
Support Ex1: "The battery dies quickly." → Negative
Support Ex2: "Screen brightness is excellent." → Positive
Support Ex3: "Delivery was faster than expected." → Positive
Rule: "Identify whether the review expresses a net positive or negative
      evaluation of a product attribute; classify accordingly."
Expected scores:
  abstraction_score: 5  (no specific details from examples)
  coverage_score: 5     (all three examples follow this principle)
  copying_score: 1      (no copying detectable)
  transferability_score: 5

[CALIBRATION EXAMPLE B — Low abstraction, High copying]
Task: Sentiment classification of product reviews
Support Ex1: "The battery dies quickly." → Negative
Support Ex2: "Screen brightness is excellent." → Positive
Support Ex3: "Delivery was faster than expected." → Positive
Rule: "If the review mentions battery life dying quickly, classify it as Negative."
Expected scores:
  abstraction_score: 1  (specific to Ex1 only)
  coverage_score: 1     (only Ex1 is covered)
  copying_score: 5      (reproduces Ex1's content directly)
  transferability_score: 1

---

## Task Description
{task_description}

## Support Examples
Example 1:
  Input:  {ex1_input}
  Output: {ex1_output}

Example 2:
  Input:  {ex2_input}
  Output: {ex2_output}

Example 3:
  Input:  {ex3_input}
  Output: {ex3_output}

## Generated Inductive Rule
{inductive_rule}

---

## Evaluation Instructions

Please evaluate the rule on the following four dimensions.
For each dimension, assign an integer score from 1 to 5, and provide
a one-sentence justification.

### Dimension 1 — abstraction_score (1-5)
Does the rule express a **task-level abstract principle** rather than
restating or paraphrasing the support examples?
- 5: The rule is fully abstract; it describes a general principle
     applicable to any instance of this task, with no surface-level
     copying from the examples.
- 3: The rule is partially abstract but still borrows specific details
     (e.g., entities, phrasing) from one or more examples.
- 1: The rule is essentially a restatement or light paraphrase of a
     single support example; no genuine abstraction is present.

### Dimension 2 — coverage_score (1-5)
Is the rule **collectively supported** by multiple support examples,
rather than derived from only one?
- 5: All three examples are consistent with and supportive of the rule;
     the rule captures what is common across all of them.
- 3: The rule fits two examples well but is inconsistent with or
     irrelevant to one example.
- 1: The rule only fits one example; the other examples are irrelevant
     or even contradicted by the rule.

### Dimension 3 — copying_score (1-5)
To what extent does the rule appear to be **copied or minimally
transformed** from a single support example?
- 5: Strong evidence of copying — the rule reproduces specific content
     (entities, structure, wording) from one example with minor changes.
- 3: Moderate borrowing — the rule draws heavily from one example
     but shows some generalization.
- 1: No copying — the rule does not closely resemble any single example;
     it reads as a genuinely inferred abstraction.
NOTE: A higher copying_score is WORSE. This dimension is reverse-scored.

### Dimension 4 — transferability_score (1-5)
Would the rule be **useful and applicable** to new, unseen instances
of this task that are different from the support examples?
- 5: The rule is clearly transferable; a new example following this
     task type would benefit directly from applying the rule.
- 3: The rule is somewhat transferable but may not apply to instances
     that differ significantly from the support examples.
- 1: The rule is not transferable; it is too specific to the given
     examples to generalize.

---

## Output Format
Return ONLY the following JSON object (no markdown, no extra text):

{
  "abstraction_score": <int 1-5>,
  "abstraction_reason": "<one sentence>",
  "coverage_score": <int 1-5>,
  "coverage_reason": "<one sentence>",
  "copying_score": <int 1-5>,
  "copying_reason": "<one sentence>",
  "transferability_score": <int 1-5>,
  "transferability_reason": "<one sentence>"
}
"""

# ============================================================
# requirements.txt
# ============================================================
REQS = """sentence-transformers>=3.0.0
scikit-learn>=1.4.0
numpy>=1.26.0
scipy>=1.13.0
google-generativeai>=0.8.0
openai>=1.50.0
anthropic>=0.30.0
tqdm>=4.66.0
matplotlib>=3.9.0
seaborn>=0.13.0
"""

# ============================================================
# README.md
# ============================================================
README = """# Figure-4 热力图归因分析流水线

## 目标

证明 GIST 训练将策略生成从"单例模仿"转变为"多示例任务级归纳"：
- **Pre-GIST**：归因集中于单一示例（H_ex 低、D_max 高）
- **Post-GIST**：归因均匀分布（H_ex 高、D_max 低）

## 环境准备

```bash
pip install -r requirements.txt
```

bge-m3 模型（约 2GB）会在首次运行时自动下载。

## 运行步骤

### Step 1：数据提取

从评测 shard 中采样 N=100 条任务（Pre-GIST 取最差，Post-GIST 取最优）：

```bash
python 1_extract_tasks.py
# 输出：data/pre_gist_tasks.json  data/post_gist_tasks.json
```

若原始 shard 路径不同，可通过 `--pre-dir` / `--post-dir` 指定。

### Step 2：M=5 规则生成

需要 Pre-GIST (Qwen3-4B) 和 Post-GIST (Qwen3-semi) 两个本地 vLLM 服务同时运行。

```bash
# Pre-GIST
python 2_gen_rules.py --split pre \\
    --input data/pre_gist_tasks.json \\
    --api-base http://localhost:8000/v1 \\
    --model Qwen3-4B \\
    --output generated_rules/pre_gist_rules.json

# Post-GIST
python 2_gen_rules.py --split post \\
    --input data/post_gist_tasks.json \\
    --api-base http://localhost:8001/v1 \\
    --model Qwen3-semi \\
    --output generated_rules/post_gist_rules.json
```

### Step 3：LOO 归因计算（核心）

使用 bge-m3 计算基准 embedding + LOO 重生成敏感度：

```bash
python 3_loo_attribution.py --split pre \\
    --tasks data/pre_gist_tasks.json \\
    --rules generated_rules/pre_gist_rules.json \\
    --api-base http://localhost:8000/v1 --model Qwen3-4B \\
    --output loo_results/pre_gist_loo.json

python 3_loo_attribution.py --split post \\
    --tasks data/post_gist_tasks.json \\
    --rules generated_rules/post_gist_rules.json \\
    --api-base http://localhost:8001/v1 --model Qwen3-semi \\
    --output loo_results/post_gist_loo.json
```

输出：loo_results/pre_gist_loo.json + pre_gist_loo_metrics.json（含 H_ex / D_max / S_total）

### Step 4：LLM-as-Judge 四维评分

使用闭源 API（支持 Gemini / OpenAI / Anthropic）：

```bash
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."

python 4_llm_judge.py --split pre \\
    --tasks data/pre_gist_tasks.json \\
    --rules generated_rules/pre_gist_rules.json \\
    --apis gemini,openai,anthropic \\
    --output judge_results/pre_gist_judge.json

python 4_llm_judge.py --split post \\
    --tasks data/post_gist_tasks.json \\
    --rules generated_rules/post_gist_rules.json \\
    --apis gemini,openai,anthropic \\
    --output judge_results/post_gist_judge.json
```

四个评分维度：
- abstraction_score：任务级抽象性（高=好）
- coverage_score：多示例覆盖性（高=好）
- copying_score：单例复制程度（高=差，自动反转为 copying_score_inv）
- transferability_score：可迁移性（高=好）

### Step 5：表征空间分析 + 汇总

```bash
python 5_analyze.py \\
    --pre-tasks data/pre_gist_tasks.json \\
    --post-tasks data/post_gist_tasks.json \\
    --pre-rules generated_rules/pre_gist_rules.json \\
    --post-rules generated_rules/post_gist_rules.json \\
    --pre-loo  loo_results/pre_gist_loo_metrics.json \\
    --post-loo loo_results/post_gist_loo_metrics.json \\
    --pre-judge  judge_results/pre_gist_judge.json \\
    --post-judge judge_results/post_gist_judge.json \\
    --judge-api gemini --judge-key $GEMINI_API_KEY \\
    --output results/summary.json
```

输出 `results/summary.json` 包含：
- LOO 指标（H_ex / D_max / S_total / 归因矩阵）
- Judge 指标（四维均值 + induction_quality）
- Silhouette Score
- Nearest-Centroid Accuracy
- 混淆矩阵（LLM 根据规则文本预测任务族）

## 断点续跑

所有脚本支持断点续跑：重新运行相同命令时，已完成的条目会自动跳过。

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| N | 100 | 每 split 采样任务数 |
| M | 5 | 每任务生成规则数 |
| K | 3 | few-shot 支持示例数 |
| Embed | BAAI/bge-m3 | LOO 敏感度计算和聚类分析用 |
| LOO workers | 2 | 并发数（LOO 计算密集，不宜过高）|
| Judge workers | 8 | 评分并发数 |
"""

if __name__ == "__main__":
    w("3_loo_attribution.py", LOO)
    w("4_llm_judge.py", JUDGE)
    w("5_analyze.py", ANALYZE)
    w("prompts/llm_judge_system.txt", SYS_PROMPT)
    w("prompts/llm_judge_user.txt", USER_PROMPT)
    w("requirements.txt", REQS)
    w("README.md", README)
    print("\nAll files written.")
