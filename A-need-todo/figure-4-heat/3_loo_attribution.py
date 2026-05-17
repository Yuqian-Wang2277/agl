#!/usr/bin/env python3
# 3_loo_attribution.py - LOO attribution (bge-m3 + local vLLM)
#
# Steps per rule R_i (M rules per task):
#   1. Baseline embedding:  e_i = bge_m3.encode(R_i)
#   2. LOO regen:           remove Ex_j, call same-condition vLLM
#   3. Sensitivity:         s_ij = 1 - cosine_sim(e_i, e_i^{-j})
#   4. Attribution:         a_ij = softmax([s_i1, s_i2, s_i3])_j
#   5. Average over M:      a_bar_j = mean_i(a_ij)
#   6. Metrics:             H_ex / D_max / S_total
#
# Usage:
#   python 3_loo_attribution.py --split pre \
#     --tasks data/pre_gist_tasks.json \
#     --rules generated_rules/pre_gist_rules.json \
#     --api-base http://localhost:8000/v1 --model Qwen3-4B \
#     --output loo_results/pre_gist_loo.json

import json
import math
import os
import re
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

K = 3
EMBED_MODEL = "BAAI/bge-m3"
TEMP = 0.7
MAX_TOK = 4096


def extract_fl(text):
    m = re.search(r"FEWSHOT_LEARNING:\s*(.*?)(?=\n[A-Z_]{3,}:|$)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def build_up(examples, problem):
    lines = ["Here are example problems and their solutions:\n"]
    for i, ex in enumerate(examples, 1):
        t = ex.get("target", [])
        inp = ex.get("input", "").strip()
        sol = t[0] if t else ""
        lines.append(f"Example {i}:\nProblem: {inp}\nSolution: {sol}\n")
    lines.append("\nExtract the two-layer problem-solving strategy following the exact format above.")
    lines.append(f"\nCurrent problem:\n{problem.strip()}")
    return "\n".join(lines)


def call_vllm(client, model, sp, up, retries=3):
    for n in range(1, retries + 1):
        try:
            r = client.chat.completions.create(
                model=model, temperature=TEMP, max_tokens=MAX_TOK,
                messages=[{"role": "system", "content": sp},
                           {"role": "user", "content": up}])
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
        per_rule.append({"rule_id": rr["rule_id"], "inductive_rule": rr["inductive_rule"],
                          "s": sl, "a": sm(sl), "loo_details": ld})
    M = len(per_rule)
    ab = [sum(pr["a"][j] for pr in per_rule) / M for j in range(K)]
    Hex = -sum(v * math.log(v) for v in ab if v > 1e-12) / math.log(K)
    return {"rollout_id": task["rollout_id"], "problem_type": task["problem_type"],
            "reward_correctness": task.get("reward_correctness", 0.0), "M_valid": M,
            "a_bar": ab, "H_ex": Hex, "D_max": max(ab),
            "S_total_task": float(np.mean(all_s)) if all_s else 0.0,
            "per_rule": per_rule, "error": None}


def global_metrics(results):
    v = [r for r in results if not r.get("error")]
    if not v:
        return {}
    ag = [sum(r["a_bar"][j] for r in v) / len(v) for j in range(K)]
    Hg = -sum(x * math.log(x) for x in ag if x > 1e-12) / math.log(K)
    return {"n_tasks": len(v),
            "H_ex_mean": float(np.mean([r["H_ex"] for r in v])), "H_ex_global": Hg,
            "D_max_mean": float(np.mean([r["D_max"] for r in v])), "D_max_global": max(ag),
            "S_total_mean": float(np.mean([r["S_total_task"] for r in v])),
            "a_global": ag, "attribution_matrix": [r["a_bar"] for r in v],
            "task_ids": [r["rollout_id"] for r in v],
            "problem_types": [r["problem_type"] for r in v]}


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
    print(f"\n[{args.split.upper()}] H_ex={m.get('H_ex_mean', 0):.4f}  "
          f"D_max={m.get('D_max_mean', 0):.4f}  S={m.get('S_total_mean', 0):.4f}")
    print(f"  a_global={[f'{x:.3f}' for x in m.get('a_global', [])]}")
    mp = args.output.replace(".json", "_metrics.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.output}  metrics: {mp}")
    if failed:
        print(f"Failed {len(failed)}: {failed[:5]}")


if __name__ == "__main__":
    main()
