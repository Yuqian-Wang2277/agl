#!/usr/bin/env python3
# 5_analyze.py - Representation space analysis + final summary.json
#
# Computes per split:
#   (a) Silhouette Score (cosine) for task-family clustering
#   (b) Nearest-Centroid Accuracy (leave-one-out)
#   (c) Confusion matrix (LLM predicts task family from rule text alone)
#   Aggregates all metrics into results/summary.json
#
# Usage:
#   python 5_analyze.py \
#     --pre-tasks  data/pre_gist_tasks.json \
#     --post-tasks data/post_gist_tasks.json \
#     --pre-rules  generated_rules/pre_gist_rules.json \
#     --post-rules generated_rules/post_gist_rules.json \
#     --pre-loo    loo_results/pre_gist_loo_metrics.json \
#     --post-loo   loo_results/post_gist_loo_metrics.json \
#     --pre-judge  judge_results/pre_gist_judge.json \
#     --post-judge judge_results/post_gist_judge.json \
#     --judge-api gemini --judge-key $GEMINI_API_KEY \
#     --output results/summary.json

import json
import os
import re
import argparse
import statistics

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

EMBED_MODEL = "BAAI/bge-m3"
DIMS = ["abstraction_score", "coverage_score", "copying_score_inv",
        "transferability_score", "induction_quality"]
DEFAULT_JUDGE_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
}


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_rules(rules_list, emb):
    texts, labels = [], []
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
        best_sim = -1.0
        y_hat = -1
        for cls in np.unique(y):
            idxs = np.where(mask & (y == cls))[0]
            if len(idxs) == 0:
                continue
            centroid = vecs[idxs].mean(axis=0)
            nrm = np.linalg.norm(centroid)
            if nrm > 1e-12:
                centroid /= nrm
            sim = float(np.dot(vecs[i], centroid))
            if sim > best_sim:
                best_sim = sim
                y_hat = cls
        if y_hat == y[i]:
            correct += 1
    return correct / len(vecs) if len(vecs) > 0 else 0.0


# ---------------------------------------------------------------------------
# LLM confusion matrix prediction
# ---------------------------------------------------------------------------

def _predict_family(inductive_rule, families, api_name, api_key, model):
    families_str = ", ".join(families)
    prompt = (
        f"Given the following inductive rule extracted from a few-shot learning strategy, "
        f"predict which task family it belongs to.\n\n"
        f"Rule:\n{inductive_rule[:800]}\n\n"
        f"Choose exactly one from: {families_str}\n"
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
                model=model, temperature=0.0, max_tokens=50,
                messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content.strip()
        elif api_name == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            r = client.messages.create(
                model=model, max_tokens=50, temperature=0.0,
                messages=[{"role": "user", "content": prompt}])
            return r.content[0].text.strip()
    except Exception:
        return ""


def judge_confusion(rules_list, tasks_map, api_name, api_key, model, sample=50):
    import random
    items = []
    for rec in rules_list:
        task = tasks_map.get(rec["rollout_id"], {})
        true_fam = task.get("problem_type", "unknown")
        for rr in rec.get("rules", []):
            txt = rr.get("inductive_rule", "").strip()
            if txt:
                items.append((true_fam, txt))
    all_families = sorted(set(x[0] for x in items))
    items = random.sample(items, min(sample, len(items)))
    true_labels, pred_labels = [], []
    for true_fam, rule_text in tqdm(items, desc="Family prediction"):
        pred = _predict_family(rule_text, all_families, api_name, api_key, model)
        matched = next((f for f in all_families if f.lower() in pred.lower()), "unknown")
        true_labels.append(true_fam)
        pred_labels.append(matched)
    return true_labels, pred_labels, all_families


# ---------------------------------------------------------------------------
# Judge score summary
# ---------------------------------------------------------------------------

def load_judge_summary(judge_path):
    if not judge_path or not os.path.exists(judge_path):
        return {}
    with open(judge_path) as f:
        data = json.load(f)
    summary = {}
    for dim in DIMS:
        vals = [r.get("ensemble", {}).get(dim) for r in data
                if isinstance(r.get("ensemble", {}).get(dim), (int, float))]
        summary[dim] = round(statistics.mean(vals), 3) if vals else None
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pa = argparse.ArgumentParser(description="Representation space analysis + summary.")
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
    pa.add_argument("--judge-model", default="")
    pa.add_argument("--embed-model", default=EMBED_MODEL)
    pa.add_argument("--output", default="results/summary.json")
    pa.add_argument("--skip-confusion", action="store_true",
                    help="Skip LLM confusion-matrix prediction (saves API calls)")
    pa.add_argument("--confusion-sample", type=int, default=50,
                    help="Number of rules to sample for confusion matrix (default 50)")
    args = pa.parse_args()

    judge_model = args.judge_model or DEFAULT_JUDGE_MODELS.get(args.judge_api, "")

    print(f"Loading embedding model: {args.embed_model}...")
    emb = SentenceTransformer(args.embed_model)

    with open(args.pre_tasks) as f: pre_tasks = json.load(f)
    with open(args.post_tasks) as f: post_tasks = json.load(f)
    with open(args.pre_rules) as f: pre_rules = json.load(f)
    with open(args.post_rules) as f: post_rules = json.load(f)
    with open(args.pre_loo) as f: pre_loo = json.load(f)
    with open(args.post_loo) as f: post_loo = json.load(f)

    tasks_map_pre = {t["rollout_id"]: t for t in pre_tasks}
    tasks_map_post = {t["rollout_id"]: t for t in post_tasks}

    summary = {"pre_gist": {}, "post_gist": {}}
    emb_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)), "..", "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    for split, rules_list, loo_m, tm in [
        ("pre_gist",  pre_rules,  pre_loo,  tasks_map_pre),
        ("post_gist", post_rules, post_loo, tasks_map_post),
    ]:
        print(f"\n=== {split} ===")

        # LOO metrics (already computed in step 3)
        summary[split]["loo"] = {k: loo_m.get(k) for k in [
            "H_ex_mean", "H_ex_global", "D_max_mean", "D_max_global",
            "S_total_mean", "a_global", "n_tasks", "attribution_matrix",
        ]}

        # Embedding
        print("  Encoding rules with bge-m3...")
        vecs, labels = embed_rules(rules_list, emb)
        np.save(os.path.join(emb_dir, f"{split}_emb.npy"), vecs)
        print(f"  Embeddings: {vecs.shape}, unique labels: {len(set(labels))}")

        # Silhouette Score
        if len(vecs) >= 2 and len(set(labels)) >= 2:
            le = LabelEncoder().fit(labels)
            y = le.transform(labels)
            sil = silhouette_score(vecs, y, metric="cosine")
            print(f"  Silhouette (cosine): {sil:.4f}")
            summary[split]["silhouette"] = round(float(sil), 4)

            # Nearest-centroid accuracy (LOO)
            print("  Nearest-centroid accuracy (LOO)...")
            nc_acc = nearest_centroid_loo(vecs, labels)
            print(f"  NC Accuracy: {nc_acc:.4f}")
            summary[split]["nc_accuracy"] = round(float(nc_acc), 4)
        else:
            summary[split]["silhouette"] = None
            summary[split]["nc_accuracy"] = None
            print("  Skipping silhouette / NC (not enough data or labels)")

        # LLM confusion matrix
        if not args.skip_confusion and args.judge_key and judge_model:
            print(f"  LLM confusion matrix ({args.judge_api})...")
            tl, pl, families = judge_confusion(
                rules_list, tm, args.judge_api, args.judge_key,
                judge_model, sample=args.confusion_sample)
            le2 = LabelEncoder().fit(families + ["unknown"])
            try:
                cm = confusion_matrix(le2.transform(tl), le2.transform(pl),
                                      labels=le2.transform(families))
                diag_acc = cm.diagonal().sum() / cm.sum() if cm.sum() > 0 else 0.0
                summary[split]["confusion_matrix"] = cm.tolist()
                summary[split]["confusion_labels"] = families
                summary[split]["confusion_diag_acc"] = round(float(diag_acc), 4)
                print(f"  Confusion diag accuracy: {diag_acc:.4f}")
            except Exception as e:
                print(f"  Confusion matrix error: {e}")
        else:
            summary[split]["confusion_matrix"] = None
            summary[split]["confusion_diag_acc"] = None

    # Save labels file
    with open(os.path.join(emb_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"pre_gist": [], "post_gist": []}, f)

    # LLM-as-Judge scores summary
    if args.pre_judge:
        summary["pre_gist"]["judge"] = load_judge_summary(args.pre_judge)
    if args.post_judge:
        summary["post_gist"]["judge"] = load_judge_summary(args.post_judge)

    # Comparison table
    print("\n" + "=" * 65)
    print("  FINAL COMPARISON: Pre-GIST vs Post-GIST")
    print("=" * 65)

    def get_val(split, key):
        parts = key.split(".")
        d = summary[split]
        for p in parts:
            d = d.get(p) if isinstance(d, dict) else None
        return d

    metrics_table = [
        ("H_ex (LOO, mean)",      "loo.H_ex_mean"),
        ("H_ex (LOO, global)",    "loo.H_ex_global"),
        ("D_max (LOO, mean)",     "loo.D_max_mean"),
        ("D_max (LOO, global)",   "loo.D_max_global"),
        ("S_total (LOO, mean)",   "loo.S_total_mean"),
        ("Silhouette Score",      "silhouette"),
        ("NC Accuracy",           "nc_accuracy"),
        ("Confusion Diag Acc",    "confusion_diag_acc"),
    ]
    for label, key in metrics_table:
        pre_v = get_val("pre_gist", key)
        post_v = get_val("post_gist", key)
        pv  = f"{pre_v:.4f}" if isinstance(pre_v, float) else str(pre_v)
        pov = f"{post_v:.4f}" if isinstance(post_v, float) else str(post_v)
        print(f"  {label:<30} Pre={pv:>8}  Post={pov:>8}")

    if summary["pre_gist"].get("judge") and summary["post_gist"].get("judge"):
        print("\n  LLM Judge scores:")
        for dim in DIMS:
            pv  = summary["pre_gist"]["judge"].get(dim)
            pov = summary["post_gist"]["judge"].get(dim)
            pv_s  = f"{pv:.3f}" if isinstance(pv, float) else str(pv)
            pov_s = f"{pov:.3f}" if isinstance(pov, float) else str(pov)
            print(f"  {dim:<30} Pre={pv_s:>6}  Post={pov_s:>6}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nFull summary saved to: {args.output}")


if __name__ == "__main__":
    main()
