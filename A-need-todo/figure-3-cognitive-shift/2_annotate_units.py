#!/usr/bin/env python3
"""
2_annotate_units.py — 对每条 <think> 轨迹逐单元标注（E/D/P/R）+ AHS 评分。

流程：
  1. 按句号/分号切分 primary_think 为命题单元 [u_1,...,u_n]
  2. Prompt 1 v1 对每个单元打 E/D/P/R 标签（round 1）
  3. Prompt 1 v2 独立打标签（round 2，措辞不同）
  4. 计算 Cohen's kappa；若 kappa < 0.7，对分歧单元第三次裁决
  5. Prompt 2 仅对最终标签为 P 的单元打 AHS 0-3 分
  6. 支持断点续跑、并发、检查点保存

用法：
  python 2_annotate_units.py --split pre --api-key $GEMINI_API_KEY
  python 2_annotate_units.py --split post --api-key $GEMINI_API_KEY
  python 2_annotate_units.py --split pre --api-key $GEMINI_API_KEY --workers 16
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(BASE, "prompts")
GEMINI_MODEL = "gemini-2.0-flash"
VALID_LABELS = {"E", "D", "P", "R"}
SENTENCE_SEP = re.compile(r"[.;。；]\s+")


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

def call_gemini(system_text: str, user_text: str, api_key: str,
                model: str = GEMINI_MODEL, retries: int = 4) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model, system_instruction=system_text)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = client.generate_content(user_text, generation_config={"temperature": 0.0})
            return r.text.strip()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini call failed after {retries} retries: {last_err}")


def parse_json_response(text: str) -> dict:
    clean = re.sub(r"```(?:json)?|```", "", text).strip()
    return json.loads(clean)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompt(filename: str) -> tuple[str, str]:
    path = os.path.join(PROMPT_DIR, filename)
    with open(path, encoding="utf-8") as f:
        content = f.read()
    # split by "USER\n" to get system and user parts
    parts = content.split("\nUSER\n", 1)
    if len(parts) == 2:
        sys_part = parts[0].replace("SYSTEM\n", "").strip()
        user_part = parts[1].strip()
    else:
        sys_part = ""
        user_part = content.strip()
    return sys_part, user_part


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def split_into_units(think_text: str) -> list[str]:
    # normalize line breaks within think
    text = think_text.replace("\n", " ").strip()
    # split on period/semicolon followed by whitespace
    raw = SENTENCE_SEP.split(text)
    units = [u.strip() for u in raw if u.strip() and len(u.strip()) >= 10]
    return units


# ---------------------------------------------------------------------------
# Annotation (Prompt 1)
# ---------------------------------------------------------------------------

def annotate_unit_label(unit_text: str, problem_text: str, examples_summary: str,
                         sys_prompt: str, user_tmpl: str, api_key: str) -> dict:
    user = (user_tmpl
            .replace("{problem_text}", problem_text[:400])
            .replace("{examples_summary}", examples_summary[:300])
            .replace("{unit_text}", unit_text[:500]))
    raw = call_gemini(sys_prompt, user, api_key)
    try:
        result = parse_json_response(raw)
        label = result.get("label", "D").upper().strip()
        if label not in VALID_LABELS:
            label = "D"
        return {"label": label, "reason": result.get("reason", ""), "raw": raw}
    except Exception:
        return {"label": "D", "reason": "parse_error", "raw": raw}


# ---------------------------------------------------------------------------
# AHS scoring (Prompt 2)
# ---------------------------------------------------------------------------

def score_ahs(unit_text: str, problem_context: str,
              sys_prompt: str, user_tmpl: str, api_key: str) -> dict:
    user = (user_tmpl
            .replace("{problem_context}", problem_context[:400])
            .replace("{unit_text}", unit_text[:500]))
    raw = call_gemini(sys_prompt, user, api_key)
    try:
        result = parse_json_response(raw)
        score = int(result.get("abstractness_score", 0))
        score = max(0, min(3, score))
        return {"abstractness_score": score, "reason": result.get("reason", ""), "raw": raw}
    except Exception:
        return {"abstractness_score": 0, "reason": "parse_error", "raw": raw}


# ---------------------------------------------------------------------------
# Per-task annotation
# ---------------------------------------------------------------------------

def annotate_task(task: dict, sys_v1: str, tmpl_v1: str,
                   sys_v2: str, tmpl_v2: str,
                   sys_ahs: str, tmpl_ahs: str,
                   api_key: str) -> dict:
    units_text = split_into_units(task["primary_think"])
    if not units_text:
        return {
            "task_id": task["task_id"],
            "problem_type": task["problem_type"],
            "units": [],
            "kappa": None,
            "error": "no units after splitting",
        }

    problem_text = task.get("problem_text", "")
    examples_summary = task.get("examples_summary", "")

    # Round 1 (v1) and Round 2 (v2) labels
    labels_v1 = []
    labels_v2 = []
    unit_results = []

    for u in units_text:
        r1 = annotate_unit_label(u, problem_text, examples_summary,
                                  sys_v1, tmpl_v1, api_key)
        r2 = annotate_unit_label(u, problem_text, examples_summary,
                                  sys_v2, tmpl_v2, api_key)
        labels_v1.append(r1["label"])
        labels_v2.append(r2["label"])
        unit_results.append({"text": u, "v1": r1, "v2": r2})

    # Cohen's kappa
    kappa = None
    if len(labels_v1) >= 2:
        try:
            kappa = float(cohen_kappa_score(labels_v1, labels_v2))
        except Exception:
            kappa = None

    # Tiebreak: if kappa < 0.7, call v1 again for disagreements
    final_labels = []
    for i, (l1, l2, ur) in enumerate(zip(labels_v1, labels_v2, unit_results)):
        if l1 == l2:
            final_labels.append(l1)
            ur["final_label"] = l1
            ur["tiebreak"] = False
        elif kappa is None or kappa < 0.7:
            # third call with v1 prompt
            r3 = annotate_unit_label(ur["text"], problem_text, examples_summary,
                                      sys_v1, tmpl_v1, api_key)
            final_labels.append(r3["label"])
            ur["v3_tiebreak"] = r3
            ur["final_label"] = r3["label"]
            ur["tiebreak"] = True
        else:
            # kappa >= 0.7 but labels differ: use v1
            final_labels.append(l1)
            ur["final_label"] = l1
            ur["tiebreak"] = False

    # AHS scoring for P units
    for ur, fl in zip(unit_results, final_labels):
        if fl == "P":
            ahs = score_ahs(ur["text"], problem_text, sys_ahs, tmpl_ahs, api_key)
            ur["ahs"] = ahs
        else:
            ur["ahs"] = None

    return {
        "task_id": task["task_id"],
        "problem_type": task["problem_type"],
        "n_units": len(units_text),
        "kappa": kappa,
        "final_labels": final_labels,
        "units": unit_results,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_ckpt(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {r["task_id"]: r for r in d if r.get("task_id")}


def save_ckpt(results: list, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Annotate <think> units with E/D/P/R + AHS.")
    parser.add_argument("--split", choices=["pre", "post"], required=True)
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", ""),
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--model", default=GEMINI_MODEL)
    parser.add_argument("--input-dir", default=os.path.join(BASE, "data"))
    parser.add_argument("--output-dir", default=os.path.join(BASE, "annotations"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Provide --api-key or set GEMINI_API_KEY environment variable.")

    input_file = os.path.join(args.input_dir, f"{args.split}_ood_tasks.json")
    output_file = os.path.join(args.output_dir, f"{args.split}_annotations.json")

    with open(input_file, encoding="utf-8") as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks from {input_file}")

    # Load prompts
    sys_v1, tmpl_v1 = load_prompt("annotate_unit_v1.txt")
    sys_v2, tmpl_v2 = load_prompt("annotate_unit_v2.txt")
    sys_ahs, tmpl_ahs = load_prompt("abstract_score.txt")

    # Checkpoint
    existing = load_ckpt(output_file)
    if existing:
        print(f"Resuming: {len(existing)} tasks already done")
    todo = [t for t in tasks if t["task_id"] not in existing]

    all_results = list(existing.values())
    failed = []

    def _annotate(task):
        return annotate_task(task, sys_v1, tmpl_v1, sys_v2, tmpl_v2,
                              sys_ahs, tmpl_ahs, args.api_key)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_annotate, t): t for t in todo}
        with tqdm(total=len(todo), desc=f"Annotating [{args.split}]", unit="task") as bar:
            for fut in as_completed(futures):
                res = fut.result()
                all_results.append(res)
                if res.get("error"):
                    failed.append(res["task_id"])
                    bar.set_postfix(fail=len(failed))
                bar.update(1)
                if len(all_results) % 10 == 0:
                    save_ckpt(all_results, output_file)

    save_ckpt(all_results, output_file)

    # Summary
    valid = [r for r in all_results if not r.get("error")]
    all_kappas = [r["kappa"] for r in valid if r.get("kappa") is not None]
    mean_kappa = sum(all_kappas) / len(all_kappas) if all_kappas else 0.0

    label_counts = {"E": 0, "D": 0, "P": 0, "R": 0}
    n_P_units = 0
    for r in valid:
        for lbl in r.get("final_labels", []):
            if lbl in label_counts:
                label_counts[lbl] += 1
        n_P_units += sum(1 for lbl in r.get("final_labels", []) if lbl == "P")

    total_units = sum(label_counts.values())
    print(f"\n=== {args.split.upper()} Annotation Summary ===")
    print(f"  Tasks annotated: {len(valid)}")
    print(f"  Mean Cohen's kappa: {mean_kappa:.3f}")
    print(f"  Total units: {total_units}")
    for lbl, cnt in label_counts.items():
        pct = 100 * cnt / total_units if total_units else 0
        print(f"    {lbl}: {cnt} ({pct:.1f}%)")
    print(f"  P units (AHS scored): {n_P_units}")
    print(f"  Saved: {output_file}")
    if failed:
        print(f"  Failed tasks: {len(failed)}: {failed[:5]}")


if __name__ == "__main__":
    main()
