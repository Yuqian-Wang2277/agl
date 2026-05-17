#!/usr/bin/env python3
"""
2_gen_rules.py — 对每个任务分别用 Pre-GIST 和 Post-GIST 模型生成 M=5 条归纳规则。

每个条件使用各自的本地 vLLM API 和 system_prompt（直接取自阶段一输出的 JSON 字段）：
  Pre-GIST：Qwen3-4B vLLM + system_prompt（来自 habit 评测数据）
  Post-GIST：Qwen3-semi vLLM + system_prompt（来自 MIST 评测数据，更严格）

LOO 控制变量原则：同一条件内唯一变化为支持集构成，模型与提示词完全固定。

用法示例：
  # 生成 Pre-GIST 规则
  python 2_gen_rules.py --split pre \
      --input data/pre_gist_tasks.json \
      --api-base http://localhost:8000/v1 \
      --model Qwen3-4B \
      --output generated_rules/pre_gist_rules.json

  # 生成 Post-GIST 规则
  python 2_gen_rules.py --split post \
      --input data/post_gist_tasks.json \
      --api-base http://localhost:8001/v1 \
      --model Qwen3-semi \
      --output generated_rules/post_gist_rules.json
"""

import json
import os
import re
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))
M_RULES = 5
TEMPERATURE = 0.7
MAX_TOKENS = 4096


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def extract_fewshot_learning(strategy_text: str) -> str:
    """从策略全文中提取 FEWSHOT_LEARNING 块。"""
    pattern = re.compile(
        r"FEWSHOT_LEARNING:\s*(.*?)(?=\n[A-Z_]{3,}:|$)",
        re.DOTALL,
    )
    m = pattern.search(strategy_text)
    if m:
        return m.group(1).strip()
    return strategy_text.strip()


def build_user_prompt(examples: list, problem: str) -> str:
    """将 K=3 示例和当前问题组装成 user prompt（与原始评测格式一致）。"""
    lines = ["Here are example problems and their solutions:\n"]
    for idx, ex in enumerate(examples, 1):
        lines.append(f"Example {idx}:")
        lines.append(f"Problem: {ex.get('input', '').strip()}")
        targets = ex.get("target", [])
        sol = targets[0] if targets else ""
        lines.append(f"Solution: {sol}\n")
    lines.append("\nExtract the two-layer problem-solving strategy following the exact format above.")
    lines.append(f"\nCurrent problem:\n{problem.strip()}")
    return "\n".join(lines)


def call_vllm(client: OpenAI, model: str, system_prompt: str,
              user_prompt: str, max_retries: int = 3) -> str:
    """调用 vLLM OpenAI 兼容接口，带重试。"""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            if attempt < max_retries:
                print(f"  [retry {attempt}] {e} — 等待 {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"vLLM 调用失败（{max_retries} 次）: {last_err}")


def load_checkpoint(output_path: str) -> dict:
    """加载已完成结果，返回 {rollout_id: record}。"""
    if not os.path.exists(output_path):
        return {}
    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)
    return {r["rollout_id"]: r for r in data if r.get("rollout_id")}


def save_results(results: list, output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def generate_for_task(task: dict, client: OpenAI, model: str, m: int) -> dict:
    """为单个任务生成 M 条规则。"""
    rules = []
    system_prompt = task["system_prompt"]
    examples = task["examples"]
    problem = task["problem"]
    user_prompt = build_user_prompt(examples, problem)

    for rule_id in range(m):
        try:
            raw = call_vllm(client, model, system_prompt, user_prompt)
            inductive_rule = extract_fewshot_learning(raw)
            rules.append({
                "rule_id": rule_id,
                "inductive_rule": inductive_rule,
                "strategy_full": raw,
                "error": None,
            })
        except Exception as e:
            rules.append({
                "rule_id": rule_id,
                "inductive_rule": "",
                "strategy_full": "",
                "error": str(e),
            })

    return {
        "rollout_id": task["rollout_id"],
        "problem_type": task["problem_type"],
        "reward_correctness": task.get("reward_correctness", 0.0),
        "split": task.get("split", ""),
        "rules": rules,
    }


def main():
    parser = argparse.ArgumentParser(description="用本地 vLLM API 为每个任务生成 M=5 条归纳规则。")
    parser.add_argument("--split", choices=["pre", "post"], required=True,
                        help="pre = Pre-GIST（Qwen3-4B），post = Post-GIST（Qwen3-semi）")
    parser.add_argument("--input", required=True,
                        help="阶段一输出的 JSON 文件（pre_gist_tasks.json 或 post_gist_tasks.json）")
    parser.add_argument("--api-base", required=True,
                        help="vLLM 服务地址，如 http://localhost:8000/v1")
    parser.add_argument("--model", required=True,
                        help="vLLM 上加载的模型名称，如 Qwen3-4B 或 Qwen3-semi")
    parser.add_argument("--output", required=True,
                        help="输出路径，如 generated_rules/pre_gist_rules.json")
    parser.add_argument("--m", type=int, default=M_RULES,
                        help=f"每任务生成规则数（默认 {M_RULES}）")
    parser.add_argument("--workers", type=int, default=4,
                        help="并发线程数（默认 4，根据 vLLM 并发能力调整）")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        tasks = json.load(f)
    print(f"加载任务数：{len(tasks)}（来自 {args.input}）")

    client = OpenAI(api_key="dummy", base_url=args.api_base)

    existing = load_checkpoint(args.output)
    if existing:
        print(f"断点续跑：已完成 {len(existing)} 条，剩余 {len(tasks) - len(existing)} 条")
    to_run = [t for t in tasks if t["rollout_id"] not in existing]

    all_results = list(existing.values())
    failed = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_for_task, task, client, args.model, args.m): task
            for task in to_run
        }
        with tqdm(total=len(to_run), desc=f"生成规则 [{args.split}]", unit="task") as pbar:
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                err_count = sum(1 for r in result["rules"] if r.get("error"))
                if err_count:
                    failed.append(result["rollout_id"])
                    pbar.set_postfix(failed=len(failed))
                pbar.update(1)
                if len(all_results) % 10 == 0:
                    save_results(all_results, args.output)

    save_results(all_results, args.output)
    print(f"\n完成。{len(all_results)} 条结果保存至 {args.output}")
    if failed:
        print(f"  警告：{len(failed)} 个任务有部分规则生成失败：{failed[:5]}")

    total_rules = sum(len(r["rules"]) for r in all_results)
    valid_rules = sum(
        1 for r in all_results for rule in r["rules"]
        if rule.get("inductive_rule") and not rule.get("error")
    )
    print(f"  规则总数：{total_rules}，有效（非空且无错）：{valid_rules}")


if __name__ == "__main__":
    main()
