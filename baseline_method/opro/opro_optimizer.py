"""OPRO core optimizer — pure test-time, no training required.

Implements the OPRO (Optimization by PROmpting) loop:
  for each task_type, iteratively generate instructions via an LLM optimizer,
  score them on eval examples, and keep the best one.
"""

import asyncio
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


class OPROOptimizer:
    """Iterative instruction optimizer using an LLM as the meta-optimizer.

    Args:
        optimizer_client: AsyncOpenAI client pointing to Gemini endpoint.
        answer_client: AsyncOpenAI client pointing to vLLM (Exp-1) or Gemini (Exp-2).
        optimizer_model: Model name for the optimizer (e.g. gemini-3-flash-preview).
        answer_model: Model name for generating answers during optimization.
        scorer_fn: Callable[[model_output, ground_truth], (hard_correct, soft_score)].
        meta_prompt: Dict loaded from meta_optimizer.toml with 'system'/'user' keys.
        num_steps: Maximum optimization steps per task_type.
        eval_batch_size: Number of examples sampled per optimization step.
        history_top_k: Max history entries shown to the optimizer.
        optimizer_temperature: Sampling temperature for the optimizer LLM.
        answer_temperature: Temperature for answer generation during optimization.
        checkpoint_dir: Directory to save/resume checkpoints. Auto-created if needed.
        concurrency: Max concurrent task_type optimizations (limits Gemini rate).
    """

    def __init__(
        self,
        optimizer_client: AsyncOpenAI,
        answer_client: AsyncOpenAI,
        optimizer_model: str,
        answer_model: str,
        scorer_fn: Callable[[str, str], tuple[bool, float]],
        meta_prompt: dict,
        num_steps: int = 10,
        eval_batch_size: int = 5,
        history_top_k: int = 8,
        optimizer_temperature: float = 1.0,
        answer_temperature: float = 0.0,
        checkpoint_dir: Optional[Path] = None,
        concurrency: int = 4,
    ) -> None:
        self.optimizer_client = optimizer_client
        self.answer_client = answer_client
        self.optimizer_model = optimizer_model
        self.answer_model = answer_model
        self.scorer_fn = scorer_fn
        self.meta_prompt = meta_prompt
        self.num_steps = num_steps
        self.eval_batch_size = eval_batch_size
        self.history_top_k = history_top_k
        self.optimizer_temperature = optimizer_temperature
        self.answer_temperature = answer_temperature
        self.checkpoint_dir = checkpoint_dir
        self.concurrency = concurrency

        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    async def optimize_all_tasks(
        self,
        benchmark: str,
        tasks: dict[str, list[dict]],
    ) -> dict[str, str]:
        """Concurrently optimize all task_types; return {task_type: best_instruction}."""
        sem = asyncio.Semaphore(self.concurrency)

        async def _run(task_type: str, examples: list[dict]) -> tuple[str, str]:
            async with sem:
                inst = await self.optimize_task(benchmark, task_type, examples)
                return task_type, inst

        results = await asyncio.gather(*[_run(tt, ex) for tt, ex in tasks.items()])
        return dict(results)

    async def optimize_task(
        self,
        benchmark: str,
        task_type: str,
        examples: list[dict],
    ) -> str:
        """Run OPRO loop for a single task_type; return best instruction found."""
        steps, batch_size = self._adapt_params(len(examples))
        ckpt_path = self._ckpt_path(benchmark, task_type)

        # Resume or initialise
        candidates: list[tuple[str, float]] = [("", 0.0)]
        completed = 0
        steps_log: list[dict] = []
        best_instruction = ""
        best_score = 0.0

        if ckpt_path and ckpt_path.exists():
            data = json.loads(ckpt_path.read_text(encoding="utf-8"))
            completed = data.get("completed_steps", 0)
            candidates = [tuple(c) for c in data.get("candidates", [("", 0.0)])]  # type: ignore[misc]
            steps_log = data.get("steps", [])
            best_instruction = data.get("best_instruction", "")
            best_score = data.get("best_score", 0.0)
            if completed >= steps:
                logger.info("[%s/%s] Already complete (%d steps), skipping.", benchmark, task_type, completed)
                return best_instruction

        logger.info("[%s/%s] Optimizing: %d examples, %d steps, batch=%d (resume from step %d)",
                    benchmark, task_type, len(examples), steps, batch_size, completed)

        rng = random.Random(42 + hash(task_type) % 10000)

        for step in range(completed + 1, steps + 1):
            # 1. Sample batch
            batch = rng.sample(examples, min(batch_size, len(examples)))
            batch_ids = [examples.index(b) if b in examples else -1 for b in batch]

            # 2. Build meta-prompt with history (ascending order, best last)
            sorted_cands = sorted(candidates, key=lambda x: x[1])
            history_slice = sorted_cands[-self.history_top_k:]
            history_str = self._format_history(history_slice)
            examples_str = self._format_examples(batch)

            user_content = self.meta_prompt["user"]["content"].format(
                task_type=task_type,
                history=history_str,
                examples=examples_str,
            )

            # 3. Call optimizer LLM → new instruction
            resp = await self.optimizer_client.chat.completions.create(
                model=self.optimizer_model,
                messages=[
                    {"role": "system", "content": self.meta_prompt["system"]["content"]},
                    {"role": "user", "content": user_content},
                ],
                temperature=self.optimizer_temperature,
                max_tokens=512,
            )
            raw_output = resp.choices[0].message.content or ""
            new_instruction = self._extract_instruction(raw_output)

            # 4. Score new instruction on batch
            per_scores: list[float] = []
            for item in batch:
                answer_text = await self._call_answer_model(new_instruction, item)
                hard, soft = self.scorer_fn(answer_text, item["ground_truth"])
                per_scores.append(1.0 if hard else soft)
            mean_score = sum(per_scores) / len(per_scores) if per_scores else 0.0

            # 5. Update candidates and best
            candidates.append((new_instruction, mean_score))
            if mean_score > best_score:
                best_score = mean_score
                best_instruction = new_instruction

            steps_log.append({
                "step": step,
                "batch_ids": batch_ids,
                "instruction": new_instruction,
                "per_problem_scores": per_scores,
                "mean_score": mean_score,
            })

            logger.info("[%s/%s] step %d/%d  score=%.3f  best=%.3f",
                        benchmark, task_type, step, steps, mean_score, best_score)

            # 6. Save checkpoint
            self._save_checkpoint(ckpt_path, benchmark, task_type, step, candidates,
                                  best_instruction, best_score, steps_log)

        return best_instruction

    # ── Answer model call ─────────────────────────────────────────────────────

    async def _call_answer_model(self, instruction: str, item: dict) -> str:
        """Call answer model with the given instruction prepended to the problem."""
        problem = item.get("problem", item.get("prompt", ""))
        user_content = f"{instruction}\n\nProblem:\n{problem}" if instruction else f"Problem:\n{problem}"
        resp = await self.answer_client.chat.completions.create(
            model=self.answer_model,
            messages=[{"role": "user", "content": user_content}],
            temperature=self.answer_temperature,
            max_tokens=1024,
            seed=42,
        )
        return resp.choices[0].message.content or ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _adapt_params(self, n: int) -> tuple[int, int]:
        """Reduce steps and batch for small task_types."""
        steps = min(self.num_steps, max(3, n))
        batch = min(self.eval_batch_size, max(1, n // 2))
        return steps, batch

    def _extract_instruction(self, text: str) -> str:
        m = re.search(r"<INST>(.*?)</INST>", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    def _format_history(self, candidates: list[tuple[str, float]]) -> str:
        if not candidates:
            return "(no history yet)"
        lines = []
        for inst, sc in candidates:
            label = inst if inst else "(empty — zero-shot baseline)"
            lines.append(f"Score {sc:.3f}: {label}")
        return "\n".join(lines)

    def _format_examples(self, batch: list[dict]) -> str:
        lines = []
        for i, item in enumerate(batch, 1):
            problem = item.get("problem", item.get("prompt", ""))
            lines.append(f"[Example {i}]\n{problem}")
        return "\n\n".join(lines)

    def _ckpt_path(self, benchmark: str, task_type: str) -> Optional[Path]:
        if self.checkpoint_dir is None:
            return None
        safe_tt = re.sub(r"[^\w\-]", "_", task_type)
        return self.checkpoint_dir / f"{benchmark}_{safe_tt}.json"

    def _save_checkpoint(
        self,
        path: Optional[Path],
        benchmark: str,
        task_type: str,
        completed_steps: int,
        candidates: list[tuple[str, float]],
        best_instruction: str,
        best_score: float,
        steps_log: list[dict],
    ) -> None:
        if path is None:
            return
        data: dict[str, Any] = {
            "benchmark": benchmark,
            "task_type": task_type,
            "completed_steps": completed_steps,
            "best_instruction": best_instruction,
            "best_score": best_score,
            "candidates": [[inst, sc] for inst, sc in candidates],
            "steps": steps_log,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
