# Copyright (c) Microsoft. All rights reserved.

"""Strategy generation agent — trains ONLY strategy generation using answer quality as reward.

This agent inherits from StrategyExtractionAgent and overrides rollout_async to:
1. Generate a strategy via the traced LLM call (this is what VERL trains)
2. Apply the strategy to solve a problem via an UN-traced HTTP call (not trained)
3. Use answer correctness as the reward signal to improve strategy generation

The answer-generation LLM call is made with raw httpx (bypassing OpenAI SDK tracing)
so that only strategy-generation tokens receive gradient updates.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, cast

import httpx
from openai import AsyncOpenAI

import agentlightning as agl

from .prompt import format_examples, load_prompt, list_versions
from .reward import RewardConfig, compute_format_reward, extract_strategy, get_reward_config

logger = logging.getLogger(__name__)


class StrategyGenerationTask(TypedDict):
    """Task structure for strategy generation training.

    Attributes:
        problem_type: The type/category of the problem.
        examples: Few-shot examples used to extract strategy (each with 'input' and 'target').
        num_shots: Number of examples included.
        problem: The problem to solve (for reward evaluation).
        ground_truth: The correct answer (for reward evaluation).
        source_problem_type: Source problem type (for cross-domain mode, None for same-domain).
    """

    problem_type: str
    examples: List[Dict[str, Any]]
    num_shots: int
    problem: str
    ground_truth: str
    source_problem_type: Optional[str]


# NOTE: Prompts are stored as TOML files under ``prompt/strategy_generation/``
# and ``prompt/answer_generation/`` (see prompt/__init__.py for details).
# Reward logic is in the ``reward/`` package.
# Select versions at init via ``strategy_prompt_version`` / ``answer_prompt_version``
# / ``reward_version``.


class StrategyGenerationAgent(agl.LitAgent["StrategyGenerationTask"]):
    """Agent that trains strategy generation using answer quality as reward.

    Rollout flow:
        1. **Traced LLM call** — generate strategy from few-shot examples (trained by VERL).
        2. **Un-traced HTTP call** — apply strategy to problem via raw httpx (NOT trained).
        3. Compute reward from answer correctness and strategy format.
    """

    def __init__(
        self,
        save_full_output: bool = True,
        rollout_traces_dir: Optional[str] = None,
        validation_output_dir: Optional[str] = None,
        experiment_id: Optional[str] = None,
        test_freq: int = 50,
        # Reward weights
        format_weight: float = 0.2,
        correctness_weight: float = 0.8,
        numeric_tolerance: float = 0.02,
        f1_threshold: float = 0.5,
        # Prompt / reward versions (see prompt/ and reward/ packages)
        strategy_prompt_version: str = "v1",
        answer_prompt_version: str = "v1",
        reward_version: str = "v1",
    ) -> None:
        super().__init__()
        self.save_full_output = save_full_output
        self.experiment_id = experiment_id
        self.rollout_traces_dir = rollout_traces_dir
        self.validation_output_dir = validation_output_dir
        self.test_freq = test_freq

        # Reward weights
        self.format_weight = format_weight
        self.correctness_weight = correctness_weight
        self.numeric_tolerance = numeric_tolerance
        self.f1_threshold = f1_threshold

        # Load TOML prompts and reward config
        self.strategy_prompt = load_prompt("strategy_generation", strategy_prompt_version)
        self.answer_prompt = load_prompt("answer_generation", answer_prompt_version)
        self.reward_config: RewardConfig = get_reward_config(reward_version)

        self._strategy_prompt_version = strategy_prompt_version
        self._answer_prompt_version = answer_prompt_version

        if rollout_traces_dir:
            if experiment_id:
                rollout_traces_dir = os.path.join(rollout_traces_dir, experiment_id)
            os.makedirs(rollout_traces_dir, exist_ok=True)
            self.traces_file = os.path.join(rollout_traces_dir, "rollout_traces.jsonl")
            logger.info(f"Rollout traces will be saved to: {self.traces_file}")
        else:
            self.traces_file = None

        # Validation tracking
        self.validation_outputs: List[Dict[str, Any]] = []
        self.last_rollout_mode: Optional[str] = None
        self.validation_step_counter: int = 0
        self._worker_id: Optional[int] = None
        self._pending_merge_step: Optional[int] = None

        if validation_output_dir:
            if experiment_id:
                validation_output_dir = os.path.join(validation_output_dir, experiment_id)
            os.makedirs(validation_output_dir, exist_ok=True)
            self.validation_output_dir = validation_output_dir
            logger.info(f"Validation outputs will be saved to: {validation_output_dir}")

        logger.info(
            f"StrategyGenerationAgent initialized "
            f"(format_w={format_weight}, correctness_w={correctness_weight}, "
            f"strategy_prompt={strategy_prompt_version}, "
            f"answer_prompt={answer_prompt_version}, "
            f"reward={self.reward_config.name}, pid={os.getpid()})"
        )

    # ------------------------------------------------------------------ #
    #  Worker / validation helpers (same pattern as StrategyApplicationAgent)
    # ------------------------------------------------------------------ #

    @property
    def worker_id(self) -> int:
        if self._worker_id is None:
            self._worker_id = os.getpid()
            logger.info(f"Resolved worker_id = {self._worker_id}")
        return self._worker_id

    def save_validation_outputs(self, step: int) -> Optional[str]:
        """Save collected validation outputs to a per-worker JSON file."""
        if not self.validation_output_dir or not self.validation_outputs:
            return None

        # Deferred merge of previous step
        if self._pending_merge_step is not None:
            self._merge_worker_validation_files(self._pending_merge_step)

        temp_file = os.path.join(
            self.validation_output_dir,
            f"validation_step{step}_worker{self.worker_id}.json",
        )
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.validation_outputs, f, ensure_ascii=False, indent=2)
            saved_count = len(self.validation_outputs)
            logger.info(f"[Worker {self.worker_id}] Saved {saved_count} validation outputs to: {temp_file}")
            self.validation_outputs = []
        except Exception as e:
            logger.error(f"[Worker {self.worker_id}] Failed to save validation outputs: {e}")
            return None

        self._pending_merge_step = step
        return temp_file

    def _merge_worker_validation_files(self, step: int) -> Optional[str]:
        """Merge all worker validation files for a given step."""
        import glob as _glob

        if not self.validation_output_dir:
            return None
        pattern = os.path.join(self.validation_output_dir, f"validation_step{step}_worker*.json")
        worker_files = _glob.glob(pattern)
        if not worker_files:
            return None

        all_outputs: List[Dict[str, Any]] = []
        for wf in sorted(worker_files):
            try:
                with open(wf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_outputs.extend(data)
            except Exception as e:
                logger.warning(f"Failed to load worker file {wf}: {e}")

        if not all_outputs:
            return None

        merged_name = f"validation_global_step{step}.json"
        merged_path = os.path.join(self.validation_output_dir, merged_name)
        try:
            with open(merged_path, "w", encoding="utf-8") as f:
                json.dump(all_outputs, f, ensure_ascii=False, indent=2)
            logger.info(
                f"[Worker {self.worker_id}] Merged {len(all_outputs)} outputs "
                f"from {len(worker_files)} workers -> {merged_path}"
            )
            return merged_path
        except Exception as e:
            logger.error(f"Failed to write merged validation file: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Un-traced answer generation (raw httpx, bypasses OpenTelemetry)
    # ------------------------------------------------------------------ #

    async def _generate_answer_untraced(
        self,
        base_url: str,
        api_key: str,
        model: str,
        strategy: str,
        problem: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate an answer using raw httpx — NOT captured by AGL tracing.

        This ensures the answer-generation tokens are excluded from VERL's
        training triplets so that only strategy-generation tokens are optimised.

        Prompts are sourced from ``self.answer_prompt`` (TOML).
        """
        url = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.answer_prompt["system"]},
                {"role": "user", "content": self.answer_prompt["user"].format(strategy=strategy, problem=problem)},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "") or ""

    # ------------------------------------------------------------------ #
    #  Main rollout
    # ------------------------------------------------------------------ #

    async def rollout_async(
        self,
        task: "StrategyGenerationTask",
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float:
        """Execute a rollout: generate strategy (trained) → verify via answer (not trained).

        Args:
            task: Task containing few-shot examples, problem, and ground truth.
            resources: Named resources including the LLM.
            rollout: Rollout metadata.

        Returns:
            Reward in [0, 1].
        """
        try:
            llm = cast(agl.LLM, resources["main_llm"])
            attempted_rollout = cast(agl.AttemptedRollout, rollout)
            base_url = llm.get_base_url(
                attempted_rollout.rollout_id,
                attempted_rollout.attempt.attempt_id,
            )

            logger.info(
                f"[Rollout {attempted_rollout.rollout_id}] START - "
                f"Type: {task['problem_type']}, Mode: {rollout.mode}"
            )

            # ---- Step 1: Generate strategy (TRACED — will be trained) ---- #
            system_prompt = self.strategy_prompt["system"]
            user_prompt = self.strategy_prompt["user"].format(
                examples_text=format_examples(task["examples"]),
            )

            client = AsyncOpenAI(
                base_url=base_url,
                api_key=llm.api_key or "dummy-key",
            )

            response = await client.chat.completions.create(
                model=llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=llm.sampling_parameters.get("temperature", 0.7),
                max_tokens=llm.sampling_parameters.get("max_tokens", 4000),
            )

            strategy_output = response.choices[0].message.content or ""
            strategy = extract_strategy(strategy_output)
            format_reward = compute_format_reward(strategy_output)

            logger.info(
                f"[Rollout {attempted_rollout.rollout_id}] Strategy: "
                f"format={format_reward}, length={len(strategy) if strategy else 0}"
            )

            # ---- Step 2: Apply strategy to problem (UN-TRACED — not trained) ---- #
            correctness = 0.0
            answer_output = ""
            extracted_answer: Optional[str] = None

            if strategy:
                try:
                    answer_output = await self._generate_answer_untraced(
                        base_url=base_url,
                        api_key=llm.api_key or "dummy-key",
                        model=llm.model,
                        strategy=strategy,
                        problem=task["problem"],
                        temperature=llm.sampling_parameters.get("temperature", 0.7),
                        max_tokens=llm.sampling_parameters.get("max_tokens", 4000),
                    )
                    extracted_answer = self.reward_config.extract_answer(answer_output)
                    if extracted_answer:
                        correctness = self.reward_config.compute_answer_correctness(
                            extracted_answer,
                            task["ground_truth"],
                            self.numeric_tolerance,
                            self.f1_threshold,
                        )
                    logger.info(
                        f"[Rollout {attempted_rollout.rollout_id}] Answer: "
                        f"correctness={correctness}, extracted={extracted_answer[:50] if extracted_answer else 'None'}"
                    )
                except Exception as e:
                    logger.warning(
                        f"[Rollout {attempted_rollout.rollout_id}] "
                        f"Answer generation failed (non-fatal): {e}"
                    )
            else:
                logger.warning(
                    f"[Rollout {attempted_rollout.rollout_id}] No strategy extracted, "
                    "skipping answer generation"
                )

            # ---- Step 3: Compute reward ---- #
            final_reward = self.reward_config.compute_final_reward(
                format_reward, correctness,
                self.format_weight, self.correctness_weight,
            )

            reward_details: Dict[str, float] = {
                "format": format_reward,
                "correctness": correctness,
                "final": final_reward,
            }

            # ---- Mode tracking & validation output saving ---- #
            current_mode = rollout.mode if hasattr(rollout, "mode") else "unknown"
            is_validation = current_mode != "train"

            # Detect transition validation → train: save buffered validation outputs
            if (
                self.last_rollout_mode is not None
                and self.last_rollout_mode != "train"
                and current_mode == "train"
            ):
                if self.validation_outputs and self.validation_output_dir:
                    actual_step = (
                        0 if self.validation_step_counter == 0
                        else self.test_freq * self.validation_step_counter
                    )
                    saved_path = self.save_validation_outputs(actual_step)
                    if saved_path:
                        logger.info(f"Validation outputs saved (step={actual_step}): {saved_path}")
                    self.validation_step_counter += 1

            if is_validation and self.validation_output_dir:
                self.validation_outputs.append(
                    {
                        "rollout_id": attempted_rollout.rollout_id,
                        "timestamp": datetime.now().isoformat(),
                        "problem_type": task["problem_type"],
                        "source_problem_type": task.get("source_problem_type"),
                        "input": {
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                            "examples": task["examples"],
                            "problem": task["problem"],
                            "ground_truth": task["ground_truth"],
                        },
                        "output": {
                            "strategy_raw": strategy_output,
                            "strategy_extracted": strategy or "",
                            "answer_raw": answer_output,
                            "answer_extracted": extracted_answer or "",
                        },
                        "reward": reward_details,
                        "metadata": {
                            "model": llm.model,
                            "temperature": llm.sampling_parameters.get("temperature", 0.7),
                            "max_tokens": llm.sampling_parameters.get("max_tokens", 4000),
                            "rollout_mode": str(current_mode),
                        },
                    }
                )

            self.last_rollout_mode = current_mode

            # ---- Save trace ---- #
            if self.traces_file:
                trace = {
                    "rollout_id": attempted_rollout.rollout_id,
                    "timestamp": datetime.now().isoformat(),
                    "mode": str(rollout.mode),
                    "problem_type": task["problem_type"],
                    "source_problem_type": task.get("source_problem_type"),
                    "input": {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "problem": task["problem"],
                        "ground_truth": task["ground_truth"],
                    },
                    "output": {
                        "strategy_raw": strategy_output,
                        "strategy_extracted": strategy or "",
                        "answer_raw": answer_output,
                        "answer_extracted": extracted_answer or "",
                    },
                    "reward": reward_details,
                }
                try:
                    with open(self.traces_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(trace, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.warning(f"Failed to save rollout trace: {e}")

            # ---- Logging ---- #
            if self.save_full_output:
                logger.info(
                    f"[Rollout {attempted_rollout.rollout_id}] FULL OUTPUT\n"
                    f"Problem Type: {task['problem_type']}\n"
                    f"Mode: {rollout.mode}\n"
                    f"Strategy Length: {len(strategy_output)} chars\n"
                    f"Answer Length: {len(answer_output)} chars\n"
                    f"Extracted Answer: {extracted_answer[:100] if extracted_answer else '(None)'}\n"
                    f"Ground Truth: {task['ground_truth'][:100]}\n"
                    f"Reward: {reward_details}\n"
                    f"{'=' * 80}"
                )

            if final_reward == 0.0:
                logger.warning(
                    f"[Rollout {attempted_rollout.rollout_id}] ZERO REWARD - "
                    f"format={format_reward}, correctness={correctness}"
                )
            else:
                logger.info(
                    f"[Rollout {attempted_rollout.rollout_id}] reward={final_reward:.3f}"
                )

            agl.emit_reward(final_reward)
            return float(final_reward)

        except Exception as e:
            logger.error(
                f"[Rollout {attempted_rollout.rollout_id}] ERROR: {e}",
                exc_info=True,
            )
            try:
                agl.emit_reward(0.0)
            except Exception as reward_err:
                logger.warning(f"Failed to emit reward: {reward_err}")
            return 0.0

