# Copyright (c) Microsoft. All rights reserved.

"""Strategy generation agent — trains ONLY strategy generation tokens.

Reward modes:
    **v2 (scorer, default)** — a separately trained strategy-scorer LLM
    directly evaluates strategy quality.  Answer generation (via a *fixed*
    base model) is optional and only used when ``correctness_weight > 0``.

    **v1 (legacy)** — answer correctness from the training model is the
    primary reward signal.

All non-strategy LLM calls use raw httpx (bypassing OpenAI SDK tracing)
so that only strategy-generation tokens receive gradient updates.

Traced strategy ``/chat/completions`` requests merge ``extra_body`` fields:
``chat_template_kwargs`` (when ``strategy_no_think``) and ``repetition_penalty``
(default 1.1; omit by setting ``strategy_repetition_penalty`` to ``None`` or ``<= 0``).
Configure vLLM so server-side defaults also use ``repetition_penalty=1.1`` when possible
(flag names vary by vLLM version).
"""

import asyncio
import hashlib
import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

import httpx
from openai import AsyncOpenAI

import agentlightning as agl

from .prompt import format_examples, load_prompt, list_versions
from .reward import RewardConfig, compute_format_reward, extract_strategy, get_reward_config
from .reward.hybrid_grounded_reward import (
    compute_hybrid_reward,
    evaluate_strategy_k_samples,
    normalize_four_dim_weights,
    parse_oc_scorer_response,
    select_effective_proxy,
    select_representative_answer,
)
from .reward.v3 import compute_answer_judgement
from .reward.strategy_ir_parser import compute_format_reward_structured

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
    ground_truths: List[str]
    task_meta: Dict[str, Any]


# NOTE: Prompts are stored as TOML files under ``prompt/strategy_generation/``
# and ``prompt/answer_generation/`` (see prompt/__init__.py for details).
# Reward logic is in the ``reward/`` package.
# Select versions at init via ``strategy_prompt_version`` / ``answer_prompt_version``
# / ``reward_version``.


class StrategyGenerationAgent(agl.LitAgent["StrategyGenerationTask"]):
    """Agent that trains strategy generation using strategy quality as reward.

    Rollout flow (v2 / scorer mode):
        1. **Traced LLM call** — generate strategy from few-shot examples (trained by VERL).
        2. **Un-traced scorer call** — evaluate strategy quality with a trained scorer LLM.
        3. *(optional)* **Un-traced answer call** — apply strategy via a *fixed* model.
        4. Compute reward: ``fw * format + sw * scorer + cw * correctness``.

    Fallback (v1 / legacy mode):
        Same as before — answer correctness from the training model.
    """

    def __init__(
        self,
        save_full_output: bool = True,
        rollout_traces_dir: Optional[str] = None,
        validation_output_dir: Optional[str] = None,
        experiment_id: Optional[str] = None,
        test_freq: int = 50,
        # Reward weights
        format_weight: float = 0.1,
        correctness_weight: float = 0.0,
        scorer_weight: float = 0.9,
        proxy_weight: float = 0.5,
        reward_mode: str = "hybrid_grounded",
        grounded_proxy_k: int = 4,
        eval_problems_per_subtask: int = 0,
        numeric_tolerance: float = 0.02,
        f1_threshold: float = 0.5,
        # Strategy scorer (trained LLM that evaluates strategy quality)
        strategy_scorer_base_url: str = "",
        strategy_scorer_model: str = "",
        strategy_scoring_prompt_version: str = "v1",
        oc_four_dim_weights: Optional[Tuple[float, float, float, float]] = None,
        strategy_scorer_timeout_sec: float = 120.0,
        # Fixed answer-generation model (frozen weights, not trained)
        answer_model_base_url: str = "",
        answer_model_name: str = "",
        use_strategy_for_answer: bool = True,
        skip_strategy_generation: bool = False,
        answer_temperature: Optional[float] = None,
        val_answer_temperature: Optional[float] = None,
        use_hard_correctness_metric: bool = False,
        strategy_no_think: bool = False,
        strategy_repetition_penalty: Optional[float] = 1.1,
        answer_no_think: bool = False,
        answer_max_tokens: Optional[int] = None,
        answer_request_retries: int = 3,
        answer_retry_delay_sec: float = 1.0,
        answer_fallback_rollout_on_failure: bool = True,
        # Prompt / reward versions (see prompt/ and reward/ packages)
        strategy_prompt_version: str = "v1",
        answer_prompt_version: str = "v1",
        # mist-inline: single-model train-free MIST.  When set, the strategy
        # prompt is loaded from answer_generation/<version>.toml and the
        # answer model (same endpoint) is used for both calls.
        inline_strategy_prompt_version: str = "",
        reward_version: str = "v2",
        # Incremental reward: R_delta = max(0, answer_soft - baseline_soft)
        baseline_cache: Optional[Dict[str, float]] = None,
        incremental_weight: float = 0.0,
        # MIST reward: R_EIR × (1 + β·R_ICR)
        beta: float = 0.0,
        eir_k: float = 10.0,
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
        self.scorer_weight = scorer_weight
        self.proxy_weight = proxy_weight
        self.reward_mode = reward_mode
        if self.reward_mode not in {"scorer_only", "hybrid_grounded"}:
            raise ValueError(f"Unsupported reward_mode: {self.reward_mode}")
        self.grounded_proxy_k = max(1, grounded_proxy_k)
        self.eval_problems_per_subtask = max(0, eval_problems_per_subtask)
        self.numeric_tolerance = numeric_tolerance
        self.f1_threshold = f1_threshold

        # Strategy scorer
        self.strategy_scorer_base_url = strategy_scorer_base_url
        self.strategy_scorer_model = strategy_scorer_model
        self.oc_four_dim_weights: Optional[Tuple[float, float, float, float]] = oc_four_dim_weights
        self.strategy_scorer_timeout_sec = max(1.0, float(strategy_scorer_timeout_sec))

        # Fixed answer model
        self.answer_model_base_url = answer_model_base_url
        self.answer_model_name = answer_model_name
        self.use_strategy_for_answer = use_strategy_for_answer
        self.skip_strategy_generation = skip_strategy_generation
        self.answer_temperature = answer_temperature
        self.val_answer_temperature = val_answer_temperature
        self.use_hard_correctness_metric = use_hard_correctness_metric
        self.strategy_no_think = strategy_no_think
        self.strategy_repetition_penalty = strategy_repetition_penalty
        self.answer_no_think = answer_no_think
        self.answer_max_tokens = answer_max_tokens
        self.answer_request_retries = max(1, int(answer_request_retries))
        self.answer_retry_delay_sec = max(0.0, float(answer_retry_delay_sec))
        self.answer_fallback_rollout_on_failure = answer_fallback_rollout_on_failure

        # Load TOML prompts and reward config
        self._inline_strategy_mode = bool(inline_strategy_prompt_version)
        if self._inline_strategy_mode:
            # Both prompts live in answer_generation/; same model handles both calls.
            self.strategy_prompt = load_prompt("answer_generation", inline_strategy_prompt_version)
        else:
            self.strategy_prompt = load_prompt("strategy_generation", strategy_prompt_version)
        self.answer_prompt = load_prompt("answer_generation", answer_prompt_version)
        self.reward_config: RewardConfig = get_reward_config(reward_version)

        self._strategy_prompt_version = (
            f"answer_generation/{inline_strategy_prompt_version}"
            if self._inline_strategy_mode else strategy_prompt_version
        )
        self._answer_prompt_version = answer_prompt_version
        self._use_structured_format_reward = strategy_prompt_version == "strategy_structured_schema"
        self.debug_baseline = os.environ.get("AGL_DEBUG_BASELINE", "0") == "1"

        # Load scorer prompt (only when a scorer is configured)
        self.scoring_prompt: Optional[Dict[str, str]] = None
        if self.strategy_scorer_base_url:
            self.scoring_prompt = load_prompt("strategy_scoring", strategy_scoring_prompt_version)

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
            f"(format_w={format_weight}, scorer_w={scorer_weight}, "
            f"correctness_w={correctness_weight}, proxy_w={proxy_weight}, "
            f"reward_mode={reward_mode}, grounded_proxy_k={self.grounded_proxy_k}, eval_problems_per_subtask={self.eval_problems_per_subtask}, "
            f"scorer_url={'SET' if strategy_scorer_base_url else 'NONE'}, "
            f"answer_url={'SET' if answer_model_base_url else 'training-model'}, "
            f"use_strategy_for_answer={self.use_strategy_for_answer}, "
            f"skip_strategy_generation={self.skip_strategy_generation}, "
            f"answer_temperature={self.answer_temperature}, "
            f"use_hard_correctness_metric={self.use_hard_correctness_metric}, "
            f"strategy_no_think={self.strategy_no_think}, strategy_repetition_penalty={self.strategy_repetition_penalty}, "
            f"answer_no_think={self.answer_no_think}, "
            f"strategy_prompt={strategy_prompt_version}, "
            f"answer_prompt={answer_prompt_version}, "
            f"reward={self.reward_config.name}, pid={os.getpid()})"
        )
        if self.debug_baseline:
            logger.warning("AGL_DEBUG_BASELINE=1 enabled: rollout diagnostics are active.")

        # Incremental reward: R_delta = max(0, answer_soft - baseline_soft)
        self._baseline_cache: Dict[str, float] = baseline_cache or {}
        self._incremental_weight: float = incremental_weight
        if self._baseline_cache and self._incremental_weight > 0.0:
            logger.info(
                f"Incremental reward enabled: {len(self._baseline_cache)} baseline entries, "
                f"incremental_weight={self._incremental_weight}"
            )

        # MIST reward: R_EIR × (1 + β·R_ICR)
        self._beta: float = float(beta)
        self._eir_k: float = float(eir_k)
        if self._baseline_cache:
            logger.info(
                f"MIST reward enabled: beta={self._beta}, eir_k={self._eir_k}, "
                f"baseline_entries={len(self._baseline_cache)}"
                + (" (pure EIR, no ICR scaling)" if self._beta == 0.0 else "")
            )

    def _problem_key(self, problem: str) -> str:
        """Stable 16-char cache key derived from problem text (MD5 prefix)."""
        return hashlib.md5(problem.encode("utf-8")).hexdigest()[:16]

    def _debug_rollout(self, rollout_id: str, stage: str, **fields: Any) -> None:
        """Emit compact rollout diagnostics when baseline debug is enabled."""
        if not self.debug_baseline:
            return
        ordered = ", ".join(f"{k}={fields[k]!r}" for k in sorted(fields))
        logger.warning("[BaselineDebug][%s][%s] %s", rollout_id, stage, ordered)

    # ------------------------------------------------------------------ #
    #  Worker / validation helpers (same pattern as StrategyApplicationAgent)
    # ------------------------------------------------------------------ #

    @property
    def worker_id(self):
        if self._worker_id is None:
            self._worker_id = os.getpid()
            logger.info(f"Resolved worker_id = {self._worker_id}")
        return self._worker_id

    @worker_id.setter
    def worker_id(self, value) -> None:
        self._worker_id = value

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

    def _append_validation_entry_to_worker_file(self, step: int, entry: Dict[str, Any]) -> Optional[str]:
        """Append one validation entry to worker shard file on disk."""
        if not self.validation_output_dir:
            return None

        worker_file = os.path.join(
            self.validation_output_dir,
            f"validation_step{step}_worker{self.worker_id}.json",
        )
        try:
            existing: List[Dict[str, Any]] = []
            if os.path.exists(worker_file):
                with open(worker_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    existing = [row for row in loaded if isinstance(row, dict)]
            existing.append(entry)
            with open(worker_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            return worker_file
        except Exception as e:
            logger.warning(
                "[Worker %s] Failed to append validation entry to %s: %s",
                self.worker_id,
                worker_file,
                e,
            )
            return None

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

    def _answer_max_tokens_for_llm(self, llm: Any) -> int:
        """Max tokens for the frozen answer API; optional override vs rollout LLM sampling."""
        if self.answer_max_tokens is not None:
            return int(self.answer_max_tokens)
        return int(llm.sampling_parameters.get("max_tokens", 16384))

    # ------------------------------------------------------------------ #
    #  Un-traced answer generation (raw httpx, bypasses OpenTelemetry)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_openai_base_url(url: str) -> str:
        return (url or "").rstrip("/")

    # ------------------------------------------------------------------ #
    #  MIST reward helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_eir(a_curr: float, a_base: float, k: float) -> float:
        """Execution Incremental Reward: sign(g_raw) · ln(1 + k·|g_raw|).

        Uses asymmetric normalization: α=0.7 when improving (a_curr≥a_base),
        α=0.3 when regressing, with denominator clamped to 0.05.
        """
        alpha = 0.7 if a_curr >= a_base else 0.3
        denom = max(a_base ** alpha * (1.0 - a_base) ** (1.0 - alpha), 0.05)
        g_raw = (a_curr - a_base) / denom
        return math.copysign(math.log(1.0 + k * abs(g_raw)), g_raw)

    @staticmethod
    def _compute_icr(token_logprobs: List[float]) -> float:
        """Internal Confidence Reward: exp(mean(log P(y_τ|...))) in (0, 1].

        Returns 0.0 when logprobs are unavailable (graceful degradation).
        """
        if not token_logprobs:
            return 0.0
        return math.exp(sum(token_logprobs) / len(token_logprobs))

    def _format_answer_user_prompt(self, strategy: str, problem: str, examples_text: str = "") -> str:
        """Format the answer user prompt, supporting both {strategy}/{problem} and
        {examples_text}/{problem} template styles."""
        return self.answer_prompt["user"].format(
            strategy=strategy,
            examples_text=examples_text,
            problem=problem,
        )

    async def _post_answer_chat_completions_mist(
        self,
        base_url: str,
        api_key: str,
        model: str,
        strategy: str,
        problem: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int] = None,
        examples_text: str = "",
    ) -> tuple[str, List[float]]:
        """Single answer call with logprobs=True for MIST ICR computation.

        Returns:
            (content, token_logprobs) where token_logprobs is a list of per-token
            log-probabilities extracted from choices[0].logprobs.content.
            Both are empty/[] on any failure.
        """
        url = f"{self._normalize_openai_base_url(base_url)}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.answer_prompt["system"]},
                {"role": "user", "content": self._format_answer_user_prompt(strategy, problem, examples_text)},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": True,
        }
        if seed is not None:
            payload["seed"] = seed
        if self.answer_no_think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"MIST answer call failed (url={url}): {e}")
            return "", []

        choices = data.get("choices", [])
        if not choices:
            logger.warning(f"MIST answer: empty choices (url={url})")
            return "", []

        content: str = choices[0].get("message", {}).get("content", "") or ""
        token_logprobs: List[float] = []
        try:
            lp_data = choices[0].get("logprobs") or {}
            lp_content = lp_data.get("content") or []
            token_logprobs = [float(tok["logprob"]) for tok in lp_content if "logprob" in tok]
        except Exception as e:
            logger.warning(f"MIST: failed to parse logprobs, ICR will be 0.0: {e}")

        return content, token_logprobs

    async def _post_answer_chat_completions_once(
        self,
        base_url: str,
        api_key: str,
        model: str,
        strategy: str,
        problem: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int] = None,
        examples_text: str = "",
    ) -> str:
        """Single /chat/completions POST; returns message content or "" on any failure."""
        url = f"{self._normalize_openai_base_url(base_url)}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.answer_prompt["system"]},
                {"role": "user", "content": self._format_answer_user_prompt(strategy, problem, examples_text)},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if self.answer_no_think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"Answer model HTTP error {e.response.status_code} "
                    f"(url={url}, model={model}): {e.response.text[:300]}"
                )
                return ""
            try:
                data = resp.json()
            except json.JSONDecodeError as e:
                logger.warning(f"Answer model invalid JSON (url={url}, model={model}): {e}")
                return ""
        except httpx.TimeoutException as e:
            logger.warning(f"Answer model timeout (url={url}, model={model}): {e}")
            return ""
        except httpx.RequestError as e:
            logger.warning(f"Answer model request error (url={url}, model={model}): {e}")
            return ""

        choices = data.get("choices", [])
        if not choices:
            logger.warning(f"Answer model empty choices (url={url}, model={model})")
            return ""
        return choices[0].get("message", {}).get("content", "") or ""

    async def _generate_answer_untraced(
        self,
        base_url: str,
        api_key: str,
        model: str,
        strategy: str,
        problem: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int] = None,
        *,
        examples_text: str = "",
        rollout_fallback_base_url: Optional[str] = None,
        rollout_fallback_model: Optional[str] = None,
    ) -> tuple[str, str]:
        """Generate an answer using raw httpx — NOT captured by AGL tracing.

        Retries transient empty/error responses on the primary (base_url, model),
        then optionally falls back to the rollout/training endpoint so answer
        service outages do not zero out correctness when a strategy exists.

        Returns:
            (content, source_tag) where source_tag is ``primary``, ``rollout_fallback``, or ``none``.
        """
        primary_u = self._normalize_openai_base_url(base_url)
        primary_m = model or ""
        fb_u = self._normalize_openai_base_url(rollout_fallback_base_url or "")
        fb_m = rollout_fallback_model or ""
        distinct_fallback = bool(
            fb_u
            and fb_m
            and (fb_u != primary_u or fb_m != primary_m)
        )

        async def _attempt_pair(bu: str, md: str, label: str) -> str:
            for attempt in range(self.answer_request_retries):
                out = await self._post_answer_chat_completions_once(
                    bu,
                    api_key,
                    md,
                    strategy,
                    problem,
                    temperature,
                    max_tokens,
                    seed=seed,
                    examples_text=examples_text,
                )
                if out:
                    return out
                if attempt + 1 < self.answer_request_retries and self.answer_retry_delay_sec > 0:
                    await asyncio.sleep(self.answer_retry_delay_sec)
            logger.warning(
                f"Answer generation exhausted retries ({self.answer_request_retries}) "
                f"for {label} url={bu} model={md}"
            )
            return ""

        text = await _attempt_pair(base_url, model, "primary")
        if text:
            return text, "primary"
        if (
            self.answer_fallback_rollout_on_failure
            and distinct_fallback
        ):
            text_fb = await _attempt_pair(fb_u, fb_m, "rollout_fallback")
            if text_fb:
                logger.info(
                    "Answer recovered via rollout/training model fallback "
                    f"(primary had failed: {primary_u} / {primary_m})"
                )
                return text_fb, "rollout_fallback"
        return "", "none"

    # ------------------------------------------------------------------ #
    #  Un-traced strategy scoring (raw httpx, bypasses OpenTelemetry)
    # ------------------------------------------------------------------ #

    class _SafeFormatDict(dict):
        """Dict that returns empty string for missing ``str.format`` keys."""
        def __missing__(self, key: str) -> str:
            return ""

    def _oc_weight_pct_strings(self) -> Tuple[str, str, str, str]:
        wa, wb, wc, wd = normalize_four_dim_weights(self.oc_four_dim_weights)
        return (
            f"{100.0 * wa:.1f}%",
            f"{100.0 * wb:.1f}%",
            f"{100.0 * wc:.1f}%",
            f"{100.0 * wd:.1f}%",
        )

    async def _score_strategy_raw_untraced(
        self,
        strategy: str,
        examples_text: str,
        problem_type: str = "",
        problem: str = "",
        answer: str = "",
        correctness_label: str = "unknown",
        ground_truth: str = "",
        expected_answer_format: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Call scorer model and return raw response text."""
        if not self.strategy_scorer_base_url or not self.scoring_prompt:
            return ""

        w_a, w_b, w_c, w_d = self._oc_weight_pct_strings()
        gt_display = (ground_truth or "").strip() or "N/A"
        fmt_display = (expected_answer_format or "").strip() or "N/A"

        format_values = self._SafeFormatDict(
            strategy=strategy,
            examples_text=examples_text,
            task=problem_type,
            problem=problem,
            answer=answer,
            correctness_label=correctness_label,
            ground_truth=gt_display,
            expected_answer_format=fmt_display,
            weight_a_pct=w_a,
            weight_b_pct=w_b,
            weight_c_pct=w_c,
            weight_d_pct=w_d,
        )

        url = f"{self.strategy_scorer_base_url}/chat/completions"
        payload = {
            "model": self.strategy_scorer_model,
            "messages": [
                {"role": "system", "content": self.scoring_prompt["system"]},
                {
                    "role": "user",
                    "content": self.scoring_prompt["user"].format_map(format_values),
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=self.strategy_scorer_timeout_sec) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            choices = data.get("choices", [])
            if not choices:
                logger.warning("Strategy scorer returned empty choices")
                return ""

            raw_output = choices[0].get("message", {}).get("content", "") or ""
            logger.debug("Strategy scorer raw response: %r", raw_output[:300].strip())
            return raw_output
        except Exception as e:
            logger.warning("Strategy scoring failed (non-fatal): %s", e)
            return ""

    async def _score_strategy_untraced(
        self,
        strategy: str,
        examples_text: str,
        problem_type: str = "",
        problem: str = "",
        answer: str = "",
        correctness_label: str = "unknown",
        ground_truth: str = "",
        expected_answer_format: str = "",
    ) -> float:
        """Score strategy quality and return score in [0, 1]."""
        raw_output = await self._score_strategy_raw_untraced(
            strategy=strategy,
            examples_text=examples_text,
            problem_type=problem_type,
            problem=problem,
            answer=answer,
            correctness_label=correctness_label,
            ground_truth=ground_truth,
            expected_answer_format=expected_answer_format,
            temperature=0.0 if self.reward_mode == "hybrid_grounded" else 0.3,
            max_tokens=1024 if self.reward_mode == "hybrid_grounded" else 512,
        )
        if not raw_output:
            return 0.0
        parsed = parse_oc_scorer_response(
            raw_output, four_dim_weights=self.oc_four_dim_weights
        )
        return float(parsed["final_score_01"])

    # ------------------------------------------------------------------ #
    #  Main rollout
    # ------------------------------------------------------------------ #

    async def rollout_async(
        self,
        task: "StrategyGenerationTask",
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> Optional[float]:
        """Execute a rollout: generate strategy (trained) → score / verify (not trained).

        On success, returns ``None`` after emitting a multi-dimensional reward span so the
        runner does not append a duplicate scalar reward span.

        v2 flow (scorer mode):
            1. Generate strategy  (traced)
            2. Score strategy via trained scorer LLM  (un-traced)
            3. Optionally generate + check answer via fixed model  (un-traced)
            4. reward = fw*format + sw*scorer + cw*correctness

        v1 flow (legacy):
            1. Generate strategy  (traced)
            2. Generate answer via training model  (un-traced)
            3. reward = fw*format + cw*correctness

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
            _seed_raw = llm.sampling_parameters.get("seed")
            llm_request_seed: Optional[int] = None
            if _seed_raw is not None:
                try:
                    llm_request_seed = int(_seed_raw)
                except (TypeError, ValueError):
                    llm_request_seed = None
            traced_strategy_call = False
            answer_call_attempted = False
            answer_call_succeeded = False
            current_mode = rollout.mode if hasattr(rollout, "mode") else "unknown"
            is_validation = current_mode != "train"

            logger.info(
                f"[Rollout {attempted_rollout.rollout_id}] START - "
                f"Type: {task['problem_type']}, Mode: {rollout.mode}"
            )
            self._debug_rollout(
                attempted_rollout.rollout_id,
                "start",
                mode=str(getattr(rollout, "mode", "unknown")),
                skip_strategy_generation=self.skip_strategy_generation,
                answer_model_base_url=bool(self.answer_model_base_url),
                answer_model_name=self.answer_model_name or llm.model,
            )

            examples_text = format_examples(task["examples"])

            strategy_output = ""
            strategy = ""
            format_reward = 0.0
            if self.skip_strategy_generation:
                system_prompt = ""
                user_prompt = ""
                logger.info(
                    f"[Rollout {attempted_rollout.rollout_id}] "
                    "Skipping strategy generation (strict no-strategy baseline)"
                )
            else:
                # ---- Step 1: Generate strategy (TRACED — will be trained) ---- #
                system_prompt = self.strategy_prompt["system"]
                user_prompt = self.strategy_prompt["user"].format(
                    examples_text=examples_text,
                    problem=task["problem"],
                )
                traced_strategy_call = True

                # mist-inline: use the answer model endpoint for strategy extraction.
                if self._inline_strategy_mode:
                    _strat_base_url = self.answer_model_base_url or base_url
                    _strat_model = self.answer_model_name or llm.model
                    _strat_no_think = self.answer_no_think
                else:
                    _strat_base_url = base_url
                    _strat_model = llm.model
                    _strat_no_think = self.strategy_no_think

                client = AsyncOpenAI(
                    base_url=_strat_base_url,
                    api_key=llm.api_key or "dummy-key",
                )

                _strategy_kwargs: Dict[str, Any] = {
                    "model": _strat_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": llm.sampling_parameters.get("temperature", 0.7),
                    "max_tokens": llm.sampling_parameters.get("max_tokens", 16384),
                }
                if llm_request_seed is not None:
                    _strategy_kwargs["seed"] = llm_request_seed
                extra_body: Dict[str, Any] = {}
                if _strat_no_think:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                if (
                    not self._inline_strategy_mode
                    and self.strategy_repetition_penalty is not None
                    and self.strategy_repetition_penalty > 0
                ):
                    extra_body["repetition_penalty"] = float(self.strategy_repetition_penalty)
                if extra_body:
                    _strategy_kwargs["extra_body"] = extra_body
                response = await client.chat.completions.create(**_strategy_kwargs)

                strategy_output = response.choices[0].message.content or ""
                strategy = extract_strategy(strategy_output)
                if self._use_structured_format_reward:
                    format_reward = compute_format_reward_structured(strategy_output)
                else:
                    format_reward = compute_format_reward(strategy_output)

                logger.info(
                    f"[Rollout {attempted_rollout.rollout_id}] Strategy: "
                    f"format={format_reward}, length={len(strategy) if strategy else 0}"
                )

            # ---- MIST reward branch (training only) ---- #
            # Activated when baseline_cache is loaded (beta=0 → pure EIR; beta>0 → EIR×(1+β·ICR)).
            # Format-invalid strategies are emitted with reward=0, format_ok=0 and skipped immediately.
            if self._baseline_cache and not is_validation:
                if strategy is None:
                    # Format gate: no valid strategy → emit zero reward, skip answer call
                    logger.info(
                        f"[Rollout {attempted_rollout.rollout_id}] MIST: format invalid, skipping"
                    )
                    agl.emit_reward(
                        {
                            "final": 0.0,
                            "format_ok": 0.0,
                            "r_eir": 0.0,
                            "r_icr": 0.0,
                            "answer_soft": 0.0,
                            "baseline_soft": -1.0,
                        },
                        primary_key="final",
                    )
                    return None

                # Baseline gate (safety fallback; dataset filter should prevent this)
                _mist_key = self._problem_key(task["problem"])
                _a_base = self._baseline_cache.get(_mist_key)
                if _a_base is None:
                    logger.warning(
                        f"[Rollout {attempted_rollout.rollout_id}] MIST: baseline not found "
                        f"(key={_mist_key}), emitting zero reward"
                    )
                    agl.emit_reward(
                        {
                            "final": 0.0,
                            "format_ok": 1.0,
                            "r_eir": 0.0,
                            "r_icr": 0.0,
                            "answer_soft": 0.0,
                            "baseline_soft": -1.0,
                        },
                        primary_key="final",
                    )
                    return None

                # M=1 answer call with logprobs for ICR
                _ans_base_url = self.answer_model_base_url or base_url
                _ans_api_key = llm.api_key or "dummy-key"
                _ans_model = self.answer_model_name or llm.model
                _ans_temp = (
                    self.answer_temperature
                    if self.answer_temperature is not None
                    else llm.sampling_parameters.get("temperature", 0.7)
                )
                _MAX_STRATEGY_CHARS = 8000
                _strategy_for_answer = strategy[:_MAX_STRATEGY_CHARS] if len(strategy) > _MAX_STRATEGY_CHARS else strategy
                if len(strategy) > _MAX_STRATEGY_CHARS:
                    logger.info(
                        f"[Rollout {attempted_rollout.rollout_id}] MIST: strategy truncated "
                        f"{len(strategy)} → {_MAX_STRATEGY_CHARS} chars for answer call"
                    )
                _ans_content, _token_logprobs = await self._post_answer_chat_completions_mist(
                    base_url=_ans_base_url,
                    api_key=_ans_api_key,
                    model=_ans_model,
                    strategy=_strategy_for_answer,
                    problem=task["problem"],
                    temperature=_ans_temp,
                    max_tokens=self._answer_max_tokens_for_llm(llm),
                    seed=llm_request_seed,
                    examples_text=examples_text,
                )
                if not _token_logprobs:
                    logger.warning(
                        f"[Rollout {attempted_rollout.rollout_id}] MIST: logprobs empty, "
                        "ICR will be 0.0"
                    )

                # Compute A_curr (soft score)
                _extracted = self.reward_config.extract_answer(_ans_content)
                if _extracted:
                    _jd = compute_answer_judgement(
                        answer=_extracted,
                        ground_truth=task["ground_truth"],
                        numeric_tolerance=self.numeric_tolerance,
                        f1_threshold=self.f1_threshold,
                        task_meta=task.get("task_meta", {}),
                    )
                    _a_curr = float(_jd.get("soft_score", 0.0))
                else:
                    _a_curr = 0.0

                # MIST formula: R = R_EIR × (1 + β·R_ICR)
                _r_icr = self._compute_icr(_token_logprobs)
                _r_eir = self._compute_eir(_a_curr, _a_base, self._eir_k)
                _final = _r_eir * (1.0 + self._beta * _r_icr)

                logger.info(
                    f"[Rollout {attempted_rollout.rollout_id}] MIST: "
                    f"a_curr={_a_curr:.3f}, a_base={_a_base:.3f}, "
                    f"r_eir={_r_eir:.4f}, r_icr={_r_icr:.4f}, final={_final:.4f}"
                )
                agl.emit_reward(
                    {
                        "final": float(_final),
                        "format_ok": 1.0,
                        "r_eir": float(_r_eir),
                        "r_icr": float(_r_icr),
                        "answer_soft": float(_a_curr),
                        "baseline_soft": float(_a_base),
                    },
                    primary_key="final",
                )
                return None

            # Outcome-conditioned scorer: any rubric TOML + vLLM base URL (even if reward-version
            # has no legacy extract_score hook, e.g. v3).
            use_scorer = bool(self.strategy_scorer_base_url and self.scoring_prompt)

            # ---- Step 2-5: reward branches ---- #
            scorer_reward = 0.0
            correctness = 0.0
            answer_output = ""
            extracted_answer: Optional[str] = None
            answer_raw_list: List[str] = []
            answer_extracted_list: List[str] = []
            router_scores: List[float] = []
            grounded_proxy = 0.0
            soft_correctness_mean = 0.0
            oc_dimension_scores: Dict[str, float] = {}
            oc_cap_flags: List[str] = []
            oc_penalties: List[str] = []
            oc_final_score_100 = 0.0
            used_soft_fallback = False
            hard_correct = 0
            hard_correct_mean = 0.0
            hard_correct_list: List[int] = []
            answer_used_rollout_fallback = False
            effective_proxy = 0.0

            if strategy and self.reward_mode == "hybrid_grounded":
                try:
                    ans_base_url = self.answer_model_base_url or base_url
                    ans_api_key = llm.api_key or "dummy-key"
                    ans_model = self.answer_model_name or llm.model
                    hybrid_answer_via: List[str] = []

                    async def _answer_once(bound_strategy: str) -> str:
                        answer_strategy = bound_strategy if self.use_strategy_for_answer else ""
                        _ans_temp = (
                            self.val_answer_temperature
                            if (is_validation and self.val_answer_temperature is not None)
                            else llm.sampling_parameters.get("temperature", 0.7)
                        )
                        _text, _via = await self._generate_answer_untraced(
                            base_url=ans_base_url,
                            api_key=ans_api_key,
                            model=ans_model,
                            strategy=answer_strategy,
                            problem=task["problem"],
                            temperature=_ans_temp,
                            max_tokens=self._answer_max_tokens_for_llm(llm),
                            seed=llm_request_seed,
                            examples_text=examples_text,
                            rollout_fallback_base_url=base_url,
                            rollout_fallback_model=llm.model,
                        )
                        hybrid_answer_via.append(_via)
                        return _text

                    eval_result = await evaluate_strategy_k_samples(
                        strategy=strategy,
                        targets=task.get("ground_truths", [task["ground_truth"]]),
                        task_meta=task.get("task_meta", {}),
                        answer_fn=_answer_once,
                        extract_answer_fn=lambda text: self.reward_config.extract_answer(text),
                        correctness_fn=lambda pred, target: self.reward_config.compute_answer_correctness(
                            pred,
                            target,
                            self.numeric_tolerance,
                            self.f1_threshold,
                        ),
                        k=self.grounded_proxy_k,
                    )
                    answer_raw_list = list(eval_result["answer_raw_list"])
                    answer_extracted_list = list(eval_result["answer_extracted_list"])
                    router_scores = list(eval_result["router_scores"])
                    grounded_proxy = float(eval_result["grounded_proxy"])
                    soft_correctness_mean = float(eval_result["soft_correctness_mean"])
                    correctness = float(eval_result["single_sample_correctness"])
                    answer_used_rollout_fallback = any(v == "rollout_fallback" for v in hybrid_answer_via)

                    representative = select_representative_answer(answer_raw_list, router_scores)
                    rep_idx = int(representative.get("representative_index", -1))
                    answer_output = representative.get("representative_answer_raw", "")
                    if 0 <= rep_idx < len(answer_extracted_list):
                        extracted_answer = answer_extracted_list[rep_idx]
                    else:
                        extracted_answer = self.reward_config.extract_answer(answer_output)
                    if extracted_answer:
                        hard_detail = compute_answer_judgement(
                            answer=extracted_answer,
                            ground_truth=task["ground_truth"],
                            numeric_tolerance=self.numeric_tolerance,
                            f1_threshold=self.f1_threshold,
                            task_meta=task.get("task_meta", {}),
                        )
                        hard_correct = int(hard_detail.get("hard_correct", 0))
                    hard_correct_list = [hard_correct]
                    hard_correct_mean = float(hard_correct)
                    rep_label = str(representative.get("representative_label", "incorrect"))

                    tm_h = task.get("task_meta") or {}
                    exp_fmt_h = str(tm_h.get("expected_answer_format", "") or "") if isinstance(tm_h, dict) else ""
                    if use_scorer:
                        raw_scorer_output = await self._score_strategy_raw_untraced(
                            strategy=strategy,
                            examples_text=examples_text,
                            problem_type=task["problem_type"],
                            problem=task["problem"],
                            answer=answer_output,
                            correctness_label=rep_label,
                            ground_truth=str(task.get("ground_truth") or ""),
                            expected_answer_format=exp_fmt_h,
                        )
                        oc_parsed = parse_oc_scorer_response(
                            raw_scorer_output, four_dim_weights=self.oc_four_dim_weights
                        )
                        scorer_reward = float(oc_parsed["final_score_01"])
                        oc_final_score_100 = float(oc_parsed["final_score_100"])
                        oc_dimension_scores = dict(oc_parsed["dimension_scores"])
                        oc_cap_flags = list(oc_parsed["cap_flags"])
                        oc_penalties = list(oc_parsed["penalties"])

                    effective_proxy = select_effective_proxy(
                        grounded_proxy=grounded_proxy,
                        soft_correctness_mean=soft_correctness_mean,
                        group_gp_values=router_scores,
                    )
                    used_soft_fallback = effective_proxy != grounded_proxy
                    final_reward = compute_hybrid_reward(
                        format_r=format_reward,
                        oc_scorer_r=scorer_reward,
                        grounded_proxy=effective_proxy,
                        format_weight=self.format_weight,
                        scorer_weight=self.scorer_weight,
                        proxy_weight=self.proxy_weight,
                    )
                    logger.info(
                        f"[Rollout {attempted_rollout.rollout_id}] Hybrid: "
                        f"scorer={scorer_reward:.3f}, gp={grounded_proxy:.3f}, soft={soft_correctness_mean:.3f}, "
                        f"final={final_reward:.3f}"
                    )
                except Exception as e:
                    logger.warning(f"[Rollout {attempted_rollout.rollout_id}] Hybrid reward failed: {e}")
                    final_reward = 0.0
                    effective_proxy = 0.0
            else:
                run_answer = bool(
                    (self.skip_strategy_generation or strategy)
                    and (
                        self.correctness_weight > 0
                        or not use_scorer
                        or self.answer_model_base_url
                    )
                )
                if run_answer:
                    answer_call_attempted = True
                    try:
                        ans_base_url = self.answer_model_base_url or base_url
                        ans_api_key = llm.api_key or "dummy-key"
                        ans_model = self.answer_model_name or llm.model
                        answer_strategy = (
                            strategy
                            if (self.use_strategy_for_answer and not self.skip_strategy_generation)
                            else ""
                        )
                        answer_temperature = (
                            self.val_answer_temperature
                            if (is_validation and self.val_answer_temperature is not None)
                            else (
                                self.answer_temperature
                                if self.answer_temperature is not None
                                else llm.sampling_parameters.get("temperature", 0.7)
                            )
                        )
                        # Multi-problem hard pass@k: evaluate strategy on sibling eval problems
                        # from the same subtask (populated at dataset creation time).
                        _eval_probs = task.get("eval_problems", [])
                        use_multi_problem = (
                            not is_validation
                            and self.eval_problems_per_subtask > 0
                            and len(_eval_probs) > 0
                        )

                        should_use_k_answers = (
                            not use_multi_problem
                            and self.reward_mode == "scorer_only"
                            and self.correctness_weight > 0
                            and self.grounded_proxy_k > 1
                        )

                        if use_multi_problem:
                            # For each eval problem: attempt grounded_proxy_k answers and mark
                            # pass=1 as soon as any answer is hard-correct (early exit).
                            # correctness = mean pass rate over eval problems.
                            passes: list[float] = []
                            for _ep in _eval_probs[:self.eval_problems_per_subtask]:
                                _ep_problem = _ep.get("problem", "")
                                _ep_gt = _ep.get("ground_truth", "")
                                if not _ep_problem or not _ep_gt:
                                    continue
                                _any_correct = 0
                                for _ in range(self.grounded_proxy_k):
                                    _ep_raw, _ep_via = await self._generate_answer_untraced(
                                        base_url=ans_base_url,
                                        api_key=ans_api_key,
                                        model=ans_model,
                                        strategy=answer_strategy,
                                        problem=_ep_problem,
                                        temperature=answer_temperature,
                                        max_tokens=self._answer_max_tokens_for_llm(llm),
                                        seed=llm_request_seed,
                                        examples_text=examples_text,
                                        rollout_fallback_base_url=base_url,
                                        rollout_fallback_model=llm.model,
                                    )
                                    if _ep_via == "rollout_fallback":
                                        answer_used_rollout_fallback = True
                                    _ep_ext = self.reward_config.extract_answer(_ep_raw) or ""
                                    if _ep_ext:
                                        _ep_hard_detail = compute_answer_judgement(
                                            answer=_ep_ext,
                                            ground_truth=_ep_gt,
                                            numeric_tolerance=self.numeric_tolerance,
                                            f1_threshold=self.f1_threshold,
                                            task_meta=task.get("task_meta", {}),
                                        )
                                        if _ep_hard_detail.get("hard_correct", 0):
                                            _any_correct = 1
                                            break
                                passes.append(float(_any_correct))
                            correctness = sum(passes) / len(passes) if passes else 0.0
                            hard_correct = int(correctness >= 1.0 - 1e-9)
                            hard_correct_mean = correctness
                            hard_correct_list = [int(p) for p in passes]
                            soft_correctness_mean = correctness
                            router_scores = [correctness]
                            answer_call_succeeded = True
                            logger.info(
                                f"[Rollout {attempted_rollout.rollout_id}] "
                                f"MultiProblem pass@{self.grounded_proxy_k}: "
                                f"{passes} → correctness={correctness:.3f}"
                            )
                        elif should_use_k_answers:
                            for _ in range(self.grounded_proxy_k):
                                answer_raw, _via = await self._generate_answer_untraced(
                                    base_url=ans_base_url,
                                    api_key=ans_api_key,
                                    model=ans_model,
                                    strategy=answer_strategy,
                                    problem=task["problem"],
                                    temperature=answer_temperature,
                                    max_tokens=self._answer_max_tokens_for_llm(llm),
                                    seed=llm_request_seed,
                                    examples_text=examples_text,
                                    rollout_fallback_base_url=base_url,
                                    rollout_fallback_model=llm.model,
                                )
                                if _via == "rollout_fallback":
                                    answer_used_rollout_fallback = True
                                answer_raw_list.append(answer_raw)
                                answer_extracted = self.reward_config.extract_answer(answer_raw) or ""
                                answer_extracted_list.append(answer_extracted)
                                soft_score = self.reward_config.compute_answer_correctness(
                                    answer_extracted,
                                    task["ground_truth"],
                                    self.numeric_tolerance,
                                    self.f1_threshold,
                                ) if answer_extracted else 0.0
                                router_scores.append(float(soft_score))
                                hard_detail = compute_answer_judgement(
                                    answer=answer_extracted,
                                    ground_truth=task["ground_truth"],
                                    numeric_tolerance=self.numeric_tolerance,
                                    f1_threshold=self.f1_threshold,
                                    task_meta=task.get("task_meta", {}),
                                ) if answer_extracted else {"hard_correct": 0}
                                hard_correct_list.append(int(hard_detail.get("hard_correct", 0)))
                                if self.use_hard_correctness_metric:
                                    router_scores[-1] = float(hard_correct_list[-1])

                            soft_correctness_mean = (
                                sum(router_scores) / len(router_scores) if router_scores else 0.0
                            )
                            correctness = soft_correctness_mean
                            hard_correct_mean = (
                                sum(hard_correct_list) / len(hard_correct_list) if hard_correct_list else 0.0
                            )

                            representative = select_representative_answer(answer_raw_list, router_scores)
                            rep_idx = int(representative.get("representative_index", 0))
                            if not (0 <= rep_idx < len(answer_raw_list)):
                                rep_idx = 0
                            answer_output = answer_raw_list[rep_idx]
                            extracted_answer = answer_extracted_list[rep_idx] if rep_idx < len(answer_extracted_list) else ""
                            hard_correct = hard_correct_list[rep_idx] if rep_idx < len(hard_correct_list) else 0
                            answer_call_succeeded = len(answer_raw_list) > 0
                        else:
                            answer_output, _via = await self._generate_answer_untraced(
                                base_url=ans_base_url,
                                api_key=ans_api_key,
                                model=ans_model,
                                strategy=answer_strategy,
                                problem=task["problem"],
                                temperature=answer_temperature,
                                max_tokens=self._answer_max_tokens_for_llm(llm),
                                seed=llm_request_seed,
                                examples_text=examples_text,
                                rollout_fallback_base_url=base_url,
                                rollout_fallback_model=llm.model,
                            )
                            answer_used_rollout_fallback = _via == "rollout_fallback"
                            extracted_answer = self.reward_config.extract_answer(answer_output)
                            if extracted_answer:
                                correctness = self.reward_config.compute_answer_correctness(
                                    extracted_answer,
                                    task["ground_truth"],
                                    self.numeric_tolerance,
                                    self.f1_threshold,
                                )
                                hard_detail = compute_answer_judgement(
                                    answer=extracted_answer,
                                    ground_truth=task["ground_truth"],
                                    numeric_tolerance=self.numeric_tolerance,
                                    f1_threshold=self.f1_threshold,
                                    task_meta=task.get("task_meta", {}),
                                )
                                hard_correct = int(hard_detail.get("hard_correct", 0))
                                if self.use_hard_correctness_metric:
                                    correctness = float(hard_correct)
                            answer_raw_list = [answer_output]
                            answer_extracted_list = [extracted_answer or ""]
                            router_scores = [correctness]
                            hard_correct_list = [hard_correct]
                            hard_correct_mean = float(hard_correct)
                            answer_call_succeeded = bool(answer_output)
                    except Exception as e:
                        logger.warning(
                            f"[Rollout {attempted_rollout.rollout_id}] "
                            f"Answer generation failed (non-fatal): {e}"
                        )

                if use_scorer and strategy:
                    tm_s = task.get("task_meta") or {}
                    exp_fmt_s = str(tm_s.get("expected_answer_format", "") or "") if isinstance(tm_s, dict) else ""
                    rep_label_s = "unknown"
                    if extracted_answer and str(task.get("ground_truth") or "").strip():
                        _hd_s = compute_answer_judgement(
                            answer=extracted_answer,
                            ground_truth=task["ground_truth"],
                            numeric_tolerance=self.numeric_tolerance,
                            f1_threshold=self.f1_threshold,
                            task_meta=task.get("task_meta", {}),
                        )
                        rep_label_s = "correct" if int(_hd_s.get("hard_correct", 0)) else "incorrect"
                    elif correctness >= 1.0 - 1e-9:
                        rep_label_s = "correct"
                    elif extracted_answer:
                        rep_label_s = "incorrect"
                    raw_oc = await self._score_strategy_raw_untraced(
                        strategy=strategy,
                        examples_text=examples_text,
                        problem_type=task["problem_type"],
                        problem=task["problem"],
                        answer=answer_output or "",
                        correctness_label=rep_label_s,
                        ground_truth=str(task.get("ground_truth") or ""),
                        expected_answer_format=exp_fmt_s,
                    )
                    oc_parsed_s = parse_oc_scorer_response(
                        raw_oc, four_dim_weights=self.oc_four_dim_weights
                    )
                    scorer_reward = float(oc_parsed_s["final_score_01"])
                    oc_final_score_100 = float(oc_parsed_s["final_score_100"])
                    oc_dimension_scores = dict(oc_parsed_s["dimension_scores"])
                    oc_cap_flags = list(oc_parsed_s["cap_flags"])
                    oc_penalties = list(oc_parsed_s["penalties"])
                    logger.info(f"[Rollout {attempted_rollout.rollout_id}] Scorer: {scorer_reward:.3f}")

                if use_scorer:
                    final_reward = self.reward_config.compute_final_reward(
                        format_reward,
                        scorer_reward,
                        correctness,
                        self.format_weight,
                        self.scorer_weight,
                        self.correctness_weight,
                    )
                else:
                    final_reward = self.reward_config.compute_final_reward(
                        format_reward,
                        correctness,
                        self.format_weight,
                        self.correctness_weight,
                    )

            # Always log v3 soft / hard for WandB (independent of use_hard_correctness_metric on the scalar `correctness`).
            answer_soft_metric = 0.0
            hard_correct_log = 0
            if extracted_answer:
                _jd_log = compute_answer_judgement(
                    answer=extracted_answer,
                    ground_truth=task["ground_truth"],
                    numeric_tolerance=self.numeric_tolerance,
                    f1_threshold=self.f1_threshold,
                    task_meta=task.get("task_meta", {}),
                )
                answer_soft_metric = float(_jd_log.get("soft_score", 0.0))
                hard_correct_log = int(_jd_log.get("hard_correct", 0))

            reward_details: Dict[str, Any] = {
                "format": format_reward,
                "scorer": scorer_reward,
                "correctness": correctness,
                "answer_soft": answer_soft_metric,
                "answer_hard": float(hard_correct_log),
                "hard_correct": hard_correct,
                "hard_correct_mean": hard_correct_mean,
                "hard_correct_list": hard_correct_list,
                "grounded_proxy": grounded_proxy,
                "soft_correctness_mean": soft_correctness_mean,
                "multi_sample_k": self.grounded_proxy_k,
                "reward_mode": self.reward_mode,
                "oc_dimension_scores": oc_dimension_scores,
                "oc_cap_flags": oc_cap_flags,
                "oc_penalties": oc_penalties,
                "oc_final_score_100": oc_final_score_100,
                "answer_model_version": self.answer_model_name or llm.model,
                "used_soft_fallback": used_soft_fallback,
                "answer_used_rollout_fallback": answer_used_rollout_fallback,
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
                actual_step = (
                    0 if self.validation_step_counter == 0 else self.test_freq * self.validation_step_counter
                )
                val_entry = {
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
                        "ground_truths": task.get("ground_truths", [task["ground_truth"]]),
                        "task_meta": task.get("task_meta", {}),
                    },
                    "output": {
                        "strategy_raw": strategy_output,
                        "strategy_extracted": strategy or "",
                        "answer_raw": answer_output,
                        "answer_extracted": extracted_answer or "",
                        "answer_raw_list": answer_raw_list,
                        "answer_extracted_list": answer_extracted_list,
                        "router_scores": router_scores,
                    },
                    "reward": reward_details,
                    "metadata": {
                        "model": llm.model,
                        "temperature": llm.sampling_parameters.get("temperature", 0.7),
                        "max_tokens": llm.sampling_parameters.get("max_tokens", 16384),
                        "rollout_mode": str(current_mode),
                    },
                }
                self.validation_outputs.append(val_entry)
                # val_only has no val->train transition; append per-sample to disk.
                self._append_validation_entry_to_worker_file(actual_step, val_entry)

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
                        "ground_truths": task.get("ground_truths", [task["ground_truth"]]),
                        "task_meta": task.get("task_meta", {}),
                        "examples": task["examples"],
                    },
                    "output": {
                        "strategy_raw": strategy_output,
                        "strategy_extracted": strategy or "",
                        "answer_raw": answer_output,
                        "answer_extracted": extracted_answer or "",
                        "answer_raw_list": answer_raw_list,
                        "answer_extracted_list": answer_extracted_list,
                        "router_scores": router_scores,
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
                    f"format={format_reward}, scorer={scorer_reward}, "
                    f"correctness={correctness}"
                )
            else:
                logger.info(
                    f"[Rollout {attempted_rollout.rollout_id}] reward={final_reward:.3f}"
                )
            self._debug_rollout(
                attempted_rollout.rollout_id,
                "finish",
                traced_strategy_call=traced_strategy_call,
                run_answer=run_answer if "run_answer" in locals() else False,
                answer_call_attempted=answer_call_attempted,
                answer_call_succeeded=answer_call_succeeded,
                strategy_len=len(strategy) if strategy else 0,
                answer_len=len(answer_output) if answer_output else 0,
                extracted_answer_present=bool(extracted_answer),
                final_reward=float(final_reward),
            )

            # R_delta：相对 SFT 基线的增量奖励（可选）
            r_delta = 0.0
            baseline_soft_logged: Optional[float] = None
            if self._incremental_weight > 0.0 and self._baseline_cache:
                _key = self._problem_key(task.get("problem", ""))
                _baseline_soft = self._baseline_cache.get(_key)
                if _baseline_soft is not None:
                    r_delta = max(0.0, answer_soft_metric - _baseline_soft)
                    final_reward = final_reward + self._incremental_weight * r_delta
                    baseline_soft_logged = _baseline_soft
                    logger.info(
                        f"[Rollout {attempted_rollout.rollout_id}] R_delta={r_delta:.3f} "
                        f"(soft={answer_soft_metric:.3f} - baseline={_baseline_soft:.3f}), "
                        f"final_with_delta={final_reward:.3f}"
                    )
                else:
                    logger.debug(
                        f"[Rollout {attempted_rollout.rollout_id}] 未找到基线分（problem key={_key}），跳过 R_delta"
                    )

            try:
                agl.emit_reward(
                    {
                        "final": float(final_reward),
                        "format": float(format_reward),
                        "scorer": float(scorer_reward),
                        "grounded_proxy": float(effective_proxy),
                        "answer_soft": float(answer_soft_metric),
                        "answer_hard": float(hard_correct_log),
                        "r_delta": float(r_delta),
                        "baseline_soft": float(baseline_soft_logged) if baseline_soft_logged is not None else -1.0,
                    },
                    primary_key="final",
                )
            except RuntimeError:
                pass  # no VERL tracer in eval_no_verl mode; validation_outputs already written
            # Return None so LitAgentRunner does not emit a second scalar reward span
            # (which would override multi-dimensional reward_dimensions in the daemon).
            return None

        except Exception as e:
            logger.error(
                f"[Rollout {attempted_rollout.rollout_id}] ERROR: {e}",
                exc_info=True,
            )
            try:
                agl.emit_reward(
                    {
                        "final": 0.0,
                        "format": 0.0,
                        "scorer": 0.0,
                        "grounded_proxy": 0.0,
                        "answer_soft": 0.0,
                        "answer_hard": 0.0,
                    },
                    primary_key="final",
                )
            except Exception as reward_err:
                logger.warning(f"Failed to emit reward: {reward_err}")
            return 0.0

