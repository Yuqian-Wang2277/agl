# Copyright (c) Microsoft. All rights reserved.

"""Strategy application agent implementation."""

import glob
import json
import logging
import os
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, cast

from openai import AsyncOpenAI

import agentlightning as agl

from .prompts import format_strategy_application_prompt, get_strategy_application_system_prompt
from .reward import compute_stage2_reward, extract_answer
from .strategy_extractor import StrategyExtractor

logger = logging.getLogger(__name__)


class StrategyApplicationTask(TypedDict):
    """Task structure for strategy application.
    
    Attributes:
        fewshot_examples: Few-shot examples used to extract strategy online.
        problem: The problem to solve.
        ground_truth: The correct answer.
        problem_type: The type/category of the problem.
        source_problem_type: Source problem type (for cross-domain mode, None for same-domain).
    """
    fewshot_examples: List[Dict[str, Any]]
    problem: str
    ground_truth: str
    problem_type: str
    source_problem_type: Optional[str]


class StrategyApplicationAgent(agl.LitAgent[StrategyApplicationTask]):
    """Agent for applying problem-solving strategies to solve problems.
    
    This agent first uses the Stage 1 strategy extractor to generate a
    problem-solving strategy from few-shot examples, then applies that
    strategy to solve the target problem. The answer should be wrapped
    in <answer>...</answer> tags.
    """
    
    def __init__(
        self,
        strategy_extractor: StrategyExtractor,
        save_full_output: bool = True,
        rollout_traces_dir: Optional[str] = None,
        validation_output_dir: Optional[str] = None,
        experiment_id: Optional[str] = None,
        test_freq: int = 50,
        consistency_threshold: float = 0.5,
        numeric_tolerance: float = 0.02,
        f1_threshold: float = 0.5,
        # Reward weights (Stage 2A defaults, can be overridden from config/CLI)
        format_weight: float = 0.2,
        consistency_weight: float = 0.8,
        correctness_weight: float = 0.0,
        coverage_weight: float = 0.4,
        order_weight: float = 0.2,
        binding_weight: float = 0.2,
        intermediate_weight: float = 0.2,
    ) -> None:
        """Initialize the strategy application agent.
        
        Args:
            save_full_output: Whether to save complete model output to log file.
            rollout_traces_dir: Directory to save rollout traces (input/output pairs).
            validation_output_dir: Base directory to save validation outputs.
            experiment_id: Unique experiment identifier (format: YYYYMMDD_HHMMSS).
            test_freq: Frequency of validation (every N steps).
            consistency_threshold: Threshold for strategy consistency check.
            numeric_tolerance: Numeric tolerance for answer correctness.
            f1_threshold: F1 threshold for answer correctness.
        """
        super().__init__()
        self.strategy_extractor = strategy_extractor
        self.save_full_output = save_full_output
        self.experiment_id = experiment_id
        self.rollout_traces_dir = rollout_traces_dir
        self.validation_output_dir = validation_output_dir
        self.test_freq = test_freq
        self.consistency_threshold = consistency_threshold
        self.numeric_tolerance = numeric_tolerance
        self.f1_threshold = f1_threshold

        # Reward weights
        self.format_weight = format_weight
        self.consistency_weight = consistency_weight
        self.correctness_weight = correctness_weight
        self.coverage_weight = coverage_weight
        self.order_weight = order_weight
        self.binding_weight = binding_weight
        self.intermediate_weight = intermediate_weight
        
        if rollout_traces_dir:
            if experiment_id:
                rollout_traces_dir = os.path.join(rollout_traces_dir, experiment_id)
            os.makedirs(rollout_traces_dir, exist_ok=True)
            self.traces_file = os.path.join(rollout_traces_dir, "rollout_traces.jsonl")
            logger.info(f"Rollout traces will be saved to: {self.traces_file}")
        else:
            self.traces_file = None
        
        # Initialize validation output tracking
        self.validation_outputs: List[Dict[str, Any]] = []
        self.last_rollout_mode: Optional[str] = None
        self.validation_step_counter: int = 0
        self.worker_id = os.getpid()  # Use process ID to distinguish workers
        self._pending_merge_step: Optional[int] = None  # Deferred merge: step awaiting merge
        
        # Batch statistics tracking for monitoring
        self.batch_rewards: List[float] = []
        self.batch_reward_details: List[Dict[str, float]] = []
        self.current_batch_id: Optional[str] = None
        
        if validation_output_dir:
            if experiment_id:
                validation_output_dir = os.path.join(validation_output_dir, experiment_id)
            os.makedirs(validation_output_dir, exist_ok=True)
            self.validation_output_dir = validation_output_dir
            logger.info(f"Validation outputs will be saved to: {validation_output_dir}")
        
        logger.info(
            f"StrategyApplicationAgent initialized "
            f"(save_full_output={save_full_output}, experiment_id={experiment_id})"
        )
    
    def save_validation_outputs(self, step: int) -> Optional[str]:
        """Save collected validation outputs to a worker-specific file.
        
        Uses deferred merging to avoid race conditions: instead of merging
        immediately (when other workers may not have saved yet), this method
        merges the *previous* step's worker files first. By the time the next
        validation round triggers a save, all workers have finished the
        previous round, so the merge is guaranteed to be complete.
        
        Args:
            step: Current training step number.
            
        Returns:
            Path to the per-worker file if successful, None otherwise.
        """
        if not self.validation_output_dir or not self.validation_outputs:
            return None
        
        # Deferred merge: merge the previous step's worker files.
        # By now all workers have finished the previous validation round
        # (at least one full training cycle has elapsed), so all per-worker
        # files are guaranteed to exist.
        if self._pending_merge_step is not None:
            self._merge_worker_validation_files(self._pending_merge_step)
        
        # Save this worker's file for the current step
        temp_file = os.path.join(self.validation_output_dir, f"validation_step{step}_worker{self.worker_id}.json")        
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.validation_outputs, f, ensure_ascii=False, indent=2)
            saved_count = len(self.validation_outputs)
            logger.info(f"[Worker {self.worker_id}] Saved {saved_count} validation outputs to: {temp_file}")
            self.validation_outputs = []
        except Exception as e:
            logger.error(f"[Worker {self.worker_id}] Failed to save validation outputs: {e}")
            return None
        
        # Record current step as pending merge (will be merged at next save)
        self._pending_merge_step = step
        return temp_file
    
    def _merge_worker_validation_files(self, step: int) -> Optional[str]:
        """Merge all worker validation files for a given step into one file.
        
        Args:
            step: Current training step number.
            
        Returns:
            Path to merged file if successful, None otherwise.
        """
        if not self.validation_output_dir:
            return None
        
        # Find all worker files for this step
        pattern = os.path.join(self.validation_output_dir, f"validation_step{step}_worker*.json")
        worker_files = glob.glob(pattern)
        
        if not worker_files:
            logger.warning(f"No worker validation files found for step {step}")
            return None
        
        # Merge all outputs
        all_outputs: List[Dict[str, Any]] = []
        for worker_file in sorted(worker_files):
            try:
                with open(worker_file, "r", encoding="utf-8") as f:
                    outputs = json.load(f)
                    if isinstance(outputs, list):
                        all_outputs.extend(outputs)
                    logger.debug(f"Loaded {len(outputs)} outputs from {worker_file}")
            except Exception as e:
                logger.warning(f"Failed to load worker file {worker_file}: {e}")
        
        if not all_outputs:
            return None
        
        # Save merged file
        if step == 0:
            merged_file = os.path.join(self.validation_output_dir, "validation_global_step0.json")
        else:
            merged_file = os.path.join(self.validation_output_dir, f"validation_global_step{step}.json")
        
        try:
            with open(merged_file, "w", encoding="utf-8") as f:
                json.dump(all_outputs, f, ensure_ascii=False, indent=2)
            logger.info(
                f"[Worker {self.worker_id}] Merged {len(all_outputs)} validation outputs "
                f"from {len(worker_files)} workers to: {merged_file}"
            )
            return merged_file
        except Exception as e:
            logger.error(f"Failed to save merged validation file: {e}")
            return None
    
    async def rollout_async(
        self,
        task: StrategyApplicationTask,
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float:
        """Execute a rollout to extract strategy and solve problem.
        
        Args:
            task: Strategy application task.
            resources: Named resources including the LLM.
            rollout: Rollout metadata.
            
        Returns:
            Final reward (0.0 to 1.0).
        """
        try:
            # Get LLM resource
            llm = cast(agl.LLM, resources["main_llm"])

            # Get rollout context
            attempted_rollout = cast(agl.AttemptedRollout, rollout)
            base_url = llm.get_base_url(
                attempted_rollout.rollout_id,
                attempted_rollout.attempt.attempt_id,
            )

            logger.info(
                f"[Rollout {attempted_rollout.rollout_id}] START - "
                f"Problem type: {task['problem_type']}, Mode: {rollout.mode}"
            )

            # 1) Use Stage 1 model to extract strategy from few-shot examples
            extraction_temperature = llm.sampling_parameters.get("temperature", 0.7)
            extraction_max_tokens = llm.sampling_parameters.get("max_tokens", 4000)

            logger.info(f"[Rollout {attempted_rollout.rollout_id}] Calling strategy extractor...")
            strategy = await self.strategy_extractor.extract_strategy(
                task["fewshot_examples"],
                temperature=extraction_temperature,
                max_tokens=extraction_max_tokens,
                base_url=base_url,
                api_key=llm.api_key or "dummy-key",
            )
            
            # DEBUG: Print strategy extraction output
            logger.info(
                f"[Rollout {attempted_rollout.rollout_id}] Strategy extraction result: "
                f"{'SUCCESS' if strategy else 'FAILED'}, "
                f"length={len(strategy) if strategy else 0} chars"
            )
            if strategy:
                logger.debug(f"[Rollout {attempted_rollout.rollout_id}] Strategy content (first 200 chars): {strategy[:200]}...")

            if not strategy:
                logger.warning(
                    f"[Rollout {attempted_rollout.rollout_id}] Strategy extraction failed; "
                    "returning reward=0.0"
                )
                final_reward = 0.0
                reward_details: Dict[str, float] = {
                    "format": 0.0,
                    "coverage": 0.0,
                    "order": 0.0,
                    "binding": 0.0,
                    "intermediate": 0.0,
                    "consistency_total": 0.0,
                    "correctness": 0.0,
                    "final": 0.0,
                    "failure_reason": "no_strategy",  # type: ignore[assignment]
                }
                output = ""
                extracted_answer: Optional[str] = None
            else:
                # 2) Apply the extracted strategy with Stage 2 policy model
                system_prompt = get_strategy_application_system_prompt()
                user_prompt = format_strategy_application_prompt(strategy, task["problem"])

                logger.debug(
                    f"[Rollout {attempted_rollout.rollout_id}] User prompt length: "
                    f"{len(user_prompt)} chars"
                )

                logger.info(f"[Rollout {attempted_rollout.rollout_id}] Calling LLM for strategy application...")
                logger.info(f"[Rollout {attempted_rollout.rollout_id}] Using base_url: {base_url}")
                
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

                # Extract response text
                output = response.choices[0].message.content or ""
                
                # DEBUG: Print LLM output
                logger.info(
                    f"[Rollout {attempted_rollout.rollout_id}] LLM response received: "
                    f"length={len(output)} chars, "
                    f"empty={not output}"
                )
                if output:
                    logger.debug(f"[Rollout {attempted_rollout.rollout_id}] Output (first 200 chars): {output[:200]}...")
                else:
                    logger.error(f"[Rollout {attempted_rollout.rollout_id}] ERROR: LLM returned empty output!")
                
                # Compute reward
                final_reward, reward_details = compute_stage2_reward(
                    strategy=strategy,
                    problem=task["problem"],
                    answer=output,
                    ground_truth=task["ground_truth"],
                    consistency_threshold=self.consistency_threshold,
                    numeric_tolerance=self.numeric_tolerance,
                    f1_threshold=self.f1_threshold,
                    format_weight=self.format_weight,
                    consistency_weight=self.consistency_weight,
                    correctness_weight=self.correctness_weight,
                    coverage_weight=self.coverage_weight,
                    order_weight=self.order_weight,
                    binding_weight=self.binding_weight,
                    intermediate_weight=self.intermediate_weight,
                )
                
                # Extract answer
                extracted_answer = extract_answer(output)
            
            # Track batch statistics (for monitoring batch quality)
            # Use rollout_id prefix as batch identifier (VERL typically groups by prefix)
            batch_id = attempted_rollout.rollout_id.split('_')[0] if '_' in attempted_rollout.rollout_id else "unknown"
            if batch_id != self.current_batch_id:
                # New batch started, log previous batch statistics
                if self.current_batch_id and self.batch_rewards:
                    self._log_batch_statistics(self.current_batch_id)
                # Reset for new batch
                self.current_batch_id = batch_id
                self.batch_rewards = []
                self.batch_reward_details = []
            
            self.batch_rewards.append(final_reward)
            self.batch_reward_details.append(reward_details)
            
            # Check mode
            current_mode = rollout.mode if hasattr(rollout, "mode") else "unknown"
            is_validation = current_mode != "train"
            
            # Detect transition from validation to training
            if (
                self.last_rollout_mode is not None
                and self.last_rollout_mode != "train"
                and current_mode == "train"
            ):
                if self.validation_outputs and self.validation_output_dir:
                    if self.validation_step_counter == 0:
                        actual_step = 0
                    else:
                        actual_step = self.test_freq * self.validation_step_counter
                    
                    saved_path = self.save_validation_outputs(actual_step)
                    if saved_path:
                        logger.info(
                            f"Validation outputs saved (global_step={actual_step}): {saved_path}"
                        )
                    self.validation_step_counter += 1
            
            # Save validation outputs
            if is_validation and self.validation_output_dir:
                validation_output = {
                    "rollout_id": attempted_rollout.rollout_id,
                    "timestamp": datetime.now().isoformat(),
                    "problem_type": task["problem_type"],
                    "source_problem_type": task.get("source_problem_type"),
                    "input": {
                        "system_prompt": get_strategy_application_system_prompt(),
                        "user_prompt": format_strategy_application_prompt(
                            strategy if strategy else "",
                            task["problem"],
                        ),
                        "strategy": strategy if strategy else "",
                        "problem": task["problem"],
                        "ground_truth": task["ground_truth"],
                    },
                    "output": {
                        "raw_response": output,
                        "response_length": len(output),
                        "extracted_answer": extracted_answer if extracted_answer is not None else "",
                        "answer_length": len(extracted_answer) if extracted_answer else 0,
                    },
                    "reward": {
                        "final_reward": final_reward,
                        "format_reward": reward_details.get("format", 0.0),
                        "coverage_reward": reward_details.get("coverage", 0.0),
                        "order_reward": reward_details.get("order", 0.0),
                        "binding_reward": reward_details.get("binding", 0.0),
                        "intermediate_reward": reward_details.get("intermediate", 0.0),
                        "consistency_reward": reward_details.get("consistency_total", 0.0),
                        "correctness_reward": reward_details.get("correctness", 0.0),
                    },
                    "metadata": {
                        "model": llm.model,
                        "temperature": llm.sampling_parameters.get("temperature", 0.7),
                        "max_tokens": llm.sampling_parameters.get("max_tokens", 4000),
                        "rollout_mode": str(current_mode),
                    },
                }
                self.validation_outputs.append(validation_output)
                logger.debug(
                    f"[Rollout {attempted_rollout.rollout_id}] Collected validation output "
                    f"(mode={current_mode}, total={len(self.validation_outputs)})"
                )
            
            # Update last mode
            self.last_rollout_mode = current_mode
            
            # Save rollout trace
            if self.traces_file:
                trace = {
                    "rollout_id": attempted_rollout.rollout_id,
                    "timestamp": datetime.now().isoformat(),
                    "mode": rollout.mode,
                    "problem_type": task["problem_type"],
                    "source_problem_type": task.get("source_problem_type"),
                    "input": {
                        "system_prompt": get_strategy_application_system_prompt(),
                        "user_prompt": format_strategy_application_prompt(
                            strategy if strategy else "",
                            task["problem"],
                        ),
                        "strategy": strategy if strategy else "",
                        "problem": task["problem"],
                        "ground_truth": task["ground_truth"],
                    },
                    "output": {
                        "raw_response": output,
                        "response_length": len(output),
                        "extracted_answer": extracted_answer if extracted_answer is not None else "",
                        "answer_length": len(extracted_answer) if extracted_answer else 0,
                    },
                    "reward": {
                        "final_reward": final_reward,
                        "format_reward": reward_details.get("format", 0.0),
                        "coverage_reward": reward_details.get("coverage", 0.0),
                        "order_reward": reward_details.get("order", 0.0),
                        "binding_reward": reward_details.get("binding", 0.0),
                        "intermediate_reward": reward_details.get("intermediate", 0.0),
                        "consistency_reward": reward_details.get("consistency_total", 0.0),
                        "correctness_reward": reward_details.get("correctness", 0.0),
                    },
                    "metadata": {
                        "model": llm.model,
                        "temperature": llm.sampling_parameters.get("temperature", 0.7),
                        "max_tokens": llm.sampling_parameters.get("max_tokens", 4000),
                    },
                }
                try:
                    with open(self.traces_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(trace, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.warning(f"Failed to save rollout trace: {e}")
            
            # Log complete output if configured
            if self.save_full_output:
                logger.info(
                    f"[Rollout {attempted_rollout.rollout_id}] FULL OUTPUT\n"
                    f"Problem Type: {task['problem_type']}\n"
                    f"Mode: {rollout.mode}\n"
                    f"Output Length: {len(output)} chars\n"
                    f"Extracted Answer: {extracted_answer if extracted_answer else '(No answer extracted)'}\n"
                    f"Final Reward: {final_reward}\n"
                    f"Reward Details: {reward_details}\n"
                    f"{'='*80}\n"
                    f"{output}\n"
                    f"{'='*80}"
                )
            
            # Terminal logs
            logger.info(
                f"[Rollout {attempted_rollout.rollout_id}] COMPLETE - "
                f"Reward: {final_reward}, Answer: {extracted_answer[:50] if extracted_answer else 'None'}..."
            )
            
            if final_reward == 0.0:
                logger.warning(
                    f"[Rollout {attempted_rollout.rollout_id}] ZERO REWARD - "
                    f"Format: {reward_details.get('format', 0.0)}, "
                    f"Consistency: {reward_details.get('consistency_total', 0.0)}, "
                    f"Correctness: {reward_details.get('correctness', 0.0)}"
                )
            else:
                logger.info(
                    f"[Rollout {attempted_rollout.rollout_id}] SUCCESS - "
                    f"Reward: {final_reward}"
                )
            
            # Record reward to spans for AgentLightning tracking
            agl.emit_reward(final_reward)
            
            return float(final_reward)
            
        except Exception as e:
            logger.error(
                f"[Rollout {attempted_rollout.rollout_id}] ERROR - "
                f"Problem type: {task['problem_type']}, Exception: {e}",
                exc_info=True,
            )
            logger.info(f"[Rollout {attempted_rollout.rollout_id}] RETURNING reward=0.0 (error)")
            
            # Record reward to spans for AgentLightning tracking
            try:
                agl.emit_reward(0.0)
            except Exception as reward_err:
                logger.warning(f"[Rollout {attempted_rollout.rollout_id}] Failed to emit reward: {reward_err}")
            
            # Track error in batch statistics
            batch_id = attempted_rollout.rollout_id.split('_')[0] if '_' in attempted_rollout.rollout_id else "unknown"
            if batch_id != self.current_batch_id:
                if self.current_batch_id and self.batch_rewards:
                    self._log_batch_statistics(self.current_batch_id)
                self.current_batch_id = batch_id
                self.batch_rewards = []
                self.batch_reward_details = []
            self.batch_rewards.append(0.0)
            self.batch_reward_details.append({"format": 0.0, "consistency": 0.0, "correctness": 0.0, "failure_reason": "error"})
            
            return 0.0
    
    def _log_batch_statistics(self, batch_id: str) -> None:
        """Log batch statistics for monitoring and debugging.
        
        Args:
            batch_id: Identifier for the batch.
        """
        if not self.batch_rewards:
            return
        
        # Calculate statistics
        num_samples = len(self.batch_rewards)
        positive_rewards = [r for r in self.batch_rewards if r > 0]
        negative_rewards = [r for r in self.batch_rewards if r < 0]
        zero_rewards = [r for r in self.batch_rewards if r == 0.0]
        
        # Count failure reasons
        failure_reasons: Dict[str, int] = {}
        for detail in self.batch_reward_details:
            reason = detail.get("failure_reason", "none")
            if reason and reason != "none":
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        # Calculate reward distribution
        reward_dist = {
            "format_fail": len([r for r in self.batch_rewards if r == -0.1]),
            "consistency_fail": len([r for r in self.batch_rewards if r == -0.05]),
            "no_strategy": len([r for r in self.batch_rewards if r == -0.15]),
            "zero_correctness": len([r for r in self.batch_rewards if r == 0.0 and r not in [-0.1, -0.05, -0.15]]),
            "f1_match": len([r for r in self.batch_rewards if r == 0.5]),
            "numeric_match": len([r for r in self.batch_rewards if r == 0.8]),
            "exact_match": len([r for r in self.batch_rewards if r == 1.0]),
        }
        
        # Log statistics
        logger.info(
            f"[Batch {batch_id}] Statistics: "
            f"size={num_samples}, "
            f"positive={len(positive_rewards)}, "
            f"negative={len(negative_rewards)}, "
            f"zero={len(zero_rewards)}, "
            f"mean={statistics.mean(self.batch_rewards):.3f}, "
            f"max={max(self.batch_rewards):.3f}, "
            f"min={min(self.batch_rewards):.3f}"
        )
        logger.info(
            f"[Batch {batch_id}] Reward distribution: {reward_dist}"
        )
        if failure_reasons:
            logger.info(
                f"[Batch {batch_id}] Failure reasons: {failure_reasons}"
            )
        
        # Check if batch has learnable rewards
        has_learnable = len(positive_rewards) > 0
        if not has_learnable:
            logger.warning(
                f"[Batch {batch_id}] WARNING: No learnable rewards (all rewards <= 0)"
            )

