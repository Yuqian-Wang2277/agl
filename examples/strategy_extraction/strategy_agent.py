# Copyright (c) Microsoft. All rights reserved.

"""Strategy extraction agent implementation."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, cast

from openai import AsyncOpenAI

import agentlightning as agl

from .prompts import format_user_prompt, get_system_prompt
from .reward import compute_format_reward, extract_strategy

logger = logging.getLogger(__name__)


class StrategyTask(TypedDict):
    """Task structure for strategy extraction.
    
    Attributes:
        problem_type: The type/category of the problem.
        examples: List of few-shot examples, each with 'input' and 'target'.
        num_shots: Number of examples included.
    """
    problem_type: str
    examples: List[Dict[str, Any]]
    num_shots: int


class StrategyExtractionAgent(agl.LitAgent[StrategyTask]):
    """Agent for extracting problem-solving strategies from examples.
    
    This agent analyzes few-shot examples and generates a general,
    executable, and transferable problem-solving strategy wrapped
    in <strategy>...</strategy> tags.
    """
    
    def __init__(self, save_full_output: bool = True, rollout_traces_dir: Optional[str] = None, validation_output_dir: Optional[str] = None, experiment_id: Optional[str] = None, test_freq: int = 50) -> None:
        """Initialize the strategy extraction agent.
        
        Args:
            save_full_output: Whether to save complete model output to log file.
            rollout_traces_dir: Directory to save rollout traces (input/output pairs). If None, traces are not saved.
            validation_output_dir: Base directory to save validation outputs. If None, validation outputs are not saved separately.
            experiment_id: Unique experiment identifier (format: YYYYMMDD_HHMMSS). Used to organize outputs by experiment.
            test_freq: Frequency of validation (every N steps). Used to calculate actual step numbers for validation outputs.
        """
        super().__init__()
        self.save_full_output = save_full_output
        self.experiment_id = experiment_id
        self.rollout_traces_dir = rollout_traces_dir
        self.validation_output_dir = validation_output_dir
        self.test_freq = test_freq
        
        if rollout_traces_dir:
            # Create experiment-specific subdirectory if experiment_id is provided
            if experiment_id:
                rollout_traces_dir = os.path.join(rollout_traces_dir, experiment_id)
            os.makedirs(rollout_traces_dir, exist_ok=True)
            self.traces_file = os.path.join(rollout_traces_dir, f"rollout_traces.jsonl")
            logger.info(f"Rollout traces will be saved to: {self.traces_file}")
        else:
            self.traces_file = None
        
        # Initialize validation output tracking
        self.validation_outputs: List[Dict[str, Any]] = []
        self.current_validation_step: Optional[int] = None
        self.last_rollout_mode: Optional[str] = None  # Track mode transitions
        self.validation_step_counter: int = 0  # Counter for validation steps (0 = before_train, 1+ = step numbers)
        
        if validation_output_dir:
            # Create experiment-specific subdirectory if experiment_id is provided
            if experiment_id:
                validation_output_dir = os.path.join(validation_output_dir, experiment_id)
            os.makedirs(validation_output_dir, exist_ok=True)
            self.validation_output_dir = validation_output_dir  # Update to experiment-specific path
            logger.info(f"Validation outputs will be saved to: {validation_output_dir}")
        
        logger.info(f"StrategyExtractionAgent initialized (save_full_output={save_full_output}, rollout_traces_dir={rollout_traces_dir}, validation_output_dir={validation_output_dir}, experiment_id={experiment_id})")
    
    def save_validation_outputs(self, step: int) -> Optional[str]:
        """Save collected validation outputs to JSON file.
        
        Args:
            step: Current training step number (0 for before_train validation, >0 for actual training step).
            
        Returns:
            Path to saved file if successful, None otherwise.
        """
        if not self.validation_output_dir or not self.validation_outputs:
            return None
        
        # Save validation outputs with actual step number in filename
        if step == 0:
            # Validation before training starts
            output_file = os.path.join(self.validation_output_dir, f"validation_global_step0.json")
        else:
            # Validation at specific training step (format: validation_global_step{step}.json)
            output_file = os.path.join(self.validation_output_dir, f"validation_global_step{step}.json")
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.validation_outputs, f, ensure_ascii=False, indent=2)
            saved_count = len(self.validation_outputs)
            logger.info(f"Saved {saved_count} validation outputs to: {output_file}")
            # Clear collected outputs after saving
            self.validation_outputs = []
            return output_file
        except Exception as e:
            logger.error(f"Failed to save validation outputs: {e}")
            return None
    
    async def rollout_async(
        self,
        task: StrategyTask,
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float:
        """Execute a rollout to extract strategy from examples.
        
        Args:
            task: Strategy extraction task containing examples.
            resources: Named resources including the LLM.
            rollout: Rollout metadata.
            
        Returns:
            Format reward (1.0 for correct format, 0.0 otherwise).
        """
        # Get LLM resource
        llm = cast(agl.LLM, resources["main_llm"])
        
        # Get rollout context for ProxyLLM
        attempted_rollout = cast(agl.AttemptedRollout, rollout)
        base_url = llm.get_base_url(attempted_rollout.rollout_id, attempted_rollout.attempt.attempt_id)
        
        logger.info(
            f"[Rollout {attempted_rollout.rollout_id}] START - "
            f"Problem type: {task['problem_type']}, Shots: {task['num_shots']}, Mode: {rollout.mode}"
        )
        
        # Build prompts
        system_prompt = get_system_prompt()
        user_prompt = format_user_prompt(task["examples"])
        
        logger.debug(
            f"[Rollout {attempted_rollout.rollout_id}] User prompt length: {len(user_prompt)} chars"
        )
        
        # Create OpenAI client
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=llm.api_key or "dummy-key",
        )
        
        try:
            # Call LLM
            response = await client.chat.completions.create(
                model=llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=llm.sampling_parameters.get("temperature", 0.7),
                max_tokens=llm.sampling_parameters.get("max_tokens", 4000),  # Reasonable output length for strategy extraction
            )
            
            # Extract response text
            output = response.choices[0].message.content or ""
            
            # Compute format reward
            reward = compute_format_reward(output)
            
            # Extract strategy content from <strategy>...</strategy> tags
            extracted_strategy = extract_strategy(output)
            if extracted_strategy is None:
                logger.warning(
                    f"[Rollout {attempted_rollout.rollout_id}] Failed to extract strategy - "
                    f"no valid <strategy> tags found in output"
                )
            
            # Save validation outputs separately if in validation mode
            # Check mode: could be "val", "validation", or check if it's not "train"
            current_mode = rollout.mode if hasattr(rollout, 'mode') else "unknown"
            is_validation = current_mode != "train"
            
            # Detect transition from validation to training (validation just completed)
            if self.last_rollout_mode is not None and self.last_rollout_mode != "train" and current_mode == "train":
                # Validation just ended, save collected outputs
                if self.validation_outputs and self.validation_output_dir:
                    # Calculate actual step number:
                    # - counter = 0: before_train validation (step = 0)
                    # - counter = 1: first validation (step = test_freq * 1 = 50)
                    # - counter = 2: second validation (step = test_freq * 2 = 100)
                    # - etc.
                    if self.validation_step_counter == 0:
                        actual_step = 0  # Before training
                    else:
                        actual_step = self.test_freq * self.validation_step_counter
                    
                    saved_path = self.save_validation_outputs(actual_step)
                    if saved_path:
                        logger.info(f"Validation outputs saved after validation completed (global_step={actual_step}): {saved_path}")
                    # Increment counter for next validation
                    self.validation_step_counter += 1
            
            if is_validation and self.validation_output_dir:
                validation_output = {
                    "rollout_id": attempted_rollout.rollout_id,
                    "timestamp": datetime.now().isoformat(),
                    "problem_type": task["problem_type"],
                    "num_shots": task["num_shots"],
                    "input": {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "examples": task["examples"],
                    },
                    "output": {
                        "raw_response": output,
                        "response_length": len(output),
                        "extracted_strategy": extracted_strategy if extracted_strategy is not None else "",
                    },
                    "reward": reward,
                    "metadata": {
                        "model": llm.model,
                        "temperature": llm.sampling_parameters.get("temperature", 0.7),
                        "max_tokens": llm.sampling_parameters.get("max_tokens", 4000),
                        "rollout_mode": str(current_mode),
                    },
                }
                self.validation_outputs.append(validation_output)
                logger.debug(f"[Rollout {attempted_rollout.rollout_id}] Collected validation output (mode={current_mode}, total={len(self.validation_outputs)})")
            
            # Update last mode
            self.last_rollout_mode = current_mode
            
            # Save rollout trace (input/output pair) if configured
            if self.traces_file:
                trace = {
                    "rollout_id": attempted_rollout.rollout_id,
                    "timestamp": datetime.now().isoformat(),
                    "mode": rollout.mode,
                    "problem_type": task["problem_type"],
                    "num_shots": task["num_shots"],
                    "input": {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "examples": task["examples"],
                    },
                    "output": {
                        "raw_response": output,
                        "response_length": len(output),
                        "extracted_strategy": extracted_strategy if extracted_strategy is not None else "",
                    },
                    "reward": reward,
                    "metadata": {
                        "model": llm.model,
                        "temperature": llm.sampling_parameters.get("temperature", 0.7),
                        "max_tokens": llm.sampling_parameters.get("max_tokens", 1000),
                    },
                }
                try:
                    with open(self.traces_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(trace, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.warning(f"Failed to save rollout trace: {e}")
            
            # Log complete output to file if configured
            if self.save_full_output:
                logger.info(
                    f"[Rollout {attempted_rollout.rollout_id}] FULL OUTPUT\n"
                    f"Problem Type: {task['problem_type']}\n"
                    f"Num Shots: {task['num_shots']}\n"
                    f"Mode: {rollout.mode}\n"
                    f"Output Length: {len(output)} chars\n"
                    f"Computed Reward: {reward}\n"
                    f"{'='*80}\n"
                    f"{output}\n"
                    f"{'='*80}\n"
                    f"Extracted Strategy:\n"
                    f"{'='*80}\n"
                    f"{extracted_strategy if extracted_strategy is not None else '(No strategy extracted)'}\n"
                    f"{'='*80}"
                )
            
            # Terminal logs (INFO/WARNING level - concise)
            logger.info(
                f"[Rollout {attempted_rollout.rollout_id}] COMPLETE - "
                f"Reward: {reward}, Output: {len(output)} chars, Problem: {task['problem_type']}"
            )
            
            if reward == 0.0:
                logger.warning(
                    f"[Rollout {attempted_rollout.rollout_id}] ZERO REWARD - "
                    f"Snippet: {output[:150]}..."
                )
            else:
                logger.info(
                    f"[Rollout {attempted_rollout.rollout_id}] SUCCESS - Valid strategy extracted"
                )
            
            # Ensure we return a proper float
            final_reward = float(reward)
            logger.info(
                f"[Rollout {attempted_rollout.rollout_id}] RETURNING reward={final_reward}"
            )
            return final_reward
            
        except Exception as e:
            logger.error(
                f"[Rollout {attempted_rollout.rollout_id}] ERROR - "
                f"Problem type: {task['problem_type']}, Exception: {e}",
                exc_info=True,
            )
            logger.info(f"[Rollout {attempted_rollout.rollout_id}] RETURNING reward=0.0 (error)")
            return 0.0


