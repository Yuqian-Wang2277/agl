# Copyright (c) Microsoft. All rights reserved.

"""Strategy extraction agent implementation."""

import logging
from typing import Any, Dict, List, TypedDict, cast

from openai import AsyncOpenAI

import agentlightning as agl

from .prompts import format_user_prompt, get_system_prompt
from .reward import compute_format_reward

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
    
    def __init__(self) -> None:
        """Initialize the strategy extraction agent."""
        super().__init__()
        logger.info("StrategyExtractionAgent initialized")
    
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
                max_tokens=llm.sampling_parameters.get("max_tokens", 600),  # Reasonable output length for strategy extraction
            )
            
            # Extract response text
            output = response.choices[0].message.content or ""
            
            # Compute format reward
            reward = compute_format_reward(output)
            
            # Log complete output to file (DEBUG level - not shown in terminal by default)
            logger.debug(
                f"[Rollout {attempted_rollout.rollout_id}] FULL OUTPUT\n"
                f"Problem Type: {task['problem_type']}\n"
                f"Num Shots: {task['num_shots']}\n"
                f"Mode: {rollout.mode}\n"
                f"Output Length: {len(output)} chars\n"
                f"Computed Reward: {reward}\n"
                f"{'='*80}\n"
                f"{output}\n"
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


