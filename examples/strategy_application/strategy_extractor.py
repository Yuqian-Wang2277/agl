# Copyright (c) Microsoft. All rights reserved.

"""Strategy extractor for loading stage 1 model and extracting strategies."""

import logging
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from examples.strategy_extraction.prompts import format_user_prompt, get_system_prompt

logger = logging.getLogger(__name__)


class StrategyExtractor:
    """Extractor for loading stage 1 model and extracting strategies from few-shot examples.
    
    This class encapsulates the strategy extraction logic from stage 1, allowing
    stage 2 to extract strategies using the trained stage 1 model.
    """
    
    def __init__(
        self,
        stage1_model_path: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the strategy extractor.
        
        Args:
            stage1_model_path: Path to the stage 1 model (used for strategy extraction).
            base_url: Base URL for LLM API (for OpenAI-compatible API).
            api_key: API key for LLM API.
        """
        self.stage1_model_path = stage1_model_path
        self.base_url = base_url
        self.api_key = api_key or "dummy-key"
        
        logger.info(f"StrategyExtractor initialized with model: {stage1_model_path}")
    
    async def extract_strategy(
        self,
        examples: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Optional[str]:
        """Extract strategy from few-shot examples using stage 1 model.
        
        Args:
            examples: List of few-shot examples, each with 'input' and 'target'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens for generation.
            base_url: Optional override for LLM base URL (per-rollout).
            api_key: Optional override for API key (per-rollout).
            
        Returns:
            Extracted strategy string, or None if extraction failed.
        """
        effective_base_url = base_url or self.base_url
        effective_api_key = api_key or self.api_key

        if not effective_base_url:
            raise ValueError("base_url must be provided before extracting strategies")
        
        # Build prompts using stage 1 format
        system_prompt = get_system_prompt()
        user_prompt = format_user_prompt(examples)
        
        # Create OpenAI client
        client = AsyncOpenAI(
            base_url=effective_base_url,
            api_key=effective_api_key,
        )
        
        try:
            # Call LLM using the stage 1 model path
            response = await client.chat.completions.create(
                model=self.stage1_model_path,  # Use stage 1 model path as model identifier
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            output = response.choices[0].message.content or ""
            
            # Extract strategy from <strategy>...</strategy> tags
            from examples.strategy_extraction.reward import extract_strategy
            extracted = extract_strategy(output)
            
            if extracted is None:
                logger.warning(f"Failed to extract strategy from output: {output[:200]}...")
            
            return extracted
            
        except Exception as e:
            logger.error(f"Error extracting strategy: {e}", exc_info=True)
            return None
    
    async def extract_strategies_batch(
        self,
        examples_list: List[List[Dict[str, Any]]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> List[Optional[str]]:
        """Extract strategies from multiple sets of examples in batch.
        
        Args:
            examples_list: List of example sets, each set is a list of examples.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens for generation.
            
        Returns:
            List of extracted strategies (may contain None for failed extractions).
        """
        import asyncio
        
        tasks = [
            self.extract_strategy(
                examples,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=base_url,
                api_key=api_key,
            )
            for examples in examples_list
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        strategies = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error extracting strategy for batch {i}: {result}")
                strategies.append(None)
            else:
                strategies.append(result)
        
        return strategies

