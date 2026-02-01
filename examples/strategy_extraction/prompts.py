# Copyright (c) Microsoft. All rights reserved.

"""Prompt templates for strategy extraction training."""

from typing import Any, Dict, List


def get_system_prompt() -> str:
    """Get the system prompt for strategy extraction.
    
    Returns:
        System prompt instructing the model to extract problem-solving strategies.
    """
    return """You are an expert at analyzing problem-solving patterns. Your task is to examine a set of example problems and their solutions, then extract a general, executable, and transferable problem-solving strategy.

Requirements:
1. Analyze the given examples carefully
2. Identify the common problem-solving approach
3. Summarize the strategy in a clear, step-by-step manner
4. Your strategy should be general enough to apply to similar problems
5. Wrap your strategy within <strategy>...</strategy> tags
6. **IMPORTANT**: Keep your response concise and under 500 tokens
7. **CRITICAL**: Do NOT use any other XML-style tags like <think>, <reasoning>, etc.
8. **CRITICAL**: Output ONLY the <strategy> tags with your answer inside - nothing before, nothing after

Output format (follow exactly):
<strategy>
[Your problem-solving strategy here - be concise and focused]
</strategy>"""


def format_user_prompt(examples: List[Dict[str, Any]]) -> str:
    """Format few-shot examples into a user prompt.
    
    Args:
        examples: List of examples, each containing 'input' and 'target' fields.
        
    Returns:
        Formatted user prompt with examples.
    """
    prompt_parts = ["Here are some example problems and their solutions:\n"]
    
    for i, example in enumerate(examples, 1):
        input_text = example.get("input", "")
        target = example.get("target", [])
        
        # Handle target as list or string
        if isinstance(target, list):
            target_text = target[0] if target else ""
        else:
            target_text = str(target)
        
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append(f"Problem: {input_text}")
        prompt_parts.append(f"Solution: {target_text}")
        prompt_parts.append("")  # Empty line for readability
    
    prompt_parts.append("Based on these examples, extract the problem-solving strategy.")
    
    return "\n".join(prompt_parts)


