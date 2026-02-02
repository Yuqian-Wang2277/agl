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
1. Analyze the given examples carefully to identify common patterns
2. Extract the underlying problem-solving approach that is shared across all examples
3. Formulate a strategy that demonstrates meta-learning capabilities - the ability to learn how to learn
4. Your strategy should be general and transferable, applicable to similar problems beyond the given examples
5. The strategy should be executable - clear enough that someone could follow it to solve new problems
6. Wrap your strategy within <strategy>...</strategy> tags
7. **CRITICAL**: Output ONLY the <strategy> tags with your answer inside - nothing before, nothing after

Key principles for strategy extraction:
- Focus on the meta-level: What is the general approach, not just the specific solution?
- Identify reusable patterns: What can be applied to similar but different problems?
- Ensure transferability: The strategy should work for problems of the same type, not just the exact examples given
- Maintain clarity: Use clear, step-by-step instructions that can guide problem-solving

Output format (follow exactly):
<strategy>
[Your problem-solving strategy here - be comprehensive and detailed as needed to capture the full strategy]
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


