# Copyright (c) Microsoft. All rights reserved.

"""Prompt templates for strategy application training."""


def get_strategy_application_system_prompt() -> str:
    """Get the system prompt for strategy application.
    
    Returns:
        System prompt instructing the model to apply a strategy to solve a problem.
    """
    return """You are an expert problem solver. Your task is to apply a given problem-solving strategy to solve a specific problem.

Requirements:
1. Carefully read and understand the provided strategy
2. Apply the strategy step-by-step to solve the given problem
3. Follow the strategy exactly as described
4. Output your final answer wrapped in <answer>...</answer> tags
5. **CRITICAL**: Output ONLY the <answer> tags with your answer inside - nothing before, nothing after

Key principles:
- Follow the strategy precisely: Each step in the strategy should be reflected in your solution
- Be methodical: Work through the problem systematically using the strategy
- Ensure correctness: Your answer should be accurate and complete
- Use the exact format: Wrap your final answer in <answer>...</answer> tags

Output format (follow exactly):
<answer>
[Your final answer here]
</answer>"""


def format_strategy_application_prompt(strategy: str, problem: str) -> str:
    """Format strategy and problem into a user prompt.
    
    Args:
        strategy: The problem-solving strategy to apply.
        problem: The problem to solve.
        
    Returns:
        Formatted user prompt with strategy and problem.
    """
    return f"""Here is a problem-solving strategy:

{strategy}

Now, apply this strategy to solve the following problem:

{problem}

Use the strategy step-by-step to solve the problem, then provide your final answer wrapped in <answer>...</answer> tags."""

