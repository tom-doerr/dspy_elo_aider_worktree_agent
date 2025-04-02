"""Compare LLM outputs using DeepSeek API"""
from typing import Tuple

def compare_llm_outputs(output1: str, output2: str) -> Tuple[int, int]:
    """
    Compare two LLM outputs and return which is better.
    Returns tuple of (winner, loser) where values are 1 or 2.
    
    Args:
        output1: First LLM output
        output2: Second LLM output
        
    Returns:
        Tuple of (winner, loser) indices (1 or 2)
    """
    # Currently using simple length comparison as placeholder
    if len(output1) > len(output2):
        return (1, 2)
    return (2, 1)
