"""Compare LLM outputs using DeepSeek model via DSPy"""

from typing import Tuple
import dspy
import re

# Configure DSPy with DeepSeek
lm = dspy.LM(model="deepseek/deepseek-chat")
dspy.configure(lm=lm)


class ComparisonModule(dspy.Module):
    """DSPy module for comparing two LLM outputs"""

    def __init__(self):
        super().__init__()
        self.compare = dspy.ChainOfThought("output1, output2 -> winner")

    def forward(self, output1: str, output2: str) -> int:
        """Return 1 if output1 is better, 2 if output2 is better"""
        result = self.compare(output1=output1, output2=output2)
        # Extract first numerical value from the response
        if match := re.search(r"\b(1|2)\b", result.winner):
            return int(match.group(1))
        # Fallback comparison if no valid number found
        return 1 if len(output1) > len(output2) else 2


comparer = ComparisonModule()


def compare_llm_outputs(output1: str, output2: str) -> Tuple[int, int]:
    """
    Compare two LLM outputs using DeepSeek model via DSPy.
    Returns tuple of (winner, loser) where values are 1 or 2.
    """
    winner = comparer(output1, output2)
    return (winner, 2 if winner == 1 else 1)
