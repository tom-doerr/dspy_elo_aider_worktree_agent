"""Compare LLM outputs using DSPy module"""

from typing import Tuple
import dspy

# Configure DSPy with DeepSeek and disable caching
lm = dspy.LM(model="deepseek/deepseek-chat")
dspy.configure(lm=lm, no_cache=True)


class ComparisonModule(dspy.Module):
    """DSPy module that compares two text outputs"""

    def __init__(self):
        super().__init__()
        self.compare = dspy.ChainOfThought("output1, output2 -> winner")

    def forward(self, output1: str, output2: str) -> int:
        """Return 1 if output1 is better, 2 if output2 is better"""
        result = self.compare(output1=output1, output2=output2)
        return 1 if "1" in result.winner else 2


# Global instance of the comparison module
comparer = ComparisonModule()


def compare_llm_outputs(output1: str, output2: str) -> Tuple[int, int]:
    """
    Compare two LLM outputs using DSPy module.
    Returns:
        Tuple of (winner, loser) where values are 1 or 2
    """
    winner = comparer(output1, output2)
    return (winner, 2 if winner == 1 else 1)
