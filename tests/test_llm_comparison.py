"""Test LLM comparison functionality"""

import pytest
import dspy
from unittest.mock import patch

try:
    from dspy_elo.llm_comparison import compare_llm_outputs, ComparisonModule
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dspy_elo.llm_comparison import compare_llm_outputs, ComparisonModule


def test_comparison_module_initialization():
    """Test the DSPy comparison module initializes correctly"""
    module = ComparisonModule()
    assert hasattr(module, "compare")
    assert isinstance(module.compare, dspy.ChainOfThought)


def test_comparison_module_forward():
    """Test the module's forward pass returns valid results"""
    module = ComparisonModule()
    with patch.object(module.compare, "__call__", return_value="1"):
        result = module("First output", "Second output")
        assert result in (1, 2)


def test_compare_llm_outputs_returns_tuple():
    """Test comparison returns (winner, loser) tuple"""
    with patch("dspy_elo.llm_comparison.comparer", return_value=1):
        result = compare_llm_outputs("A", "B")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result == (1, 2)


@pytest.mark.integration
def test_compare_llm_outputs_integration():
    """Test actual LLM comparison with clear differences"""
    long_response = "Detailed response with examples and explanations."
    short_response = "Short"
    
    winner, loser = compare_llm_outputs(long_response, short_response)
    assert winner != loser
    assert winner in (1, 2)
    assert loser in (1, 2)
