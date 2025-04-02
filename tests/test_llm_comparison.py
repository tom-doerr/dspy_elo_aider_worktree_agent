"""Test LLM comparison functionality"""

import pytest

try:
    from dspy_elo.llm_comparison import compare_llm_outputs
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dspy_elo.llm_comparison import compare_llm_outputs


def test_compare_llm_outputs_returns_tuple():
    """Test comparison returns (winner, loser) tuple"""
    result = compare_llm_outputs("Hello", "Hi there")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] in (1, 2)
    assert result[1] in (1, 2)
    assert result[0] != result[1]


def test_compare_llm_outputs_consistent():
    """Test same inputs produce consistent results"""
    result1 = compare_llm_outputs("A", "B")
    result2 = compare_llm_outputs("A", "B")
    assert result1 == result2


@pytest.mark.skip(reason="Integration test - requires actual LLM calls")
def test_compare_llm_outputs_integration():
    """Test actual LLM comparison (skipped by default)"""
    winner, loser = compare_llm_outputs("Detailed response", "Short response")
    assert winner != loser
