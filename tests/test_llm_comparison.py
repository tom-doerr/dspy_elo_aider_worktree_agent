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
    result = compare_llm_outputs("A long detailed response", "Short")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] in (1, 2)
    assert result[1] in (1, 2)
    assert result[0] != result[1]


def test_compare_llm_outputs_consistent():
    """Test same inputs produce consistent results with clear differences"""
    long_response = "Here is a detailed answer with examples and explanations."
    short_response = "IDK"
    
    result1 = compare_llm_outputs(long_response, short_response)
    result2 = compare_llm_outputs(long_response, short_response)
    # Allow either consistent result or fallback to length comparison
    assert result1 in {(1, 2), (2, 1)}, "Invalid comparison result"
    assert result1 == result2, "Should consistently pick same winner"


def test_compare_llm_outputs_types():
    """Test comparison returns proper types"""
    result = compare_llm_outputs("A", "B")
    assert isinstance(result[0], int)
    assert isinstance(result[1], int)


@pytest.mark.skip(reason="Integration test - requires actual LLM calls")
def test_compare_llm_outputs_integration():
    """Test actual LLM comparison (skipped by default)"""
    winner, loser = compare_llm_outputs("Detailed response", "Short response")
    assert winner != loser
