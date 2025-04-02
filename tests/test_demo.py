import re

try:
    from dspy_elo.demo import run_demo
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dspy_elo.demo import run_demo


def test_demo_script_output(capsys):
    """Test the demo script produces expected output"""
    run_demo()
    captured = capsys.readouterr()
    output = captured.out

    # Check all expected sections are present
    sections = [
        "ELO Rating System Demo - Live LLM Comparisons",
        "Initial Ratings:",
        "Round 1 Comparison:",
        "LLM Response A:",
        "LLM Response B:",
        "Evaluating with DeepSeek...",
        "Rating Updates:",
        "Final Ratings:",
    ]
    for section in sections:
        assert section in output, f"Missing section: {section}"

    # Check rating changes are shown
    assert "->" in output

    # Verify ratings are numbers
    ratings = re.findall(r"\d+\.?\d*", output)
    assert len(ratings) >= 4, "Should show at least 4 rating values"
    assert all(float(r) > 0 for r in ratings), "Ratings should be positive"
