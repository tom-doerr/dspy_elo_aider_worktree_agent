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
    
    # Check basic output structure
    assert "Initial ratings" in captured.out
    assert "After comparison" in captured.out
    assert "Final ratings" in captured.out
    
    # Check it shows rating changes
    assert "->" in captured.out
