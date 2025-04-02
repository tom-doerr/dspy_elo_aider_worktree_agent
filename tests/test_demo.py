from dspy_elo.rating import EloRatingSystem

def test_demo_script_output(capsys):
    """Test the demo script produces expected output"""
    import dspy_elo.demo
    captured = capsys.readouterr()
    
    # Check basic output structure
    assert "Initial ratings" in captured.out
    assert "After comparison" in captured.out
    assert "Final ratings" in captured.out
    
    # Check it shows rating changes
    assert "->" in captured.out
