"""Test README documentation examples work as advertised"""

from pathlib import Path
import pandas as pd

try:
    from dspy_elo import train_elo_predictor, compare_llm_outputs
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dspy_elo import train_elo_predictor, compare_llm_outputs


def test_readme_training_example(tmp_path):
    """Test the training example from README works"""
    # Create minimal dataset matching README example
    data = pd.DataFrame({
        "text": ["Detailed analysis", "Brief summary"],
        "rating": [8, 3]  # Using 1-9 scale as documented
    })
    
    # Train as shown in README
    predictor = train_elo_predictor(data, output_dir=tmp_path)
    
    # Verify it works
    result = predictor.predict("Detailed analysis", "Brief summary")
    assert result == (1, 2)  # Higher rated should win


def test_readme_inference_example():
    """Test the inference example from README works"""
    # These are new samples not in training data
    winner, loser = compare_llm_outputs(
        "Well-reasoned argument with citations",
        "Short opinion without evidence"
    )
    assert winner in (1, 2)
    assert loser in (1, 2)
    assert winner != loser


def test_readme_custom_dataset_example(tmp_path):
    """Test custom dataset example from README works"""
    # Create custom dataset as shown in README
    custom_data = pd.DataFrame({
        "text": ["Best response", "Worst response"],
        "rating": [9, 1]  # Using min/max ratings
    })
    
    # Train and verify
    predictor = train_elo_predictor(custom_data, output_dir=tmp_path)
    result = predictor.predict("Best response", "Worst response")
    assert result == (1, 2)
