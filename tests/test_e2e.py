"""End-to-end tests for DSPy ELO predictor matching spec in info.md"""

import pytest
import pandas as pd
import sys
from pathlib import Path

try:
    from dspy_elo import train_elo_predictor, compare_llm_outputs
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dspy_elo import train_elo_predictor, compare_llm_outputs

@pytest.fixture(name="training_data")
def training_data_fixture():
    """Minimal valid training data"""
    return pd.DataFrame({
        "text": ["Excellent response", "Poor response"],
        "rating": [9, 1]  # Using min/max ratings for clear test cases
    })

def test_bootstrap_fewshot_training(tmp_path, training_data):
    """Test training with bootstrap few-shot as per spec"""
    predictor = train_elo_predictor(sample_data, output_dir=tmp_path)
    
    # Verify basic functionality
    assert hasattr(predictor, 'predict')
    assert hasattr(predictor, 'get_rating_difference')
    
    # Test prediction directionality
    result = predictor.predict("Excellent response", "Poor response")
    assert result == (1, 2)  # Higher rated should win

def test_inference_on_new_samples(training_data):
    """Test comparing new LLM outputs not in training data"""
    # These are new samples not in any training data
    winner, loser = compare_llm_outputs(
        "Detailed analysis with examples", 
        "Vague response"
    )
    assert winner in (1, 2)
    assert loser in (1, 2)
    assert winner != loser

def test_custom_dataset_workflow(tmp_path, training_data):
    """Test full workflow with custom dataset"""
    # Create minimal custom dataset
    custom_data = pd.DataFrame({
        "text": ["Custom best", "Custom worst"],
        "rating": [9, 1]
    })
    
    # Train and verify
    predictor = train_elo_predictor(custom_data, output_dir=tmp_path)
    result = predictor.predict("Custom best", "Custom worst")
    assert result == (1, 2)
