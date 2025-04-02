"""End-to-end tests for DSPy ELO predictor"""

import pytest
import pandas as pd
from pathlib import Path
from dspy_elo import train_elo_predictor, compare_llm_outputs

@pytest.fixture
def sample_data():
    """Sample training data matching spec requirements"""
    return pd.DataFrame({
        "text": [
            "Detailed technical explanation with references",
            "Simple summary of key points", 
            "Concise answer with examples",
            "Vague response with no specifics"
        ],
        "rating": [9, 7, 5, 2]  # Ratings 1-9 as per spec
    })

def test_training_workflow(tmp_path, sample_data):
    """Test complete training workflow from spec"""
    # Train the model
    predictor = train_elo_predictor(sample_data, output_dir=tmp_path)
    
    # Verify outputs exist
    assert (tmp_path / "training_info.txt").exists()
    
    # Test predictions match rating order
    result = predictor.predict(
        "Detailed technical explanation with references",
        "Vague response with no specifics"
    )
    assert result == (1, 2)  # Higher rated should win

def test_inference_on_new_samples(tmp_path, sample_data):
    """Test inference on unseen samples"""
    predictor = train_elo_predictor(sample_data, output_dir=tmp_path)
    
    # Compare new samples not in training data
    winner, loser = compare_llm_outputs(
        "Clear explanation with step-by-step reasoning",
        "Unclear response lacking details"
    )
    assert winner != loser
    assert winner in (1, 2)

def test_custom_dataset(tmp_path):
    """Test using custom dataset as per spec"""
    custom_data = pd.DataFrame({
        "text": ["Custom response 1", "Custom response 2"],
        "rating": [8, 3]  # Custom ratings in 1-9 range
    })
    
    predictor = train_elo_predictor(custom_data, output_dir=tmp_path)
    result = predictor.predict("Custom response 1", "Custom response 2")
    assert result == (1, 2)  # Higher rated should win
