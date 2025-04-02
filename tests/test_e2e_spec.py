"""End-to-end tests verifying all functionality in info.md spec"""

import pandas as pd
import dspy
from dspy_elo import train_elo_predictor, compare_llm_outputs


def test_model_uses_deepseek():
    """Verify we're using deepseek model as specified"""
    assert isinstance(dspy.settings.lm, dspy.LM)
    assert "deepseek" in dspy.settings.lm.model.lower()


def test_training_with_bootstrap_fewshot(tmp_path):
    """Test training with bootstrap few-shot as specified"""
    data = pd.DataFrame({
        "text": ["Detailed analysis", "Brief response"],
        "rating": [8, 3]  # Using 1-9 scale
    })
    
    predictor = train_elo_predictor(data, output_dir=tmp_path)
    
    # Verify basic functionality
    assert hasattr(predictor, "predict")
    assert hasattr(predictor, "get_rating_difference")
    
    # Test prediction directionality
    result = predictor.predict(data.iloc[0]["text"], data.iloc[1]["text"])
    assert result == (1, 2)  # Higher rated should win


def test_inference_on_new_samples():
    """Test comparing new LLM outputs not in training data"""
    # These are new samples not in any training data
    winner, loser = compare_llm_outputs(
        "Detailed analysis with examples", 
        "Vague response"
    )
    assert winner in (1, 2)
    assert loser in (1, 2)
    assert winner != loser


def test_custom_dataset_workflow(tmp_path):
    """Test full workflow with custom dataset"""
    # Create minimal custom dataset
    custom_data = pd.DataFrame({
        "text": ["Custom best", "Custom worst"],
        "rating": [9, 1]  # Using min/max ratings
    })

    # Train and verify
    predictor = train_elo_predictor(custom_data, output_dir=tmp_path)
    result = predictor.predict("Custom best", "Custom worst")
    assert result == (1, 2)
