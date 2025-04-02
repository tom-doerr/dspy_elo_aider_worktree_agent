"""End-to-end tests verifying all functionality in info.md spec"""

import pytest
import pandas as pd
from pathlib import Path
import dspy
from dspy_elo import train_elo_predictor, compare_llm_outputs

def test_model_uses_deepseek():
    """Verify we're using deepseek model as specified"""
    assert isinstance(dspy.settings.lm, dspy.LM)
    assert "deepseek" in dspy.settings.lm.model.lower()

def test_training_with_bootstrap_fewshot(tmp_path):
    """Test training workflow matches spec"""
    # Create minimal training data matching spec requirements
    data = pd.DataFrame({
        "text": ["Detailed analysis", "Brief response"],
        "rating": [8, 3]  # Using 1-9 scale as specified
    })

    # Train model
    predictor = train_elo_predictor(data, output_dir=tmp_path)

    # Verify predictions match rating order
    result = predictor.predict(data.iloc[0]["text"], data.iloc[1]["text"])
    assert result == (1, 2)  # Higher rated should win

def test_llm_comparison_workflow():
    """Test LLM output comparison as specified"""
    # Use clearly different responses to test comparison
    detailed = "Detailed response with examples and explanations."
    vague = "I'm not sure."

    winner, loser = compare_llm_outputs(detailed, vague)
    assert winner in (1, 2)
    assert loser in (1, 2)
    assert winner != loser

def test_custom_dataset_workflow(tmp_path):
    """Test full workflow with custom dataset as specified"""
    # Create custom dataset matching spec format
    custom_data = pd.DataFrame({
        "text": ["Custom best response", "Custom worst response"],
        "rating": [9, 1]  # Using min/max of 1-9 scale
    })

    # Train and verify
    predictor = train_elo_predictor(custom_data, output_dir=tmp_path)
    result = predictor.predict(custom_data.iloc[0]["text"], custom_data.iloc[1]["text"])
    assert result == (1, 2)  # Higher rated should win
