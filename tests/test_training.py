"""Test training the ELO predictor using bootstrap few-shot"""

import pytest
import pandas as pd
from pathlib import Path

try:
    from dspy_elo.training import train_elo_predictor
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dspy_elo.training import train_elo_predictor


@pytest.fixture(name="training_sample_data")
def training_sample_data_fixture():
    """Sample data for testing"""
    return pd.DataFrame(
        {
            "text": ["Great response", "Okay response", "Poor response"],
            "rating": [9, 5, 1],
        }
    )


def test_train_elo_predictor(training_sample_data, tmp_path):
    """Test training produces a usable predictor"""
    predictor = train_elo_predictor(training_sample_data, output_dir=tmp_path)

    # Test it can make predictions
    result = predictor.predict("Great response", "Poor response")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] in (1, 2)
    assert result[1] in (1, 2)
    assert result[0] != result[1]


def test_train_elo_predictor_saves_model(training_sample_data, tmp_path):
    """Test training saves model files"""
    train_elo_predictor(training_sample_data, output_dir=tmp_path)

    # Check training info file was created
    assert (tmp_path / "training_info.txt").exists()


def test_training_scales_ratings_correctly(training_sample_data, tmp_path):
    """Test ratings are scaled from 1-9 to 100-900"""
    predictor = train_elo_predictor(training_sample_data, output_dir=tmp_path)
    
    # Check ratings were scaled properly
    assert predictor.predict("Great response", "Okay response") == (1, 2)
    assert predictor.predict("Okay response", "Poor response") == (1, 2)
    assert predictor.predict("Great response", "Poor response") == (1, 2)


def test_training_with_empty_data(tmp_path):
    """Test training handles empty data gracefully"""
    empty_data = pd.DataFrame(columns=["text", "rating"])
    with pytest.raises(ValueError):
        train_elo_predictor(empty_data, output_dir=tmp_path)
