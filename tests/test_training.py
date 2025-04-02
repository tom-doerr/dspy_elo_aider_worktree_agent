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


@pytest.fixture
def sample_data():
    """Sample data for testing"""
    return pd.DataFrame({
        'text': ['Great response', 'Okay response', 'Poor response'],
        'rating': [9, 5, 1]
    })


def test_train_elo_predictor(sample_data, tmp_path):
    """Test training produces a usable predictor"""
    predictor = train_elo_predictor(sample_data, output_dir=tmp_path)
    
    # Test it can make predictions
    result = predictor.predict("Great response", "Poor response")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] in (1, 2)
    assert result[1] in (1, 2)
    assert result[0] != result[1]


def test_train_elo_predictor_saves_model(sample_data, tmp_path):
    """Test training saves model files"""
    train_elo_predictor(sample_data, output_dir=tmp_path)
    
    # Check some expected files were created
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "training_args.bin").exists()
