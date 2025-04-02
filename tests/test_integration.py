"""Integration tests with real data"""

import pytest
import pandas as pd
from pathlib import Path

try:
    from dspy_elo.training import train_elo_predictor
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dspy_elo.training import train_elo_predictor


@pytest.mark.integration
def test_with_real_data(tmp_path):
    """Test training with actual CSV data"""
    data_path = Path("data") / "ratings.csv"
    if not data_path.exists():
        pytest.skip("No real data file found")

    df = pd.read_csv(data_path)
    predictor = train_elo_predictor(df, output_dir=tmp_path)

    # Test prediction on some samples
    samples = df.sample(2)
    result = predictor.predict(samples.iloc[0]["text"], samples.iloc[1]["text"])
    assert isinstance(result, tuple)
    assert len(result) == 2
