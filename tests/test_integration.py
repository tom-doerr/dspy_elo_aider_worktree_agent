"""Integration tests with real data"""

import pytest
import pandas as pd
import sys
from pathlib import Path

try:
    from dspy_elo.training import train_elo_predictor
    from dspy_elo.train import main
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dspy_elo.training import train_elo_predictor
    from dspy_elo.train import main


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


@pytest.mark.integration
def test_training_script_cli(tmp_path, monkeypatch):
    """Test the training script via command line interface"""
    # Create sample CSV
    csv_path = tmp_path / "training.csv"
    pd.DataFrame(
        {"text": ["Detailed response", "Brief answer"], "rating": [8, 4]}
    ).to_csv(csv_path, index=False)

    output_dir = tmp_path / "output"

    # Mock command line arguments
    monkeypatch.setattr(
        sys, "argv", ["train.py", str(csv_path), "--output-dir", str(output_dir)]
    )

    # Run main
    main()

    # Verify outputs
    assert (output_dir / "training_info.txt").exists()
