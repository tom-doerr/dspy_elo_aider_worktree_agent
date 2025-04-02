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
    sample1 = samples.iloc[0]
    sample2 = samples.iloc[1]

    result = predictor.predict(sample1["text"], sample2["text"])
    assert isinstance(result, tuple)
    assert len(result) == 2

    # Verify prediction aligns with rating comparison
    if sample1["rating"] > sample2["rating"]:
        assert result == (1, 2)
    else:
        assert result == (2, 1)

    # Test rating difference calculation
    diff = predictor.get_rating_difference(sample1["text"], sample2["text"])
    expected_diff = abs((sample1["rating"] * 100) - (sample2["rating"] * 100))
    assert diff == expected_diff


@pytest.mark.integration
def test_real_data_training_flow(tmp_path):
    """Test complete training and prediction flow with realistic data"""
    # Create test data that matches spec requirements
    data = pd.DataFrame(
        {
            "text": [
                "Detailed analysis with multiple examples",
                "Concise summary",
                "Technical breakdown with diagrams",
            ],
            "rating": [8, 5, 9],
        }
    )

    # Test training
    predictor = train_elo_predictor(data, output_dir=tmp_path)

    # Test predictions
    result = predictor.predict(data.iloc[0]["text"], data.iloc[2]["text"])
    assert result == (2, 1)  # Higher rated item should win

    # Verify rating scaling
    assert predictor.elo.get_rating(data.iloc[0]["text"]) == 800
    assert predictor.elo.get_rating(data.iloc[2]["text"]) == 900


@pytest.mark.integration
def test_demo_training_data_validation():
    """Test the demo training data file exists and has valid structure"""
    data_path = Path("data") / "demo_training_data.csv"
    if not data_path.exists():
        pytest.fail(f"Demo training data not found at {data_path}")

    df = pd.read_csv(data_path)

    # Validate required columns
    assert {"text", "rating"}.issubset(df.columns), "Missing required columns"

    # Validate rating range
    assert df["rating"].between(1, 9).all(), "Ratings must be between 1-9"

    # Validate sample count
    assert len(df) >= 10, "Demo data should contain at least 10 samples"


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
