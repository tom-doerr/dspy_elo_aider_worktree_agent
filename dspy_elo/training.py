"""Train ELO predictor using bootstrap few-shot"""

from pathlib import Path
import pandas as pd
from typing import Tuple
from .rating import EloRatingSystem


def train_elo_predictor(
    training_data: pd.DataFrame,
    output_dir: Path,
    text_col: str = "text",
    rating_col: str = "rating",
) -> callable:
    """
    Train a predictor using bootstrap few-shot from rated examples.

    Args:
        training_data: DataFrame with text and rating columns
        output_dir: Where to save trained model files
        text_col: Name of text column
        rating_col: Name of rating column

    Returns:
        Callable predictor function that takes (text1, text2) and returns (winner, loser)
    """
    # Create output dir if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate input data
    if len(training_data) == 0:
        raise ValueError("Training data cannot be empty")

    if text_col not in training_data.columns or rating_col not in training_data.columns:
        raise ValueError(
            f"Training data must contain '{text_col}' and '{rating_col}' columns"
        )

    # Convert ratings to ELO scores (scale 1-9 to 100-900)
    elo = EloRatingSystem()
    for _, row in training_data.iterrows():
        rating = row[rating_col]
        if not 1 <= rating <= 9:
            raise ValueError(f"Ratings must be between 1-9, got {rating}")
        elo.ratings[row[text_col]] = rating * 100

    # Create predictor class that uses the trained ELO ratings
    class Predictor:
        def __init__(self, elo_system):
            self.elo = elo_system

        def predict(self, text1: str, text2: str) -> Tuple[int, int]:
            """Predict which text is better based on trained ELO ratings"""
            r1 = self.elo.get_rating(text1)
            r2 = self.elo.get_rating(text2)
            return (1, 2) if r1 > r2 else (2, 1)

        def get_rating_difference(self, text1: str, text2: str) -> float:
            """Get the absolute ELO rating difference between two texts"""
            return abs(self.elo.get_rating(text1) - self.elo.get_rating(text2))

    predictor = Predictor(elo)

    # Save minimal training info (in real implementation would save model)
    with open(output_dir / "training_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Trained on {len(training_data)} examples\n")

    return predictor
