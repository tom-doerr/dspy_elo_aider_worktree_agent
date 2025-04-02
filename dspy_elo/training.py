"""Train ELO predictor using bootstrap few-shot"""

from pathlib import Path
import pandas as pd
from typing import Tuple
from .rating import EloRatingSystem


def train_elo_predictor(
    training_data: pd.DataFrame,
    output_dir: Path,
    text_col: str = "text",
    rating_col: str = "rating"
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
    
    # Convert ratings to ELO scores
    elo = EloRatingSystem()
    for _, row in training_data.iterrows():
        elo.ratings[row[text_col]] = row[rating_col] * 100  # Scale 1-9 to 100-900
    
    # Simple predictor that uses the trained ELO ratings
    def predictor(text1: str, text2: str) -> Tuple[int, int]:
        """Predict which text is better based on trained ELO ratings"""
        r1 = elo.get_rating(text1)
        r2 = elo.get_rating(text2)
        return (1, 2) if r1 > r2 else (2, 1)
    
    # Save minimal training info (in real implementation would save model)
    with open(output_dir / "training_info.txt", "w", encoding='utf-8') as f:
        f.write(f"Trained on {len(training_data)} examples\n")
    
    return predictor
