from .rating import EloRatingSystem
from .demo import run_demo
from .llm_comparison import compare_llm_outputs
from .training import train_elo_predictor

__version__ = "0.1.1"
__author__ = "Your Name <your.email@example.com>"
__license__ = "MIT"
__all__ = [
    "EloRatingSystem",
    "run_demo",
    "compare_llm_outputs",
    "train_elo_predictor",
    "__version__",
]
