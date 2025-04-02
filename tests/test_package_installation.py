"""Test package installation and basic imports"""

import importlib
import sys
from pathlib import Path

try:
    from dspy_elo.rating import EloRatingSystem
    from dspy_elo.demo import run_demo
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dspy_elo.rating import EloRatingSystem
    from dspy_elo.demo import run_demo


def test_package_import():
    """Test the package can be imported"""
    dspy_elo = importlib.import_module("dspy_elo")
    assert dspy_elo is not None


def test_import_rating_system():
    """Test the EloRatingSystem can be imported"""
    assert EloRatingSystem is not None


def test_import_demo():
    """Test the demo module can be imported"""
    assert callable(run_demo)
