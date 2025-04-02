"""Test package installation and basic imports"""
import importlib

def test_package_import():
    """Test the package can be imported"""
    dspy_elo = importlib.import_module("dspy_elo")
    assert dspy_elo is not None

def test_import_rating_system():
    """Test the EloRatingSystem can be imported"""
    from dspy_elo.rating import EloRatingSystem
    assert EloRatingSystem is not None

def test_import_demo():
    """Test the demo module can be imported"""
    from dspy_elo.demo import run_demo
    assert callable(run_demo)
