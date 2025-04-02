"""Test package installation and basic imports"""

import importlib
import sys
from pathlib import Path

try:
    import dspy_elo
    from dspy_elo.rating import EloRatingSystem
    from dspy_elo.demo import run_demo
    from dspy_elo import __version__
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import dspy_elo
    from dspy_elo.rating import EloRatingSystem
    from dspy_elo.demo import run_demo
    from dspy_elo import __version__



def test_import_rating_system():
    """Test the EloRatingSystem can be imported"""
    assert EloRatingSystem is not None


def test_import_demo():
    """Test the demo module can be imported"""
    assert callable(run_demo)


def test_package_version():
    """Test package has a version string"""
    assert isinstance(__version__, str)
    assert len(__version__) > 0
    assert all(c.isdigit() or c == '.' for c in __version__), "Version should be numeric"


def test_package_metadata():
    """Test package metadata is complete"""
    assert hasattr(dspy_elo, '__version__')
    assert hasattr(dspy_elo, '__author__')
    assert hasattr(dspy_elo, '__license__')
    assert isinstance(dspy_elo.__author__, str)
    assert isinstance(dspy_elo.__license__, str)
    assert len(dspy_elo.__author__) > 0
