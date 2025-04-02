try:
    from dspy_elo.rating import EloRatingSystem
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dspy_elo.rating import EloRatingSystem


def test_elo_initial_ratings():
    """Test that new modules get default ELO ratings"""
    elo = EloRatingSystem()
    module_id = "module1"
    assert elo.get_rating(module_id) == 1000


def test_elo_update_after_comparison():
    """Test ELO ratings update after a comparison"""
    elo = EloRatingSystem()
    winner = "module1"
    loser = "module2"

    initial_winner = elo.get_rating(winner)
    initial_loser = elo.get_rating(loser)

    elo.update_ratings(winner, loser)

    assert elo.get_rating(winner) > initial_winner
    assert elo.get_rating(loser) < initial_loser
    assert (elo.get_rating(winner) + elo.get_rating(loser)) == (
        initial_winner + initial_loser
    )


def test_elo_custom_k_factor():
    """Test that custom k_factor affects rating changes"""
    default_k = EloRatingSystem()
    high_k = EloRatingSystem(k_factor=64)

    default_k.update_ratings("A", "B")
    high_k.update_ratings("A", "B")

    default_change = default_k.get_rating("A") - 1000
    high_change = high_k.get_rating("A") - 1000

    assert high_change > default_change
