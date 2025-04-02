import pytest
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
    assert (elo.get_rating(winner) + elo.get_rating(loser)) == (initial_winner + initial_loser)
