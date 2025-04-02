"""
Demo script showing ELO rating changes from LLM output comparisons.
Run with: python -m dspy_elo.demo
"""

from .rating import EloRatingSystem
from .llm_comparison import compare_llm_outputs


def run_demo():
    """Run the ELO rating system demo"""
    elo = EloRatingSystem()

    print("\nELO Rating System Demo")
    print("=" * 40)

    # Initial state
    print("\nInitial ratings:")
    print(f"Module A: {elo.get_rating('A')}")
    print(f"Module B: {elo.get_rating('B')}")

    # Generate some sample LLM outputs
    outputs = {
        "A": "Here is a detailed and thoughtful response to the query.",
        "B": "Short answer."
    }

    # Compare outputs
    print("\nComparing LLM outputs...")
    winner_idx, _ = compare_llm_outputs(outputs["A"], outputs["B"])
    winner = "A" if winner_idx == 1 else "B"
    loser = "B" if winner == "A" else "A"
    
    print(f"Result: {winner} beats {loser}")
    old_a = elo.get_rating("A")
    old_b = elo.get_rating("B")
    elo.update_ratings(winner, loser)
    print(f"Module A: {old_a} -> {elo.get_rating('A')}")
    print(f"Module B: {old_b} -> {elo.get_rating('B')}")

    # Final state
    print("\nFinal ratings:")
    print(f"Module A: {elo.get_rating('A')}")
    print(f"Module B: {elo.get_rating('B')}")
    print("\n" + "=" * 40)


if __name__ == "__main__":
    run_demo()
