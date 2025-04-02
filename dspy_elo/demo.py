"""
Demo script showing ELO rating changes step by step.
Run with: python -m dspy_elo.demo
"""

from .rating import EloRatingSystem


def run_demo():
    """Run the ELO rating system demo"""
    elo = EloRatingSystem()

    print("\nELO Rating System Demo")
    print("=" * 40)

    # Initial state
    print("\nInitial ratings:")
    print(f"Module A: {elo.get_rating('A')}")
    print(f"Module B: {elo.get_rating('B')}")

    # First comparison
    print("\nAfter comparison (A beats B):")
    old_a = elo.get_rating("A")
    old_b = elo.get_rating("B")
    elo.update_ratings("A", "B")
    print(f"Module A: {old_a} -> {elo.get_rating('A')}")
    print(f"Module B: {old_b} -> {elo.get_rating('B')}")

    # Second comparison
    print("\nAfter comparison (B beats A):")
    old_a = elo.get_rating("A")
    old_b = elo.get_rating("B")
    elo.update_ratings("B", "A")
    print(f"Module A: {old_a} -> {elo.get_rating('A')}")
    print(f"Module B: {old_b} -> {elo.get_rating('B')}")

    # Final state
    print("\nFinal ratings:")
    print(f"Module A: {elo.get_rating('A')}")
    print(f"Module B: {elo.get_rating('B')}")
    print("\n" + "=" * 40)


if __name__ == "__main__":
    run_demo()
