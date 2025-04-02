"""
Demo script showing ELO rating changes from LLM output comparisons.
Run with: python -m dspy_elo.demo
"""

from .rating import EloRatingSystem
from .llm_comparison import compare_llm_outputs


def run_demo():  # pylint: disable=too-many-locals
    """Run the ELO rating system demo with detailed output"""
    elo = EloRatingSystem()
    modules = ["A", "B", "C", "D"]
    
    print("\nELO Rating System Demo - Live LLM Comparisons")
    print("=" * 60)
    
    # Initialize all module ratings
    print("\nInitial Ratings:")
    for mod in modules:
        print(f"  {mod}: {elo.get_rating(mod)}")
    
    # Define comparison pairs with varied response types
    comparisons = [
        ("A", "B", 
         "Here is a comprehensive analysis with multiple data points and references to recent studies...",
         "It's probably that one."),
        ("C", "D",
         "Clear step-by-step explanation:\n1. First, consider the core factors...\n2. Then evaluate...",
         "The answer is 42. Trust me."),
        ("A", "C",
         "Updated analysis incorporating new information from recent research papers...",
         "Based on fundamental principles, the key factors are...")
    ]
    
    for i, (mod1, mod2, resp1, resp2) in enumerate(comparisons, 1):
        print(f"\nRound {i} Comparison: {mod1} vs {mod2}")
        print("-" * 50)
        
        print(f"\nLLM Response {mod1}:")
        print(resp1)
        print(f"\nLLM Response {mod2}:")
        print(resp2)
        
        try:
            print("\nEvaluating with DeepSeek...")
            winner_idx, _ = compare_llm_outputs(resp1, resp2)
            winner = mod1 if winner_idx == 1 else mod2
            loser = mod2 if winner == mod1 else mod1
        except (ValueError, RuntimeError) as e:
            print(f"Comparison error: {e}")
            continue
            
        print(f"\nResult: {winner} preferred over {loser}")
        
        # Show rating changes
        old_ratings = {mod: elo.get_rating(mod) for mod in (mod1, mod2)}
        elo.update_ratings(winner, loser)
        
        print("\nRating Updates:")
        for mod in (mod1, mod2):
            change = elo.get_rating(mod) - old_ratings[mod]
            arrow = "↑" if change > 0 else "↓"
            print(f"  {mod}: {old_ratings[mod]:.1f} → {elo.get_rating(mod):.1f} ({arrow}{abs(change):.1f})")
    
    print("\nFinal Ratings:")
    for mod in modules:
        print(f"  {mod}: {elo.get_rating(mod):.1f}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_demo()
