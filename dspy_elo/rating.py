class EloRatingSystem:
    def __init__(self, initial_rating=1000, k_factor=32):
        self.ratings = {}
        self.initial_rating = initial_rating
        self.k_factor = k_factor

    def get_rating(self, module_id):
        """Get current rating for a module, initializing if not present"""
        return self.ratings.get(module_id, self.initial_rating)

    def update_ratings(self, winner_id, loser_id):
        """Update ratings after a comparison where winner_id beat loser_id"""
        r1 = self.get_rating(winner_id)
        r2 = self.get_rating(loser_id)
        
        expected1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        expected2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
        
        self.ratings[winner_id] = r1 + self.k_factor * (1 - expected1)
        self.ratings[loser_id] = r2 + self.k_factor * (0 - expected2)
