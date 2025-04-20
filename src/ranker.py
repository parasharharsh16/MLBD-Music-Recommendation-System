from collections import defaultdict
from config import WEIGHTS

class Ranker:
    def __init__(self, k=10):
        self.k = k

    def rank_songs(self, recs: dict[str, list[tuple[str, float]]]) -> tuple[list[tuple[str, float]], dict[str, list[tuple[str, float]]]]:
        """
        Rank songs by weighted scores from multiple models.
        Returns top-k ranked list and updated recs with weighted scores.
        """
        combined_scores = defaultdict(float)
        weighted_recs = {}

        for model_name, track_list in recs.items():
            weight = WEIGHTS.get(model_name, 1.0)
            updated = []
            for track, score in track_list:
                weighted_score = score * weight
                combined_scores[track] += weighted_score
                updated.append((track, weighted_score))
            weighted_recs[model_name] = updated

        # Top-k global ranking
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        return ranked[:self.k], weighted_recs
