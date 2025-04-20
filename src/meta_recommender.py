import logging
from sklearn.neighbors import NearestNeighbors
from config import KNN_K

def normalize(q: str) -> str:
    return q.lower().strip()

class MetadataRecommender:
    def __init__(self, tfidf_mat, df, tfidf):
        self.df = df
        self.tfidf = tfidf
        self.nn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.nn.fit(tfidf_mat)
        logging.getLogger(__name__).info('Initialized MetadataRecommender')

    def recommend(self, q: str, k: int = KNN_K) -> list[tuple[str, float]]:
        """
        Recommend tracks using metadata-based KNN (TF-IDF), returning (track_name, normalized similarity_score).
        Scores are between 0 and 1, higher = more similar.
        """
        if not q:
            return [(name, 0.0) for name in self.df.sample(k)['track_name'].tolist()]
        
        vec = self.tfidf.transform([normalize(q)])
        distances, idx = self.nn.kneighbors(vec, n_neighbors=k + 1)

        rec_indices = idx[0][1:]
        rec_distances = distances[0][1:]

        min_dist = min(rec_distances)
        max_dist = max(rec_distances)

        # Avoid division by zero and force scores between 0 and 1
        if max_dist == min_dist:
            similarity_scores = [0.5 for _ in rec_distances] 
        else:
            similarity_scores = [
                (max_dist - d) / (max_dist - min_dist)
                for d in rec_distances
            ]

        return [
            (self.df.iloc[i]['track_name'], round(float(score), 3))
            for i, score in zip(rec_indices, similarity_scores)
        ]
