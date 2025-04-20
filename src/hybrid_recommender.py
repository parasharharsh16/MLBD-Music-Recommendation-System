import logging
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from config import KNN_K

logger = logging.getLogger(__name__)

def normalize(q: str) -> str:
    return q.lower().strip()

class HybridRecommender:
    def __init__(self, tfidf, tfidf_mat, audio_mat, df):
        """
        Combine metadata and audio neighbors to produce hybrid recommendations.
        """
        logger.info('Initialized HybridRecommender')
        self.df = df
        self.tfidf = tfidf
        self.audio_mat = audio_mat
        # Metadata KNN
        self.nn_meta = NearestNeighbors(metric='cosine', algorithm='brute')
        self.nn_meta.fit(tfidf_mat)
        # Audio KNN
        self.nn_audio = NearestNeighbors(metric='euclidean')
        self.nn_audio.fit(audio_mat)

    def recommend(self, q: str, k: int = KNN_K) -> list[tuple[str, float]]:
        """
        Hybrid recommender combining metadata and audio similarity.
        Returns a list of (track_name, normalized score [0,1]).
        """
        processed = normalize(q)
        
        # Metadata neighbors
        vec = self.tfidf.transform([processed])
        _, m_idx = self.nn_meta.kneighbors(vec, n_neighbors=k * 2)

        # Audio neighbors
        mask = self.df['track_name'].str.contains(processed, case=False, na=False)
        if not mask.any():
            return [(name, 0.0) for name in self.df.sample(k)['track_name'].tolist()]
        
        seed_idx = mask.idxmax()
        _, a_idx = self.nn_audio.kneighbors(self.audio_mat[seed_idx].reshape(1, -1), n_neighbors=k * 2)

        # Combine and count frequencies
        combined = list(m_idx[0]) + list(a_idx[0])
        counts = Counter(combined)
        counts.pop(seed_idx, None)

        # Normalize scores to [0, 1]
        max_score = max(counts.values()) if counts else 1
        top_indices = [idx for idx, _ in counts.most_common(k)]

        return [(self.df.iloc[idx]['track_name'], counts[idx] / max_score) for idx in top_indices]
