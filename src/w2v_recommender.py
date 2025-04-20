import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
from config import KNN_K

logger = logging.getLogger(__name__)

class Word2VecRecommender:
    def __init__(self, w2v_model, w2v_emb, df):
        self.df = df
        self.emb = w2v_emb
        self.nn = NearestNeighbors(metric='cosine')
        self.nn.fit(self.emb)
        self.w2v_model = w2v_model
        logger.info('Initialized Word2VecRecommender')
    def recommend(self, q: str, k: int = KNN_K) -> list:
        tokens = q.lower().split()
        vecs = [self.w2v_model.wv[t] for t in tokens if t in self.w2v_model.wv]
        
        if not vecs:
            # If no valid tokens are found, return random songs with their default score (0 or any value)
            sample_tracks = self.df.sample(k)
            return [(track, 0) for track in sample_tracks['track_name'].tolist()]
        
        qv = np.mean(vecs, axis=0).reshape(1, -1)
        _, idx = self.nn.kneighbors(qv, n_neighbors=k)
        
        # Get the track names and the corresponding distances (or similarities)
        track_names = self.df.iloc[idx[0]]['track_name'].tolist()
        distances = _[0]  # This is the distance to the neighbors
        
        # Safeguard for possible zero distances or invalid normalization
        max_distance = np.max(distances) if np.max(distances) > 0 else 1  # Ensure max_distance isn't zero
        
        # Normalize the score based on distance
        scores = [1 - (dist / max_distance) for dist in distances]
        
        # Return list of tuples with track names and their respective scores
        return list(zip(track_names, scores))
