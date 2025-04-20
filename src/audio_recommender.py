import logging
from sklearn.neighbors import NearestNeighbors
from config import KNN_K

def normalize(q: str) -> str:
    return q.lower().strip()

class AudioRecommender:
    def __init__(self, audio_mat, df):
        self.df = df
        self.mat = audio_mat
        self.nn = NearestNeighbors(metric='euclidean')
        self.nn.fit(audio_mat)
        logging.getLogger(__name__).info('Initialized AudioRecommender')

    def recommend(self, q: str, k: int = KNN_K) -> list[tuple[str, float]]:
        """
        Recommend songs based on KNN similarity, returning (track_name, score).
        Lower score means higher similarity.
        """
        if not q:
            # Fallback: random tracks with dummy score
            return [(name, 0.0) for name in self.df.sample(k)['track_name'].tolist()]
        
        mask = self.df['track_name'].str.contains(normalize(q), case=False, na=False)
        if not mask.any():
            return [(name, 0.0) for name in self.df.sample(k)['track_name'].tolist()]
        
        idx0 = mask.idxmax()
        distances, neigh = self.nn.kneighbors(self.mat[idx0].reshape(1, -1), n_neighbors=k + 1)

        # Skip the first neighbor (it’s the query track itself)
        rec_indices = neigh[0][1:]
        rec_distances = distances[0][1:]

        return [(self.df.iloc[idx]['track_name'], float(dist)) for idx, dist in zip(rec_indices, rec_distances)]
