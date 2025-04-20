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

    def recommend(self, q: str, k: int = KNN_K) -> list:
        if not q:
            return self.df.sample(k)['track_name'].tolist()
        mask = self.df['track_name'].str.contains(normalize(q), case=False, na=False)
        if not mask.any():
            return self.df.sample(k)['track_name'].tolist()
        idx0 = mask.idxmax()
        _, neigh = self.nn.kneighbors(self.mat[idx0].reshape(1, -1), n_neighbors=k + 1)
        return self.df.iloc[neigh[0][1:]]['track_name'].tolist()