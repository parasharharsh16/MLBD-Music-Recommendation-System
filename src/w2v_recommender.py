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
            return self.df.sample(k)['track_name'].tolist()
        qv = np.mean(vecs, axis=0).reshape(1, -1)
        _, idx = self.nn.kneighbors(qv, n_neighbors=k)
        return self.df.iloc[idx[0]]['track_name'].tolist()