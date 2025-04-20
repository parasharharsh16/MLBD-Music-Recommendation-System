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

    def recommend(self, q: str, k: int = KNN_K) -> list:
        if not q:
            return self.df.sample(k)['track_name'].tolist()
        vec = self.tfidf.transform([normalize(q)])
        _, idx = self.nn.kneighbors(vec, n_neighbors=k + 1)
        return self.df.iloc[idx[0][1:]]['track_name'].tolist()