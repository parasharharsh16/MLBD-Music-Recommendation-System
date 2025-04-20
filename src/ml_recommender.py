import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import TruncatedSVD
from config import RF_PARAMS, SVD_PARAMS, MODE, KNN_K
from .model_persistence import load_sklearn, save_sklearn, load_numpy, save_numpy

logger = logging.getLogger(__name__)

def normalize(q: str) -> str:
    return q.lower().strip()

class MLRecommender:
    def __init__(self, df, audio_mat, tfidf, tfidf_mat):
        logger.info('Initializing MLRecommender (mode=%s)', MODE)
        from .model_persistence import MODELS_DIR
        if MODE == 'train':
            import shutil, os
            shutil.rmtree(MODELS_DIR, ignore_errors=True)
            os.makedirs(MODELS_DIR, exist_ok=True)

        # Train or load SVD for text embeddings
        svd = load_sklearn('svd.pkl')
        txt_emb = load_numpy('txt_emb')
        if svd is None or txt_emb is None:
            logger.info('Training SVD embeddings')
            svd = TruncatedSVD(**SVD_PARAMS)
            txt_emb = svd.fit_transform(tfidf_mat)
            save_sklearn(svd, 'svd.pkl')
            save_numpy(txt_emb, 'txt_emb')

        # Train or load RandomForest model
        rf = load_sklearn('ml_model.pkl')
        if rf is None:
            logger.info('Training RandomForest model')
            X = np.hstack([audio_mat, txt_emb])
            y = df['popularity'].fillna(0).values
            rf = RandomForestRegressor(**RF_PARAMS)
            rf.fit(X, y)
            save_sklearn(rf, 'ml_model.pkl')

        self.model = rf
        self.audio_mat = audio_mat
        self.txt_emb = txt_emb
        self.df = df

    def recommend(self, q: str, k: int = KNN_K) -> list:
        """
        Recommend tracks matching query, ranked by predicted popularity.
        If no match, fallback to global popularity.
        """
        logger.info('MLRecommender: recommending for query "%s"', q)
        processed = normalize(q)
        # Filter candidates by query
        mask = self.df['track_name'].str.contains(processed, case=False, na=False)
        if mask.any():
            # Predict popularity for matched tracks
            X_cand = np.hstack([
                self.audio_mat[mask],
                self.txt_emb[mask]
            ])
            preds = self.model.predict(X_cand)
            cand_df = self.df[mask].reset_index(drop=True)
            top_idx = np.argsort(preds)[-k:][::-1]
            return cand_df.loc[top_idx, 'track_name'].tolist()
        # Fallback: top global predicted popularity
        X_all = np.hstack([self.audio_mat, self.txt_emb])
        preds_all = self.model.predict(X_all)
        top_all = np.argsort(preds_all)[-k:][::-1]
        return self.df.iloc[top_all]['track_name'].tolist()