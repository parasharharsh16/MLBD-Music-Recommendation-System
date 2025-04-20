import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np
from .model_persistence import save_sklearn, load_sklearn, save_numpy, load_numpy
from config import TFIDF_PARAMS, W2V_PARAMS

logger = logging.getLogger(__name__)

# Audio features used for KNN
AUDIO_FEATURES = ['danceability', 'energy', 'loudness', 'tempo', 'valence']
TFIDF_NAME = 'tfidf.pkl'
W2V_MODEL_NAME = 'w2v_model.model'
W2V_EMB_NAME = 'w2v_embeddings'

def preprocess_features(df):
    """
    Generate TF-IDF, audio feature matrix, and Word2Vec embeddings.
    Returns: tfidf, tfidf_mat, scaler, audio_mat, w2v_model, w2v_emb
    """
    logger.info('Preprocessing features')

    # TF-IDF
    tfidf = load_sklearn(TFIDF_NAME)
    if tfidf is None:
        tfidf = TfidfVectorizer(**TFIDF_PARAMS)
        tfidf.fit(df['metadata'])
        save_sklearn(tfidf, TFIDF_NAME)
    tfidf_mat = tfidf.transform(df['metadata'])
    logger.info('TF-IDF matrix shape: %s', tfidf_mat.shape)

    # Audio features
    X = df[AUDIO_FEATURES].fillna(df[AUDIO_FEATURES].mean())
    scaler = StandardScaler()
    audio_mat = scaler.fit_transform(X)
    logger.info('Audio matrix shape: %s', audio_mat.shape)

    # Word2Vec on metadata tokens using simple_preprocess
    w2v_emb = load_numpy(W2V_EMB_NAME)
    if w2v_emb is None:
        logger.info('Training Word2Vec model for metadata')
        sentences = [simple_preprocess(text) for text in df['metadata']]
        w2v_model = Word2Vec(sentences, **W2V_PARAMS)
        w2v_model.save(W2V_MODEL_NAME)
        embeds = []
        for tokens in sentences:
            vecs = [w2v_model.wv[token] for token in tokens if token in w2v_model.wv]
            embeds.append(np.mean(vecs, axis=0) if vecs else np.zeros(W2V_PARAMS['vector_size']))
        w2v_emb = np.vstack(embeds)
        save_numpy(w2v_emb, W2V_EMB_NAME)
    else:
        w2v_model = Word2Vec.load(W2V_MODEL_NAME)
        logger.info('Loaded Word2Vec embeddings')

    return tfidf, tfidf_mat, scaler, audio_mat, w2v_model, w2v_emb