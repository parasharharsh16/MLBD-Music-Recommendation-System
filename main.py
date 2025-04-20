import os
import logging
import datetime
import shutil
import streamlit as st
import torch, types
import config
from src.data_loader import load_data
from src.preprocessing import preprocess_features
from src.content_recommender import MetadataRecommender
from src.audio_recommender import AudioRecommender
from src.ml_recommender import MLRecommender
from src.dl_recommender import DLRecommender
from src.w2v_recommender import Word2VecRecommender
from src.hybrid_recommender import HybridRecommender
from src.evaluation import compare_models

# Disable file watcher for torch
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
try:
    torch._classes = types.ModuleType('torch._classes')
    torch._classes.__path__ = []
except:
    pass

logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s: %(message)s', level=logging.INFO)
logger = logging.getLogger('music_recommender_app')

# Handle train mode cleanup
if config.MODE == 'train':
    shutil.rmtree('models', ignore_errors=True)
    os.makedirs('models', exist_ok=True)
    logger.info('Cleared models directory')

st.title('🎵 Music Recommender')
st.info('Loading data and models...')

df = load_data('data')
tfidf, tfidf_mat, scaler, audio_mat, w2v_model, w2v_emb = preprocess_features(df)

# Initialize models
recommenders = {
    'Metadata': MetadataRecommender(tfidf_mat, df, tfidf),
    'Audio': AudioRecommender(audio_mat, df),
    'Hybrid': HybridRecommender(tfidf, tfidf_mat, audio_mat, df),
    'ML': MLRecommender(df, audio_mat, tfidf, tfidf_mat),
    'DL': DLRecommender(audio_mat, df),
    'Word2Vec': Word2VecRecommender(w2v_model, w2v_emb, df)
}
st.success('Models ready!')

query = st.text_input('Artist, track, year, genre, or mood:')
if st.button('Recommend'):
    if not query:
        hour = datetime.datetime.now().hour
        query = 'morning' if 5<=hour<12 else 'afternoon' if 12<=hour<17 else 'evening' if 17<=hour<21 else 'night'

    # Collect top-K from each model
    recs = {name: mod.recommend(query, k=config.KNN_K) for name, mod in recommenders.items()}

    # Ensemble scoring
    scores = {}
    for name, tracks in recs.items():
        weight = config.WEIGHTS.get(name, 1)
        for i, tr in enumerate(tracks):
            pts = config.KNN_K - i
            scores[tr] = scores.get(tr, 0) + weight * pts
    final = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:config.KNN_K]

    st.subheader('🎯 Final Top Recommendations')
    for tr, _ in final:
        st.write(f'- {tr}')

    st.subheader('🔍 Per-model Recommendations')
    for name, tracks in recs.items():
        st.markdown(f'**{name}:** ' + ', '.join(tracks))