"""
Global config: MODE train/prod, model hyperparams, ensemble weights, Word2Vec settings.
"""

# Modes
MODE = 'prod'  # 'train' to clear & retrain models, 'prod' to load persisted

# Random Forest parameters
RF_PARAMS = {'n_estimators': 50, 'max_depth': 10, 'n_jobs': -1, 'random_state': 0}
# SVD parameters
SVD_PARAMS = {'n_components': 50, 'random_state': 0}
# TF-IDF parameters
TFIDF_PARAMS = {'stop_words': 'english', 'max_features': 5000}
# Autoencoder parameters
AE_PARAMS = {'latent_dim': 3, 'epochs': 30, 'lr': 1e-3}
# Word2Vec parameters
W2V_PARAMS = {'vector_size': 100, 'window': 5, 'min_count': 5, 'epochs': 20}

# Number of neighbors per model (top-K)
KNN_K = 5

# Ensemble weights for pointwise scoring
WEIGHTS = {
    'Collaborative Filtering': 0.65,
    'Content-Based Filtering': 1.5,
    'Hybrid': 0.72,
    'Random Forest': 0.8,
    'Autoencoder': 0.85,
    'Semantic Model': 1.43
}

# = {
#     'Metadata': 1.0,
#     'Audio': 1.0,
#     'Hybrid': 1.0,
#     'ML': 1.0,
#     'DL': 1.0,
#     'Word2Vec': 1.0
# }