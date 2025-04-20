import logging
logger = logging.getLogger(__name__)

def compare_models(models: dict, query: str, k: int, df) -> tuple:
    scores = {}
    for name, model in models.items():
        recs = model.recommend(query, k=k)
        pops = df[df['track_name'].isin(recs)]['popularity'].fillna(0)
        scores[name] = pops.mean()
    best = max(scores, key=scores.get)
    logger.info('Best model: %s', best)
    return best, scores