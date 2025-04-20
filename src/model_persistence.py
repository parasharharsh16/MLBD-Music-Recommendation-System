import os
import joblib
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def save_sklearn(obj, name):
    joblib.dump(obj, os.path.join(MODELS_DIR, name))
    logger.info('Saved sklearn: %s', name)

def load_sklearn(name):
    p = os.path.join(MODELS_DIR, name)
    if os.path.exists(p):
        logger.info('Loaded sklearn: %s', name)
        return joblib.load(p)
    return None

def save_torch(model, name):
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, name))
    logger.info('Saved torch: %s', name)

def load_torch(cls, name, *args, **kwargs):
    p = os.path.join(MODELS_DIR, name)
    if os.path.exists(p):
        m = cls(*args, **kwargs)
        m.load_state_dict(torch.load(p))
        m.eval()
        logger.info('Loaded torch: %s', name)
        return m
    return None

def save_numpy(arr, name):
    np.save(os.path.join(MODELS_DIR, name + '.npy'), arr)
    logger.info('Saved numpy: %s', name)

def load_numpy(name):
    p = os.path.join(MODELS_DIR, name + '.npy')
    if os.path.exists(p):
        logger.info('Loaded numpy: %s', name)
        return np.load(p)
    return None