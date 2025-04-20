import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import joblib

import os
import pickle
# Function to prepare data for training
def prepare_data_for_training(data):
    """
    Prepare the data for collaborative filtering using Surprise.
    Converts artist and song names to user_id and item_id respectively.
    """
    data = data[['artists', 'name', 'popularity']].dropna()
    data.columns = ['user_id', 'item_id', 'rating']
    return data

# Function to train the SVD model (ALS-like)
def train_als_model(data, model_path="als_model.pkl"):
    """
    Train SVD (ALS-like) model and save it.
    Parameters:
    - data: pandas DataFrame with 'user_id', 'item_id', 'rating'
    """
    reader = Reader(rating_scale=(0, 100))  # Popularity range
    dataset = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)
    
    trainset, testset = train_test_split(dataset, test_size=0.2)
    model = SVD(n_factors=50, biased=True)

    model.fit(trainset)
    predictions = model.test(testset)

    print("RMSE:", accuracy.rmse(predictions))
    
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model

# Function to load a trained model
def load_model(model_path="als_model.pkl"):
    """
    Load a saved model using joblib.
    """
    return joblib.load(model_path)

# Function to recommend songs for a given user
def recommend_songs(user_id, model, data, num_recommendations=10):
    """
    Recommend top N songs for a given user_id (artist).
    Parameters:
    - user_id: artist name (used as user ID)
    - model: Trained Surprise model
    - data: pandas DataFrame used during training
    - num_recommendations: how many recommendations to return
    """
    all_items = data['item_id'].unique()
    rated_items = data[data['user_id'] == user_id]['item_id'].unique()
    unrated_items = [item for item in all_items if item not in rated_items]

    predictions = [(item, model.predict(user_id, item).est) for item in unrated_items]
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return [item for item, _ in predictions[:num_recommendations]]
def save_model(model, path):
    """
    Save a trained model to the given path using pickle.
    
    Parameters:
    - model: Trained collaborative filtering model (e.g., from Surprise)
    - path: File path to save the model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)