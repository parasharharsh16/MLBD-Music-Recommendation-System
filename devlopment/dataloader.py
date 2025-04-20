import pandas as pd
import numpy as np
def load_data(dataset_path = None):
    # Load the datasets
    data_by_artist = pd.read_csv(f"{dataset_path}/data_by_artist.csv")
    data_by_genre = pd.read_csv(f"{dataset_path}/data_by_genres.csv")
    data_by_year = pd.read_csv(f"{dataset_path}/data_by_year.csv")
    data_with_genres = pd.read_csv(f"{dataset_path}/data_w_genres.csv")
    data = pd.read_csv(f"{dataset_path}/data.csv")
    # clean data
    data_by_artist = clean_data(data_by_artist)
    data_by_genre = clean_data(data_by_genre)
    data_by_year = clean_data(data_by_year)
    data_with_genres = clean_data(data_with_genres)
    data = clean_data(data)
    return data_by_artist, data_by_genre, data_by_year, data_with_genres, data

def clean_data(data):
    # Clean the data: Drop rows with missing values in relevant columns
    columns_to_check = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    data = data.dropna(subset=columns_to_check)
    
    # Ensure that we have no duplicates
    data = data.drop_duplicates()
    
    return data

import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

def prepare_data_for_als(df):
    """
    Converts the dataframe into user-item-rating format and saves necessary mappings.

    Parameters:
    - df: DataFrame with at least 'artists', 'name', and 'popularity'

    Returns:
    - A new DataFrame with columns: user_id, item_id, rating
    """

    # Drop missing values
    df = df.dropna(subset=['artists', 'name', 'popularity'])

    # Encode artists to user_ids
    user_encoder = LabelEncoder()
    df['user_id'] = user_encoder.fit_transform(df['artists'])

    # Encode song names to item_ids
    item_encoder = LabelEncoder()
    df['item_id'] = item_encoder.fit_transform(df['name'])

    # Use 'popularity' as the implicit rating
    df['rating'] = df['popularity']

    # Save mappings
    user_id_mapping = dict(zip(df['artists'], df['user_id']))
    item_id_mapping = dict(zip(df['name'], df['item_id']))
    id_songname_mapping = dict(zip(df['item_id'], df['name']))

    with open("user_id_mapping.pkl", "wb") as f:
        pickle.dump(user_id_mapping, f)

    with open("item_id_mapping.pkl", "wb") as f:
        pickle.dump(item_id_mapping, f)

    with open("id_songname_mapping.pkl", "wb") as f:
        pickle.dump(id_songname_mapping, f)

    return df[['user_id', 'item_id', 'rating']]
