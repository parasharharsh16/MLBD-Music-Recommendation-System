from devlopment.collabortaive_filtering import train_als_model,prepare_data_for_training
from devlopment.dataloader import load_data
from config import collaborative_filtering_model_path
from devlopment.collabortaive_filtering import train_als_model, save_model
from devlopment.dataloader import load_data, prepare_data_for_als
from config import collaborative_filtering_model_path

# Load data
data_by_artist, data_by_genre, data_by_year, data_with_genres, data = load_data("data/spotify_dataset")

# Prepare training data and generate mappings
data_for_training = prepare_data_for_als(data)

# Train and save model
model = train_als_model(data_for_training)
save_model(model, collaborative_filtering_model_path)
