import streamlit as st
import pickle
import pandas as pd
from config import collaborative_filtering_model_path

# Load the trained collaborative filtering model
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

# Load ID mappings (user_id ↔ artist, item_id ↔ song name)
@st.cache_data
def load_mappings():
    with open("user_id_mapping.pkl", "rb") as f:
        user_id_mapping = pickle.load(f)
    with open("item_id_mapping.pkl", "rb") as f:
        item_id_mapping = pickle.load(f)
    with open("id_songname_mapping.pkl", "rb") as f:
        id_songname_mapping = pickle.load(f)
    return user_id_mapping, item_id_mapping, id_songname_mapping

# Generate recommendations for a given artist
def recommend_songs(model, user_id, all_song_ids, id_to_songname, top_n=10):
    predictions = [model.predict(user_id, item_id) for item_id in all_song_ids]
    sorted_preds = sorted(predictions, key=lambda x: x.est, reverse=True)
    top_recs = sorted_preds[:top_n]
    return [(id_to_songname[p.iid], round(p.est, 2)) for p in top_recs]

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="🎵 AI Music Recommender", layout="centered")

st.title("🎵 AI-Powered Music Recommender")
st.markdown("Find top song recommendations based on your favorite artist using collaborative filtering.")

# Load model and mappings
model = load_model(collaborative_filtering_model_path)
user_to_artist, item_to_song, id_to_songname = load_mappings()
artist_to_user_id = user_to_artist
all_song_ids = list(id_to_songname.keys())

# UI for collaborative filtering
artist_name = st.selectbox("🎤 Select an Artist", sorted(artist_to_user_id.keys()))

if st.button("🎧 Recommend Songs"):
    user_id = artist_to_user_id.get(artist_name)
    if user_id is None:
        st.error("❌ Artist not found in the trained model.")
    else:
        recommendations = recommend_songs(model, user_id, all_song_ids, id_to_songname)
        st.success("✅ Here are your song recommendations:")
        for i, (song, score) in enumerate(recommendations, 1):
            st.write(f"{i}. {song} — Predicted Score: {score}")
