import os
import shutil
import logging
import datetime

import pandas as pd
import altair as alt
import torch
import types
import streamlit as st

import config

from src.data_loader import load_data
from src.preprocessing import preprocess_features

from src.meta_recommender import MetadataRecommender
from src.audio_recommender import AudioRecommender
from src.ml_recommender import MLRecommender
from src.dl_recommender import DLRecommender
from src.w2v_recommender import Word2VecRecommender
from src.hybrid_recommender import HybridRecommender
from src.evaluation import compare_models
from src.ranker import Ranker  # Import Ranker class

# Disable file watcher for torch
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
st.markdown(""" 
    <style> 
        body { 
            background: linear-gradient(135deg, #1f1c2c, #928dab); 
            background-attachment: fixed; 
            color: white; 
        } 
        .stApp { 
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364); 
            background-attachment: fixed; 
        } 
        section.main > div { 
            padding-top: 2rem; 
        } 
        .block-container { 
            padding: 2rem 2rem 2rem 2rem; 
        } 
    </style>
""", unsafe_allow_html=True)

st.markdown(""" 
    <style> 
        html, body, .stApp { 
            height: 100%; 
            width: 100%; 
            margin: 0; 
            padding: 0; 
            overflow-x: hidden; 
        }

        .stApp { 
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364); 
            background-attachment: fixed; 
        }

        section.main > div { 
            max-width: 100vw; 
            padding-top: 2rem; 
            padding-bottom: 2rem; 
        }

        .block-container { 
            max-width: 100%; 
            padding: 0 4vw; 
        }

        .element-container, .stTextInput, .stButton, .stMarkdown { 
            width: 100% !important; 
        }

        @media screen and (min-aspect-ratio: 16/9) { 
            .block-container { 
                height: 100vh; 
            } 
        } 
    </style>
""", unsafe_allow_html=True)

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

st.markdown("""
    <div style="background:radial-gradient(circle at center, #3a1c71, #d76d77, #ffaf7b);padding:30px;border-radius:12px;">
        <h1 style="color:white;text-align:center;font-size:46px;font-weight:700;">EchoMind</h1>
        <p style="color:white;text-align:center;font-size:17px;margin-top:10px;">Personalized soundscapes powered by machine learning</p>
        <hr style="margin:20px auto; width:50%; border: 1px solid rgba(255,255,255,0.3)">
        <p style="color:white;text-align:center;font-size:14px;margin-top:0;">Built by Prateek, Harsh and Ayush</p>
    </div>
""", unsafe_allow_html=True)
@st.experimental_fragment()
def chart_container(recs):
    # Prepare chart data with Rank, Track, and Score from recs
    chart_data = []
    for name, tracks in recs.items():
        for rank, (track, score) in enumerate(tracks):  # Extract track and score
            chart_data.append({'Model': name, 'Track': track, 'Score': score})

    chart_df = pd.DataFrame(chart_data)

    # Dropdown to select chart type
    chart_type = st.selectbox(
        "Choose a chart type:",
        ["Bar Chart", "Line Chart", "Pie Chart"]
    )

    # Dynamic chart rendering based on selected chart type
    if chart_type == "Bar Chart":
        bar_chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X('Track:N', title='Track'),
            y=alt.Y('Score:Q', title='Score'),
            color=alt.Color('Model:N', title='Model'),
            tooltip=['Model', 'Track', 'Score']
        ).properties(width=700, height=400).configure_axis(
            labelAngle=-45
        ).interactive()

        st.markdown("### Model Contribution Breakdown")
        st.altair_chart(bar_chart, use_container_width=True)


    elif chart_type == "Line Chart":
        line_chart = alt.Chart(chart_df).mark_line().encode(
            x=alt.X('Score:Q'),
            y=alt.Y('Track:N', sort='-x'),
            color='Model:N',
            tooltip=['Model', 'Track', 'Score']
        ).properties(width=700, height=300).interactive()
        st.markdown("### Model Contribution Breakdown")
        st.altair_chart(line_chart, use_container_width=True)

    elif chart_type == "Pie Chart":
        pie_chart = alt.Chart(chart_df).mark_arc().encode(
            theta='Score:Q',
            color='Model:N',
            tooltip=['Track', 'Score']
        ).properties(width=700, height=300).interactive()
        st.markdown("### Model Contribution Breakdown")
        st.altair_chart(pie_chart, use_container_width=True)

with st.spinner('Loading data and models...'):

    df = load_data('data')
    tfidf, tfidf_mat, scaler, audio_mat, w2v_model, w2v_emb = preprocess_features(df)

    # Initialize models
    recommenders = {
        'Collaborative Filtering': MetadataRecommender(tfidf_mat, df, tfidf),
        'Content-Based Filtering': AudioRecommender(audio_mat, df),
        'Hybrid': HybridRecommender(tfidf, tfidf_mat, audio_mat, df),
        'Random Forest': MLRecommender(df, audio_mat, tfidf, tfidf_mat),
        'Autoencoder': DLRecommender(audio_mat, df),
        'Semantic Model': Word2VecRecommender(w2v_model, w2v_emb, df)
    }

st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.05); 
                padding: 15px 20px; 
                border-left: 6px solid #ffaf7b;
                border-radius: 8px; 
                margin-bottom: 20px;
                color: #ffffff;">
        <strong>Ready for recommendations!</strong>
    </div>
""", unsafe_allow_html=True)


st.markdown("### Search")
query = st.text_input('Enter artist, track, year, genre, or mood:', placeholder="e.g., jazz 2020, workout, classical")

if st.button('Get Recommendations'):
    if not query:
        hour = datetime.datetime.now().hour
        query = 'morning' if 5 <= hour < 12 else 'afternoon' if 12 <= hour < 17 else 'evening' if 17 <= hour < 21 else 'night'

    recs = {name: mod.recommend(query, k=config.KNN_K) for name, mod in recommenders.items()}

    # Initialize Ranker class for ensemble scoring
    ranker = Ranker(k=config.KNN_K)

    # Rank songs based on weighted scores using the Ranker
    top_n_recs,recs = ranker.rank_songs(recs)

    st.markdown("### Top Recommendations")
    for tr, _ in top_n_recs:
        st.write(f"- {tr}")

    with st.expander("View Model-Specific Recommendations", expanded=False):
        st.markdown("#### Model-Wise Top Recommendations")

        model_names = list(recs.keys())
        cols = st.columns(2) if len(model_names) > 1 else [st]

        for idx, (name, tracks) in enumerate(recs.items()):
            with cols[idx % 2]:
                st.markdown(f"**{name}**")

                # Unpack each tuple (track, score) into separate columns
                df_tracks = pd.DataFrame({
                    'Rank': range(1, len(tracks) + 1),
                    'Track': [tr[0] for tr in tracks],  # Track names
                    'Score': [tr[1] for tr in tracks]   # Scores
                })

                # Display the dataframe with three columns: Rank, Track, and Score
                st.dataframe(df_tracks, hide_index=True, use_container_width=True)

        chart_container(recs)



# Footer
st.markdown("""
    <hr style="margin-top:40px;">
    <div style="text-align:center; color: #888888; font-size: 14px;">
        © 2025 Music Recommender · All rights reserved
    </div>
""", unsafe_allow_html=True)
