# EchoMind : Music Recommender App
 
 Personalized soundscapes powered by machine learning


A Streamlit application that provides song recommendations based on flexible user input (artist, track, year, genre, mood, or time of day). Implements six recommendation models: Metadata KNN, Audio KNN, Hybrid KNN, scmentic(word2vec), Random Forest (ML), and Autoencoder (DL).

## 👥 Contributors
- Prateek Singhal (M22AIE215)
- Harsh Parashar (M22AIE210)
- Ayush Mishra (M22AIE251)



#### Folder Structure
```plaintext
music_recommender_app/
├── config.py
├── data/
│   ├── data.csv
│   ├── data_by_artist.csv
│   ├── data_by_genres.csv
│   ├── data_by_year.csv
│   └── data_w_genres.csv
├── models/
│   ├── ml_model.pkl
│   ├── svd.pkl
│   ├── tfidf.pkl
│   ├── autoencoder.pth
│   ├── dl_embeddings.npy
│   ├── w2v_model.model
│   └── w2v_embeddings.npy
├── main.py
├── requirements.txt
├── README.md
└── src/
    ├── data_loader.py
    ├── model_persistence.py
    ├── preprocessing.py
    ├── content_recommender.py
    ├── audio_recommender.py
    ├── ml_recommender.py
    ├── dl_recommender.py
    ├── w2v_recommender.py
    ├── hybrid_recommender.py
    └── evaluation.py
```

## Features
- **MODE switch** (train/prod) in `config.py`
- **Model persistence** in `models/`
- **Aggregated stats** from artist/genre/year CSVs
- **INFO-level logging** for each step

## Setup
```bash
git clone https://github.com/parasharharsh16/MLBD-Music-Recommendation-System.git
cd music_recommender_app
pip install -r requirements.txt
```

## Run
```bash
streamlit run main.py
```
- In `train` mode, models retrain and overwrite.
- In `prod` mode, persisted models load quickly.

## Usage
1. Enter a query (artist, track, year, genre, mood).
2. Leave blank to use time-of-day defaults (morning/afternoon/evening/night).
3. See 5 recommendations per model; view the best model by weighted average popularity.