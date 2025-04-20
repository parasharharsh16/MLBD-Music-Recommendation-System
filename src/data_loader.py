import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def load_data(folder: str) -> pd.DataFrame:
    """
    Load track-level data from data.csv, standardize columns,
    merge aggregated stats, and build `metadata`.
    """
    base = os.path.join(folder, 'data.csv')
    if not os.path.exists(base):
        raise FileNotFoundError(f"data.csv not found in {folder}")
    df = pd.read_csv(base)
    logger.info('Loaded data.csv: %s', df.shape)

    # Standardize names
    if 'track_name' not in df.columns:
        if 'name' in df.columns:
            df.rename(columns={'name': 'track_name'}, inplace=True)
        else:
            raise KeyError("Missing 'track_name' or 'name' column")
    if 'artist_name' not in df.columns:
        if 'artists' in df.columns:
            df.rename(columns={'artists': 'artist_name'}, inplace=True)
        else:
            raise KeyError("Missing 'artist_name' or 'artists' column")

    # Extract year
    if 'year' not in df.columns and 'release_date' in df.columns:
        df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

    # Merge aggregates
    for fname, key in [('data_by_artist.csv', 'artist_name'),
                       ('data_by_genres.csv', 'genres'),
                       ('data_by_year.csv', 'year')]:
        path = os.path.join(folder, fname)
        if os.path.exists(path) and key in df.columns:
            agg = pd.read_csv(path)
            if key == 'artist_name' and 'artists' in agg.columns:
                agg.rename(columns={'artists': 'artist_name'}, inplace=True)
            df = df.merge(agg, how='left', on=key, suffixes=('', f'_{fname.split(".")[0]}'))
            logger.info('Merged %s', fname)

    # Build metadata
    meta = df['track_name'].astype(str) + ' ' + df['artist_name'].astype(str)
    if 'genres' in df.columns:
        meta += ' ' + df['genres'].astype(str)
    if 'year' in df.columns:
        meta += ' ' + df['year'].astype(str)
    df['metadata'] = meta
    logger.info('Constructed metadata; shape %s', df.shape)

    return df.dropna(subset=['track_name', 'artist_name'])