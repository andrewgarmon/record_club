import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def load_sheet(sheets_doc_id: str) -> pd.DataFrame:
    df = pd.read_csv(
        f'https://docs.google.com/spreadsheets/d/{sheets_doc_id}/export?format=csv&sheet=Albums',
        encoding='utf_8',
    )
    return _normalize_columns(df)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the whitespace date column to 'Date'."""
    renames = {}
    for col in df.columns:
        if col.strip() == '' and 'Date' not in df.columns:
            renames[col] = 'Date'
            break
    if renames:
        df = df.rename(columns=renames)
    return df


def get_listeners(df: pd.DataFrame) -> list[str]:
    """Detect listener columns positionally: everything between 'Release Year' (or 'Album') and 'Average'."""
    cols = list(df.columns)
    avg_idx = cols.index('Average')
    # Find the last metadata column before Average
    metadata = ['Date', 'Requester', 'Artist', 'Album', 'Release Year', 'Decade']
    last_meta_idx = max(
        (cols.index(c) for c in metadata if c in cols and cols.index(c) < avg_idx),
        default=0,
    )
    return [
        c for c in cols[last_meta_idx + 1 : avg_idx]
        if not c.endswith(('.1', '.2')) and not c.startswith('Unnamed')
    ]


def build_albums_df(df: pd.DataFrame) -> pd.DataFrame:
    columns = {
        'Artist': 'artist',
        'Album': 'album',
        'Requester': 'requester',
        'Date': 'date',
    }
    if 'Release Year' in df.columns:
        columns['Release Year'] = 'release_year'
    available = {k: v for k, v in columns.items() if k in df.columns}
    return df[list(available.keys())].rename(columns=available).reset_index(drop=True)


def build_reviews_df(df: pd.DataFrame, listeners: list[str]) -> pd.DataFrame:
    reviews_rows = [
        {
            'listener': listener,
            'artist': row['Artist'],
            'album': row['Album'],
            'score': score,
            'favorite_track': row.get(f'{listener}.1'),
            'least_favorite_track': row.get(f'{listener}.2'),
        }
        for _, row in df.iterrows()
        if pd.notnull(row['Artist']) and pd.notnull(row['Album'])
        for listener in listeners
        if pd.notnull(score := row.get(listener))
    ]
    return pd.DataFrame(reviews_rows)


def build_deviation_df(reviews_df: pd.DataFrame) -> pd.DataFrame:
    users = reviews_df['listener'].unique()
    similarity_matrix = pd.DataFrame(index=users, columns=users)

    for user1 in users:
        for user2 in users:
            if user1 == user2:
                continue

            reviews1 = reviews_df[reviews_df['listener'] == user1]
            reviews2 = reviews_df[reviews_df['listener'] == user2]
            common_albums = set(reviews1['album']).intersection(reviews2['album'])

            if common_albums:
                merged_reviews = pd.merge(
                    reviews1[reviews1['album'].isin(common_albums)],
                    reviews2[reviews2['album'].isin(common_albums)],
                    on=['artist', 'album'],
                    suffixes=('_1', '_2'),
                )
                deviation_metric = np.sqrt(
                    np.mean(
                        (merged_reviews['score_1'] - merged_reviews['score_2']) ** 2
                    )
                )
                similarity_matrix.loc[user1, user2] = deviation_metric.round(2)

    similarity_matrix = similarity_matrix.fillna(0).infer_objects()
    similarity_matrix.loc['average'] = similarity_matrix.mean().round(2)
    return similarity_matrix


def build_listener_requester_df(
    reviews_df: pd.DataFrame, albums_df: pd.DataFrame
) -> pd.DataFrame:
    merged_df = pd.merge(reviews_df, albums_df, on=['artist', 'album'])
    avg_scores_df = (
        merged_df.groupby(['listener', 'requester'])['score']
        .mean()
        .reset_index()
    )
    avg_scores_df['score'] = avg_scores_df['score'].round(1)
    return avg_scores_df.pivot(index='listener', columns='requester', values='score')
