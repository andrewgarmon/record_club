import numpy as np
import pandas as pd
import streamlit as st
import matplotlib as plt

LISTENERS = [
    "Riley",
    "Sky",
    "Alex",
    "Harry",
    "Jose",
    "Thomas",
    "Manny",
    "Drew",
]


def make_albums_df(df: pd.DataFrame) -> pd.DataFrame:
    albums_df = (
        df[["Artist", "Album", "Requester", "Date"]]
        .rename(
            columns={
                "Artist": "artist",
                "Album": "album",
                "Requester": "requester",
                "Date": "date",
            }
        )
        .reset_index(drop=True)
    )
    return albums_df


def make_reviews_df(df: pd.DataFrame, listeners: list[str]) -> pd.DataFrame:
    reviews_rows = []
    for index, row in df.iterrows():
        if pd.isnull(row["Artist"]) or pd.isnull(row["Album"]):
            continue
        for listener in listeners:
            score = row.get(listener)
            if pd.isnull(score):
                continue
            reviews_rows.append(
                {
                    "listener": listener,
                    "artist": row["Artist"],
                    "album": row["Album"],
                    "score": score,
                    "favorite_track": row.get(f"{listener}.1"),
                    "least_favorite_track": row.get(f"{listener}.2"),
                }
            )

    return pd.DataFrame(reviews_rows)


def make_deviation_df(reviews_df: pd.DataFrame) -> pd.DataFrame:
    users = reviews_df['listener'].unique()
    similarity_matrix = pd.DataFrame(index=users, columns=users)
    for user1 in users:
        for user2 in users:
            if user1 != user2:
                reviews1 = reviews_df[reviews_df['listener'] == user1]
                reviews2 = reviews_df[reviews_df['listener'] == user2]
                common_albums = list(
                    set(reviews1['album'].unique()).intersection(
                        set(reviews2['album'].unique())
                    )
                )
                if common_albums:
                    merged_reviews = pd.merge(
                        reviews1[reviews1['album'].isin(common_albums)],
                        reviews2[reviews2['album'].isin(common_albums)],
                        on=['artist', 'album'],
                        suffixes=('_1', '_2'),
                    )

                    squared_errors = (
                        merged_reviews['score_1'] - merged_reviews['score_2']
                    ) ** 2
                    avg_squared_error = np.mean(squared_errors)
                    deviation_metric = np.sqrt(avg_squared_error)
                    similarity_matrix.loc[user1, user2] = (
                        deviation_metric.round(2)
                    )
    similarity_matrix.fillna(0, inplace=True)
    similarity_matrix.loc['average'] = similarity_matrix.mean().round(2)
    return similarity_matrix


def make_listener_requester_df(
    albums_df: pd.DataFrame, reviews_df: pd.DataFrame
) -> pd.DataFrame:
    # the chart
    merged_df = pd.merge(reviews_df, albums_df, on=['artist', 'album'])
    avg_scores_df = (
        merged_df.groupby(['listener', 'requester'])['score']
        .mean()
        .reset_index()
    )
    avg_scores_df['score'] = avg_scores_df['score'].round(1)
    return avg_scores_df.pivot(
        index='listener', columns='requester', values='score'
    )


if __name__ == "__main__":
    df = pd.read_csv('data.csv', encoding='utf_8')

    albums_df = make_albums_df(df)
    reviews_df = make_reviews_df(df, LISTENERS)

    st.markdown('#### Album Scores')
    st.dataframe(
        reviews_df.groupby(["artist", "album"])["score"]
        .agg(["mean", "median"])
        .reset_index()
        .round(2)
        .sort_values(by='mean', ascending=False)
        .style.background_gradient(axis=None, cmap='RdYlGn'),
        hide_index=True,
    )

    st.markdown('#### Favorite Tracks')
    st.dataframe(
        reviews_df.groupby(["artist", "album", "favorite_track"])["listener"]
        .count()
        .reset_index()
        .sort_values(by="listener", ascending=False)
        .rename(columns={"listener": "count"}),
        hide_index=True,
    )

    st.markdown('#### Least Favorite Tracks')
    st.dataframe(
        reviews_df.groupby(["artist", "album", "least_favorite_track"])[
            "listener"
        ]
        .count()
        .reset_index()
        .sort_values(by="listener", ascending=False)
        .rename(columns={"listener": "count"}),
        hide_index=True,
    )

    st.markdown('#### Average Score by Listener/Requester')
    st.dataframe(
        make_listener_requester_df(
            albums_df, reviews_df
        ).style.background_gradient(axis=None, cmap='RdYlGn')
    )

    st.markdown('#### Deviation from other listeners\' scores')
    st.dataframe(
        make_deviation_df(reviews_df).style.background_gradient(
            axis=None, cmap='RdYlGn_r'
        )
    )
