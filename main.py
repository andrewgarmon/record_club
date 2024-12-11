import numpy as np
import pandas as pd
import streamlit as st
import matplotlib as plt

import last_fm


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


def make_reviews_df(df: pd.DataFrame) -> pd.DataFrame:
    listener_columns = [
        col
        for col in df.columns
        if not col.endswith(('.1', '.2', 'Unnamed'))
        and col
        not in [
            'Date',
            'Requester',
            'Artist',
            'Album',
            'Average',
            'Median',
            'Favorite',
            'Worst',
        ]
    ]

    reviews_rows = []
    for index, row in df.iterrows():
        if pd.isnull(row["Artist"]) or pd.isnull(row["Album"]):
            continue
        for listener in listener_columns:
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


@st.cache_data
def _get_df_from_sheets(sheets_doc_id):
    return pd.read_csv(
        f'https://docs.google.com/spreadsheets/d/{sheets_doc_id}/export?format=csv&gid=1242904482',
        encoding='utf_8',
    )


if __name__ == "__main__":
    sheets_doc_id = st.secrets['SHEETS_DOC_ID']
    df = _get_df_from_sheets(sheets_doc_id)
    albums_df = make_albums_df(df)
    reviews_df = make_reviews_df(df)
    lf_client = last_fm.LastFmClient(st.secrets['LAST_FM_API_KEY'])

    listeners = list(reviews_df["listener"].drop_duplicates())
    tabs = st.tabs(["Main"] + listeners)

    with tabs[0]:
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
            reviews_df.groupby(["artist", "album", "favorite_track"])[
                "listener"
            ]
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

        top_albums = (
            reviews_df.groupby(["artist", "album"])["score"]
            .agg(["mean", "median"])
            .reset_index()
            .round(2)
            .sort_values(by="mean", ascending=False)
            .head(25)
        )
        album_list = list(zip(top_albums["artist"], top_albums["album"]))
        columns = [st.columns(5) for _ in range(5)]  # 5 rows, 5 columns per row
        for index, (artist, album) in enumerate(album_list):
            row, col = divmod(
                index, 5
            )  # Determine the row and column dynamically
            with columns[row][col]:
                try:
                    # Fetch album and render image
                    album_data = lf_client.get_album(artist, album)
                    st.image(
                        album_data.get_album_art(), use_container_width=True
                    )
                except Exception as e:
                    print(e)
                    st.error(f"Error loading album: {artist} - {album}")

    for i, listener in enumerate(listeners):
        with tabs[i + 1]:
            listener_reviews = reviews_df[reviews_df["listener"] == listener]

            if not listener_reviews.empty:
                highest_rated = listener_reviews.loc[
                    listener_reviews["score"].idxmax()
                ]
                album = highest_rated["album"]
                score = highest_rated["score"]

                st.subheader(f"Highest Rated Album for {listener}")
                st.write(f"**Album:** {album}")
                st.write(f"**Score:** {score}")
            else:
                st.write("No reviews available for this listener.")
