import numpy as np
import pandas as pd
import streamlit as st
import last_fm


def make_albums_df(df: pd.DataFrame) -> pd.DataFrame:
    return (
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

    reviews_rows = [
        {
            "listener": listener,
            "artist": row["Artist"],
            "album": row["Album"],
            "score": score,
            "favorite_track": row.get(f"{listener}.1"),
            "least_favorite_track": row.get(f"{listener}.2"),
        }
        for _, row in df.iterrows()
        if pd.notnull(row["Artist"]) and pd.notnull(row["Album"])
        for listener in listener_columns
        if pd.notnull(score := row.get(listener))
    ]

    return pd.DataFrame(reviews_rows)


def make_deviation_df() -> pd.DataFrame:
    reviews_df = st.session_state['reviews_df']
    users = reviews_df['listener'].unique()
    similarity_matrix = pd.DataFrame(index=users, columns=users)

    for user1 in users:
        for user2 in users:
            if user1 == user2:
                continue

            reviews1 = reviews_df[reviews_df['listener'] == user1]
            reviews2 = reviews_df[reviews_df['listener'] == user2]
            common_albums = set(reviews1['album']).intersection(
                reviews2['album']
            )

            if common_albums:
                merged_reviews = pd.merge(
                    reviews1[reviews1['album'].isin(common_albums)],
                    reviews2[reviews2['album'].isin(common_albums)],
                    on=['artist', 'album'],
                    suffixes=('_1', '_2'),
                )
                deviation_metric = np.sqrt(
                    np.mean(
                        (merged_reviews['score_1'] - merged_reviews['score_2'])
                        ** 2
                    )
                )
                similarity_matrix.loc[user1, user2] = deviation_metric.round(2)

    similarity_matrix.fillna(0, inplace=True)
    similarity_matrix.loc['average'] = similarity_matrix.mean().round(2)
    return similarity_matrix


def make_listener_requester_df() -> pd.DataFrame:
    albums_df = st.session_state['albums_df']
    reviews_df = st.session_state['reviews_df']
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
def _get_df_from_sheets(sheets_doc_id: str) -> pd.DataFrame:
    return pd.read_csv(
        f'https://docs.google.com/spreadsheets/d/{sheets_doc_id}/export?format=csv&gid=1242904482',
        encoding='utf_8',
    )


def display_summary_tables() -> None:
    reviews_df = st.session_state["reviews_df"]
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


def display_listener_analysis() -> None:
    st.markdown('#### Average Score by Listener/Requester')
    st.dataframe(
        st.session_state["listener_requester_df"]
        .style.format(precision=2)
        .background_gradient(axis=None, cmap='RdYlGn')
    )

    st.markdown('#### Deviation from other listeners\' scores')
    st.dataframe(
        st.session_state["deviation_df"]
        .style.format(precision=2)
        .background_gradient(axis=None, cmap='RdYlGn_r')
    )


def display_top_albums(lf_client: last_fm.LastFmClient) -> None:
    reviews_df = st.session_state["reviews_df"]
    st.markdown('#### Top Albums')
    top_albums = (
        reviews_df.groupby(["artist", "album"])["score"]
        .agg(["mean", "median"])
        .reset_index()
        .round(2)
        .sort_values(by="mean", ascending=False)
        .head(25)
    )
    album_list = list(zip(top_albums["artist"], top_albums["album"]))

    progress_bar = st.progress(0)
    album_images = []
    errors = []

    for index, (artist, album) in enumerate(album_list):
        try:
            album_data = lf_client.get_album(artist, album)
            album_art = album_data.get_album_art()
            album_images.append((artist, album, album_art))
        except Exception as e:
            errors.append((artist, album, str(e)))
        finally:
            progress_bar.progress((index + 1) / len(album_list))

    progress_bar.empty()

    columns = [st.columns(5) for _ in range(5)]
    for index, (artist, album, album_art) in enumerate(album_images):
        row, col = divmod(index, 5)
        with columns[row][col]:
            st.image(album_art, use_container_width=True)
            st.caption(f"{artist} - {album}")

    if errors:
        st.markdown("### Errors")
        for artist, album, error_message in errors:
            st.error(
                f"Failed to load: {artist} - {album}. Error: {error_message}"
            )


if __name__ == "__main__":
    st.set_page_config(page_title='Records and Rebuttals')
    sheets_doc_id = st.secrets['SHEETS_DOC_ID']
    df = _get_df_from_sheets(sheets_doc_id)
    lf_client = last_fm.LastFmClient(st.secrets['LAST_FM_API_KEY'])

    st.session_state["albums_df"] = make_albums_df(df)
    st.session_state["reviews_df"] = make_reviews_df(df)
    st.session_state["deviation_df"] = make_deviation_df()
    st.session_state["listener_requester_df"] = make_listener_requester_df()

    display_summary_tables()
    display_listener_analysis()
    display_top_albums(lf_client)
