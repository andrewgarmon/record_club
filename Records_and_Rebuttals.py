import streamlit as st
import data
import last_fm


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


st.set_page_config(page_title='Records and Rebuttals')
sheets_doc_id = st.secrets['SHEETS_DOC_ID']
df = data.load_sheet(sheets_doc_id)
lf_client = last_fm.LastFmClient(st.secrets['LAST_FM_API_KEY'])

listeners = data.get_listeners(df)
st.session_state["albums_df"] = data.build_albums_df(df)
st.session_state["reviews_df"] = data.build_reviews_df(df, listeners)
st.session_state["deviation_df"] = data.build_deviation_df(st.session_state["reviews_df"])
st.session_state["listener_requester_df"] = data.build_listener_requester_df(
    st.session_state["reviews_df"], st.session_state["albums_df"]
)
st.session_state["album_stats_df"] = data.build_album_stats_df(
    st.session_state["reviews_df"], st.session_state["albums_df"]
)

display_summary_tables()
display_listener_analysis()
display_top_albums(lf_client)
