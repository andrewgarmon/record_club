import streamlit as st
import pandas as pd
import last_fm


def _display_album(
    artist: str, album: str, lf_client: last_fm.LastFmClient, width: int = 300
):
    try:
        album_data = lf_client.get_album(artist, album)
        st.image(
            album_data.get_album_art(),
            caption=f"{artist} - {album}",
            width=width,
        )
    except Exception:
        st.write(f"{artist} - {album}")


def _display_scores_given(listener_requester_df, reviews_df, listener):
    given_scores = listener_requester_df.loc[[listener]]
    avg_score_given = reviews_df[reviews_df["listener"] == listener][
        "score"
    ].mean()
    given_scores["Average"] = avg_score_given
    valid_scores = given_scores.loc[listener].dropna()

    sorted_columns = valid_scores.argsort()[::-1]  # Sort remaining valid scores
    sorted_given_scores = given_scores[
        valid_scores.index[sorted_columns]
    ].reset_index(drop=True)

    # Calculate the average score the listener gave
    avg_score_given = reviews_df[reviews_df["listener"] == listener][
        "score"
    ].mean()
    # sorted_given_scores["Average"] = avg_score_given

    styled_given_scores = sorted_given_scores.style.format(
        precision=2
    ).background_gradient(axis=1, cmap="RdYlGn")
    st.dataframe(
        styled_given_scores,
        hide_index=True,
    )


def _display_scores_received(
    listener_requester_df, albums_df, reviews_df, listener
):
    received_scores = listener_requester_df[[listener]].dropna().T
    listener_albums = albums_df[albums_df["requester"] == listener][
        "album"
    ].unique()
    avg_score_received = reviews_df[
        reviews_df["album"].isin(listener_albums)
        & (reviews_df["listener"] != listener)
    ]["score"].mean()
    received_scores["Average"] = avg_score_received

    sorted_received_scores = received_scores[
        received_scores.columns[received_scores.loc[listener].argsort()[::-1]]
    ]

    styled_received_scores = sorted_received_scores.style.format(
        precision=2
    ).background_gradient(axis=1, cmap="RdYlGn")
    st.dataframe(styled_received_scores, hide_index=True)


def _display_deviation(deviation_df, listener):
    sorted_deviation = (
        deviation_df.loc[[listener]]
        .drop(columns=listener, errors="ignore")
        .sort_values(by=listener, axis=1)
    )
    st.dataframe(
        sorted_deviation.style.format(precision=2).background_gradient(
            axis=1, cmap="RdYlGn_r"
        ),
        hide_index=True,
    )


def _display_listener_details(
    listener,
    listener_requester_df,
    deviation_df,
    reviews_df,
    albums_df,
    lf_client,
):
    listener_reviews = reviews_df[reviews_df["listener"] == listener]

    if listener_reviews.empty:
        st.write("No reviews available for this listener.")
        return

    favorite_album = listener_reviews.loc[listener_reviews["score"].idxmax()]
    least_favorite_album = listener_reviews.loc[
        listener_reviews["score"].idxmin()
    ]

    fav_col, least_col = st.columns(2)
    with fav_col:
        st.markdown("### Favorite Album")
        _display_album(
            favorite_album['artist'], favorite_album['album'], lf_client
        )

    with least_col:
        st.markdown("### Least Favorite Album")
        _display_album(
            least_favorite_album['artist'],
            least_favorite_album['album'],
            lf_client,
        )

    st.markdown("#### Deviation from Other Listeners")
    if listener in deviation_df.index:
        _display_deviation(deviation_df, listener)
    else:
        st.write("No deviation data available.")

    st.markdown("#### Scores Given to Albums")
    if listener in listener_requester_df.index:
        _display_scores_given(listener_requester_df, reviews_df, listener)
    else:
        st.write("No scores available for this listener.")

    st.markdown("#### Scores Received by Listeners")
    if listener in listener_requester_df.columns:
        _display_scores_received(
            listener_requester_df, albums_df, reviews_df, listener
        )
    else:
        st.write("No scores available for this listener.")


if "reviews_df" in st.session_state and "albums_df" in st.session_state:
    reviews_df = st.session_state["reviews_df"]
    albums_df = st.session_state["albums_df"]
    deviation_df = st.session_state["deviation_df"]
    listener_requester_df = st.session_state["listener_requester_df"]
    lf_client = last_fm.LastFmClient(st.secrets['LAST_FM_API_KEY'])
    listeners = reviews_df["listener"].drop_duplicates().tolist()
    tabs = st.tabs(listeners)
    for listener, tab in zip(listeners, tabs):
        with tab:
            _display_listener_details(
                listener,
                listener_requester_df,
                deviation_df,
                reviews_df,
                albums_df,
                lf_client,
            )
else:
    st.error("No reviews data available. Please visit the main page first.")
