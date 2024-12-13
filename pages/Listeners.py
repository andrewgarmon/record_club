import streamlit as st
import pandas as pd
from typing import List


def display_listener_tabs(reviews_df: pd.DataFrame) -> None:
    listeners = list(reviews_df["listener"].drop_duplicates())
    tabs = st.tabs(listeners)

    for listener, tab in zip(listeners, tabs):
        with tab:
            display_listener_info(reviews_df, listener)


def display_listener_info(reviews_df: pd.DataFrame, listener: str) -> None:
    listener_reviews = reviews_df[reviews_df["listener"] == listener]

    if not listener_reviews.empty:
        highest_rated = listener_reviews.loc[listener_reviews["score"].idxmax()]
        album = highest_rated["album"]
        score = highest_rated["score"]

        st.subheader(f"Highest Rated Album for {listener}")
        st.write(f"**Album:** {album}")
        st.write(f"**Score:** {score}")
    else:
        st.write("No reviews available for this listener.")


if "reviews_df" in st.session_state:
    reviews_df = st.session_state["reviews_df"]
    display_listener_tabs(reviews_df)
else:
    st.error("No reviews data available. Please visit the main page first.")
