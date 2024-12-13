import streamlit as st

if "reviews_df" in st.session_state:
    reviews_df = st.session_state["reviews_df"]
    listeners = list(reviews_df["listener"].drop_duplicates())
    tabs = st.tabs(listeners)
    for i, listener in enumerate(listeners):
        with tabs[i]:
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
else:
    st.error("No reviews data available. Please visit the main page first.")
