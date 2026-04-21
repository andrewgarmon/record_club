"""
Record Club Stats dashboard.

A mix of leaderboards, quirky superlatives, and charts built entirely off the
reviews/albums dataframes that the home page loaded into session state.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import data
import last_fm

st.set_page_config(page_title="Stats - Records and Rebuttals", layout="wide")
st.title("Record Club Stats")

data.ensure_session_state(st.secrets["SHEETS_DOC_ID"])


@st.cache_data(show_spinner=False)
def _load_album_tags(albums_key: tuple, _lf_client) -> pd.DataFrame:
    """Cache-wrapped genre-tag fetch. ``albums_key`` is just an (artist, album)
    tuple used as the cache key; ``_lf_client`` starts with an underscore so
    Streamlit skips hashing it."""
    albums = pd.DataFrame(list(albums_key), columns=["artist", "album"])
    return data.build_album_tags_df(albums, _lf_client)

reviews_df: pd.DataFrame = st.session_state["reviews_df"].copy()
albums_df: pd.DataFrame = st.session_state["albums_df"].copy()
album_stats: pd.DataFrame = st.session_state["album_stats_df"].copy()

# Defensive: if album_stats_df was stashed by a prior (pre-update) run,
# rebuild it rather than bailing with an empty page.
if album_stats.empty and not reviews_df.empty:
    album_stats = data.build_album_stats_df(reviews_df, albums_df)
    st.session_state["album_stats_df"] = album_stats

if reviews_df.empty:
    st.warning("No reviews yet — nothing to stat.")
    st.stop()


# ---------------------------------------------------------------------------
# Hero metrics
# ---------------------------------------------------------------------------
top_row = album_stats.sort_values("mean", ascending=False).iloc[0]
bottom_row = album_stats.sort_values("mean", ascending=True).iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Albums", len(album_stats))
c2.metric("Reviews", len(reviews_df))
c3.metric("Listeners", reviews_df["listener"].nunique())
c4.metric("Club Average", f"{reviews_df['score'].mean():.2f}")
c5.metric(
    "Top Album",
    f"{top_row['mean']:.2f}",
    f"{top_row['artist']} — {top_row['album']}",
)

st.caption(
    f"The club's lowest scoring record so far: **{bottom_row['artist']} — "
    f"{bottom_row['album']}** at **{bottom_row['mean']:.2f}**."
)

st.divider()

# ---------------------------------------------------------------------------
# Score distribution
# ---------------------------------------------------------------------------
st.subheader("Score Distribution")

dist_col, stat_col = st.columns([3, 1])
with dist_col:
    hist = (
        alt.Chart(reviews_df)
        .mark_bar(color="#e45756")
        .encode(
            x=alt.X(
                "score:Q",
                bin=alt.Bin(step=0.5),
                title="Score",
            ),
            y=alt.Y("count()", title="Reviews"),
            tooltip=[alt.Tooltip("count()", title="Reviews")],
        )
        .properties(height=260)
    )
    st.altair_chart(hist, use_container_width=True)

with stat_col:
    desc = reviews_df["score"].describe()
    st.metric("Mean", f"{desc['mean']:.2f}")
    st.metric("Median", f"{desc['50%']:.2f}")
    st.metric("Std Dev", f"{desc['std']:.2f}")
    perfect = (reviews_df["score"] >= 10).sum()
    zeros = (reviews_df["score"] <= 0).sum()
    st.metric("Perfect 10s", int(perfect))
    st.metric("Zeros", int(zeros))


# ---------------------------------------------------------------------------
# Decade breakdown
# ---------------------------------------------------------------------------
if "decade" in album_stats.columns and album_stats["decade"].notna().any():
    st.divider()
    st.subheader("By Decade")

    decade_df = (
        album_stats.dropna(subset=["decade"])
        .groupby("decade")
        .agg(albums=("album", "count"), avg_score=("mean", "mean"))
        .reset_index()
    )
    # Sort decades numerically when possible (handles "60s", "1960s", etc.)
    decade_df["_sort_key"] = (
        decade_df["decade"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
        .fillna(-1)
    )
    decade_df = decade_df.sort_values("_sort_key").drop(columns="_sort_key")
    decade_df["avg_score"] = decade_df["avg_score"].round(2)
    score_min = decade_df["avg_score"].min() - 5
    score_max = decade_df["avg_score"].max() + 5

    left, right = st.columns(2)
    with left:
        st.markdown("**Albums per decade**")
        chart = (
            alt.Chart(decade_df)
            .mark_bar(color="#4c78a8")
            .encode(
                x=alt.X("decade:N", sort=list(decade_df["decade"]), title=None),
                y=alt.Y("albums:Q", title="Albums"),
                tooltip=["decade", "albums"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)
    with right:
        st.markdown("**Average score by decade**")
        chart = (
            alt.Chart(decade_df)
            .mark_bar(color="#54a24b")
            .encode(
                x=alt.X("decade:N", sort=list(decade_df["decade"]), title=None),
                y=alt.Y(
                    "avg_score:Q",
                    title="Avg score",
                    scale=alt.Scale(domain=[score_min, score_max]),
                ),
                tooltip=["decade", "avg_score"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)


# ---------------------------------------------------------------------------
# Release year trend
# ---------------------------------------------------------------------------
if "release_year" in album_stats.columns and album_stats["release_year"].notna().any():
    st.divider()
    st.subheader("Does the Club Prefer Old or New?")

    year_df = album_stats.dropna(subset=["release_year"]).copy()
    year_df["release_year"] = pd.to_numeric(
        year_df["release_year"], errors="coerce"
    )
    year_df = year_df.dropna(subset=["release_year"])

    if not year_df.empty:
        points = (
            alt.Chart(year_df)
            .mark_circle(size=90, opacity=0.75)
            .encode(
                x=alt.X(
                    "release_year:Q",
                    title="Release year",
                    scale=alt.Scale(zero=False),
                ),
                y=alt.Y(
                    "mean:Q",
                    title="Average score",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color("mean:Q", scale=alt.Scale(scheme="redyellowgreen")),
                tooltip=["artist", "album", "release_year", "mean"],
            )
        )
        trend = (
            alt.Chart(year_df)
            .transform_regression("release_year", "mean")
            .mark_line(color="#e45756", strokeWidth=3)
            .encode(x="release_year:Q", y="mean:Q")
        )
        st.altair_chart(
            (points + trend).properties(height=340),
            use_container_width=True,
        )

        # Correlation — does the club skew old or new?
        corr = year_df[["release_year", "mean"]].corr().iloc[0, 1]
        if pd.notna(corr):
            if abs(corr) < 0.1:
                verdict = "basically indifferent to release year"
            elif corr > 0:
                verdict = "slightly partial to **newer** albums"
            else:
                verdict = "slightly partial to **older** albums"
            st.caption(
                f"Release-year/score correlation is **{corr:+.2f}** — the club is "
                f"{verdict}."
            )


# ---------------------------------------------------------------------------
# Genre / tag breakdown (Last.fm)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("What Does the Club Actually Listen To?")
st.caption(
    "Genre tags pulled from Last.fm. Each album contributes up to its top five "
    "tags; personal labels (favorites, seen live, nationalities) and decade "
    "tags are filtered out."
)

_lf_client = last_fm.LastFmClient(st.secrets["LAST_FM_API_KEY"])
_albums_key = tuple(
    albums_df[["artist", "album"]]
    .dropna()
    .itertuples(index=False, name=None)
)
with st.spinner("Fetching genre tags from Last.fm..."):
    tags_df = _load_album_tags(_albums_key, _lf_client)

if tags_df.empty:
    st.info("Couldn't load any genre tags — check the Last.fm API key or network.")
else:
    tag_counts = (
        tags_df.groupby("tag")
        .size()
        .reset_index(name="albums")
        .sort_values("albums", ascending=False)
    )

    # Join tags with per-album mean score for score-by-genre.
    tag_scores_all = tags_df.merge(
        album_stats[["artist", "album", "mean"]],
        on=["artist", "album"],
        how="left",
    ).dropna(subset=["mean"])
    tag_scores = (
        tag_scores_all.groupby("tag")
        .agg(albums=("album", "count"), avg_score=("mean", "mean"))
        .reset_index()
        .round({"avg_score": 2})
    )

    top_n = 15
    top_genres = tag_counts.head(top_n)
    # Only rank score-by-genre on tags that appear at least twice — a single
    # album's score isn't really a "genre average".
    scored_genres = (
        tag_scores[tag_scores["albums"] >= 2]
        .sort_values("avg_score", ascending=False)
        .head(top_n)
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Distinct genres", int(tag_counts["tag"].nunique()))
    m2.metric("Most common genre", top_genres.iloc[0]["tag"])
    if not scored_genres.empty:
        best = scored_genres.iloc[0]
        m3.metric(
            "Club's favorite genre",
            best["tag"],
            f"avg {best['avg_score']:.2f} over {int(best['albums'])} albums",
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**Most common genres**")
        chart = (
            alt.Chart(top_genres)
            .mark_bar(color="#6f42c1")
            .encode(
                x=alt.X("albums:Q", title="Albums"),
                y=alt.Y("tag:N", sort="-x", title=None),
                tooltip=["tag", "albums"],
            )
            .properties(height=28 * len(top_genres) + 40)
        )
        st.altair_chart(chart, use_container_width=True)
    with right:
        st.markdown("**Best-received genres** (≥ 2 albums)")
        if scored_genres.empty:
            st.caption(
                "No genre has two or more albums yet — check back once the "
                "club's racked up a few more picks."
            )
        else:
            score_min = max(0, scored_genres["avg_score"].min() - 1)
            score_max = min(10, scored_genres["avg_score"].max() + 1)
            chart = (
                alt.Chart(scored_genres)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "avg_score:Q",
                        title="Avg score",
                        scale=alt.Scale(domain=[score_min, score_max]),
                    ),
                    y=alt.Y("tag:N", sort="-x", title=None),
                    color=alt.Color(
                        "avg_score:Q",
                        scale=alt.Scale(scheme="redyellowgreen"),
                        legend=None,
                    ),
                    tooltip=["tag", "avg_score", "albums"],
                )
                .properties(height=28 * len(scored_genres) + 40)
            )
            st.altair_chart(chart, use_container_width=True)

    # Scatter: how much the club likes a genre vs. how often it shows up.
    if (tag_scores["albums"] >= 2).any():
        st.markdown("**Genre popularity vs. club reception**")
        scatter_df = tag_scores[tag_scores["albums"] >= 2]
        scatter = (
            alt.Chart(scatter_df)
            .mark_circle(size=140, opacity=0.8)
            .encode(
                x=alt.X("albums:Q", title="Albums in this genre"),
                y=alt.Y(
                    "avg_score:Q",
                    title="Avg club score",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "avg_score:Q",
                    scale=alt.Scale(scheme="redyellowgreen"),
                    legend=None,
                ),
                tooltip=["tag", "albums", "avg_score"],
            )
        )
        labels = (
            alt.Chart(scatter_df)
            .mark_text(align="left", dx=8, dy=-2, fontSize=11)
            .encode(x="albums:Q", y="avg_score:Q", text="tag:N")
        )
        st.altair_chart(
            (scatter + labels).properties(height=360),
            use_container_width=True,
        )

    with st.expander("Per-album genre tags"):
        per_album = (
            tags_df.groupby(["artist", "album"])["tag"]
            .apply(lambda tags: ", ".join(sorted(set(tags))))
            .reset_index()
            .rename(columns={"tag": "tags"})
            .sort_values(["artist", "album"])
        )
        st.dataframe(per_album, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Top / bottom / divisive / unanimous
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Leaderboards")

tab_top, tab_bottom, tab_div, tab_unan = st.tabs(
    ["Top 10", "Bottom 10", "Most Divisive", "Most Unanimous"]
)

display_cols = ["artist", "album", "mean", "median", "std", "count"]
if "requester" in album_stats.columns:
    display_cols.insert(2, "requester")

with tab_top:
    top10 = album_stats.sort_values("mean", ascending=False).head(10)[display_cols]
    st.dataframe(
        top10.style.background_gradient(subset=["mean"], cmap="RdYlGn"),
        hide_index=True,
        use_container_width=True,
    )

with tab_bottom:
    bottom10 = album_stats.sort_values("mean", ascending=True).head(10)[
        display_cols
    ]
    st.dataframe(
        bottom10.style.background_gradient(subset=["mean"], cmap="RdYlGn"),
        hide_index=True,
        use_container_width=True,
    )

with tab_div:
    divisive = album_stats[album_stats["count"] >= 2].sort_values(
        "std", ascending=False
    ).head(10)[display_cols]
    st.caption("Biggest score spread between listeners — the albums that started fights.")
    st.dataframe(
        divisive.style.background_gradient(subset=["std"], cmap="Reds"),
        hide_index=True,
        use_container_width=True,
    )

with tab_unan:
    unanimous = album_stats[album_stats["count"] >= 2].sort_values(
        ["std", "mean"], ascending=[True, False]
    ).head(10)[display_cols]
    st.caption("Lowest score spread — the club's rare moments of harmony.")
    st.dataframe(
        unanimous.style.background_gradient(subset=["mean"], cmap="RdYlGn"),
        hide_index=True,
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Listener superlatives
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Listener Superlatives")

listener_stats = (
    reviews_df.groupby("listener")["score"]
    .agg(["mean", "median", "std", "min", "max", "count"])
    .reset_index()
    .rename(
        columns={
            "mean": "avg",
            "median": "median",
            "std": "spread",
            "min": "floor",
            "max": "ceiling",
            "count": "reviews",
        }
    )
    .round(2)
)

harshest = listener_stats.sort_values("avg", ascending=True).iloc[0]
kindest = listener_stats.sort_values("avg", ascending=False).iloc[0]
most_consistent = listener_stats[listener_stats["reviews"] >= 2].sort_values(
    "spread", ascending=True
).iloc[0] if (listener_stats["reviews"] >= 2).any() else None
most_volatile = listener_stats[listener_stats["reviews"] >= 2].sort_values(
    "spread", ascending=False
).iloc[0] if (listener_stats["reviews"] >= 2).any() else None
most_active = listener_stats.sort_values("reviews", ascending=False).iloc[0]

cols = st.columns(5)
cols[0].metric("Harshest Critic", harshest["listener"], f"avg {harshest['avg']:.2f}")
cols[1].metric("Most Generous", kindest["listener"], f"avg {kindest['avg']:.2f}")
if most_consistent is not None:
    cols[2].metric(
        "Most Consistent",
        most_consistent["listener"],
        f"σ {most_consistent['spread']:.2f}",
    )
if most_volatile is not None:
    cols[3].metric(
        "Most All-Over-the-Place",
        most_volatile["listener"],
        f"σ {most_volatile['spread']:.2f}",
    )
cols[4].metric(
    "Most Active",
    most_active["listener"],
    f"{int(most_active['reviews'])} reviews",
)

st.markdown("**Full listener breakdown**")
st.dataframe(
    listener_stats.style.background_gradient(subset=["avg"], cmap="RdYlGn")
    .background_gradient(subset=["spread"], cmap="Reds")
    .format(precision=2),
    hide_index=True,
    use_container_width=True,
)


# ---------------------------------------------------------------------------
# Requester leaderboard — who picks the best records?
# ---------------------------------------------------------------------------
if "requester" in album_stats.columns and album_stats["requester"].notna().any():
    st.divider()
    st.subheader("Who Picks the Best Records?")

    requester_stats = (
        album_stats.dropna(subset=["requester"])
        .groupby("requester")
        .agg(
            picks=("album", "count"),
            avg_reception=("mean", "mean"),
            best=("mean", "max"),
            worst=("mean", "min"),
        )
        .reset_index()
        .round(2)
        .sort_values("avg_reception", ascending=False)
    )
    st.dataframe(
        requester_stats.style.background_gradient(
            subset=["avg_reception"], cmap="RdYlGn"
        ),
        hide_index=True,
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Cumulative timeline
# ---------------------------------------------------------------------------
if (
    "date" in albums_df.columns
    and pd.api.types.is_datetime64_any_dtype(albums_df["date"])
    and albums_df["date"].notna().any()
):
    st.divider()
    st.subheader("Club Timeline")

    timeline = albums_df.dropna(subset=["date"]).sort_values("date").copy()
    timeline["cumulative"] = np.arange(1, len(timeline) + 1)

    # Merge in mean scores to color the timeline
    timeline = timeline.merge(
        album_stats[["artist", "album", "mean"]],
        on=["artist", "album"],
        how="left",
    )

    line = (
        alt.Chart(timeline)
        .mark_line(color="#4c78a8", strokeWidth=2)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("cumulative:Q", title="Albums reviewed"),
        )
    )
    dots = (
        alt.Chart(timeline)
        .mark_circle(size=80)
        .encode(
            x="date:T",
            y="cumulative:Q",
            color=alt.Color(
                "mean:Q",
                scale=alt.Scale(scheme="redyellowgreen"),
                title="Score",
            ),
            tooltip=["date:T", "artist", "album", "mean"],
        )
    )
    st.altair_chart((line + dots).properties(height=320), use_container_width=True)


# ---------------------------------------------------------------------------
# Hot takes — biggest deviation from club average on a single album
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Hottest Takes")
st.caption(
    "Where a single listener's score strayed the furthest from the rest of the "
    "club on one album."
)

takes = reviews_df.merge(
    album_stats[["artist", "album", "mean"]], on=["artist", "album"]
)
takes["delta"] = takes["score"] - takes["mean"]
takes["abs_delta"] = takes["delta"].abs()
hot_takes = takes.sort_values("abs_delta", ascending=False).head(10)[
    ["listener", "artist", "album", "score", "mean", "delta"]
].rename(columns={"mean": "club_avg"})
st.dataframe(
    hot_takes.style.background_gradient(subset=["delta"], cmap="RdBu")
    .format(precision=2),
    hide_index=True,
    use_container_width=True,
)
