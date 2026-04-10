"""Tests for the pure data-munging helpers in ``data.py``."""
import numpy as np
import pandas as pd
import pytest

import data


class TestNormalizeColumns:
    def test_renames_whitespace_column_to_date(self):
        df = pd.DataFrame({" ": ["2024-01-01"], "Artist": ["Beatles"]})
        result = data._normalize_columns(df)
        assert "Date" in result.columns
        assert " " not in result.columns

    def test_leaves_existing_date_column_alone(self):
        df = pd.DataFrame(
            {"Date": ["2024-01-01"], " ": ["junk"], "Artist": ["Beatles"]}
        )
        result = data._normalize_columns(df)
        # Existing Date column wins, whitespace column is untouched.
        assert list(result.columns) == ["Date", " ", "Artist"]

    def test_noop_when_no_whitespace_column(self):
        df = pd.DataFrame({"Artist": ["Beatles"], "Album": ["Abbey Road"]})
        result = data._normalize_columns(df)
        pd.testing.assert_frame_equal(result, df)


class TestGetListeners:
    def test_detects_listeners_between_metadata_and_average(self, raw_sheet_df):
        df = data._normalize_columns(raw_sheet_df)
        assert data.get_listeners(df) == ["Alice", "Bob", "Carol"]

    def test_excludes_favorite_and_least_favorite_columns(self, raw_sheet_df):
        df = data._normalize_columns(raw_sheet_df)
        listeners = data.get_listeners(df)
        assert not any(name.endswith((".1", ".2")) for name in listeners)

    def test_excludes_unnamed_columns(self, raw_sheet_df):
        df = data._normalize_columns(raw_sheet_df)
        df.insert(len(df.columns) - 1, "Unnamed: 42", [None] * len(df))
        listeners = data.get_listeners(df)
        assert "Unnamed: 42" not in listeners

    def test_raises_if_average_column_missing(self):
        df = pd.DataFrame({"Artist": ["Beatles"], "Alice": [8]})
        with pytest.raises(ValueError):
            data.get_listeners(df)


class TestBuildAlbumsDf:
    def test_selects_and_renames_columns(self, raw_sheet_df):
        df = data._normalize_columns(raw_sheet_df)
        albums = data.build_albums_df(df)
        assert list(albums.columns) == [
            "artist",
            "album",
            "requester",
            "date",
            "release_year",
        ]
        assert len(albums) == 3
        assert albums.iloc[0]["artist"] == "Beatles"
        assert albums.iloc[0]["album"] == "Abbey Road"

    def test_works_without_release_year_column(self, raw_sheet_df):
        df = data._normalize_columns(raw_sheet_df).drop(columns=["Release Year"])
        albums = data.build_albums_df(df)
        assert "release_year" not in albums.columns
        assert "artist" in albums.columns


class TestBuildReviewsDf:
    def test_flattens_wide_sheet_into_long_reviews(self, raw_sheet_df):
        df = data._normalize_columns(raw_sheet_df)
        listeners = data.get_listeners(df)
        reviews = data.build_reviews_df(df, listeners)

        # 3 albums x 3 listeners = 9 reviews
        assert len(reviews) == 9
        assert set(reviews.columns) == {
            "listener",
            "artist",
            "album",
            "score",
            "favorite_track",
            "least_favorite_track",
        }

    def test_captures_favorite_and_least_favorite_tracks(self, raw_sheet_df):
        df = data._normalize_columns(raw_sheet_df)
        reviews = data.build_reviews_df(df, data.get_listeners(df))

        alice_zep = reviews[
            (reviews["listener"] == "Alice") & (reviews["album"] == "IV")
        ].iloc[0]
        assert alice_zep["favorite_track"] == "Black Dog"
        assert alice_zep["least_favorite_track"] == "Four Sticks"

    def test_skips_rows_with_missing_artist_or_album(self, raw_sheet_df):
        df = data._normalize_columns(raw_sheet_df)
        df.loc[1, "Artist"] = None
        reviews = data.build_reviews_df(df, data.get_listeners(df))
        assert "Let It Bleed" not in reviews["album"].tolist()

    def test_skips_reviews_with_null_scores(self, raw_sheet_df):
        df = data._normalize_columns(raw_sheet_df)
        df.loc[0, "Alice"] = None
        reviews = data.build_reviews_df(df, data.get_listeners(df))
        alice_abbey = reviews[
            (reviews["listener"] == "Alice") & (reviews["album"] == "Abbey Road")
        ]
        assert alice_abbey.empty


class TestBuildDeviationDf:
    def test_returns_square_matrix_plus_average_row(self, raw_sheet_df):
        df = data._normalize_columns(raw_sheet_df)
        reviews = data.build_reviews_df(df, data.get_listeners(df))
        deviation = data.build_deviation_df(reviews)

        listeners = sorted(reviews["listener"].unique())
        assert sorted(deviation.columns.tolist()) == listeners
        assert "average" in deviation.index
        # The non-average rows are the listener names.
        non_avg = [i for i in deviation.index if i != "average"]
        assert sorted(non_avg) == listeners

    def test_diagonal_is_zero(self, raw_sheet_df):
        df = data._normalize_columns(raw_sheet_df)
        reviews = data.build_reviews_df(df, data.get_listeners(df))
        deviation = data.build_deviation_df(reviews)

        for listener in reviews["listener"].unique():
            assert deviation.loc[listener, listener] == 0

    def test_deviation_is_symmetric(self, raw_sheet_df):
        df = data._normalize_columns(raw_sheet_df)
        reviews = data.build_reviews_df(df, data.get_listeners(df))
        deviation = data.build_deviation_df(reviews)

        listeners = reviews["listener"].unique()
        for a in listeners:
            for b in listeners:
                assert deviation.loc[a, b] == deviation.loc[b, a]

    def test_known_rmse_value(self):
        # Alice and Bob both rate two albums; Alice=[10, 6], Bob=[8, 8].
        # Differences: 2, -2. RMSE = sqrt((4+4)/2) = 2.0
        reviews = pd.DataFrame(
            [
                {"listener": "Alice", "artist": "X", "album": "A", "score": 10},
                {"listener": "Alice", "artist": "X", "album": "B", "score": 6},
                {"listener": "Bob", "artist": "X", "album": "A", "score": 8},
                {"listener": "Bob", "artist": "X", "album": "B", "score": 8},
            ]
        )
        deviation = data.build_deviation_df(reviews)
        assert deviation.loc["Alice", "Bob"] == pytest.approx(2.0)


class TestBuildListenerRequesterDf:
    def test_produces_pivot_keyed_by_listener_and_requester(self, raw_sheet_df):
        df = data._normalize_columns(raw_sheet_df)
        listeners = data.get_listeners(df)
        reviews = data.build_reviews_df(df, listeners)
        albums = data.build_albums_df(df)

        pivot = data.build_listener_requester_df(reviews, albums)

        assert pivot.index.name == "listener"
        assert pivot.columns.name == "requester"
        assert set(pivot.index) <= {"Alice", "Bob", "Carol"}
        assert set(pivot.columns) <= {"Alice", "Bob", "Carol"}

    def test_values_are_mean_scores_rounded_to_one_decimal(self):
        reviews = pd.DataFrame(
            [
                {"listener": "Alice", "artist": "X", "album": "A", "score": 9},
                {"listener": "Alice", "artist": "X", "album": "B", "score": 6},
            ]
        )
        albums = pd.DataFrame(
            [
                {"artist": "X", "album": "A", "requester": "Bob"},
                {"artist": "X", "album": "B", "requester": "Bob"},
            ]
        )
        pivot = data.build_listener_requester_df(reviews, albums)
        assert pivot.loc["Alice", "Bob"] == pytest.approx(7.5)
