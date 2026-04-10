"""Shared fixtures for the test suite.

Adds the project root to ``sys.path`` so tests can import the top-level
modules (``data``, ``last_fm``) the same way the Streamlit app does.
"""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def raw_sheet_df() -> pd.DataFrame:
    """A DataFrame shaped like the Google Sheet ``load_sheet`` returns.

    The real sheet has an unnamed whitespace-only date column, three
    columns per listener (``Name``, ``Name.1``, ``Name.2`` for score,
    favorite track, least favorite track), and an ``Average`` column at
    the end.
    """
    return pd.DataFrame(
        {
            " ": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "Requester": ["Alice", "Bob", "Carol"],
            "Artist": ["Beatles", "Stones", "Zeppelin"],
            "Album": ["Abbey Road", "Let It Bleed", "IV"],
            "Release Year": [1969, 1969, 1971],
            "Decade": ["60s", "60s", "70s"],
            "Alice": [9, 8, 7],
            "Alice.1": ["Come Together", "Gimme Shelter", "Black Dog"],
            "Alice.2": ["Octopus's Garden", None, "Four Sticks"],
            "Bob": [7, 9, 10],
            "Bob.1": ["Something", "Monkey Man", "Stairway to Heaven"],
            "Bob.2": [None, None, None],
            "Carol": [8, 7, 9],
            "Carol.1": ["Here Comes the Sun", "Midnight Rambler", "Rock and Roll"],
            "Carol.2": [None, None, None],
            "Average": [8.0, 8.0, 8.67],
        }
    )
