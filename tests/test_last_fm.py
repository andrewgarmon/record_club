"""Tests for the Last.fm API client and ``Album`` response parser."""
from unittest.mock import patch, MagicMock
from urllib.parse import parse_qs, urlparse

import pytest

import last_fm


@pytest.fixture
def album_response():
    """A trimmed-down copy of a real ``album.getinfo`` response."""
    return {
        "album": {
            "artist": "Radiohead",
            "name": "OK Computer",
            "image": [
                {"#text": "https://img/small.png", "size": "small"},
                {"#text": "https://img/medium.png", "size": "medium"},
                {"#text": "https://img/large.png", "size": "large"},
                {"#text": "https://img/xl.png", "size": "extralarge"},
            ],
            "tracks": {
                "track": [
                    {
                        "name": "Airbag",
                        "duration": "284",
                        "@attr": {"rank": "1"},
                    },
                    {
                        "name": "Paranoid Android",
                        "duration": "383",
                        "@attr": {"rank": "2"},
                    },
                ]
            },
            "tags": {
                "tag": [
                    {"name": "alternative"},
                    {"name": "rock"},
                ]
            },
            "listeners": "1234567",
            "playcount": "98765432",
        }
    }


class TestAlbum:
    def test_parses_basic_fields(self, album_response):
        album = last_fm.Album(album_response)
        assert album.artist == "Radiohead"
        assert album.title == "OK Computer"
        assert album.listeners == "1234567"
        assert album.playcount == "98765432"

    def test_picks_large_image(self, album_response):
        album = last_fm.Album(album_response)
        assert album.image_url == "https://img/large.png"

    def test_parses_tracks_with_rank_and_duration(self, album_response):
        album = last_fm.Album(album_response)
        assert album.tracks == [
            {"title": "Airbag", "duration": "284", "track_number": "1"},
            {
                "title": "Paranoid Android",
                "duration": "383",
                "track_number": "2",
            },
        ]

    def test_parses_tags(self, album_response):
        album = last_fm.Album(album_response)
        assert album.tags == [{"name": "alternative"}, {"name": "rock"}]

    def test_handles_missing_tags(self, album_response):
        album_response["album"].pop("tags")
        album = last_fm.Album(album_response)
        assert album.tags == []

    def test_handles_empty_response(self):
        album = last_fm.Album({})
        assert album.artist is None
        assert album.title is None
        assert album.tracks == []
        assert album.tags == []
        assert album.image_url is None

    def test_image_url_is_none_when_no_large_image(self, album_response):
        album_response["album"]["image"] = [
            {"#text": "https://img/small.png", "size": "small"}
        ]
        album = last_fm.Album(album_response)
        assert album.image_url is None


class TestLastFmClient:
    def test_build_url_adds_api_key_method_and_format(self):
        client = last_fm.LastFmClient("secret-key")
        url = client._build_url(
            "album.getinfo", {"artist": "Radiohead", "album": "OK Computer"}
        )

        parsed = urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "ws.audioscrobbler.com"
        assert parsed.path == "/2.0/"

        qs = parse_qs(parsed.query)
        assert qs["api_key"] == ["secret-key"]
        assert qs["method"] == ["album.getinfo"]
        assert qs["format"] == ["json"]
        assert qs["artist"] == ["Radiohead"]
        assert qs["album"] == ["OK Computer"]

    def test_get_album_calls_api_and_returns_album(self, album_response):
        client = last_fm.LastFmClient("k")
        with patch.object(last_fm, "_make_call", return_value=album_response) as mk:
            album = client.get_album("Radiohead", "OK Computer")

        mk.assert_called_once()
        called_url = mk.call_args[0][0]
        qs = parse_qs(urlparse(called_url).query)
        assert qs["artist"] == ["Radiohead"]
        assert qs["album"] == ["OK Computer"]
        assert qs["autocorrect"] == ["1"]
        assert isinstance(album, last_fm.Album)
        assert album.title == "OK Computer"

    def test_get_album_returns_none_when_api_returns_none(self):
        client = last_fm.LastFmClient("k")
        with patch.object(last_fm, "_make_call", return_value=None):
            assert client.get_album("X", "Y") is None

    def test_make_call_raises_for_status(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("boom")
        with patch.object(last_fm.requests, "get", return_value=mock_response):
            with pytest.raises(Exception, match="boom"):
                last_fm._make_call("https://example.com")

    def test_make_call_returns_parsed_json(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status.return_value = None
        with patch.object(last_fm.requests, "get", return_value=mock_response):
            assert last_fm._make_call("https://example.com") == {"ok": True}

    def test_get_album_art_fetches_image_bytes(self, album_response):
        album = last_fm.Album(album_response)
        mock_response = MagicMock()
        mock_response.content = b"\x89PNG fake bytes"
        with patch.object(last_fm.requests, "get", return_value=mock_response):
            buf = album.get_album_art()
        assert buf.read() == b"\x89PNG fake bytes"
