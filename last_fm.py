from urllib.parse import urlencode
from io import BytesIO
import requests
import streamlit as st

_DEFAULT_IMAGE_SIZE = 'large'
_USER_AGENT = 'RecordClub/1.0'


@st.cache_resource
def _get_album_art(url):
    headers = {'User-Agent': _USER_AGENT}
    res = requests.get(url, headers=headers)
    return BytesIO(res.content)


@st.cache_resource
def _make_call(url):
    headers = {'User-Agent': _USER_AGENT}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


class Album:
    def __init__(self, get_album_response):
        album_data = get_album_response.get('album', {})
        self.artist = album_data.get('artist')
        self.title = album_data.get('name')
        self.image_url = next(
            (
                img.get('#text')
                for img in album_data.get('image', [])
                if img.get('size') == _DEFAULT_IMAGE_SIZE
            ),
            None,
        )
        self.tracks = [
            {
                'title': track.get('name'),
                'duration': track.get('duration'),
                'track_number': track.get('@attr', {}).get('rank'),
            }
            for track in album_data.get('tracks', {}).get('track', [])
        ]
        self.tags = (
            [
                {'name': tag.get('name')}
                for tag in album_data.get('tags', {}).get('tag', [])
            ]
            if album_data.get('tags')
            else []
        )
        self.listeners = album_data.get('listeners')
        self.playcount = album_data.get('playcount')

    def get_album_art(self):
        return _get_album_art(self.image_url)


class LastFmClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://ws.audioscrobbler.com/2.0/'

    def _build_url(self, method, params):
        params.update(
            {'api_key': self.api_key, 'format': 'json', 'method': method}
        )
        return f"{self.base_url}?{urlencode(params)}"

    def get_album(self, artist, album):
        params = {'artist': artist, 'album': album, 'autocorrect': 1}
        url = self._build_url('album.getinfo', params)
        res = _make_call(url)
        return Album(res) if res else None
