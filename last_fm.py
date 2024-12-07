from urllib.parse import urlencode
import requests
import streamlit as st

_DEFAULT_IMAGE_SIZE = 'large'


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
        #  TODO: This breaks with invalid call of get() on str
        # self.tags = [
        #     {'name': tag.get('name')}
        #     for tag in album_data.get('tags', {}).get('tag', [])
        # ]
        self.listeners = album_data.get('listeners')
        self.playcount = album_data.get('playcount')

    def get_album_art(self):
        res = requests.get(self.image_url)
        return res.content


class LastFmClient:
    def __init__(self, api_key, user_agent='RecordClub/1.0'):
        self.api_key = api_key
        self.base_url = 'https://ws.audioscrobbler.com/2.0/'
        self.user_agent = user_agent

    def _build_url(self, method, params):
        params.update(
            {'api_key': self.api_key, 'format': 'json', 'method': method}
        )
        return f"{self.base_url}?{urlencode(params)}"

    def make_call(self, method, params):
        url = self._build_url(method, params)
        headers = {'User-Agent': self.user_agent}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def get_album(self, artist, album):
        params = {'artist': artist, 'album': album, 'autocorrect': 1}
        res = self.make_call('album.getinfo', params)
        return Album(res) if res else None
