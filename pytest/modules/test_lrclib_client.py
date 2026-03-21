"""Tests for LRCLIB API client."""

import json
import unittest
from unittest.mock import patch, MagicMock

from src.modules.lrclib_client import (
    LyricsInfo,
    search_lyrics,
    _do_search,
    _search_by_fields,
    _search_by_query,
)


def _make_response(entries: list[dict]) -> bytes:
    """Create a mock HTTP response body."""
    return json.dumps(entries).encode("utf-8")


def _lrclib_entry(
    track="Test Song",
    artist="Test Artist",
    plain="Hello world\nGoodbye world",
    synced=None,
    duration=180.0,
    instrumental=False,
) -> dict:
    return {
        "trackName": track,
        "artistName": artist,
        "plainLyrics": plain,
        "syncedLyrics": synced,
        "duration": duration,
        "instrumental": instrumental,
    }


class TestLyricsInfo(unittest.TestCase):
    def test_defaults(self):
        info = LyricsInfo(track_name="Song", artist_name="Artist")
        self.assertIsNone(info.plain_lyrics)
        self.assertIsNone(info.synced_lyrics)
        self.assertIsNone(info.duration)
        self.assertFalse(info.instrumental)


class TestSearchLyrics(unittest.TestCase):
    @patch("src.modules.lrclib_client._search_by_query")
    @patch("src.modules.lrclib_client._search_by_fields")
    def test_returns_field_search_result(self, mock_fields, mock_query):
        mock_fields.return_value = LyricsInfo(
            track_name="Song", artist_name="Artist", plain_lyrics="lyrics"
        )
        result = search_lyrics("Artist", "Song")
        self.assertIsNotNone(result)
        self.assertEqual(result.plain_lyrics, "lyrics")
        mock_query.assert_not_called()

    @patch("src.modules.lrclib_client._search_by_query")
    @patch("src.modules.lrclib_client._search_by_fields")
    def test_falls_back_to_query_search(self, mock_fields, mock_query):
        mock_fields.return_value = None
        mock_query.return_value = LyricsInfo(
            track_name="Song", artist_name="Artist", plain_lyrics="fallback"
        )
        result = search_lyrics("Artist", "Song")
        self.assertIsNotNone(result)
        self.assertEqual(result.plain_lyrics, "fallback")

    @patch("src.modules.lrclib_client._search_by_query")
    @patch("src.modules.lrclib_client._search_by_fields")
    def test_returns_none_when_no_results(self, mock_fields, mock_query):
        mock_fields.return_value = None
        mock_query.return_value = None
        result = search_lyrics("Artist", "Song")
        self.assertIsNone(result)


class TestDoSearch(unittest.TestCase):
    @patch("src.modules.lrclib_client.urllib.request.urlopen")
    def test_parses_response(self, mock_urlopen):
        entry = _lrclib_entry()
        mock_resp = MagicMock()
        mock_resp.read.return_value = _make_response([entry])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _do_search("https://lrclib.net/api/search?q=test")
        self.assertIsNotNone(result)
        self.assertEqual(result.track_name, "Test Song")
        self.assertEqual(result.artist_name, "Test Artist")
        self.assertEqual(result.plain_lyrics, "Hello world\nGoodbye world")

    @patch("src.modules.lrclib_client.urllib.request.urlopen")
    def test_empty_response(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = _make_response([])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _do_search("https://lrclib.net/api/search?q=test")
        self.assertIsNone(result)

    @patch("src.modules.lrclib_client.urllib.request.urlopen")
    def test_prefers_non_instrumental(self, mock_urlopen):
        entries = [
            _lrclib_entry(instrumental=True, plain=None),
            _lrclib_entry(track="Real Song", plain="real lyrics"),
        ]
        mock_resp = MagicMock()
        mock_resp.read.return_value = _make_response(entries)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _do_search("https://lrclib.net/api/search?q=test")
        self.assertEqual(result.track_name, "Real Song")
        self.assertEqual(result.plain_lyrics, "real lyrics")

    @patch("src.modules.lrclib_client.urllib.request.urlopen")
    def test_network_error_returns_none(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("connection failed")
        result = _do_search("https://lrclib.net/api/search?q=test")
        self.assertIsNone(result)

    @patch("src.modules.lrclib_client.urllib.request.urlopen")
    def test_invalid_json_returns_none(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _do_search("https://lrclib.net/api/search?q=test")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
