"""Tests for LRCLIB API client."""

import json
import unittest
import urllib.error
import urllib.parse
from unittest.mock import patch, MagicMock

from src.modules.lrclib_client import (
    LyricsInfo,
    find_lyrics,
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
        mock_query.return_value = None
        result = search_lyrics("Artist", "Song")
        self.assertIsNotNone(result)
        self.assertEqual(result.plain_lyrics, "lyrics")

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

    @patch("src.modules.lrclib_client.time.sleep")
    @patch("src.modules.lrclib_client.urllib.request.urlopen")
    def test_busy_server_is_asked_again(self, mock_urlopen, mock_sleep):
        busy = urllib.error.HTTPError("https://lrclib.net/api/search?q=test", 503,
                                      "Service Unavailable", {}, None)
        mock_resp = MagicMock()
        mock_resp.read.return_value = _make_response([_lrclib_entry()])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.side_effect = [busy, mock_resp]

        result = _do_search("https://lrclib.net/api/search?q=test")
        self.assertIsNotNone(result)
        self.assertEqual(result.track_name, "Test Song")
        mock_sleep.assert_called_once_with(1)


def _version(duration, synced=True, plain=None, track="Song Title", artist="Some Artist"):
    """A version of the wanted song as LRCLIB lists it."""
    return _lrclib_entry(track=track, artist=artist, duration=duration,
                         plain=plain or f"lyrics of the {duration} s version",
                         synced="[00:01.00] la la" if synced else None)


class TestFindLyrics(unittest.TestCase):
    def _find(self, answer, artist="Some Artist", title="Song Title", duration=None):
        """find_lyrics with LRCLIB answering ``answer(params)``; returns the
        result and the parameters of all requests."""
        requests = []

        def get_json(url):
            params = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
            requests.append(params)
            return answer(params)

        with patch("src.modules.lrclib_client._get_json", side_effect=get_json):
            result = find_lyrics(artist, title, duration)
        return result, requests

    def test_searches_again_without_title_additions(self):
        def answer(params):
            return [_version(200.0)] if params.get("track_name") == "Song Title" else []
        result, requests = self._find(answer, title="Song Title - US Version (Official Video)")
        self.assertIsNotNone(result)
        self.assertEqual(result.track_name, "Song Title")
        self.assertEqual([p.get("track_name") or p.get("q") for p in requests], [
            "Song Title - US Version (Official Video)",
            "Some Artist Song Title - US Version (Official Video)",
            "Song Title",
            "Some Artist Song Title",
        ])

    def test_searches_last_with_the_title_before_a_description(self):
        def answer(params):
            return [_version(237.0)] if params.get("track_name") == "Song Title" else []
        result, requests = self._find(answer, title="Song Title - Singer Name in the studio (Official Video)")
        self.assertEqual(result.track_name, "Song Title")
        self.assertEqual([p.get("track_name") or p.get("q") for p in requests], [
            "Song Title - Singer Name in the studio (Official Video)",
            "Some Artist Song Title - Singer Name in the studio (Official Video)",
            "Song Title - Singer Name in the studio",
            "Some Artist Song Title - Singer Name in the studio",
            "Song Title",
            "Some Artist Song Title",
        ])

    def test_a_part_after_the_dash_is_kept(self):
        result, requests = self._find(lambda params: [_version(200.0)] if params.get("track_name") == "Song Title"
                                      else [], title="Song Title - Part Two of It")
        self.assertIsNone(result)
        self.assertEqual([p.get("track_name") or p.get("q") for p in requests],
                         ["Song Title - Part Two of It", "Some Artist Song Title - Part Two of It"])

    def test_the_title_before_any_dash_for_a_version_of_the_length_of_the_audio(self):
        """"Chapter 2" after the dash is no description, but a version of the length of
        the audio shows that "Song Title" is the song."""
        def answer(params):
            return [_version(239.0)] if params.get("track_name") == "Song Title" else []
        title = "Song Title [OFFICIAL VIDEO] - Chapter 2"
        result, requests = self._find(answer, title=title, duration=238.0)
        self.assertEqual((result.track_name, result.duration), ("Song Title", 239.0))
        self.assertEqual([p.get("track_name") or p.get("q") for p in requests], [
            title, f"Some Artist {title}",
            "Song Title - Chapter 2", "Some Artist Song Title - Chapter 2",
            "Song Title", "Some Artist Song Title",
        ])

    def test_the_title_before_any_dash_takes_no_version_of_another_length(self):
        """"Song Title - Part 2" can be another song than "Song Title"."""
        def answer(params):
            return [_version(180.0)] if params.get("track_name") == "Song Title" else []
        result, _ = self._find(answer, title="Song Title - Part 2", duration=238.0)
        self.assertIsNone(result)

    def test_the_title_before_any_dash_needs_the_length_of_the_audio(self):
        def answer(params):
            return [_version(239.0)] if params.get("track_name") == "Song Title" else []
        result, requests = self._find(answer, title="Song Title - Part 2")
        self.assertIsNone(result)
        self.assertEqual(len(requests), 2)

    def test_up_to_5_s_off_the_length_of_the_audio(self):
        for length, taken in ((243.0, True), (243.01, False), (None, False)):
            def answer(params, length=length):
                return [_version(length)] if params.get("track_name") == "Song Title" else []
            with self.subTest(length=length):
                result, _ = self._find(answer, title="Song Title - Part 2", duration=238.0)
                self.assertEqual(result is not None, taken)

    def test_the_song_found_as_instrumental_stands(self):
        """LRCLIB knows "Song Title - Part 2" as instrumental: the lyrics of "Song Title",
        another song, are not taken for it."""
        def answer(params):
            if (params.get("track_name") or params.get("q", "")).endswith("Part 2"):
                return [_lrclib_entry(track="Song Title - Part 2", artist="Some Artist", plain=None,
                                      duration=238.0, instrumental=True)]
            return [_version(236.0)]
        result, requests = self._find(answer, title="Song Title - Part 2", duration=238.0)
        self.assertTrue(result.instrumental)
        self.assertEqual(len(requests), 2)

    def test_a_description_is_searched_without_the_length(self):
        """The title before a description takes a version of any length, as before."""
        def answer(params):
            return [_version(300.0)] if params.get("track_name") == "Song Title" else []
        result, requests = self._find(answer, title="Song Title - Singer Name in the studio", duration=238.0)
        self.assertEqual(result.duration, 300.0)
        self.assertEqual(len(requests), 4)

    def test_full_title_is_searched_first(self):
        result, requests = self._find(lambda params: [_version(200.0, track="Song Title (Live)")],
                                      title="Song Title (Live)")
        self.assertIsNotNone(result)
        self.assertEqual([p.get("track_name") or p.get("q") for p in requests],
                         ["Song Title (Live)", "Some Artist Song Title (Live)"])

    def test_closer_version_of_the_combined_query_wins(self):
        def answer(params):
            return [_version(306.0)] if "track_name" in params else [_version(234.0)]
        result, _ = self._find(answer, duration=235.0)
        self.assertEqual(result.duration, 234.0)

    def test_other_part_is_not_taken(self):
        result, _ = self._find(lambda params: [_version(200.0, track="Song Title (Part 2)")],
                               title="Song Title (Part 1)")
        self.assertIsNone(result)

    def test_version_closest_in_length_wins(self):
        versions = [_version(392.0), _version(306.0), _version(234.0)]
        result, _ = self._find(lambda params: versions, duration=235.2)
        self.assertEqual(result.duration, 234.0)

    def test_synced_version_preferred_among_about_equally_long_ones(self):
        versions = [_version(246.0, synced=False), _version(242.0)]
        result, _ = self._find(lambda params: versions, duration=252.5)
        self.assertEqual(result.duration, 242.0)
        self.assertIsNotNone(result.synced_lyrics)

    def test_much_closer_version_beats_synced_lyrics(self):
        # A synced album version of another length would bring the lyrics of
        # another arrangement (an extra chorus, a verse more or less)
        versions = [_version(285.0), _version(211.0, synced=False)]
        result, _ = self._find(lambda params: versions, duration=210.0)
        self.assertEqual(result.duration, 211.0)

    def test_length_never_rules_a_version_out(self):
        result, _ = self._find(lambda params: [_version(306.0)], duration=235.0)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.synced_lyrics)

    def test_synced_lyrics_alone_are_enough(self):
        synced_only = _lrclib_entry(track="Song Title", artist="Some Artist", plain=None,
                                    synced="[00:01.00] la la")
        result, _ = self._find(lambda params: [synced_only])
        self.assertEqual(result.synced_lyrics, "[00:01.00] la la")

    def test_without_a_length_synced_lyrics_first(self):
        versions = [_version(200.0, synced=False), _version(300.0)]
        result, _ = self._find(lambda params: versions)
        self.assertEqual(result.duration, 300.0)

    def test_other_song_is_not_taken(self):
        result, _ = self._find(lambda params: [_version(200.0, track="Other Song", artist="Other Artist")])
        self.assertIsNone(result)

    def test_longer_title_with_the_same_start_is_not_taken(self):
        result, _ = self._find(lambda params: [_version(200.0, track="Song Title Goes On")])
        self.assertIsNone(result)

    def test_artist_credits_written_differently(self):
        for credit in ("SOME ARTIST", "Some Artist & Guest Name", "The Some Artist", "Some Artist feat. Guest",
                       "Guest Name, Some Artist", "Some Artist and Guest Name"):
            with self.subTest(credit=credit):
                result, _ = self._find(lambda params: [_version(200.0, artist=credit)])
                self.assertIsNotNone(result)

    def test_other_artist_with_a_longer_name_is_not_taken(self):
        result, _ = self._find(lambda params: [_version(200.0, artist="Some Artist Tribute")])
        self.assertIsNone(result)
        result, _ = self._find(lambda params: [_version(200.0, artist="Some Much Longer Artist Name Tribute")],
                               artist="Some Much Longer Artist Name")
        self.assertIsNone(result)

    def test_credits_split_only_at_list_separators(self):
        from src.modules.lrclib_client import _artist_matches
        self.assertTrue(_artist_matches("Some Artist / Guest Name", "Some Artist"))
        self.assertFalse(_artist_matches("Ab/Cd", "Cd"))           # a slash inside a name
        self.assertFalse(_artist_matches("And Some", "Some"))      # a name that starts with "And"
        # a separator before an article belongs to a band name
        self.assertFalse(_artist_matches("Singer and the Band Name", "Singer"))
        self.assertFalse(_artist_matches("Singer & the Band Name", "Singer"))
        self.assertTrue(_artist_matches("Singer and the Band Name", "Singer and the Band Name"))
        # the same band name with another connector
        self.assertTrue(_artist_matches("Singer & the Band Name", "Singer and the Band Name"))
        self.assertTrue(_artist_matches("Singer + the Band Name", "Singer and the Band Name"))

    def test_empty_artist_or_title_finds_nothing(self):
        for artist, title in (("", "Song Title"), ("Some Artist", ""), (None, "Song Title"),
                              ("   ", "Song Title"), ("Some Artist", "  ")):
            with self.subTest(artist=artist, title=title):
                result, requests = self._find(lambda params: [_version(200.0)], artist=artist, title=title)
                self.assertIsNone(result)
                self.assertEqual(requests, [])

    def test_whitespace_synced_lyrics_count_as_none(self):
        from src.modules.lrclib_client import _best_version
        blank = {"duration": 200.0, "syncedLyrics": "  ", "plainLyrics": "blank"}
        synced = {"duration": 200.0, "syncedLyrics": "[00:01.00] la", "plainLyrics": "synced"}
        self.assertIs(_best_version([blank, synced], 200.0), synced)

    def test_song_named_before_the_artist(self):
        # Video title "Song Title - Some Artist": artist and title come swapped
        def answer(params):
            return [_version(200.0)] if "q" in params else []
        result, _ = self._find(answer, artist="Song Title", title="Some Artist")
        self.assertIsNotNone(result)

    def test_instrumental_only_when_no_lyrics_anywhere(self):
        instrumental = _lrclib_entry(track="Song Title", artist="Some Artist", plain=None, instrumental=True)

        def answer(params):
            if params.get("track_name") == "Song Title" or params.get("q") == "Some Artist Song Title":
                return [_version(200.0)]
            return [instrumental]
        result, _ = self._find(answer, title="Song Title (Official Video)")
        self.assertEqual(result.plain_lyrics, "lyrics of the 200.0 s version")

        result, _ = self._find(lambda params: [instrumental], title="Song Title (Official Video)")
        self.assertTrue(result.instrumental)


class TestSearchLyricsReport(unittest.TestCase):
    def test_reports_the_lengths(self):
        found = LyricsInfo(track_name="Song Title", artist_name="Some Artist",
                           plain_lyrics="la la la", synced_lyrics="[00:01.00] la la la", duration=234.0)
        with patch("src.modules.lrclib_client.find_lyrics", return_value=found), \
                patch("builtins.print") as mock_print:
            search_lyrics("Some Artist", "Song Title", duration=235.2)
        report = " ".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn("[plain (3 words), synced], 234 s long (the audio: 235 s)", report)

    def test_reports_synced_lyrics_alone(self):
        found = LyricsInfo(track_name="Song Title", artist_name="Some Artist",
                           plain_lyrics="", synced_lyrics="[00:01.00] la la la")
        with patch("src.modules.lrclib_client.find_lyrics", return_value=found), \
                patch("builtins.print") as mock_print:
            search_lyrics("Some Artist", "Song Title")
        report = " ".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn("[synced]", report)


if __name__ == "__main__":
    unittest.main()
