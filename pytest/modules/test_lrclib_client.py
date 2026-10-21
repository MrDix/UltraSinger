"""Tests for LRCLIB API client."""

import json
import unittest
import urllib.error
import urllib.parse
from unittest.mock import patch, MagicMock

import pytest

from src.modules.lrclib_client import (
    _is_credit,
    _second_db_lyrics,
    lyric_words,
    too_few_words,
    LyricsInfo,
    find_lyrics,
    find_on_second_db,
    lrc_text,
    other_recording,
    search_lyrics,
    _do_search,
    _search_by_fields,
    _search_by_query,
)


@pytest.fixture(autouse=True)
def _no_second_db(request):
    """The short test lyrics are far too few words for most audio: the
    second lyrics database is not asked, except by its own tests."""
    if getattr(request.cls, "asks_second_db", False):
        yield
        return
    with patch("src.modules.lrclib_client._find_on_second_db", return_value=None):
        yield


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
        # the title before its part number is searched for that part only
        self.assertEqual([p.get("track_name") or p.get("q") for p in requests],
                         ["Song Title - Part 2", "Some Artist Song Title - Part 2", "Song Title"])

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
        # the search for the parts of "Song Title" finds none with lyrics
        self.assertEqual(len(requests), 3)

    def test_the_same_search_is_sent_once(self):
        """The title before the part number and the title before any dash are the same search:
        LRCLIB's answer to it is filtered again by the length, not fetched again."""
        for title in ("Song Title - Part 2", "Song Title - I"):
            with self.subTest(title=title):
                result, requests = self._find(lambda params: [], title=title, duration=238.0)
                self.assertIsNone(result)
                self.assertEqual([p.get("track_name") or p.get("q") for p in requests],
                                 [title, f"Some Artist {title}", "Song Title", "Some Artist Song Title"])

    def test_the_part_under_another_name(self):
        """"Song Title - I" is listed as "Song Title, Part I", found by the title before its part number."""
        def answer(params):
            if params.get("track_name") != "Song Title":
                return []
            return [_version(200.0), _version(210.0, track="Song Title, Part II"),
                    _version(220.0, track="Song Title, Part I")]
        for title in ("Song Title - I", "Song Title, Pt. 1", "Song Title (Part 1)", "Song Title Part One"):
            with self.subTest(title=title):
                result, requests = self._find(answer, title=title)
                self.assertEqual(result.track_name, "Song Title, Part I")
                self.assertEqual([p.get("track_name") or p.get("q") for p in requests],
                                 [title, f"Some Artist {title}", "Song Title"])

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
        self.assertIn("[plain (3 words), synced], 234.0 s long (the audio: 235.2 s)", report)

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


class TestOtherRecording(unittest.TestCase):
    """Found lyrics of a version much longer or shorter than the audio may be of another recording."""

    def test_both_limits(self):
        # a live version with an extra part: 236 s of lyrics for 391 s of audio
        self.assertTrue(other_recording(236.0, 391.0))
        self.assertTrue(other_recording(391.0, 236.0))
        # 21 s off, but not 10 % of the audio
        self.assertFalse(other_recording(379.0, 400.0))
        # 10 % of the audio, but not 20 s off
        self.assertFalse(other_recording(80.0, 100.0))
        self.assertFalse(other_recording(200.0, 205.0))

    def test_unknown_length(self):
        self.assertFalse(other_recording(None, 391.0))
        self.assertFalse(other_recording(0, 391.0))
        self.assertFalse(other_recording(236.0, None))


class TestLrcText(unittest.TestCase):
    def test_time_tags_are_left_out(self):
        synced = "[ar:Artist]\n[00:01.00] Hello <00:01.50>world\n[00:03.00]\n[00:04.00][00:20.00] Again\n"
        self.assertEqual(lrc_text(synced), "Hello world\nAgain\nAgain")

    def test_lines_in_the_order_they_are_sung(self):
        synced = "[00:04.00][00:20.00] Again\n[00:10.00] Between\n[00:01.00] First\n"
        self.assertEqual(lrc_text(synced), "First\nAgain\nBetween\nAgain")


def _words(count: int) -> str:
    return " ".join(f"word{n}" for n in range(count))


def _second_db_song(song_id, name="Song Title", artists=("Some Artist",), duration_s=300.0):
    return {"id": song_id, "name": name, "artists": [{"name": artist} for artist in artists],
            "duration": int(duration_s * 1000)}


def _second_db_lrc(words: int) -> dict:
    """Its answer with lyrics of ``words`` words, after two credit lines."""
    lines = ["[00:00.000] 作词 : Some Writer", "[00:01.000] Composer : Someone Else"]
    lines += [f"[00:{10 + n:02d}.500] {_words(10)}" for n in range(words // 10)]
    return {"lrc": {"lyric": "\n".join(lines)}, "code": 200}


class TestSecondLyricsDatabase(unittest.TestCase):
    """The second lyrics database, asked when LRCLIB has no lyrics or far too few."""
    asks_second_db = True

    @staticmethod
    def _get_json(lrclib, songs, lyrics, asked):
        """Answers of LRCLIB (``lrclib`` to each search) and of the second
        database (``songs`` to a search, ``lyrics[id]`` for a song's lyrics)."""
        def get_json(url, attempts=None):
            asked.append(url)
            if url.startswith("https://lrclib.net/"):
                return lrclib
            if "/search/get?" in url:
                searched = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))["s"]
                return {"result": {"songs": list(songs(searched) if callable(songs) else songs)}, "code": 200}
            song_id = int(dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))["id"])
            return (lyrics or {}).get(song_id, {"nolyric": True})
        return get_json

    def _find(self, lrclib, songs=(), lyrics=None, duration=300.0, title="Song Title"):
        """find_lyrics with these answers; returns the result and the URLs asked."""
        asked = []
        with patch("src.modules.lrclib_client._get_json",
                   side_effect=self._get_json(lrclib, songs, lyrics, asked)):
            result = find_lyrics("Some Artist", title, duration)
        return result, asked

    def test_asked_when_lrclib_has_no_lyrics(self):
        result, _ = self._find([], songs=[_second_db_song(7)], lyrics={7: _second_db_lrc(100)})
        self.assertEqual(result.source, "music.163.com")
        self.assertEqual(result.duration, 300.0)
        # Its credit lines are no lyrics
        self.assertNotIn("Writer", result.synced_lyrics)
        self.assertNotIn("Composer", result.synced_lyrics)
        self.assertEqual(len(result.plain_lyrics.split()), 100)
        self.assertTrue(result.synced_lyrics.startswith("[00:10.500] word0"))

    def test_asked_alone(self):
        asked = []
        with patch("src.modules.lrclib_client._get_json",
                   side_effect=self._get_json([], [_second_db_song(7)], {7: _second_db_lrc(100)}, asked)):
            result = find_on_second_db("Some Artist", "Song Title", 300.0)
            self.assertIsNone(find_on_second_db(" ", "Song Title", 300.0))
        self.assertEqual(result.source, "music.163.com")
        # LRCLIB is not asked
        self.assertFalse(any(url.startswith("https://lrclib.net/") for url in asked))

    def test_fuller_lyrics_replace_far_too_few(self):
        # 40 words for 5 minutes: 8 per minute
        sparse = [_version(300.0, plain=_words(40))]
        result, _ = self._find(sparse, songs=[_second_db_song(7)], lyrics={7: _second_db_lrc(200)})
        self.assertEqual(result.source, "music.163.com")

    def test_lyrics_not_fuller_enough_are_not_taken(self):
        sparse = [_version(300.0, plain=_words(80))]
        result, _ = self._find(sparse, songs=[_second_db_song(7)], lyrics={7: _second_db_lrc(110)})
        self.assertEqual(result.source, "LRCLIB")

    def test_fuller_lyrics_of_another_recording_are_not_taken(self):
        sparse = [_version(300.0, plain=_words(40))]
        result, _ = self._find(sparse, songs=[_second_db_song(7, duration_s=200.0)],
                               lyrics={7: _second_db_lrc(200)})
        self.assertEqual(result.source, "LRCLIB")

    def test_not_asked_for_enough_words(self):
        result, asked = self._find([_version(300.0, plain=_words(150))])
        self.assertEqual(result.source, "LRCLIB")
        self.assertTrue(all(url.startswith("https://lrclib.net/") for url in asked))

    def test_not_asked_for_a_song_known_as_instrumental(self):
        result, asked = self._find([_lrclib_entry(track="Song Title", artist="Some Artist", plain=None,
                                                  instrumental=True)])
        self.assertTrue(result.instrumental)
        self.assertTrue(all(url.startswith("https://lrclib.net/") for url in asked))

    def test_the_version_closest_in_length_of_the_wanted_song(self):
        songs = [_second_db_song(1, duration_s=200.0), _second_db_song(2, name="Other Song"),
                 _second_db_song(3, artists=("Other Artist",)), _second_db_song(4, duration_s=301.0)]
        lyrics = {n: _second_db_lrc(100 + n * 10) for n in (1, 2, 3, 4)}
        result, _ = self._find([], songs=songs, lyrics=lyrics)
        self.assertEqual(len(result.plain_lyrics.split()), 140)

    def test_a_guest_artist_in_its_credit(self):
        songs = [_second_db_song(5, artists=("Some Artist", "Guest"))]
        result, _ = self._find([], songs=songs, lyrics={5: _second_db_lrc(100)})
        self.assertEqual(result.artist_name, "Some Artist, Guest")

    def test_the_next_version_when_one_has_no_lyrics(self):
        songs = [_second_db_song(1), _second_db_song(2, duration_s=310.0)]
        result, _ = self._find([], songs=songs, lyrics={2: _second_db_lrc(100)})
        self.assertEqual(result.duration, 310.0)

    def test_none_of_an_instrumental(self):
        lyrics = {7: {"lrc": {"lyric": "[00:00.00] 纯音乐，请欣赏"}}}
        result, _ = self._find([], songs=[_second_db_song(7)], lyrics=lyrics)
        self.assertIsNone(result)

    def test_none_when_it_finds_nothing(self):
        result, _ = self._find([], songs=[])
        self.assertIsNone(result)

    def test_asked_once_when_unreachable(self):
        asked = []

        def get_json(url, attempts=None):
            asked.append((url, attempts))
            return [] if url.startswith("https://lrclib.net/") else None

        with patch("src.modules.lrclib_client._get_json", side_effect=get_json), \
                patch("builtins.print") as printed:
            self.assertIsNone(search_lyrics("Some Artist", "Song Title (Official Video)", 300.0))
        self.assertEqual([attempts for url, attempts in asked if "music.163.com" in url], [1])
        self.assertIn("No lyrics found on LRCLIB or music.163.com", str(printed.call_args_list[-1].args[0]))

    def test_searched_again_without_the_title_additions(self):
        def songs(searched):
            return [] if "Video" in searched else [_second_db_song(7)]
        result, asked = self._find([], songs=songs, lyrics={7: _second_db_lrc(100)},
                                   title="Song Title (Official Video)")
        self.assertEqual(result.source, "music.163.com")
        searches = [dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))["s"]
                    for url in asked if "/search/get?" in url]
        self.assertEqual(searches, ["Some Artist Song Title (Official Video)", "Some Artist Song Title"])

    def test_colon_time_tags_and_id_tags(self):
        lyrics = {7: {"lrc": {"lyric": "[by:someone]\n[00:12:50] " + _words(10) + "\n[01:02:25] " + _words(10)}}}
        result, _ = self._find([], songs=[_second_db_song(7)], lyrics=lyrics)
        self.assertEqual(result.synced_lyrics.splitlines()[0], "[00:12.50] " + _words(10))
        self.assertNotIn("someone", result.synced_lyrics)

    def test_lyrics_without_times_are_plain(self):
        lyrics = {7: {"lrc": {"lyric": _words(10) + "\n" + _words(10)}}}
        result, _ = self._find([], songs=[_second_db_song(7)], lyrics=lyrics)
        self.assertIsNone(result.synced_lyrics)
        self.assertEqual(result.plain_lyrics, _words(10) + "\n" + _words(10))

    def test_its_search_reported(self):
        sparse = [_version(300.0, plain=_words(40))]
        get_json = self._get_json(sparse, [_second_db_song(7)], {7: _second_db_lrc(200)}, [])
        with patch("builtins.print") as printed, \
                patch("src.modules.lrclib_client._get_json", side_effect=get_json):
            search_lyrics("Some Artist", "Song Title", 300.0)
        lines = [str(call.args[0]) for call in printed.call_args_list]
        self.assertTrue(any("Searching music.163.com" in line and "have 40 words, for 5:00 of audio" in line
                            for line in lines))
        self.assertTrue(any("Found lyrics on music.163.com" in line for line in lines))


class TestTooFewWords(unittest.TestCase):
    def test_words_per_minute_of_the_audio(self):
        info = LyricsInfo(track_name="T", artist_name="A", plain_lyrics=_words(99))
        self.assertTrue(too_few_words(info, 300.0))
        info.plain_lyrics = _words(100)
        self.assertFalse(too_few_words(info, 300.0))

    def test_the_length_of_the_version_without_the_audio(self):
        info = LyricsInfo(track_name="T", artist_name="A", plain_lyrics=_words(10), duration=300.0)
        self.assertTrue(too_few_words(info, None))

    def test_not_for_short_or_unknown_lengths(self):
        info = LyricsInfo(track_name="T", artist_name="A", plain_lyrics=_words(5))
        self.assertFalse(too_few_words(info, 50.0))
        self.assertFalse(too_few_words(info, None))

    def test_the_synced_or_plain_text_with_more_words(self):
        info = LyricsInfo(track_name="T", artist_name="A", plain_lyrics=_words(10),
                          synced_lyrics="[00:01.00][00:30.00] " + _words(60))
        self.assertEqual(lyric_words(info), 120)


class TestIsCredit(unittest.TestCase):
    def test_credit_lines(self):
        for line in ("作词 : Some Writer", "制作人：Some Producer",
                     "Composer : Someone", "Lyrics by: Someone", "PGM : 杨阳"):
            self.assertTrue(_is_credit(line), line)

    def test_english_roles(self):
        for line in ("Guitar : Alex Smith", "Mixing Engineer : John Doe", "Backing Vocals : Jane",
                     "Executive Producer : Someone", "Recording Studio : Some Place"):
            self.assertTrue(_is_credit(line), line)

    def test_sung_lines(self):
        for line in ("Listen: this is a song", "Oh baby", "HEY: you there", "Time: three o'clock",
                     "Music: it's all I need", "Words: they fail me now"):
            self.assertFalse(_is_credit(line), line)


class TestSecondDbLyricsLines(unittest.TestCase):
    """Which lines of the lyrics of the second database are kept."""

    @staticmethod
    def _lines(*lines, times=None):
        version = {"id": 7, "trackName": "Song Title", "artistName": "Some Artist"}
        times = times or range(len(lines))
        answer = {"lrc": {"lyric": "\n".join(f"[00:{t:02d}.00] {line}" for t, line in zip(times, lines))}}
        with patch("src.modules.lrclib_client._get_json", return_value=answer) as get_json:
            synced, plain = _second_db_lyrics(version)
        get_json.assert_called_once()
        assert get_json.call_args.kwargs == {"attempts": 1}
        return plain.splitlines() if plain else []

    def test_the_title_and_the_artist_before_the_lyrics(self):
        self.assertEqual(self._lines("作词 : Someone", "Song Title", "Some Artist", "first line", "last line"),
                         ["first line", "last line"])

    def test_credits_after_the_lyrics(self):
        self.assertEqual(self._lines("first line", "last line", "Guitar : Alex Smith", "混音 : Someone"),
                         ["first line", "last line"])

    def test_title_and_artist_in_one_heading_before_the_title_sung(self):
        self.assertEqual(self._lines("作词 : Someone", "Song Title - Some Artist", "Song Title", "next line",
                                     times=(0, 0, 18, 21)),
                         ["Song Title", "next line"])

    def test_the_artist_right_after_the_title(self):
        self.assertEqual(self._lines("Song Title", "Some Artist", "first line", times=(1, 15, 20)), ["first line"])

    def test_the_title_sung_after_the_heading_of_title_and_artist(self):
        self.assertEqual(self._lines("Song Title", "Some Artist", "Song Title", "next line", times=(1, 15, 20, 23)),
                         ["Song Title", "next line"])

    def test_the_title_sung_after_the_artist(self):
        self.assertEqual(self._lines("Some Artist", "Song Title", "next line", times=(0, 18, 21)),
                         ["Song Title", "next line"])

    def test_credits_of_other_roles_among_credits(self):
        # "lyrics", "vocal arrangement", "recording assistant" ... "audio assistant"
        lines = ("作词 : A", "合声编写：B", "录音助理 : C",
                 "first line", "last line", "音频助理 : D", "电吉他 : E")
        self.assertEqual(self._lines(*lines), ["first line", "last line"])

    def test_a_credit_block_that_starts_with_another_role(self):
        # "planning", then "lyrics"
        self.assertEqual(self._lines("企划 : X", "作词 : Y", "first line"), ["first line"])

    def test_english_credits_without_a_colon_at_the_end(self):
        lines = ("first line", "last line", "Producers : Some One", "Lead & Background Vocals : Some One",
                 "Mixed at Some Studio (London)",
                 "Published by Some Music Limited", "Mastered at Some Mastering (London)")
        self.assertEqual(self._lines(*lines), ["first line", "last line"])

    def test_html_entities(self):
        self.assertEqual(self._lines("let&apos;s go &amp; sing", "a &quot;word&quot;"),
                         ["let's go & sing", 'a "word"'])

    def test_a_first_sung_line_that_is_the_title(self):
        self.assertEqual(self._lines("Song Title", "next line", times=(18, 21)), ["Song Title", "next line"])

    def test_a_sung_line_in_chinese_with_a_colon(self):
        # "I said: goodbye", at the end
        self.assertEqual(self._lines("first line", "我说：再见"),
                         ["first line", "我说：再见"])

    def test_lines_like_a_credit_or_the_title_among_the_lyrics_are_sung(self):
        lines = ("first line", "Song Title", "Music: All I Need", "last line")
        self.assertEqual(self._lines(*lines), list(lines))
