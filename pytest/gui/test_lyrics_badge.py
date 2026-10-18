"""The browser tab's lyrics badge finds the lyrics that a conversion of the video finds."""

import json
import logging
import os
import urllib.parse
from contextlib import ExitStack
from unittest.mock import patch
from urllib.error import HTTPError

import musicbrainzngs
import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

# The modules as the GUI and the conversion import them
from modules import lrclib_client
from modules.Audio import video_platform
from src.gui.browser_tab import _FormatProbeWorker

_app = QApplication.instance() or QApplication([])

SYNCED = "✅ Synced lyrics"
PLAIN = "⚠️ Plain lyrics only"
NO_LYRICS = "\U0001f4dd No lyrics"


def _recording(artist, title, length=None):
    """A recording as the song database lists it (with its ``length`` in seconds, when it knows it)."""
    recording = {"id": f"{artist} - {title}", "title": title, "artist-credit-phrase": artist,
                 "artist-credit": [{"artist": {"id": artist}}], "release-list": []}
    if length is not None:
        recording["length"] = str(length * 1000)
    return recording


def _lyrics(artist, title, duration, synced=True):
    """A version of a song as the lyrics service lists it."""
    return {"artistName": artist, "trackName": title, "duration": duration, "instrumental": False,
            "plainLyrics": "first line\nsecond line",
            "syncedLyrics": "[00:12.00] first line\n[00:15.00] second line" if synced else None}


class _Services:
    """The song database and the lyrics service, each knowing one song. Like search
    engines they list it for any search that names its title (the clients check that
    it fits). They record what they are asked."""

    def __init__(self, recording, lyrics):
        self.recording = recording
        self.lyrics = lyrics
        self.asked = []

    def _search_recordings(self, query="", **fields):
        self.asked.append(("song database", query, fields))
        return {"recording-count": 1, "recording-list": [self.recording]}

    def _get_json(self, url):
        params = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        self.asked.append(("lyrics", params))
        searched = (params.get("track_name") or params.get("q") or "").casefold()
        return [self.lyrics] if self.lyrics["trackName"].casefold() in searched else []

    def __enter__(self):
        self._stack = ExitStack()
        for target, kwargs in (("musicbrainzngs.search_recordings", {"side_effect": self._search_recordings}),
                               ("musicbrainzngs.search_artists", {"return_value": None}),
                               ("musicbrainzngs.get_release_group_by_id", {"return_value": None}),
                               ("musicbrainzngs.get_image_front", {"return_value": None})):
            self._stack.enter_context(patch(target, **kwargs))
        self._stack.enter_context(patch.object(lrclib_client, "_get_json", side_effect=self._get_json))
        return self

    def __exit__(self, *exc):
        self._stack.close()

    def take(self):
        """What the services were asked since the last call."""
        asked, self.asked = self.asked, []
        return asked


def _conversion(info, folder):
    """The lyrics that a conversion of the video finds: the download names the song
    (from yt-dlp's info), then UltraSinger.py looks the lyrics up with the length of
    the audio."""
    video = os.path.join(folder, "download.mp4")

    def extract_info(url, download=False):
        if not download:
            return info
        open(video, "wb").close()
        return {"requested_downloads": [{"filepath": video}]}

    with patch.object(video_platform.yt_dlp, "YoutubeDL") as ydl, \
            patch.object(video_platform, "separate_audio_video", return_value=("song.ogg", None, "ogg", None)):
        ydl.return_value.__enter__.return_value.extract_info.side_effect = extract_info
        *_, media_info = video_platform.download_from_video_platform("https://video.example/watch?v=vid", folder)
    return lrclib_client.search_lyrics(media_info.artist, media_info.title, duration=info["duration"])


def _badge(info):
    """The lyrics line of the badge that the browser tab shows for the video."""
    worker = _FormatProbeWorker("vid")
    shown = []
    worker.finished.connect(lambda _video_id, text, _language: shown.append(text))
    with patch.object(_FormatProbeWorker, "_dump_json", return_value=json.dumps(info)):
        worker.run()
    return shown[0].splitlines()[-1]


def _status(found):
    """The badge line for the lyrics that the conversion found."""
    if found is not None and found.synced_lyrics:
        return SYNCED
    if found is not None and found.plain_lyrics:
        return PLAIN
    return NO_LYRICS


_CASES = {
    # The creator field holds the channels that uploaded the video, not the artist
    "official video": (
        {"title": "BAND - Song Title (OFFICIAL VIDEO)", "creator": "Label Records, Management Co, Band Official",
         "channel": "Label Records", "duration": 215, "formats": []},
        _recording("Band", "Song Title"), _lyrics("Band", "Song Title", 214), SYNCED),
    "live video with a label after the title": (
        {"title": "BAND - Song Title (Live at Festival 2019) | Label Records", "channel": "Label Records",
         "duration": 251, "formats": []},
        _recording("Band", "Song Title"), _lyrics("Band", "Song Title", 250, synced=False), PLAIN),
    # A separate, known limitation: the singer named after a second dash stays in the
    # title, so the conversion does not find the song either. The badge has to agree.
    "performance with the singer after the title": (
        {"title": "Band - Song Title - Singer live one take performance", "channel": "Band",
         "duration": 230, "formats": []},
        _recording("Band", "Song Title"), _lyrics("Band", "Song Title", 229), None),
    # "Chapter 2" after the dash is no description: the title before the dash is searched for a
    # recording and lyrics of the length of the video
    "chapter after the title": (
        {"title": "Band - Song Title [OFFICIAL VIDEO] - Chapter 2", "channel": "Band", "duration": 238,
         "formats": []},
        _recording("Band", "Song Title", 239), _lyrics("Band", "Song Title", 239), SYNCED),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_badge_shows_what_the_conversion_finds(case, tmp_path, capsys):
    info, recording, lyrics, expected = _CASES[case]
    with _Services(recording, lyrics) as services:
        found = _conversion(info, str(tmp_path))
        conversion_asked = services.take()
        capsys.readouterr()
        badge = _badge(info)
        badge_asked = services.take()
    # The same song-database searches and the same lyrics queries
    assert badge_asked == conversion_asked
    assert badge == _status(found)
    if expected is not None:
        assert badge == expected
    # The badge runs in the GUI: it prints nothing
    assert capsys.readouterr() == ("", "")


def test_the_song_database_names_the_song_by_the_title_before_any_dash(tmp_path):
    """Both with the length of the video: the conversion names the song folder after the recording
    of that length, and the badge asks the song database as the conversion does."""
    info, recording, lyrics, _ = _CASES["chapter after the title"]
    with _Services(recording, lyrics) as services:
        _conversion(info, str(tmp_path))
        conversion_asked = services.take()
        _badge(info)
        badge_asked = services.take()
    assert (tmp_path / "Band - Song Title").is_dir()
    searched = [asked[2]["recording"] for asked in badge_asked if asked[0] == "song database"]
    # Each title is searched as the title and, swapped, as the artist
    assert searched[::2] == ["song title - chapter 2", "Song Title"]
    assert badge_asked == conversion_asked


def test_the_lookup_prints_and_logs_nothing(capsys, caplog):
    """Also not what a conversion reports (a stripped suffix, a failed request), nor what the
    song database's client library logs: the GUI's log shows INFO lines."""
    caplog.set_level(logging.INFO)
    # The music service adds "(live)" to the track, the video is no live one
    info = {"title": "Band - Song Title (Official Video)", "artist": "Band", "track": "Song Title (live)",
            "duration": 215, "formats": []}

    def search_recordings(*args, **kwargs):
        # The client library's parser logs each part of an answer that it does not read
        logging.getLogger("musicbrainzngs").info("in <recording>, uncaught <first-release-date>")
        raise musicbrainzngs.ResponseError(cause=HTTPError("u", 400, "Bad Request", None, None))

    with patch("musicbrainzngs.search_recordings", side_effect=search_recordings) as search, \
            patch.object(lrclib_client, "_get_json", return_value=[]) as lyrics:
        assert _badge(info) == NO_LYRICS
    assert search.called and lyrics.called
    assert capsys.readouterr() == ("", "")
    assert [record for record in caplog.records if record.name.startswith("musicbrainzngs")] == []


def test_probe_stopped_during_the_song_lookup_asks_for_no_lyrics():
    worker = _FormatProbeWorker("vid")
    shown = []
    worker.finished.connect(lambda *args: shown.append(args))

    def search_recordings(*args, **kwargs):
        worker.cancel()  # the window closes, or another video is opened
        return {"recording-count": 0, "recording-list": []}

    info = _CASES["official video"][0]
    with patch.object(_FormatProbeWorker, "_dump_json", return_value=json.dumps(info)), \
            patch("musicbrainzngs.search_recordings", side_effect=search_recordings), \
            patch.object(lrclib_client, "_get_json", return_value=[]) as lyrics:
        worker.run()
    lyrics.assert_not_called()
    assert shown == []
