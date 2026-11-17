"""The own lyrics of a song in the queue: only the song's own settings carry them."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.gui import queue_file
from src.gui.config import _DEFAULTS
from src.gui.queue_manager import QueueManager

_app = QApplication.instance() or QApplication([])

LYRICS = "first line of the song\nsecond line of it"


def _manager(**global_settings):
    manager = QueueManager()
    manager._runner.start = MagicMock()
    manager.set_global_config({**_DEFAULTS, "cookie_file": "", **global_settings})
    return manager


def _started_args(manager) -> list[str]:
    return manager._runner.start.call_args.args[0]


class TestOwnLyricsInTheQueue:
    def test_the_lyrics_of_a_song_go_to_its_conversion(self, tmp_path):
        audio = tmp_path / "song.mp3"
        audio.write_bytes(b"x")
        manager = _manager()
        song = manager.add_item(str(audio), "file", "Song")
        song.settings_overrides = {"lyrics_text": LYRICS}
        manager.start_all()
        args = _started_args(manager)
        assert Path(args[args.index("--lyrics_file") + 1]).read_text(encoding="utf-8") == LYRICS
        assert song.resolved_config["lyrics_text"] == LYRICS

    def test_a_song_whose_lyrics_cannot_be_saved_fails_before_it_starts(self, tmp_path):
        """Its conversion would make the chart from other lyrics and report success."""
        audio = tmp_path / "song.mp3"
        audio.write_bytes(b"x")
        manager = _manager()
        lines, statuses = [], []
        manager.line_output.connect(lines.append)
        manager.item_status_changed.connect(lambda item_id, status: statuses.append((item_id, status)))
        song = manager.add_item(str(audio), "file", "Song with lyrics")
        song.settings_overrides = {"lyrics_text": LYRICS}
        other = manager.add_item(str(audio), "file", "Song without")
        with patch("src.gui.queue_manager.write_own_lyrics", side_effect=OSError("disk full")):
            manager.start_all()
            manager._runner.start.assert_not_called()
            _app.processEvents()  # (the song ends from the event loop)
        assert (song.status, song.exit_code) == ("failed", 1)
        assert any("could not be saved" in line and "disk full" in line for line in lines)
        assert (song.id, "failed") in statuses
        # the queue goes on with the next song, which is the only one that was started
        assert other.status == "running"
        manager._runner.start.assert_called_once()
        assert "--lyrics_file" not in _started_args(manager)

    def test_a_long_row_of_such_songs_ends_one_by_one(self, tmp_path):
        """Each ends from the event loop: the row is no call in a call that gets too deep."""
        audio = tmp_path / "song.mp3"
        audio.write_bytes(b"x")
        manager = _manager()
        songs = []
        for number in range(600):
            song = manager.add_item(str(audio), "file", f"Song {number}")
            song.settings_overrides = {"lyrics_text": LYRICS}
            songs.append(song)
        finished = []
        manager.queue_finished.connect(lambda failed, cancelled: finished.append((failed, cancelled)))
        with patch("src.gui.queue_manager.write_own_lyrics", side_effect=OSError("disk full")):
            manager.start_all()
            for _ in range(700):
                if finished:
                    break
                _app.processEvents()
        assert finished == [(600, False)]
        assert {song.status for song in songs} == {"failed"}
        manager._runner.start.assert_not_called()

    def test_cancel_all_while_such_a_song_ends(self, tmp_path):
        audio = tmp_path / "song.mp3"
        audio.write_bytes(b"x")
        manager = _manager()
        song = manager.add_item(str(audio), "file", "Song with lyrics")
        song.settings_overrides = {"lyrics_text": LYRICS}
        other = manager.add_item(str(audio), "file", "Song without")
        finished = []
        manager.queue_finished.connect(lambda failed, cancelled: finished.append((failed, cancelled)))
        with patch("src.gui.queue_manager.write_own_lyrics", side_effect=OSError("disk full")):
            manager.start_all()
            manager.cancel_all()
            _app.processEvents()
        assert (song.status, other.status) == ("failed", "cancelled")
        assert finished == [(1, False)]
        manager._runner.start.assert_not_called()
        assert not manager._running

    def test_lyrics_in_the_global_settings_are_no_song_s_lyrics(self, tmp_path):
        audio = tmp_path / "song.mp3"
        audio.write_bytes(b"x")
        manager = _manager(lyrics_text=LYRICS)
        song = manager.add_item(str(audio), "file", "Song")
        manager.start_all()
        assert "--lyrics_file" not in _started_args(manager)
        assert song.resolved_config["lyrics_text"] == ""


class TestOwnLyricsInAQueueFile:
    def test_they_are_kept_and_imported(self):
        assert "lyrics_text" in queue_file.PER_SONG_SETTINGS
        for imported in (False, True):
            problems = []
            stored = queue_file._stored_settings({"lyrics_text": LYRICS}, imported, problems, "song 1")
            assert stored == {"lyrics_text": LYRICS} and problems == []

    def test_no_lyrics_of_a_wrong_kind(self):
        problems = []
        assert queue_file._stored_settings({"lyrics_text": ["a", "b"]}, True, problems, "song 1") == {}
        assert queue_file._stored_settings({"lyrics_text": "x" * 100_001}, True, problems, "song 1") == {}
        assert len(problems) == 2
