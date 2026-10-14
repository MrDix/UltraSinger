"""Keeping the conversion queue between sessions, and exporting and importing it.

The queue is saved in the private settings directory whenever it changes and
comes back on the next start without starting; an export or import uses the
same file format. A queue file never holds a secret, and a damaged or
hand-edited file never stops the GUI.
"""

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

from src.gui import queue_file
from src.gui.config import _DEFAULTS, _is_secret_key
from src.gui.models import QueueItem
from src.gui.queue_file import (
    FORMAT_VERSION,
    PER_SONG_SETTINGS,
    load_saved_queue,
    read_queue_file,
    without_duplicates,
    write_queue_file,
)
from src.gui.queue_manager import QueueManager
from src.gui.settings_tab import ConversionSettingsForm
from src.gui.ultrasinger_runner import UltraSingerRunner

_app = QApplication.instance() or QApplication([])

_URL = "https://video.invalid/watch?v=abc"


def _process_events_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.01)
    return condition()


def _song(source=_URL, input_type="url", title="Artist - Song Title", status="pending", **fields):
    return QueueItem(input_source=source, input_type=input_type, title=title, status=status,
                     **fields)


def _local_file(tmp_path, name="song.mp3"):
    return str(tmp_path / name)


def _write_raw(path, items, version=FORMAT_VERSION):
    path.write_text(json.dumps({"version": version, "items": items}), encoding="utf-8")


def _summary(item):
    return (item.input_type, item.input_source, item.title, item.status, item.video_id,
            item.settings_overrides)


class TestRoundTrip:
    def test_songs_that_are_not_finished_come_back(self, tmp_path):
        local = _local_file(tmp_path)
        items = [
            _song(video_id="abc", settings_overrides={"language_mode": "manual", "language": "de"}),
            _song(local, "file", "Local Song"),
            _song("https://video.invalid/watch?v=run", title="Running", status="running",
                  resolved_config={"whisper_model": "large-v2"}),
            _song("https://video.invalid/watch?v=cancel", title="Cancelled", status="cancelled"),
            _song("https://video.invalid/watch?v=fail", title="Failed", status="failed",
                  exit_code=1),
            _song("https://video.invalid/watch?v=done", title="Done", status="done", exit_code=0,
                  result_info={"language": "en"}),
        ]
        path = tmp_path / "queue.json"
        assert write_queue_file(path, items) == 5

        restored, problems = read_queue_file(path)
        assert problems == []
        assert [_summary(item) for item in restored] == [
            ("url", _URL, "Artist - Song Title", "pending", "abc",
             {"language_mode": "manual", "language": "de"}),
            ("file", local, "Local Song", "pending", "", {}),
            # The song that ran when the queue was saved, and a cancelled one, wait again
            ("url", "https://video.invalid/watch?v=run", "Running", "pending", "", {}),
            ("url", "https://video.invalid/watch?v=cancel", "Cancelled", "pending", "", {}),
            # A failed song stays marked, so the user sees it
            ("url", "https://video.invalid/watch?v=fail", "Failed", "failed", "", {}),
        ]
        for item in restored:  # nothing of a run comes back
            assert item.exit_code is None and item.resolved_config == {} and item.result_info == {}
        assert len({item.id for item in restored + items}) == len(restored + items)

    def test_file_format(self, tmp_path):
        path = tmp_path / "queue.json"
        write_queue_file(path, [_song(video_id="abc", settings_overrides={"golden_notes": True})])
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "version": 1,
            "items": [{
                "input_type": "url", "input_source": _URL, "title": "Artist - Song Title",
                "video_id": "abc", "status": "pending",
                "settings_overrides": {"golden_notes": True},
            }],
        }

    def test_a_failed_write_keeps_the_old_file(self, tmp_path):
        path = tmp_path / "queue.json"
        write_queue_file(path, [_song()])
        before = path.read_bytes()
        with patch.object(queue_file, "_replace", side_effect=PermissionError("locked")):
            with pytest.raises(PermissionError):
                write_queue_file(path, [_song(title="Other")])
        assert path.read_bytes() == before
        assert list(tmp_path.iterdir()) == [path]  # no temporary file left

    def test_saving_logs_a_failure(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            queue_file.save_queue(tmp_path / "missing" / "queue.json", [_song()])
        assert "Failed to save the queue" in caplog.text

    def test_saving_never_raises(self, tmp_path, caplog):
        """A save runs from a timer and on close, where an error would skip the shutdown."""
        with patch.object(queue_file, "write_queue_file", side_effect=TypeError("odd value")), \
                caplog.at_level(logging.WARNING):
            queue_file.save_queue(tmp_path / "queue.json", [_song()])
        assert "Failed to save the queue" in caplog.text

    def test_a_lone_surrogate_in_a_name_is_written_and_read_back(self, tmp_path):
        """A Windows file name can hold one; it has no UTF-8 form."""
        title = "Song " + chr(0xD800) + " Title"
        local = str(tmp_path / ("song" + chr(0xDC80) + ".mp3"))
        path = tmp_path / "queue.json"
        write_queue_file(path, [_song(local, "file", title)])
        [item], problems = read_queue_file(path)
        assert (item.title, item.input_source, problems) == (title, local, [])

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
    def test_file_is_private(self, tmp_path):
        path = tmp_path / "queue.json"
        write_queue_file(path, [_song()])
        assert path.stat().st_mode & 0o777 == 0o600


class TestNoSecrets:
    def test_keys_tokens_and_run_data_are_never_written(self, tmp_path):
        item = _song(
            settings_overrides={
                "language": "de",
                "remote_stt_api_key": "stt-secret",
                "segmentation_model_token": "token-secret",
                "llm_api_key_abc": "llm-secret",
                "llm_api_key": "legacy-secret",
                "yt_po_token": "po-secret",
                "proxy_url": "http://user:proxy-secret@proxy.invalid:8080",
                "cookie_file": "C:/cookies-secret.txt",
            },
            resolved_config={"llm_api_key_abc": "llm-secret-2", "yt_po_token": "po-secret-2",
                             "remote_stt_api_key": "stt-secret-2"},
            result_info={"output_folder": "C:/Songs/out"},
        )
        path = tmp_path / "queue.json"
        write_queue_file(path, [item])
        text = path.read_text(encoding="utf-8")
        assert "secret" not in text
        for key in ("api_key", "token", "proxy", "cookie", "resolved_config", "result_info"):
            assert key not in text
        assert json.loads(text)["items"][0]["settings_overrides"] == {"language": "de"}

    def test_a_secret_in_a_hand_edited_file_is_ignored_without_logging_it(self, tmp_path, caplog):
        path = tmp_path / "queue.json"
        _write_raw(path, [{"input_type": "url", "input_source": _URL,
                           "settings_overrides": {"remote_stt_api_key": "stt-secret",
                                                  "language": "de"}}])
        with caplog.at_level(logging.WARNING):
            [item], problems = read_queue_file(path)
        assert item.settings_overrides == {"language": "de"}
        assert "remote_stt_api_key" in problems[0]
        assert "stt-secret" not in caplog.text and "stt-secret" not in " ".join(problems)


class _BlockingDownload(QObject):
    """Stand-in for MediaDownloadWorker: runs until cancelled, then reports a failure."""

    finished = Signal(bool, str, str)
    progress = Signal(str)

    def __init__(self):
        super().__init__()
        self._cancel = threading.Event()
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True
        self._cancel.set()

    def run(self):
        self._cancel.wait(10)
        self.finished.emit(False, "", "killed")


def test_a_song_downloading_its_browser_stream_comes_back_as_its_url(tmp_path):
    """The browser stream goes to a temporary file while the song runs. The queue
    keeps the video URL instead: after a restart the file is gone, and the song
    downloads again."""
    config = {**_DEFAULTS, "output_folder": str(tmp_path), "cookie_file": ""}
    manager = QueueManager()
    manager._runner.start = MagicMock()
    manager.set_global_config(config)
    manager.set_media_interceptor(MagicMock(
        get_po_token=MagicMock(return_value=""),
        get_stream_with_status=MagicMock(return_value=(
            SimpleNamespace(url="https://media.invalid/x", mime_type="audio/webm",
                            seconds_until_expiry=600, is_muxed=False), "ok")),
    ))
    item = manager.add_item(_URL, "url", "Artist - Song Title")
    item.video_id = "abc"
    worker = _BlockingDownload()

    def fake_start(url, path, keep_video=False, parent=None):
        Path(path).write_bytes(b"partial")
        thread = QThread(parent)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        return thread, worker

    path = tmp_path / "queue.json"
    with patch("src.gui.media_downloader.start_media_download", side_effect=fake_start):
        manager.start_all()
    try:
        assert item.status == "running" and list(tmp_path.glob("_intercepted_*"))
        write_queue_file(path, manager.items)
    finally:
        manager.shutdown()
        _process_events_until(lambda: item.status == "cancelled")
    assert "_intercepted_" not in path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("_intercepted_*"))

    restored, problems = read_queue_file(path)
    assert problems == []
    assert [_summary(it) for it in restored] == [
        ("url", _URL, "Artist - Song Title", "pending", "abc", {})]
    # Without a captured stream the conversion downloads the URL
    again = QueueManager()
    again._runner.start = MagicMock()
    again.set_global_config(config)
    again.add_items(restored)
    again.start_all()
    assert again._runner.start.call_args.args[0][:2] == ["-i", _URL]


def _backups(directory):
    return sorted(directory.glob("queue.json.*.bak"))


class TestUnreadableFile:
    @pytest.mark.parametrize("raw", [
        b'{"version": 1, "items": [',                                   # cut off
        '{"version": 1, "items": [{"title": "Gr\xfc\xdfe"}]}'.encode("cp1252"),  # not UTF-8
        b'[{"input_type": "url"}]',                                      # no queue file
        b'{"items": []}',                                                # no version
        b'{"version": 1, "items": {}}',                                  # items no list
        b"[" * 100_000,                                                  # nested too deeply
    ], ids=["cut-off", "encoding", "list", "no-version", "items", "nested"])
    def test_starts_empty_and_keeps_a_copy(self, tmp_path, caplog, raw):
        path = tmp_path / "queue.json"
        path.write_bytes(raw)
        with caplog.at_level(logging.WARNING):
            items, notes, may_save = load_saved_queue(path)
        assert items == [] and may_save
        backups = _backups(tmp_path)
        assert len(backups) == 1 and backups[0].read_bytes() == raw
        assert "cannot be read" in caplog.text
        assert notes[0].startswith("cannot be read") and str(backups[0]) in notes[-1]

    def test_a_newer_version_is_kept_aside(self, tmp_path):
        path = tmp_path / "queue.json"
        _write_raw(path, [{"input_type": "url", "input_source": _URL}], version=2)
        items, notes, _ = load_saved_queue(path)
        assert items == [] and "version 2 " in notes[0]
        assert len(_backups(tmp_path)) == 1

    def test_a_version_as_text_is_named_as_such(self, tmp_path):
        path = tmp_path / "queue.json"
        _write_raw(path, [], version="1")
        _, notes, _ = load_saved_queue(path)
        assert "version '1' (this version of UltraSinger reads version 1)" in notes[0]

    def test_entries_left_out_are_kept_in_a_copy(self, tmp_path):
        """A newer version may write entries this one cannot read: the save after
        the restore would drop them."""
        path = tmp_path / "queue.json"
        _write_raw(path, [{"input_type": "url", "input_source": _URL},
                          {"input_type": "stream", "input_source": _URL, "title": "Newer"}])
        raw = path.read_bytes()
        items, notes, may_save = load_saved_queue(path)
        assert [item.input_source for item in items] == [_URL] and may_save
        [backup] = _backups(tmp_path)
        assert backup.read_bytes() == raw
        assert "'Newer' skipped" in notes[0] and str(backup) in notes[-1]

    def test_a_short_lock_is_waited_out(self, tmp_path):
        """On Windows a virus scanner can hold the file open for a moment."""
        path = tmp_path / "queue.json"
        _write_raw(path, [{"input_type": "url", "input_source": _URL}])
        result = read_queue_file(path)
        with patch.object(queue_file, "read_queue_file",
                          side_effect=[PermissionError("locked"), PermissionError("locked"), result]):
            items, notes, may_save = load_saved_queue(path)
        assert [item.input_source for item in items] == [_URL] and notes == [] and may_save
        assert _backups(tmp_path) == []

    def test_a_file_that_cannot_be_read_or_copied_is_not_saved_over(self, tmp_path, monkeypatch):
        monkeypatch.setattr(queue_file, "time", SimpleNamespace(sleep=lambda _seconds: None))
        path = tmp_path / "queue.json"
        _write_raw(path, [{"input_type": "url", "input_source": _URL}])
        with patch.object(queue_file, "read_queue_file",
                          side_effect=PermissionError("locked")) as read, \
                patch.object(queue_file, "_back_up_file", return_value=None):
            items, notes, may_save = load_saved_queue(path)
        assert read.call_count == 20  # retried for a moment
        assert items == [] and not may_save
        assert notes[-1] == ("The file could not be copied, so the queue is not saved "
                             "in this session.")

    def test_no_file_is_an_empty_queue(self, tmp_path):
        assert load_saved_queue(tmp_path / "queue.json") == ([], [], True)
        assert list(tmp_path.iterdir()) == []

    def test_byte_order_mark_is_accepted(self, tmp_path):
        path = tmp_path / "queue.json"
        path.write_bytes(b"\xef\xbb\xbf" + json.dumps(
            {"version": 1, "items": [{"input_type": "url", "input_source": _URL}]}).encode())
        items, notes, _ = load_saved_queue(path)
        assert [it.input_source for it in items] == [_URL] and notes == []
        assert _backups(tmp_path) == []


class TestInvalidEntries:
    def test_skipped_with_a_note_the_rest_comes_back(self, tmp_path, caplog):
        local = _local_file(tmp_path)
        path = tmp_path / "queue.json"
        _write_raw(path, [
            {"input_type": "url", "input_source": _URL, "title": "Good URL"},
            "not a song",
            {"input_type": "stream", "input_source": _URL, "title": "Bad Type"},
            {"input_type": "url", "title": "No Source"},
            {"input_type": "url", "input_source": "file:///etc/passwd", "title": "Not Web"},
            {"input_type": "file", "input_source": "relative/song.mp3", "title": "Relative"},
            {"input_type": "url", "input_source": _URL, "status": "paused", "title": "Status"},
            {"input_type": "url", "input_source": _URL, "status": "done", "title": "Finished"},
            {"input_type": "file", "input_source": local, "title": ["no", "text"],
             "video_id": 5, "settings_overrides": {
                 "whisper_batch_size": "eight",   # text for a number
                 "golden_notes": True,
                 "whisper_compute_type": None,     # no value: no override
                 "no_such_setting": 1,
                 "output_folder": "C:/elsewhere",  # not a per-song setting
             }},
        ])
        with caplog.at_level(logging.WARNING):
            items, problems = read_queue_file(path)
        assert [_summary(item) for item in items] == [
            ("url", _URL, "Good URL", "pending", "", {}),
            ("file", local, local, "pending", "", {"golden_notes": True}),
        ]
        joined = "\n".join(problems)
        for note in ("entry 2 skipped (not a song)", "'Bad Type' skipped (unknown source type)",
                     "'No Source' skipped (no valid source)",
                     "'Not Web' skipped (the source is not a web address)",
                     "'Relative' skipped (the source is not a full file path)",
                     "'Status' skipped (unknown status)", "'Finished' skipped (the song is finished)",
                     "entry 9: video ID ignored (not a plain video ID)",
                     "setting 'whisper_batch_size' ignored (wrong type)",
                     "setting 'no_such_setting' ignored", "setting 'output_folder' ignored"):
            assert note in joined
        assert len(problems) == 11
        assert "unknown status" in caplog.text

    def test_a_file_with_too_many_songs_is_read_up_to_the_limit(self, tmp_path, monkeypatch):
        """Each song becomes a row in the queue list; tens of thousands froze the GUI."""
        monkeypatch.setattr(queue_file, "_MAX_SONGS", 3)
        path = tmp_path / "queue.json"
        _write_raw(path, [{"input_type": "url", "input_source": f"{_URL}{n}", "title": "x" * 2000}
                          for n in range(5)])
        items, problems = read_queue_file(path)
        assert [item.input_source for item in items] == [f"{_URL}{n}" for n in range(3)]
        assert problems[0] == "only the first 3 songs are read, 2 more are left out"
        # Each change to the file has a note, so a restore keeps a copy before saving
        assert len(problems) == 4
        for n, problem in enumerate(problems[1:], start=1):
            assert problem.startswith(f"entry {n} 'xxx")
            assert problem.endswith(": title cut to 500 characters")
        assert all(len(item.title) == queue_file._MAX_TITLE for item in items)

    def test_a_video_id_is_no_path(self, tmp_path):
        """The video ID becomes part of the file name a browser stream is downloaded to."""
        path = tmp_path / "queue.json"
        _write_raw(path, [{"input_type": "url", "input_source": _URL, "video_id": video_id}
                          for video_id in ("abc-_9XyZ01", "../../startup/x", "a\\b", "")])
        items, problems = read_queue_file(path)
        assert [item.video_id for item in items] == ["abc-_9XyZ01", "", "", ""]
        assert problems == ["entry 2: video ID ignored (not a plain video ID)",
                            "entry 3: video ID ignored (not a plain video ID)"]

    def test_the_notes_on_one_entry_are_limited(self, tmp_path):
        """One entry with very many unknown settings flooded the console and froze the GUI."""
        path = tmp_path / "queue.json"
        _write_raw(path, [{"input_type": "url", "input_source": _URL,
                           "settings_overrides": {f"setting_{n}": 0 for n in range(5000)}}])
        [item], problems = read_queue_file(path)
        assert item.settings_overrides == {}
        assert len(problems) == 11
        assert problems[-1] == "entry 1: 4990 more settings ignored"

    def test_the_notes_on_one_file_are_limited(self, tmp_path):
        path = tmp_path / "queue.json"
        _write_raw(path, ["not a song"] * 150)
        items, problems = read_queue_file(path)
        assert items == [] and len(problems) == 101
        assert problems[-1] == "50 more notes left out"

    def test_a_saved_queue_reads_back_without_notes(self, tmp_path):
        """The writer cuts and checks like the reader: no copy of the file at every start."""
        path = tmp_path / "queue.json"
        write_queue_file(path, [_song(title="x" * 2000, video_id="../x")])
        [item], problems = read_queue_file(path)
        assert problems == []
        assert (len(item.title), item.video_id) == (queue_file._MAX_TITLE, "")

    def test_wrong_values_never_reach_the_settings_dialog(self, tmp_path):
        """The per-song settings dialog shows global settings plus the item's; a value
        of a wrong type stopped it (e.g. QSpinBox.setValue with text)."""
        path = tmp_path / "queue.json"
        _write_raw(path, [{"input_type": "url", "input_source": _URL, "settings_overrides": {
            "whisper_batch_size": 99999999999, "denoise_nr": "loud", "hyphenation": "yes",
            "bpm_override": 120, "language": "de"}}])
        [item], _ = read_queue_file(path)
        assert item.settings_overrides == {"bpm_override": "120", "language": "de"}
        ConversionSettingsForm({**_DEFAULTS, **item.settings_overrides})


class TestImport:
    def test_no_path_web_address_or_model_setting_from_an_imported_file(self, tmp_path):
        """An imported file may come from someone else: a program or model it names could
        run code, a server would get the API key. The saved queue keeps them."""
        settings = {
            "ffmpeg_path": r"\\server.invalid\share\ffmpeg.exe",
            "musescore_path": "C:/Tools/mscore.exe",
            "segmentation_model": "C:/models/x.pt",
            "segmentation_model_repo": "someone/models",
            "whisper_align_model": "someone/align",
            "remote_stt_api_base_url": "https://collector.invalid/v1",
            "cache_folder": "C:/cache",
            "language_mode": "manual",
            "language": "de",
            "remote_stt": True,
        }
        path = tmp_path / "queue.json"
        write_queue_file(path, [_song(settings_overrides=settings)])
        [restored], _ = read_queue_file(path)
        assert restored.settings_overrides == settings
        [imported], problems = read_queue_file(path, imported=True)
        assert imported.settings_overrides == {"language_mode": "manual", "language": "de",
                                               "remote_stt": True}
        assert len(problems) == 7 and all("not imported" in p for p in problems)

    def test_an_imported_file_sets_only_values_the_dialog_offers(self, tmp_path):
        """Such a text goes into the command line and into names of cache files: a path
        in it could send a file of the conversion elsewhere."""
        good = {"transcriber": "qwen3", "whisper_compute_type": "int8",
                "language_mode": "manual", "language": "de", "octave_consistency": "all",
                "format_version": "1.0.0", "linebreak_timing": "usdx", "pitcher": "fcpe",
                "separator_backend": "demucs", "chart_style": "score",
                "bpm_override": "120,5", "octave_shift": "-2", "llm_provider_id": "0a1b2c3d"}
        bad = {"transcriber": "../x", "whisper_compute_type": "/../../x", "language_mode": "x",
               "language": "../../x", "octave_consistency": "x", "format_version": "9.9",
               "linebreak_timing": "x", "pitcher": "C:\\x", "separator_backend": "x",
               "chart_style": "x", "bpm_override": "1e9", "octave_shift": "../1",
               "llm_provider_id": "../../x"}
        path = tmp_path / "queue.json"
        write_queue_file(path, [_song(settings_overrides=good), _song(settings_overrides=bad)])
        [kept, cleaned], problems = read_queue_file(path, imported=True)
        assert kept.settings_overrides == good
        assert cleaned.settings_overrides == {}
        assert sum("a value the per-song settings do not offer" in p for p in problems) == 10
        assert problems[-1] == "entry 2 'Artist - Song Title': 3 more settings ignored"
        # The saved queue keeps what the user set
        [_, restored], _ = read_queue_file(path)
        assert restored.settings_overrides == bad

    def test_a_failed_song_is_imported_to_be_converted(self, tmp_path):
        path = tmp_path / "queue.json"
        write_queue_file(path, [_song(status="failed")])
        [item], _ = read_queue_file(path, imported=True)
        assert item.status == "pending"

    def test_songs_already_waiting_are_skipped(self):
        german = {"language_mode": "manual", "language": "de"}
        queue = [
            _song(settings_overrides=dict(german)),
            _song("https://video.invalid/watch?v=done", status="done"),
            _song("https://video.invalid/watch?v=run", status="running"),
            _song("https://video.invalid/watch?v=fail", status="failed"),
        ]
        imported = [
            _song(settings_overrides=dict(german)),                       # waits already
            _song(settings_overrides={"language_mode": "manual", "language": "en"}),
            _song(),                                                       # other settings
            _song("https://video.invalid/watch?v=done"),                   # finished: again
            _song("https://video.invalid/watch?v=run"),                    # runs already
            _song("https://video.invalid/watch?v=fail"),                   # failed: again
            _song("https://video.invalid/watch?v=new"),
        ]
        added, skipped = without_duplicates(imported, queue)
        assert skipped == 2
        assert added == [imported[1], imported[2], imported[3], imported[5], imported[6]]

    def test_reimporting_an_export_adds_nothing_twice(self, tmp_path):
        """Also for a song with a setting an imported file does not set."""
        queue = [_song(settings_overrides={"whisper_model": "medium", "language": "de"})]
        path = tmp_path / "exported.json"
        write_queue_file(path, queue)
        imported, _ = read_queue_file(path, imported=True)
        assert imported[0].settings_overrides == {"language": "de"}
        assert without_duplicates(imported, queue) == ([], 1)


class TestPerSongSettings:
    def test_the_settings_of_the_per_song_dialog_without_secrets(self):
        """A new setting must be sorted in: per-song (in the dialog) or for all songs
        (queue_file._GLOBAL_ONLY); a secret is never written."""
        form_keys = set(ConversionSettingsForm(dict(_DEFAULTS)).collect_config())
        assert PER_SONG_SETTINGS == {key for key in form_keys if not _is_secret_key(key)}

    def test_the_text_settings_an_imported_file_may_set(self):
        """A new text setting must be sorted in: if it names a path, web address or model
        it belongs in queue_file._NOT_IMPORTED."""
        text_settings = {key for key in PER_SONG_SETTINGS
                         if isinstance(_DEFAULTS.get(key, ""), str)}
        assert text_settings - queue_file._NOT_IMPORTED == {
            "transcriber", "whisper_compute_type", "language_mode", "language",
            "octave_consistency", "format_version", "linebreak_timing", "bpm_override",
            "octave_shift", "pitcher", "separator_backend", "chart_style", "llm_provider_id",
        }
        assert queue_file._NOT_IMPORTED <= PER_SONG_SETTINGS

    def test_the_values_an_imported_file_may_set_are_those_of_the_dialog(self):
        """A new text setting, or a new choice in the dialog, must be added to
        queue_file._IMPORT_VALUES."""
        form = ConversionSettingsForm(dict(_DEFAULTS))

        def texts(combo):
            return {combo.itemText(i) for i in range(combo.count())}

        def data(combo):
            return {combo.itemData(i) for i in range(combo.count())}

        values = queue_file._IMPORT_VALUES
        text_settings = {key for key in PER_SONG_SETTINGS
                         if isinstance(_DEFAULTS.get(key, ""), str)}
        assert set(values) == text_settings - queue_file._NOT_IMPORTED
        assert set(values["transcriber"]) == data(form._transcriber)
        assert set(values["octave_consistency"]) == data(form._octave_consistency)
        assert set(values["linebreak_timing"]) == data(form._linebreak_timing)
        assert set(values["whisper_compute_type"]) == {
            "" if text == "auto" else text for text in texts(form._whisper_compute)}
        for key, combo in (("format_version", form._format_version),
                           ("pitcher", form._pitcher),
                           ("separator_backend", form._separator_backend),
                           ("chart_style", form._chart_style)):
            assert set(values[key]) == texts(combo), key
        assert all(values["language"].fullmatch(code) for code in texts(form._language_combo))
        for text in ("120", "120.5", "120,5", "500"):  # the BPM field takes 1 to 500
            assert values["bpm_override"].fullmatch(text), text
        for text in ("-10", "0", "+3", "10"):  # the octave field takes -10 to 10
            assert values["octave_shift"].fullmatch(text), text
        from src.gui.models import LLMProvider
        assert values["llm_provider_id"].fullmatch(LLMProvider().id)
        # Whatever the dialog gives, an imported file may set
        for key, value in form.collect_config().items():
            if key in values:
                assert queue_file._checked_setting(key, value, imported=True) == (value, None), key


# ── The main window: restore at start, saving, export and import ──────────


class _FakeBrowserTab(QWidget):
    convert_requested = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.cookie_manager = MagicMock(has_video_cookies=False)
        self.media_interceptor = MagicMock()
        self.probe_cookie_file = ""
        self.shutdown = MagicMock()
        self._view = MagicMock()


@pytest.fixture
def settings_dir(tmp_path, monkeypatch):
    """The settings directory (with queue.json) in a temporary directory."""
    from src.gui import config as config_module

    directory = tmp_path / "settings"
    directory.mkdir()
    monkeypatch.setattr(config_module, "_CONFIG_DIR", directory)
    return directory


@pytest.fixture
def open_window(settings_dir, monkeypatch):
    """Opens main windows on the temporary settings directory; no conversion really starts."""
    from src.gui import main_window
    from src.gui.widgets import llm_provider_list

    monkeypatch.setattr(llm_provider_list, "_shutting_down", False)
    monkeypatch.setattr(main_window.MainWindow, "_QUEUE_SAVE_DELAY_MS", 10)
    monkeypatch.setattr(main_window, "save_config", MagicMock())
    start = MagicMock()
    monkeypatch.setattr(UltraSingerRunner, "start", start)
    windows = []

    def open_():
        with patch.object(main_window, "BrowserTab", _FakeBrowserTab), \
                patch.object(main_window.MainWindow, "_start_potoken_provider", lambda self: None), \
                patch.object(main_window.MainWindow, "_start_ytdlp_update_check",
                             lambda self: None):
            win = main_window.MainWindow({**_DEFAULTS, "cookie_file": ""})
        win.conversion_start = start
        windows.append(win)
        return win

    yield open_
    for win in windows:  # no save after the test
        win._queue_saving = False
        win._queue_save_timer.stop()


def _close(win):
    from PySide6.QtGui import QCloseEvent

    event = QCloseEvent()
    with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
        win.closeEvent(event)
    assert event.isAccepted()


def _saved(settings_dir):
    path = settings_dir / "queue.json"
    if not path.exists():
        return None
    return [(entry["input_source"], entry.get("status", "pending"))
            for entry in json.loads(path.read_text(encoding="utf-8"))["items"]]


def _console(win):
    return win._queue_tab._log.toPlainText()


def _export_dialog(target, seen=None):
    """The export dialog picks ``target`` (None: the user cancels); ``seen`` gets its setup."""
    def exec_(dialog):
        if seen is not None:
            seen.append((dialog.acceptMode(), dialog.defaultSuffix()))
        return 0 if target is None else 1

    return patch.multiple(QFileDialog, exec=exec_,
                          selectedFiles=lambda dialog: [] if target is None else [str(target)])


class TestMainWindow:
    def test_the_last_queue_comes_back_and_does_not_start(self, settings_dir, open_window):
        _write_raw(settings_dir / "queue.json", [
            {"input_type": "url", "input_source": _URL, "title": "First", "video_id": "abc",
             "settings_overrides": {"language_mode": "manual", "language": "de"}},
            {"input_type": "url", "input_source": "https://video.invalid/watch?v=2",
             "title": "Second"},
            {"input_type": "url", "input_source": "https://video.invalid/watch?v=3",
             "title": "Third", "status": "failed"},
        ])
        win = open_window()
        items = win._queue_mgr.items
        assert [(it.title, it.status) for it in items] == [
            ("First", "pending"), ("Second", "pending"), ("Third", "failed")]
        widgets = win._sidebar.queue_list._item_widgets
        assert list(widgets) == [it.id for it in items]
        assert widgets[items[0].id]._gear_btn.toolTip() == "Per-song settings (custom)"
        assert "Restored 3 song(s)" in _console(win) and "1 of them failed" in _console(win)
        assert win._sidebar._start_btn.isEnabled()
        # Never started by itself
        _process_events_until(lambda: False, timeout=0.2)
        assert not win._queue_mgr.is_running
        assert all(it.status != "running" for it in win._queue_mgr.items)
        win.conversion_start.assert_not_called()

    def test_a_damaged_queue_file_starts_an_empty_queue(self, settings_dir, open_window):
        (settings_dir / "queue.json").write_text("{not json", encoding="utf-8")
        win = open_window()
        assert win._queue_mgr.items == []
        assert "[Queue] Saved queue: cannot be read" in _console(win)
        assert len(_backups(settings_dir)) == 1
        _close(win)
        assert _saved(settings_dir) == []  # replaced on close; the copy keeps the content

    def test_changes_are_saved(self, settings_dir, open_window):
        from src.gui import main_window

        win = open_window()
        manager = win._queue_mgr
        first = manager.add_item(_URL, "url", "First")
        second = manager.add_item("https://video.invalid/watch?v=2", "url", "Second")
        assert _process_events_until(lambda: _saved(settings_dir) == [
            (_URL, "pending"), ("https://video.invalid/watch?v=2", "pending")])

        manager.remove_item(second.id)
        assert _process_events_until(lambda: _saved(settings_dir) == [(_URL, "pending")])

        dialog = MagicMock()
        dialog.return_value.exec.return_value = True
        dialog.return_value.get_overrides.return_value = {"language_mode": "manual",
                                                          "language": "de"}
        with patch.object(main_window, "PerSongSettingsDialog", dialog):
            win._on_per_song_settings(first.id)

        def overrides():
            entries = json.loads((settings_dir / "queue.json").read_text(encoding="utf-8"))
            return entries["items"][0]["settings_overrides"]
        assert _process_events_until(lambda: overrides() == {"language_mode": "manual",
                                                             "language": "de"})

        first.status = "done"
        manager.item_status_changed.emit(first.id, "done")
        assert _process_events_until(lambda: _saved(settings_dir) == [])

    def test_closing_saves_the_songs_as_waiting(self, settings_dir, open_window):
        """The shutdown cancels the waiting songs; the queue is saved before, and not again."""
        win = open_window()
        manager = win._queue_mgr
        running = manager.add_item(_URL, "url", "Running")
        waiting = manager.add_item("https://video.invalid/watch?v=2", "url", "Waiting")
        running.status = "running"
        _close(win)  # before the delayed save of the additions
        assert waiting.status == "cancelled"  # by the shutdown
        assert _saved(settings_dir) == [(_URL, "pending"),
                                        ("https://video.invalid/watch?v=2", "pending")]
        (settings_dir / "queue.json").unlink()
        manager.add_item("https://video.invalid/watch?v=3", "url", "Late")
        _process_events_until(lambda: False, timeout=0.2)
        assert _saved(settings_dir) is None  # nothing is saved after the close

    def test_export_and_import(self, settings_dir, open_window, tmp_path):
        win = open_window()
        manager = win._queue_mgr
        manager.add_item(_URL, "url", "First")
        manager.add_item("https://video.invalid/watch?v=2", "url", "Second")
        exported = tmp_path / "exported.json"
        seen = []

        with _export_dialog(exported, seen):
            win._sidebar._export_action.trigger()
        # A name typed without .json gets it from the dialog, which then asks before replacing
        assert seen == [(QFileDialog.AcceptMode.AcceptSave, "json")]
        assert len(json.loads(exported.read_text(encoding="utf-8"))["items"]) == 2
        assert f"Exported 2 song(s) to {exported}" in _console(win)

        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(exported), "")):
            win._sidebar._import_action.trigger()
        assert len(manager.items) == 2  # both wait already
        assert "Imported 0 song(s)" in _console(win) and "skipped 2" in _console(win)

        manager.clear_pending()
        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(exported), "")):
            win._sidebar._import_action.trigger()
        assert [(it.title, it.status) for it in manager.items] == [
            ("First", "pending"), ("Second", "pending")]
        assert not manager.is_running
        win.conversion_start.assert_not_called()
        assert win._sidebar._start_btn.isEnabled()

    def test_import_of_an_unreadable_file_warns(self, settings_dir, open_window, tmp_path):
        win = open_window()
        bad = tmp_path / "bad.json"
        bad.write_text("no queue", encoding="utf-8")
        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(bad), "")), \
                patch.object(QMessageBox, "warning") as warning:
            win._sidebar._import_action.trigger()
        warning.assert_called_once()
        assert win._queue_mgr.items == []
        # An imported file is never copied or changed
        assert sorted(p.name for p in tmp_path.iterdir()) == ["bad.json", "settings"]
        assert bad.read_text(encoding="utf-8") == "no queue"

    def test_export_failure_warns(self, settings_dir, open_window, tmp_path):
        win = open_window()
        win._queue_mgr.add_item(_URL, "url", "First")
        target = tmp_path / "missing" / "queue.json"
        with _export_dialog(target), patch.object(QMessageBox, "warning") as warning:
            win._sidebar._export_action.trigger()
        warning.assert_called_once()
        assert not target.exists()

    def test_cancelled_dialogs_do_nothing(self, settings_dir, open_window):
        win = open_window()
        with _export_dialog(None), \
                patch.object(QFileDialog, "getOpenFileName", return_value=("", "")), \
                patch.object(QMessageBox, "warning") as warning:
            win._sidebar._export_action.trigger()
            win._sidebar._import_action.trigger()
        warning.assert_not_called()
        assert win._queue_mgr.items == []

    def test_a_failed_song_can_be_removed(self, settings_dir, open_window):
        """Failed songs are kept between sessions, so each can be removed, not only with Clear."""
        _write_raw(settings_dir / "queue.json", [
            {"input_type": "url", "input_source": _URL, "title": "Waiting"},
            {"input_type": "url", "input_source": "https://video.invalid/watch?v=3",
             "title": "Failed", "status": "failed"},
        ])
        win = open_window()
        waiting, failed = win._queue_mgr.items
        widget = win._sidebar.queue_list._item_widgets[failed.id]
        assert not widget._remove_btn.isHidden()
        widget._remove_btn.click()
        assert win._queue_mgr.items == [waiting]
        assert _process_events_until(lambda: _saved(settings_dir) == [(_URL, "pending")])

    @staticmethod
    def _open_on_a_locked_queue_file(settings_dir, open_window, monkeypatch):
        """A window whose saved queue can neither be read nor copied at start."""
        monkeypatch.setattr(queue_file, "time", SimpleNamespace(sleep=lambda _seconds: None))
        path = settings_dir / "queue.json"
        _write_raw(path, [{"input_type": "url", "input_source": _URL}])
        with patch.object(queue_file, "read_queue_file", side_effect=PermissionError("locked")), \
                patch.object(queue_file, "_back_up_file", return_value=None), \
                patch.object(QMessageBox, "warning") as warning:
            win = open_window()
        warning.assert_called_once()  # told at start, not only in the console
        assert win._queue_mgr.items == []
        return win, path, path.read_bytes()

    def test_a_locked_queue_file_is_kept_until_it_can_be_copied(self, settings_dir, open_window,
                                                                monkeypatch):
        win, path, raw = self._open_on_a_locked_queue_file(settings_dir, open_window, monkeypatch)
        with patch.object(queue_file, "_back_up_file", return_value=None):  # still locked
            win._queue_mgr.add_item("https://video.invalid/watch?v=2", "url", "New")
            _process_events_until(lambda: False, timeout=0.2)
        assert path.read_bytes() == raw and _backups(settings_dir) == []
        # The next change copies it aside, then the queue of this session is saved
        win._queue_mgr.add_item("https://video.invalid/watch?v=3", "url", "Newer")
        [backup] = _backups(settings_dir)
        assert backup.read_bytes() == raw
        assert _process_events_until(lambda: _saved(settings_dir) == [
            ("https://video.invalid/watch?v=2", "pending"),
            ("https://video.invalid/watch?v=3", "pending")])
        assert f"kept in {backup}" in _console(win)

    def test_closing_tries_the_copy_again(self, settings_dir, open_window, monkeypatch):
        win, _path, raw = self._open_on_a_locked_queue_file(settings_dir, open_window,
                                                            monkeypatch)
        with patch.object(queue_file, "_back_up_file", return_value=None):
            win._queue_mgr.add_item("https://video.invalid/watch?v=2", "url", "New")
        _close(win)
        [backup] = _backups(settings_dir)
        assert backup.read_bytes() == raw
        assert _saved(settings_dir) == [("https://video.invalid/watch?v=2", "pending")]

    def test_nothing_is_saved_over_a_file_that_stays_locked(self, settings_dir, open_window,
                                                            monkeypatch):
        win, path, raw = self._open_on_a_locked_queue_file(settings_dir, open_window, monkeypatch)
        with patch.object(queue_file, "_back_up_file", return_value=None):
            win._queue_mgr.add_item("https://video.invalid/watch?v=2", "url", "New")
            _close(win)
        assert path.read_bytes() == raw and _backups(settings_dir) == []

    def test_a_failing_save_does_not_stop_the_close(self, settings_dir, open_window):
        win = open_window()
        win._queue_mgr.add_item(_URL, "url", "First")
        with patch.object(queue_file, "write_queue_file", side_effect=ValueError("odd value")):
            _close(win)
        win._browser_tab.shutdown.assert_called_once()


def test_running_and_finished_songs_are_not_removed_one_by_one():
    manager = QueueManager()
    running, done, cancelled = (manager.add_item(f"{_URL}{n}", "url", "Song Title")
                                for n in range(3))
    running.status, done.status, cancelled.status = "running", "done", "cancelled"
    for item in (running, done, cancelled):
        manager.remove_item(item.id)
    assert manager.items == [running, done]


def test_the_queue_menu_keeps_the_menu_style():
    """The bigger font of the menu button must not reach its menu, a child of the button."""
    from src.gui.widgets.sidebar import Sidebar

    sidebar = Sidebar()
    button = sidebar._queue_menu_btn
    button.ensurePolished()
    button.menu().ensurePolished()
    assert button.font().pixelSize() == 18
    assert button.menu().font().pixelSize() != 18
