"""Cancelling the queue and closing the window while work runs.

A cancel during the browser-stream pre-download must end the queue; closing
the window must ask first, then stop all work and wait for its threads (a
QThread destroyed while it runs aborts the application).
"""

import functools
import os
import subprocess
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
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from src.gui.config import _DEFAULTS
from src.gui.media_downloader import MediaDownloadWorker
from src.gui.queue_manager import QueueManager
from src.gui.ultrasinger_runner import ConversionWorker, UltraSingerRunner

_app = QApplication.instance() or QApplication([])


def _process_events_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.01)
    return condition()


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


def _start_with_blocking_download(tmp_path, monkeypatch):
    slot_threads = []
    original = QueueManager._on_predownload_finished

    @functools.wraps(original)  # PySide finds the slot by its name
    def record_thread(self, *args):
        slot_threads.append(threading.current_thread() is threading.main_thread())
        original(self, *args)

    monkeypatch.setattr(QueueManager, "_on_predownload_finished", record_thread)
    manager = QueueManager()
    manager._runner.start = MagicMock()
    manager.set_global_config({**_DEFAULTS, "output_folder": str(tmp_path), "cookie_file": ""})
    manager.set_media_interceptor(MagicMock(
        get_po_token=MagicMock(return_value=""),
        get_stream_with_status=MagicMock(return_value=(
            SimpleNamespace(url="https://media.invalid/x", mime_type="audio/webm",
                            seconds_until_expiry=600, is_muxed=False), "ok")),
    ))
    item = manager.add_item("https://video.invalid/watch?v=abc", "url", "Song Title")
    item.video_id = "abc"
    finished = []
    manager.queue_finished.connect(lambda failed, cancelled: finished.append(cancelled))
    worker = _BlockingDownload()

    def fake_start(url, path, keep_video=False, parent=None):
        Path(path).write_bytes(b"partial")
        thread = QThread(parent)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        return thread, worker

    with patch("src.gui.media_downloader.start_media_download", side_effect=fake_start):
        manager.start_all()
    return manager, item, finished, slot_threads


class TestCancelDuringPreDownload:
    def test_cancel_all_ends_the_item_and_the_queue(self, tmp_path, monkeypatch):
        manager, item, finished, slot_threads = _start_with_blocking_download(tmp_path, monkeypatch)
        assert manager.is_running and item.status == "running"

        manager.cancel_all()
        assert _process_events_until(lambda: finished), "queue did not finish"

        assert finished == [True]
        assert item.status == "cancelled"
        assert not manager.is_running
        manager._runner.start.assert_not_called()  # no fallback conversion
        assert slot_threads == [True]  # the queue continued on the GUI thread
        assert not list(tmp_path.glob("_intercepted_*"))  # partial download removed

    def test_shutdown_waits_for_the_download_thread(self, tmp_path, monkeypatch):
        manager, item, finished, _ = _start_with_blocking_download(tmp_path, monkeypatch)
        thread = manager._download_thread
        manager.shutdown()
        assert thread.isFinished()
        assert _process_events_until(lambda: finished)
        assert item.status == "cancelled"


class TestMediaDownloadWorker:
    def test_cancel_before_start_runs_no_ffmpeg(self, tmp_path):
        worker = MediaDownloadWorker("https://media.invalid/x", str(tmp_path / "a.webm"))
        results = []
        worker.finished.connect(lambda ok, path, err: results.append(ok))
        worker.cancel()
        with patch("src.gui.media_downloader.subprocess.Popen") as popen:
            worker.run()
        popen.assert_not_called()
        assert results == [False] and worker.is_cancelled

    def test_cancel_after_start_kills_ffmpeg(self, tmp_path):
        worker = MediaDownloadWorker("https://media.invalid/x", str(tmp_path / "a.webm"))
        process = MagicMock(returncode=-9)
        process.communicate.return_value = (b"", b"killed")

        def popen(*args, **kwargs):
            worker.cancel()  # the user cancels while ffmpeg starts
            return process

        with patch("src.gui.media_downloader._find_ffmpeg", return_value=Path("ffmpeg")), \
                patch("src.gui.media_downloader.subprocess.Popen", side_effect=popen):
            worker.run()
        process.kill.assert_called()


class _EndlessConversion(ConversionWorker):
    """A conversion whose process runs until cancel() stops it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop = threading.Event()

    def run(self):
        self._stop.wait(10)
        self.finished.emit(-2 if self._cancelled else 0)

    def cancel(self):
        self._cancelled = True
        self._stop.set()


class _StubbornConversion(_EndlessConversion):
    """A conversion whose processes survive the cancel (e.g. a child ignores SIGTERM)."""

    def run(self):
        self._stop.wait(60)
        self.finished.emit(-2)

    def cancel(self):
        self._cancelled = True

    def force_kill(self):
        self._stop.set()


class TestRunnerShutdown:
    def test_shutdown_cancels_and_joins_the_thread(self):
        runner = UltraSingerRunner()
        with patch("src.gui.ultrasinger_runner.ConversionWorker", _EndlessConversion):
            runner.start(["-i", "x.mp3"])
        thread = runner._thread
        assert _process_events_until(thread.isRunning)
        runner.shutdown()
        assert thread.isFinished()
        assert not runner.is_running

    def test_processes_surviving_the_cancel_are_killed(self, monkeypatch):
        monkeypatch.setattr(UltraSingerRunner, "_SHUTDOWN_GRACE_MS", 100)
        runner = UltraSingerRunner()
        with patch("src.gui.ultrasinger_runner.ConversionWorker", _StubbornConversion):
            runner.start(["-i", "x.mp3"])
        thread = runner._thread
        assert _process_events_until(thread.isRunning)
        start = time.monotonic()
        runner.shutdown()
        assert thread.isFinished()
        assert time.monotonic() - start < 10  # killed, not waited out

    def test_shutdown_without_a_conversion(self):
        UltraSingerRunner().shutdown()


class TestLanguageInAutoMode:
    def test_auto_mode_passes_no_language(self):
        """The CLI takes the video platform's language itself, as a hint it does not trust fully."""
        manager = QueueManager()
        manager._runner.start = MagicMock()
        manager.set_global_config({**_DEFAULTS, "language_mode": "auto", "language": "de",
                                   "cookie_file": ""})
        manager.add_item("https://video.invalid/watch?v=abc", "url", "Song Title")
        manager.start_all()
        args = manager._runner.start.call_args.args[0]
        assert "--language" not in args


class _FakeBrowserTab(QWidget):
    convert_requested = Signal(str, str)
    settings_requested = Signal()

    def __init__(self, signin_allowed=False):
        super().__init__()
        self.cookie_manager = MagicMock(has_video_cookies=False)
        self.media_interceptor = MagicMock()
        self.probe_cookie_file = ""
        self.set_signin_allowed = MagicMock()
        self.shutdown = MagicMock()
        self._view = MagicMock()


@pytest.fixture
def settings_dir(tmp_path, monkeypatch):
    """The settings directory, cleaned of stale private files when the window starts."""
    from src.gui import config as config_module

    monkeypatch.setattr(config_module, "_CONFIG_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def window(monkeypatch, settings_dir):
    from src.gui import main_window
    from src.gui.widgets import llm_provider_list

    # Closing the window stops model fetches for good; later tests start them again
    monkeypatch.setattr(llm_provider_list, "_shutting_down", False)
    # Plain functions: PySide crashes on mocks in the class while connecting slots
    with patch.object(main_window, "BrowserTab", _FakeBrowserTab), \
            patch.object(main_window.MainWindow, "_start_potoken_provider", lambda self: None), \
            patch.object(main_window.MainWindow, "_start_app_update_check", lambda self: None), \
            patch.object(main_window.MainWindow, "_start_ytdlp_update_check", lambda self: None):
        win = main_window.MainWindow({**_DEFAULTS, "cookie_file": ""})
    with patch.object(main_window, "save_config"):
        yield win


def test_start_removes_stale_private_files(settings_dir, monkeypatch):
    """A crash can leave a cookie copy or a PO-token option file; the next start removes them,
    also without a cookie file setting."""
    from src.gui import main_window
    from src.gui.widgets import llm_provider_list

    monkeypatch.setattr(llm_provider_list, "_shutting_down", False)
    crashed = subprocess.Popen([sys.executable, "-c", "pass"])  # an earlier, ended session
    crashed.wait()
    options = settings_dir / f"ytdlp-options-{crashed.pid}-a1.conf"
    copy = settings_dir / f"cookies-copy-{crashed.pid}-b2.txt"
    for leftover in (options, copy):
        leftover.write_text("secret", encoding="utf-8")
    with patch.object(main_window, "BrowserTab", _FakeBrowserTab), \
            patch.object(main_window.MainWindow, "_start_potoken_provider", lambda self: None), \
            patch.object(main_window.MainWindow, "_start_app_update_check", lambda self: None), \
            patch.object(main_window.MainWindow, "_start_ytdlp_update_check", lambda self: None):
        main_window.MainWindow({**_DEFAULTS, "cookie_file": ""})
    assert not options.exists() and not copy.exists()


class TestCloseWindow:
    def _close(self, win, answer=None):
        from PySide6.QtGui import QCloseEvent

        event = QCloseEvent()
        with patch.object(QMessageBox, "question", return_value=answer) as question, \
                patch.object(QMessageBox, "information") as information:
            win.closeEvent(event)
        return event, question, information

    def test_asks_before_cancelling_a_conversion(self, window):
        window._queue_mgr._running = True
        window._queue_mgr.shutdown = MagicMock()
        event, question, _ = self._close(window, QMessageBox.StandardButton.No)
        question.assert_called_once()
        assert not event.isAccepted()
        window._queue_mgr.shutdown.assert_not_called()

    def test_quitting_stops_the_queue(self, window):
        window._queue_mgr._running = True
        window._queue_mgr.shutdown = MagicMock()
        event, _, _ = self._close(window, QMessageBox.StandardButton.Yes)
        assert event.isAccepted()
        window._queue_mgr.shutdown.assert_called_once()
        window._browser_tab.shutdown.assert_called_once()

    def test_not_while_yt_dlp_is_updated(self, window):
        running = MagicMock(isRunning=MagicMock(return_value=True))
        window._settings_tab._ytdlp_update_thread = running
        event, _, information = self._close(window)
        information.assert_called_once()
        assert not event.isAccepted()
        window._settings_tab._ytdlp_update_thread = None

    def test_without_running_work_no_question(self, window):
        event, question, _ = self._close(window)
        question.assert_not_called()
        assert event.isAccepted()

    @pytest.mark.parametrize("deleted", [True, False])
    def test_bot_reset_logs_only_a_real_deletion(self, window, caplog, deleted):
        """The reset happens while signing in is allowed (without it no cookies are used)."""
        from src.gui import main_window

        window._config["allow_platform_signin"] = True
        window._config["cookie_file"] = "cookies.txt"
        with patch.object(main_window, "remove_cookie_file", return_value=deleted) as remove, \
                patch.object(QMessageBox, "information"), \
                caplog.at_level("INFO", logger=main_window.logger.name):
            window._check_for_bot_detection("ERROR: Sign in to confirm you're not a bot")
        remove.assert_called_once_with("cookies.txt")
        assert ("Deleted cookie file" in caplog.text) is deleted

    def test_closing_stops_model_fetches(self, window):
        from src.gui.widgets import llm_provider_list

        self._close(window)
        assert llm_provider_list._shutting_down

    def test_quitting_the_application_stops_model_fetches(self, window):
        """Second line of defence: a fetch started after closeEvent is joined on quit."""
        from src.gui.widgets import llm_provider_list

        release = threading.Event()
        fetcher = llm_provider_list._ModelFetcher("https://one.invalid/v1", "")
        with patch.object(llm_provider_list._ModelFetcher, "run",
                          lambda self: (release.wait(5), self.done.emit())):
            assert llm_provider_list.start_model_fetch(fetcher)
            thread = llm_provider_list._running_fetches[-1][0]
            threading.Timer(0.2, release.set).start()
            _app.aboutToQuit.emit()
        assert thread.isFinished()
        assert llm_provider_list._shutting_down


class TestUpdateQuestion:
    """A newer UltraSinger version found on start: the question and the update on close."""

    @staticmethod
    def _info(commit="abc123", count=2):
        from src.gui.app_updater import UpdateInfo

        return UpdateInfo("origin/main", commit, count, ["Newest change", "Older change"])

    @staticmethod
    def _answer(label):
        """Answer the next question with the button of this label."""
        return patch.multiple(
            QMessageBox,
            exec=lambda self: None,
            clickedButton=lambda self: next(b for b in self.buttons() if b.text() == label),
        )

    def test_no_check_when_switched_off(self, window):
        window._config["check_for_updates"] = False
        window._start_app_update_check()
        assert window._app_update_thread is None

    def test_nothing_without_an_update(self, window):
        with patch.object(window, "_ask_to_update") as ask:
            window._on_app_update_checked(None)
        ask.assert_not_called()

    def test_asks_at_once(self, window):
        info = self._info()
        with patch.object(window, "_ask_to_update") as ask:
            window._on_app_update_checked(info)
        ask.assert_called_once_with(info)

    def test_asks_after_a_running_conversion(self, window):
        info = self._info()
        window._queue_mgr._running = True
        with patch.object(window, "_ask_to_update") as ask:
            window._on_app_update_checked(info)
            ask.assert_not_called()
            window._queue_mgr._running = False
            window._on_queue_finished(0, False)
            window._on_queue_finished(0, False)
        ask.assert_called_once_with(info)

    def test_skipped_version_is_not_asked_again(self, window):
        window._config["update_skipped_commit"] = "abc123"
        with patch.object(window, "_ask_to_update") as ask:
            window._on_app_update_checked(self._info("abc123"))
            ask.assert_not_called()
            window._on_app_update_checked(self._info("def456"))
        ask.assert_called_once()

    def test_skip_this_version(self, window):
        from src.gui import main_window

        with self._answer("Skip This Version"), patch.object(window, "_update_now") as update:
            window._ask_to_update(self._info("abc123"))
        update.assert_not_called()
        assert window._config["update_skipped_commit"] == "abc123"
        main_window.save_config.assert_called()

    def test_later(self, window):
        with self._answer("Later"), patch.object(window, "_update_now") as update:
            window._ask_to_update(self._info())
        update.assert_not_called()
        assert window._config.get("update_skipped_commit", "") == ""

    def test_update_now(self, window):
        with self._answer("Update Now"), patch("src.gui.app_updater.can_start_update", return_value=True), \
                patch.object(window, "_update_now") as update:
            window._ask_to_update(self._info())
        update.assert_called_once()

    def test_without_an_update_script_to_start_only_tells(self, window):
        buttons = []
        with patch.multiple(QMessageBox, clickedButton=lambda self: None,
                            exec=lambda self: buttons.extend(b.text() for b in self.buttons())),                 patch("src.gui.app_updater.can_start_update", return_value=False):
            window._ask_to_update(self._info())
        assert "Update Now" not in buttons and {"Later", "Skip This Version"} <= set(buttons)

    def test_buttons_are_not_squeezed(self, window):
        """The box grows instead of cutting off "Skip This Version" (#217 follow-up)."""
        from PySide6.QtWidgets import QPushButton

        widths = []

        def show(box):
            widths.extend((b.minimumWidth(), b.sizeHint().width()) for b in box.findChildren(QPushButton))

        with patch.multiple(QMessageBox, clickedButton=lambda self: None, exec=show),                 patch("src.gui.app_updater.can_start_update", return_value=True):
            window._ask_to_update(self._info())
        assert len(widths) == 4  # Update Now, Later, Skip This Version, Show Details...
        assert all(minimum >= hint > 0 for minimum, hint in widths)

    def test_closing_starts_the_update(self, window):
        from PySide6.QtGui import QCloseEvent

        window._update_on_close = True
        with patch("src.gui.app_updater.start_update", return_value=True) as start:
            window.closeEvent(QCloseEvent())
        start.assert_called_once_with()

    def test_a_failed_start_is_reported(self, window):
        from PySide6.QtGui import QCloseEvent

        window._update_on_close = True
        with patch("src.gui.app_updater.start_update", return_value=False), \
                patch.object(QMessageBox, "warning") as warning:
            window.closeEvent(QCloseEvent())
        warning.assert_called_once()

    def test_no_update_when_closing_is_cancelled(self, window):
        window._queue_mgr._running = True
        window._queue_mgr.shutdown = MagicMock()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No), \
                patch("src.gui.app_updater.start_update") as start:
            window._update_now()
        start.assert_not_called()
        assert not window._update_on_close

    def test_plain_closing_starts_no_update(self, window):
        with patch("src.gui.app_updater.start_update") as start:
            TestCloseWindow()._close(window)
        start.assert_not_called()
