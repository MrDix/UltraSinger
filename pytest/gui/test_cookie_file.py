"""The exported cookie file: written in one step, one writer at a time,
private to the user, and never written back by a yt-dlp run."""

import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.gui import cookie_file
from src.gui.browser_tab import BrowserTab, _FormatProbeWorker
from src.gui.cookie_manager import CookieManager
from src.gui.ultrasinger_runner import ConversionWorker

_app = QApplication.instance() or QApplication([])


def test_write_replaces_the_file_in_one_step(tmp_path):
    target = tmp_path / "cookies.txt"
    target.write_text("old", encoding="utf-8")
    with patch("src.gui.cookie_file.os.replace", wraps=os.replace) as replace:
        cookie_file.write_cookie_file(target, "new content")
    replace.assert_called_once()
    assert target.read_text(encoding="utf-8") == "new content"
    assert [p.name for p in tmp_path.iterdir()] == ["cookies.txt"]  # no temporary file left


def test_writes_do_not_interleave(tmp_path):
    target = tmp_path / "cookies.txt"
    contents = [f"# writer {i}\n" + "x" * 200_000 for i in range(8)]
    threads = [threading.Thread(target=cookie_file.write_cookie_file, args=(target, text))
               for text in contents]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert target.read_text(encoding="utf-8") in contents


def test_failed_write_keeps_the_old_file(tmp_path):
    target = tmp_path / "cookies.txt"
    target.write_text("old", encoding="utf-8")
    with patch("src.gui.cookie_file.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            cookie_file.write_cookie_file(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["cookies.txt"]


def test_copy_and_stale_copies(tmp_path):
    target = tmp_path / "cookies.txt"
    target.write_text("cookies", encoding="utf-8")
    copy = cookie_file.copy_cookie_file(target)
    assert copy.parent == tmp_path and copy != target
    assert copy.read_text(encoding="utf-8") == "cookies"
    half_written = tmp_path / ".cookies-abc.tmp"  # a write a crash interrupted
    half_written.write_text("cook", encoding="utf-8")
    cookie_file.remove_stale_copies(tmp_path)
    assert not copy.exists() and not half_written.exists() and target.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_files_are_private(tmp_path):
    directory = tmp_path / "new_dir"
    target = cookie_file.write_cookie_file(directory / "cookies.txt", "cookies")
    assert directory.stat().st_mode & 0o777 == 0o700
    assert target.stat().st_mode & 0o777 == 0o600
    assert cookie_file.copy_cookie_file(target).stat().st_mode & 0o777 == 0o600
    assert cookie_file.write_private_copy(directory, "x").stat().st_mode & 0o777 == 0o600


def _cookie_manager():
    manager = CookieManager.__new__(CookieManager)  # no browser profile needed
    manager._cookies = []
    return manager


def test_cookie_manager_export_uses_the_atomic_write(tmp_path):
    target = tmp_path / "cookies.txt"
    with patch("src.gui.cookie_manager.write_cookie_file", wraps=cookie_file.write_cookie_file) as write:
        _cookie_manager().export_netscape(target)
    write.assert_called_once()
    assert target.read_text(encoding="utf-8").startswith("# Netscape HTTP Cookie File")


def test_reset_is_not_undone_by_a_running_conversion(tmp_path):
    """The bot-detection reset deletes the shared file while yt-dlp runs; its write-back
    at the end goes to the run's private copy."""
    shared = tmp_path / "cookies.txt"
    shared.write_text("stale cookies", encoding="utf-8")

    def fake_popen(cmd, **kwargs):
        cookie_file.remove_cookie_file(shared)  # the reset
        Path(cmd[cmd.index("--cookiefile") + 1]).write_text("stale cookies", encoding="utf-8")
        return MagicMock(stdout=iter([]), returncode=0)

    with patch("src.gui.ultrasinger_runner.subprocess.Popen", side_effect=fake_popen):
        ConversionWorker(["-i", "x", "--cookiefile", str(shared)]).run()
    assert list(tmp_path.iterdir()) == []


class _ProbeHost:
    """The attributes of BrowserTab that _probe_formats and shutdown use."""

    def __init__(self, tmp_path):
        self._page = None
        self._view = MagicMock()
        self.probe_cookie_file = str(tmp_path / "cookies.txt")
        self.cookie_manager = MagicMock(has_video_cookies=True)
        self.cookie_manager.export_private_copy.side_effect = (
            lambda directory: cookie_file.write_private_copy(directory, "cookies"))
        self.media_interceptor = MagicMock(get_po_token=MagicMock(return_value=""))
        self._probes = []
        self.results = []

    def _on_format_probed(self, *args):
        self.results.append(args)


def test_probe_uses_a_private_copy_and_removes_it(tmp_path):
    host = _ProbeHost(tmp_path)
    seen = {}

    def fake_dump(worker):
        seen["cookie_file"] = worker._cookie_file
        seen["existed"] = Path(worker._cookie_file).exists()
        return None

    with patch.object(_FormatProbeWorker, "_dump_json", fake_dump):
        BrowserTab._probe_formats(host, "vid")
        for thread, _worker in host._probes:
            thread.join(5)
    assert seen["existed"]
    assert Path(seen["cookie_file"]).parent == tmp_path
    assert Path(seen["cookie_file"]).name != "cookies.txt"  # the shared file is not touched
    assert list(tmp_path.iterdir()) == []  # the copy is gone


def test_shutdown_stops_running_probes(tmp_path):
    host = _ProbeHost(tmp_path)
    release = threading.Event()
    worker = MagicMock()
    worker.cancel.side_effect = release.set
    thread = threading.Thread(target=release.wait, args=(5,), daemon=True)
    thread.start()
    host._probes = [(thread, worker)]
    host._page = MagicMock()
    BrowserTab.shutdown(host)
    worker.cancel.assert_called_once()
    assert not thread.is_alive()
    assert host._probes == []


def test_probe_cancel_kills_yt_dlp():
    worker = _FormatProbeWorker("vid")
    process = MagicMock(returncode=-9)
    process.communicate.return_value = ("", "")

    def popen(*args, **kwargs):
        worker.cancel()  # the window closes while yt-dlp starts
        return process

    with patch("src.gui.browser_tab.shutil.which", return_value="yt-dlp"), \
            patch("src.gui.browser_tab.subprocess.Popen", side_effect=popen):
        assert worker._dump_json() is None
    process.kill.assert_called()
