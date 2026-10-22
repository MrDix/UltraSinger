"""Round-trip tests for the download limits of a video link in the GUI.

ConversionSettingsForm widgets -> collect_config() -> UltraSingerRunner.build_args()
-> UltraSinger.py CLI option parsing (init_settings) -> Settings fields; and the
quality badge of the browser tab, which names the resolution a conversion downloads.
"""

import json
import logging
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from src.gui import browser_tab
from src.gui.config import _DEFAULTS, is_limit, limit_setting
from src.gui.queue_file import PER_SONG_SETTINGS, read_queue_file
from src.gui.settings_dialog import PerSongSettingsDialog
from src.gui.settings_tab import ConversionSettingsForm
from src.gui.ultrasinger_runner import UltraSingerRunner
from src.Settings import Settings
from src.UltraSinger import init_settings

_app = QApplication.instance() or QApplication([])

_KEYS = ("download_rate_limit", "max_video_resolution", "max_video_fps")


@pytest.fixture(autouse=True)
def _limit_warnings_anew():
    """Each test sees the warnings about stored limits that are no limit (given once a run)."""
    from src.gui import config as config_module
    config_module._warned_limits.clear()
    yield
    config_module._warned_limits.clear()
_URL = "https://video.invalid/watch?v=x"


def _combos(form):
    return form._download_rate_limit, form._max_video_resolution, form._max_video_fps


def _shown(form):
    return tuple(combo.currentData() for combo in _combos(form))


def _cli(config):
    """The download limits of the CLI run that the GUI starts with ``config``."""
    args = UltraSingerRunner().build_args(config, _URL)
    settings = init_settings(args)
    return tuple(getattr(settings, key) for key in _KEYS)


class TestDownloadLimitWidgets(unittest.TestCase):

    def tearDown(self):
        init_settings(["-i", "test.mp3"])  # reset the shared settings

    def test_defaults_are_those_of_the_cli(self):
        self.assertEqual(tuple(_DEFAULTS[key] for key in _KEYS), (2000, 1080, 60))
        self.assertEqual(tuple(getattr(Settings(), key) for key in _KEYS), (2000, 1080, 60))
        form = ConversionSettingsForm({})
        self.assertEqual(_shown(form), (2000, 1080, 60))
        config = form.collect_config()
        self.assertEqual(tuple(config[key] for key in _KEYS), (2000, 1080, 60))

    def test_always_passed_so_the_gui_choice_wins(self):
        args = UltraSingerRunner().build_args(ConversionSettingsForm({}).collect_config(), _URL)
        for key, value in zip(_KEYS, ("2000", "1080", "60")):
            self.assertEqual(args[args.index(f"--{key}") + 1], value)
        # Also without the settings in the config, and as whole numbers
        self.assertEqual(_cli({}), (2000, 1080, 60))
        self.assertEqual(_cli({"max_video_resolution": 720.0}), (2000, 720, 60))

    def test_no_limit_round_trip(self):
        form = ConversionSettingsForm({key: 0 for key in _KEYS})
        self.assertEqual([combo.currentText() for combo in _combos(form)], ["No limit"] * 3)
        self.assertEqual(_cli(form.collect_config()), (0, 0, 0))

    def test_every_choice_reaches_the_cli(self):
        form = ConversionSettingsForm({})
        for index, (key, combo) in enumerate(zip(_KEYS, _combos(form))):
            for i in range(combo.count()):
                with self.subTest(key=key, choice=combo.itemText(i)):
                    combo.setCurrentIndex(i)
                    self.assertEqual(_cli(form.collect_config())[index], combo.itemData(i))
            combo.setCurrentIndex(combo.findData(_DEFAULTS[key]))

    def test_a_value_that_is_no_limit_is_shown_and_run_as_the_default(self):
        """A hand-edited -1 made every conversion end: the command line refuses it."""
        config = {"download_rate_limit": -1, "max_video_resolution": 1080.5, "max_video_fps": None}
        with self.assertLogs(level="WARNING") as logs:
            form = ConversionSettingsForm(config)
        self.assertEqual(_shown(form), (2000, 1080, 60))
        self.assertEqual(_cli(config), (2000, 1080, 60))
        self.assertEqual(_cli(form.collect_config()), (2000, 1080, 60))
        for key in _KEYS:
            self.assertIn(f"'{key}'", "\n".join(logs.output))

    def test_another_limit_is_shown_as_it_is_used(self):
        """From a hand-edited config or a queue file: an entry of its own, among the others."""
        config = {"download_rate_limit": 3000, "max_video_resolution": 900, "max_video_fps": 50}
        form = ConversionSettingsForm(config)
        self.assertEqual([combo.currentText() for combo in _combos(form)], ["3000 KiB/s", "900p", "50 fps"])
        resolutions = form._max_video_resolution
        self.assertEqual([resolutions.itemData(i) for i in range(resolutions.count())],
                         [360, 480, 720, 900, 1080, 1440, 2160, 0])
        self.assertEqual(_cli(config), (3000, 900, 50))
        self.assertEqual(_cli(form.collect_config()), (3000, 900, 50))
        # A whole number saved as a float is the value of its entry
        form = ConversionSettingsForm({"download_rate_limit": 1000.0})
        self.assertEqual(form._download_rate_limit.currentText(), "1000 KiB/s (about 8 Mbit/s)")
        self.assertEqual(_shown(form), (1000, 1080, 60))

    def test_reset_buttons_select_the_defaults(self):
        form = ConversionSettingsForm({"download_rate_limit": 0, "max_video_resolution": 720, "max_video_fps": 30})
        self.assertEqual(_shown(form), (0, 720, 30))
        for combo in _combos(form):
            reset = next(button for button in combo.parentWidget().findChildren(QPushButton)
                         if button.text() == "↺")
            reset.click()
        self.assertEqual(_shown(form), (2000, 1080, 60))

    def test_a_per_song_choice_is_an_override(self):
        self.assertTrue(set(_KEYS) <= PER_SONG_SETTINGS)
        dialog = PerSongSettingsDialog(dict(_DEFAULTS), {}, [])
        dialog._form._max_video_fps.setCurrentIndex(dialog._form._max_video_fps.findData(30))
        overrides = dialog.get_overrides()
        self.assertEqual(overrides.get("max_video_fps"), 30)
        self.assertNotIn("download_rate_limit", overrides)
        self.assertNotIn("max_video_resolution", overrides)
        # The override wins over the global setting
        self.assertEqual(_cli({**_DEFAULTS, **overrides}), (2000, 1080, 30))


class TestLimitValues:
    """One check of a download limit for the settings, the command line and the queue file."""

    @pytest.mark.parametrize("value", [0, 60, 2000, 1080.0, 2**31 - 1])
    def test_whole_numbers_of_0_or_more(self, value):
        assert is_limit(value)

    @pytest.mark.parametrize("value", [-1, 1.5, float("nan"), float("inf"), 2**31, True, None, "60", [60]])
    def test_no_limit(self, value):
        assert not is_limit(value)

    def test_the_default_for_a_value_that_is_no_limit(self, caplog):
        assert limit_setting({"max_video_fps": 30.0}, "max_video_fps") == 30
        assert limit_setting({}, "max_video_fps") == 60
        with caplog.at_level(logging.WARNING):
            assert limit_setting({"max_video_fps": -1}, "max_video_fps") == 60
        assert "'max_video_fps'" in caplog.text

    def test_the_warning_comes_once_for_each_value(self, caplog):
        """The badge and each queued song read the setting: the same bad value warns once."""
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                assert limit_setting({"max_video_fps": -1}, "max_video_fps") == 60
            assert limit_setting({"max_video_resolution": -1}, "max_video_resolution") == 1080
            assert limit_setting({"max_video_fps": [60]}, "max_video_fps") == 60
            assert limit_setting({"max_video_fps": [60]}, "max_video_fps") == 60
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 3
        assert "'max_video_fps': -1" in warnings[0]
        assert "'max_video_resolution': -1" in warnings[1]
        assert "'max_video_fps': [60]" in warnings[2]

    @pytest.mark.parametrize("imported", [False, True])
    def test_a_queue_file_keeps_only_limits(self, tmp_path, imported):
        """The saved queue and an imported file alike: a value the command line refuses ends the song."""
        path = tmp_path / "queue.json"
        settings = {"download_rate_limit": -1, "max_video_resolution": 900, "max_video_fps": 1.5}
        path.write_text(json.dumps({"version": 1, "items": [
            {"input_type": "url", "input_source": _URL, "settings_overrides": settings}]}), encoding="utf-8")
        [item], problems = read_queue_file(path, imported=imported)
        assert item.settings_overrides == {"max_video_resolution": 900}
        assert sorted(problems) == [
            "entry 1: setting 'download_rate_limit' ignored (no whole number of 0 or more)",
            "entry 1: setting 'max_video_fps' ignored (no whole number of 0 or more)"]


# A 4K video as yt-dlp lists its formats: VP9 (webm) and AV1 (mp4), 1080p, and the audio
_FORMATS = [
    {"format_id": "313", "ext": "webm", "vcodec": "vp9", "acodec": "none", "width": 3840, "height": 2160,
     "fps": 30, "url": "https://video.invalid/313"},
    {"format_id": "401", "ext": "mp4", "vcodec": "av01.0.12M.08", "acodec": "none", "width": 3840, "height": 2160,
     "fps": 30, "url": "https://video.invalid/401"},
    {"format_id": "399", "ext": "mp4", "vcodec": "av01.0.08M.08", "acodec": "none", "width": 1920, "height": 1080,
     "fps": 30, "url": "https://video.invalid/399"},
    {"format_id": "251", "ext": "webm", "vcodec": "none", "acodec": "opus", "abr": 130,
     "url": "https://audio.invalid/251"},
]


def _badge_video_line(formats, **limits):
    """The video line of the browser tab's quality badge (no request: yt-dlp's output stands in)."""
    worker = browser_tab._FormatProbeWorker("vid", **limits)
    shown = []
    worker.finished.connect(lambda _video_id, text, _language: shown.append(text))
    info = {"title": "Artist - Title", "duration": 200, "formats": formats}
    with patch.object(browser_tab._FormatProbeWorker, "_dump_json", return_value=json.dumps(info)), \
            patch.object(browser_tab._FormatProbeWorker, "_check_lrclib", return_value="lyrics"):
        worker.run()
    return shown[0].splitlines()[0]


class TestQualityBadge:
    def test_names_the_resolution_that_is_downloaded(self, capsys):
        assert _badge_video_line(_FORMATS, max_resolution=1080, max_fps=60) == "\U0001f4f9 2160p VP9 (downloads 1080p)"
        assert _badge_video_line(_FORMATS, max_resolution=720, max_fps=60) == "\U0001f4f9 2160p VP9 (downloads 1080p)"
        # The badge runs in the GUI: it prints nothing
        assert capsys.readouterr() == ("", "")

    def test_nothing_more_when_the_best_is_downloaded(self):
        assert _badge_video_line(_FORMATS) == "\U0001f4f9 2160p VP9"
        assert _badge_video_line(_FORMATS[2:], max_resolution=1080) == "\U0001f4f9 1080p AV1"

    def test_the_probe_gets_the_saved_limits(self):
        """BrowserTab hands the limits of its settings to the probe; one that is none gets the default."""
        tab = SimpleNamespace(_probes=[], _page=None, probe_cookie_file="", signin_allowed=False,
                              cookie_manager=MagicMock(has_video_cookies=False),
                              media_interceptor=MagicMock(get_po_token=MagicMock(return_value="")),
                              _on_format_probed=MagicMock(), _on_badge_ready=MagicMock())
        for config, expected in (({}, {"max_resolution": 1080, "max_fps": 60}),
                                 ({"max_video_resolution": 720, "max_video_fps": 0}, {"max_resolution": 720, "max_fps": 0}),
                                 ({"max_video_resolution": -5}, {"max_resolution": 1080, "max_fps": 60})):
            tab.probe_config = config
            with patch.object(browser_tab, "_FormatProbeWorker") as worker, \
                    patch.object(browser_tab.threading, "Thread"):
                browser_tab.BrowserTab._probe_formats(tab, "vid")
            assert {key: worker.call_args.kwargs[key] for key in expected} == expected


class _BadgeHost:
    """The badge part of BrowserTab, on a page that records its scripts."""

    _on_badge_ready = browser_tab.BrowserTab._on_badge_ready
    _on_format_probed = browser_tab.BrowserTab._on_format_probed
    _from_earlier_probe = browser_tab.BrowserTab._from_earlier_probe
    _probe_limits = browser_tab.BrowserTab._probe_limits
    refresh_quality_badge = browser_tab.BrowserTab.refresh_quality_badge

    def __init__(self, config):
        self.probe_config = config
        self._page = MagicMock()
        self._probe_video_id = "vid"
        self._badge = None

    def shown(self):
        """The video line of the badge that the page shows last."""
        script = self._page.runJavaScript.call_args.args[0]
        return script.split("b.textContent = '")[1].split("\\n")[0]


def _probed_badge(config):
    """A host with the badge of a probe made with the limits of ``config``."""
    host = _BadgeHost(config)
    worker = browser_tab._FormatProbeWorker(
        "vid", max_resolution=limit_setting(config, "max_video_resolution"),
        max_fps=limit_setting(config, "max_video_fps"))
    worker.badge_ready.connect(host._on_badge_ready)
    worker.finished.connect(host._on_format_probed)
    host._active_probe = 0  # the worker's default number
    info = {"title": "Artist - Title", "duration": 200, "formats": _FORMATS}
    with patch.object(browser_tab._FormatProbeWorker, "_dump_json", return_value=json.dumps(info)), \
            patch.object(browser_tab._FormatProbeWorker, "_check_lrclib", return_value="lyrics"):
        worker.run()
    return host


class TestBadgeAfterSavedLimits:
    def test_saved_limits_update_the_badge_without_a_new_probe(self):
        config = {"max_video_resolution": 1080, "max_video_fps": 60}
        host = _probed_badge(config)
        assert host.shown() == "\U0001f4f9 2160p VP9 (downloads 1080p)"
        config["max_video_resolution"] = 0  # saved in place, as the window does
        with patch.object(browser_tab, "_FormatProbeWorker") as worker, \
                patch.object(browser_tab.subprocess, "Popen") as popen:
            host.refresh_quality_badge()
        assert not worker.called and not popen.called
        assert host.shown() == "\U0001f4f9 2160p VP9"
        config["max_video_resolution"] = 1080
        host.refresh_quality_badge()
        assert host.shown() == "\U0001f4f9 2160p VP9 (downloads 1080p)"

    def test_a_probe_that_finishes_after_a_save_shows_the_saved_limits(self):
        host = _BadgeHost({"max_video_resolution": 0, "max_video_fps": 0})
        badge = browser_tab._QualityBadge({"formats": _FORMATS}, 2160, "VP9", ["lyrics"])
        host._on_badge_ready("vid", badge)
        host._on_format_probed("vid", badge.text(1080, 60), "")  # started with the old limits
        assert host.shown() == "\U0001f4f9 2160p VP9"

    def test_nothing_is_shown_without_a_probed_badge(self):
        host = _BadgeHost({})
        host.refresh_quality_badge()
        host._on_badge_ready("other", browser_tab._QualityBadge({"formats": _FORMATS}, 2160, "VP9", []))
        host.refresh_quality_badge()  # the badge of another video is not kept
        assert not host._page.runJavaScript.called


def test_the_result_of_a_replaced_probe_of_the_same_video_is_not_taken():
    """A second probe of the same video (e.g. once the PO token provider is ready)
    replaces the first. A result of the first that is still queued is not shown
    or kept: each result names its probe (the worker may be gone by then)."""
    host = _BadgeHost({"max_video_resolution": 0, "max_video_fps": 0})
    earlier, newer = 1, 2
    host._active_probe = newer
    host._on_badge_ready("vid", browser_tab._QualityBadge({"formats": _FORMATS}, 2160, "VP9", []), earlier)
    host._on_format_probed("vid", "old text", "", earlier)
    assert host._badge is None
    assert not host._page.runJavaScript.called
    badge = browser_tab._QualityBadge({"formats": _FORMATS[2:]}, 1080, "AV1", [])
    host._on_badge_ready("vid", badge, newer)
    host._on_format_probed("vid", badge.text(0, 0), "", newer)
    assert host._badge is badge
    assert host._page.runJavaScript.call_count == 1
    assert "1080p AV1" in host._page.runJavaScript.call_args.args[0]


def test_a_probe_result_names_its_probe():
    """The worker sends its number with both results, so that a replaced probe's can be told apart."""
    worker = browser_tab._FormatProbeWorker("vid", probe_number=7)
    got = []
    worker.badge_ready.connect(lambda *args: got.append(("badge", args[-1])))
    worker.finished.connect(lambda *args: got.append(("finished", args[-1])))
    info = {"title": "Artist - Title", "duration": 200, "formats": _FORMATS}
    with patch.object(browser_tab._FormatProbeWorker, "_dump_json", return_value=json.dumps(info)),             patch.object(browser_tab._FormatProbeWorker, "_check_lrclib", return_value="lyrics"):
        worker.run()
    assert got == [("badge", 7), ("finished", 7)]


class _FakeBrowserTab(QWidget):
    convert_requested = Signal(str, str)
    settings_requested = Signal()

    def __init__(self, signin_allowed=False):
        super().__init__()
        self.cookie_manager = MagicMock(has_video_cookies=False)
        self.media_interceptor = MagicMock()
        self.probe_cookie_file = ""
        self.refresh_quality_badge = MagicMock()
        self.shutdown = MagicMock()
        self._view = MagicMock()


def test_the_window_gives_the_browser_tab_its_saved_settings(tmp_path, monkeypatch):
    """The settings that the window saves in place: the badge follows a saved change."""
    from src.gui import config as config_module
    from src.gui import main_window
    from src.gui.widgets import llm_provider_list

    monkeypatch.setattr(config_module, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(llm_provider_list, "_shutting_down", False)
    with patch.object(main_window, "BrowserTab", _FakeBrowserTab), \
            patch.object(main_window.MainWindow, "_start_potoken_provider", lambda self: None), \
            patch.object(main_window.MainWindow, "_start_app_update_check", lambda self: None), \
            patch.object(main_window.MainWindow, "_start_ytdlp_update_check", lambda self: None):
        win = main_window.MainWindow({**_DEFAULTS, "cookie_file": ""})
    assert win._browser_tab.probe_config is win._config
    # "Save Settings" shows the badge with the saved limits
    win._settings_tab._save()
    assert win._browser_tab.refresh_quality_badge.called


if __name__ == "__main__":
    unittest.main()
