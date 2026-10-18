"""Round-trip tests for the download limits of a video link in the GUI.

ConversionSettingsForm widgets -> collect_config() -> UltraSingerRunner.build_args()
-> UltraSinger.py CLI option parsing (init_settings) -> Settings fields.
"""

import os
import unittest

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton

from src.gui.config import _DEFAULTS
from src.gui.queue_file import PER_SONG_SETTINGS
from src.gui.settings_dialog import PerSongSettingsDialog
from src.gui.settings_tab import ConversionSettingsForm
from src.gui.ultrasinger_runner import UltraSingerRunner
from src.Settings import Settings
from src.UltraSinger import init_settings

_app = QApplication.instance() or QApplication([])

_KEYS = ("download_rate_limit", "max_video_resolution", "max_video_fps")
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

    def test_unknown_config_values_show_the_default(self):
        with self.assertLogs(level="WARNING") as logs:
            form = ConversionSettingsForm({"download_rate_limit": 3000, "max_video_resolution": 900,
                                           "max_video_fps": None})
        self.assertEqual(_shown(form), (2000, 1080, 60))
        self.assertIn("max_video_resolution", "\n".join(logs.output))
        # A whole number saved as a float is no unknown value
        self.assertEqual(_shown(ConversionSettingsForm({"download_rate_limit": 1000.0})), (1000, 1080, 60))

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


if __name__ == "__main__":
    unittest.main()
