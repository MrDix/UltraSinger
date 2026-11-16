"""Round-trip tests for the freestyle detection GUI switch.

Covers the full chain: ConversionSettingsForm widget -> collect_config()
-> UltraSingerRunner.build_args() -> UltraSinger.py CLI option parsing
(init_settings) -> Settings field.
"""

import os
import unittest

import pytest

# Skip the whole module (instead of failing collection) when PySide6 or its
# native Qt libraries are unavailable, e.g. on a runner without the gui extra.
pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.gui.config import _DEFAULTS
from src.gui.settings_tab import ConversionSettingsForm
from src.gui.ultrasinger_runner import UltraSingerRunner
from src.UltraSinger import init_settings

_app = QApplication.instance() or QApplication([])


class TestFreestyleDetectionWidget(unittest.TestCase):
    def test_enabled_by_default(self):
        self.assertTrue(_DEFAULTS["freestyle_detection"])
        form = ConversionSettingsForm({})
        self.assertTrue(form._freestyle_detection.isChecked())
        self.assertTrue(form.collect_config()["freestyle_detection"])

    def test_the_saved_choice_is_shown_and_collected(self):
        form = ConversionSettingsForm({"freestyle_detection": False})
        self.assertFalse(form._freestyle_detection.isChecked())
        self.assertFalse(form.collect_config()["freestyle_detection"])

    def test_settings_of_the_earlier_detection_do_not_turn_it_off(self):
        # A settings file saved before: its switch (off by default then) and
        # thresholds belonged to another detection
        form = ConversionSettingsForm({"detect_growl": False, "freestyle_harmonicity": 0.2})
        self.assertTrue(form._freestyle_detection.isChecked())
        config = form.collect_config()
        self.assertTrue(config["freestyle_detection"])
        self.assertNotIn("freestyle_harmonicity", config)


class TestBuildArgsAndCli(unittest.TestCase):
    def setUp(self):
        self.runner = UltraSingerRunner()

    def _args(self, **changes):
        config = dict(_DEFAULTS)
        config.update(changes)
        return self.runner.build_args(config, "test.mp3")

    def test_enabled_sends_no_option(self):
        args = self._args()
        self.assertFalse([arg for arg in args if "freestyle" in arg])
        self.assertTrue(init_settings(args).freestyle_detection)

    def test_disabled_sends_the_disable(self):
        args = self._args(freestyle_detection=False)
        self.assertIn("--disable_freestyle_detection", args)
        self.assertFalse(init_settings(args).freestyle_detection)

    def test_a_missing_setting_counts_as_enabled(self):
        config = dict(_DEFAULTS)
        del config["freestyle_detection"]
        self.assertNotIn("--disable_freestyle_detection", self.runner.build_args(config, "test.mp3"))


if __name__ == "__main__":
    unittest.main()
