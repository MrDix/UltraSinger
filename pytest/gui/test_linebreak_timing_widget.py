"""Round-trip tests for the "Line Break Timing" GUI setting.

ConversionSettingsForm widget -> collect_config() -> UltraSingerRunner.build_args()
-> UltraSinger.py CLI option parsing (init_settings) -> Settings field.
"""

import os
import unittest

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.gui.config import _DEFAULTS
from src.gui.settings_tab import ConversionSettingsForm
from src.gui.ultrasinger_runner import UltraSingerRunner
from src.UltraSinger import init_settings

_app = QApplication.instance() or QApplication([])


def _cli(args: list[str]) -> list[str]:
    return args[args.index("-i"):] if "-i" in args else ["-i", "test.mp3"] + args


class TestLinebreakTimingSetting(unittest.TestCase):

    def test_default_yass(self):
        self.assertEqual(_DEFAULTS["linebreak_timing"], "yass")
        config = ConversionSettingsForm({}).collect_config()
        self.assertEqual(config["linebreak_timing"], "yass")
        args = UltraSingerRunner().build_args(config, "test.mp3")
        self.assertEqual(args[args.index("--linebreak_timing") + 1], "yass")
        self.assertEqual(init_settings(_cli(args)).linebreak_timing, "yass")

    def test_round_trip(self):
        for value in ("usdx", "after_last_note"):
            with self.subTest(value=value):
                form = ConversionSettingsForm({"linebreak_timing": value})
                args = UltraSingerRunner().build_args(form.collect_config(), "test.mp3")
                self.assertEqual(init_settings(_cli(args)).linebreak_timing, value)
        init_settings(["-i", "test.mp3"])  # reset the shared settings

    def test_unknown_config_value_shows_the_default(self):
        form = ConversionSettingsForm({"linebreak_timing": "bogus"})
        self.assertEqual(form.collect_config()["linebreak_timing"], "yass")
