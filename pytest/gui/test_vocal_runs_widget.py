"""Round-trip tests for the "Vocal runs, riffs and acrobatics as notes" GUI setting.

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


class TestVocalRunsSetting(unittest.TestCase):

    def test_default_off(self):
        self.assertFalse(_DEFAULTS["vocal_runs_as_notes"])
        config = ConversionSettingsForm({}).collect_config()
        self.assertFalse(config["vocal_runs_as_notes"])
        args = UltraSingerRunner().build_args(config, "test.mp3")
        self.assertNotIn("--vocal_runs_as_notes", args)
        self.assertFalse(init_settings(_cli(args)).vocal_runs_as_notes)

    def test_round_trip_on(self):
        form = ConversionSettingsForm({"vocal_runs_as_notes": True})
        config = form.collect_config()
        self.assertTrue(config["vocal_runs_as_notes"])
        args = UltraSingerRunner().build_args(config, "test.mp3")
        self.assertIn("--vocal_runs_as_notes", args)
        self.assertTrue(init_settings(_cli(args)).vocal_runs_as_notes)
        init_settings(["-i", "test.mp3"])  # reset the shared settings

    def test_threshold_round_trip(self):
        form = ConversionSettingsForm({"vocal_runs_as_notes": True, "vocal_runs_threshold": 0.6})
        config = form.collect_config()
        self.assertEqual(config["vocal_runs_threshold"], 0.6)
        args = UltraSingerRunner().build_args(config, "test.mp3")
        self.assertEqual(args[args.index("--vocal_runs_threshold") + 1], "0.6")
        self.assertEqual(init_settings(_cli(args)).vocal_runs_threshold, 0.6)
        init_settings(["-i", "test.mp3"])  # reset the shared settings

    def test_threshold_only_with_the_switch(self):
        form = ConversionSettingsForm({"vocal_runs_as_notes": False, "vocal_runs_threshold": 0.6})
        self.assertFalse(form._vocal_runs_threshold.isEnabled())
        args = UltraSingerRunner().build_args(form.collect_config(), "test.mp3")
        self.assertNotIn("--vocal_runs_threshold", args)
        form._vocal_runs_as_notes.setChecked(True)
        self.assertTrue(form._vocal_runs_threshold.isEnabled())

    def test_a_queued_threshold_out_of_range_runs_with_the_default(self):
        # A queue file can bring any number; the CLI would stop instead of converting
        for value in (5, 0.1, "x"):
            with self.subTest(value=value):
                args = UltraSingerRunner().build_args(
                    {**_DEFAULTS, "vocal_runs_as_notes": True, "vocal_runs_threshold": value}, "test.mp3")
                self.assertIn("--vocal_runs_as_notes", args)
                self.assertNotIn("--vocal_runs_threshold", args)

    def test_threshold_out_of_range_is_an_error(self):
        for value in ("0.1", "5"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                init_settings(["-i", "test.mp3", "--vocal_runs_threshold", value])
        init_settings(["-i", "test.mp3"])  # reset the shared settings
