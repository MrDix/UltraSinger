"""Round-trip tests for the "Refine GAP" GUI setting.

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


class TestRefineGapSetting(unittest.TestCase):

    def test_default_on_and_no_flag(self):
        self.assertTrue(_DEFAULTS["refine_gap"])
        config = ConversionSettingsForm({"refine_from_vocal": True}).collect_config()
        self.assertTrue(config["refine_gap"])
        args = UltraSingerRunner().build_args(config, "test.mp3")
        self.assertIn("--refine_from_vocal", args)
        self.assertNotIn("--disable_refine_gap", args)
        self.assertTrue(init_settings(_cli(args)).refine_gap)

    def test_off_round_trip(self):
        form = ConversionSettingsForm({"refine_from_vocal": True, "refine_gap": False})
        self.assertFalse(form._refine_gap.isChecked())
        args = UltraSingerRunner().build_args(form.collect_config(), "test.mp3")
        self.assertIn("--disable_refine_gap", args)
        self.assertFalse(init_settings(_cli(args)).refine_gap)
        init_settings(["-i", "test.mp3"])  # reset the shared settings

    def test_follows_the_refinement_switch(self):
        form = ConversionSettingsForm({"refine_from_vocal": False})
        self.assertFalse(form._refine_gap.isEnabled())
        form._refine_from_vocal.setChecked(True)
        self.assertTrue(form._refine_gap.isEnabled())
