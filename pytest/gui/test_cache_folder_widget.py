"""Round-trip tests for the "Cache Folder" GUI setting.

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


class TestCacheFolderSetting(unittest.TestCase):

    def test_default_is_automatic_and_no_flag(self):
        self.assertEqual(_DEFAULTS["cache_folder"], "")
        config = ConversionSettingsForm({}).collect_config()
        self.assertEqual(config["cache_folder"], "")
        args = UltraSingerRunner().build_args(config, "test.mp3")
        self.assertNotIn("--cache_folder", args)
        self.assertIsNone(init_settings(_cli(args)).cache_folder)

    def test_folder_round_trip(self):
        form = ConversionSettingsForm({"cache_folder": " D:/UltraSinger cache "})
        args = UltraSingerRunner().build_args(form.collect_config(), "test.mp3")
        self.assertEqual(args[args.index("--cache_folder") + 1], "D:/UltraSinger cache")
        self.assertEqual(init_settings(_cli(args)).cache_folder, "D:/UltraSinger cache")
        init_settings(["-i", "test.mp3"])  # reset the shared settings
