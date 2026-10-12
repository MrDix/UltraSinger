"""Round-trip tests for the "Transcriber" GUI setting.

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


class TestTranscriberSetting(unittest.TestCase):

    def test_default_auto(self):
        self.assertEqual(_DEFAULTS["transcriber"], "auto")
        config = ConversionSettingsForm({}).collect_config()
        self.assertEqual(config["transcriber"], "auto")
        args = UltraSingerRunner().build_args(config, "test.mp3")
        self.assertEqual(args[args.index("--transcriber") + 1], "auto")
        # the Whisper model the GUI always passes does not turn "auto" into Whisper
        self.assertIn("--whisper", args)
        self.assertEqual(init_settings(_cli(args)).transcriber, "auto")

    def test_round_trip(self):
        for value in ("qwen3", "whisper"):
            with self.subTest(value=value):
                form = ConversionSettingsForm({"transcriber": value})
                args = UltraSingerRunner().build_args(form.collect_config(), "test.mp3")
                self.assertEqual(init_settings(_cli(args)).transcriber, value)
        init_settings(["-i", "test.mp3"])  # reset the shared settings

    def test_unknown_config_value_shows_the_default(self):
        form = ConversionSettingsForm({"transcriber": "bogus"})
        self.assertEqual(form.collect_config()["transcriber"], "auto")


if __name__ == "__main__":
    unittest.main()
