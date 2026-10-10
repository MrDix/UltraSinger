"""Round-trip tests for the "Octave Consistency" GUI setting.

ConversionSettingsForm widget -> collect_config() -> UltraSingerRunner.build_args()
-> UltraSinger.py CLI option parsing (init_settings) -> Settings field.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import src.gui.config as config_module
from src.gui.config import _DEFAULTS, octave_consistency_choice
from src.gui.settings_tab import ConversionSettingsForm
from src.gui.ultrasinger_runner import UltraSingerRunner
from src.UltraSinger import init_settings

_app = QApplication.instance() or QApplication([])


def _cli(args: list[str]) -> list[str]:
    return args[args.index("-i"):] if "-i" in args else ["-i", "test.mp3"] + args


class TestOctaveConsistencySetting(unittest.TestCase):

    def tearDown(self):
        init_settings(["-i", "test.mp3"])  # reset the shared settings

    def _round_trip(self, config: dict):
        config = ConversionSettingsForm(config).collect_config()
        args = UltraSingerRunner().build_args(config, "test.mp3")
        return config["octave_consistency"], args, init_settings(_cli(args)).octave_consistency

    def test_default_is_the_model_notes_without_a_flag(self):
        self.assertEqual(_DEFAULTS["octave_consistency"], "model")
        value, args, setting = self._round_trip({})
        self.assertEqual(value, "model")
        self.assertNotIn("--octave_consistency", args)
        self.assertNotIn("--disable_octave_consistency", args)
        self.assertIsNone(setting)

    def test_all_notes_round_trip(self):
        value, args, setting = self._round_trip({"octave_consistency": "all"})
        self.assertEqual(value, "all")
        self.assertIn("--octave_consistency", args)
        self.assertIs(setting, True)

    def test_off_round_trip(self):
        value, args, setting = self._round_trip({"octave_consistency": "off"})
        self.assertEqual(value, "off")
        self.assertIn("--disable_octave_consistency", args)
        self.assertIs(setting, False)

    def test_switch_of_older_configs(self):
        """On applied the pass to all notes; off was the default."""
        self.assertEqual(octave_consistency_choice(True), "all")
        self.assertEqual(octave_consistency_choice(False), "model")
        self.assertEqual(octave_consistency_choice(None), "model")
        self.assertEqual(self._round_trip({"octave_consistency": True})[0], "all")
        self.assertEqual(self._round_trip({"octave_consistency": False})[0], "model")
        self.assertIn("--octave_consistency", UltraSingerRunner().build_args({"octave_consistency": True}, "x.mp3"))

    def test_load_config_migrates_the_switch(self):
        for stored, expected in ((True, "all"), (False, "model"), ("off", "off")):
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "config.json"
                path.write_text(json.dumps({"octave_consistency": stored}), encoding="utf-8")
                with patch.object(config_module, "_CONFIG_FILE", path), \
                        patch("src.gui.secrets.get_secret", return_value=""), \
                        patch("src.gui.secrets.store_secret", return_value=False):
                    self.assertEqual(config_module.load_config()["octave_consistency"], expected)
