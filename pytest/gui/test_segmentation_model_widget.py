"""Round-trip tests for the segmentation model GUI setting.

ConversionSettingsForm widget -> collect_config() -> UltraSingerRunner.build_args()
-> UltraSinger.py CLI option parsing (init_settings) -> Settings field.
"""

import os
import tempfile
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


class TestSegmentationModelSetting(unittest.TestCase):

    def test_default_is_empty(self):
        self.assertEqual(_DEFAULTS["segmentation_model"], "")
        form = ConversionSettingsForm({})
        self.assertEqual(form._segmentation_model.text(), "")
        self.assertEqual(form.collect_config()["segmentation_model"], "")

    def test_no_flag_when_empty(self):
        args = UltraSingerRunner().build_args({"segmentation_model": ""}, "test.mp3")
        self.assertNotIn("--segmentation_model", args)

    def test_full_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "segmentation.pt")
            open(path, "wb").close()
            form = ConversionSettingsForm({"segmentation_model": path})
            self.assertEqual(form._segmentation_model.text(), path)
            config = form.collect_config()
            self.assertEqual(config["segmentation_model"], path)
            args = UltraSingerRunner().build_args(config, "test.mp3")
            i = args.index("--segmentation_model")
            self.assertEqual(args[i + 1], path)
            settings = init_settings(args[args.index("-i"):] if "-i" in args else ["-i", "test.mp3"] + args)
            self.assertEqual(settings.segmentation_model, path)

    def test_whitespace_is_trimmed(self):
        form = ConversionSettingsForm({"segmentation_model": "  "})
        self.assertEqual(form.collect_config()["segmentation_model"], "")
