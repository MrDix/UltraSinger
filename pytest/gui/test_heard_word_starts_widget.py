"""Round-trip test for the "Heard Word Starts" switch.

Settings form -> collect_config() -> UltraSingerRunner.build_args() -> UltraSinger.py CLI option
parsing (init_settings) -> Settings field.
"""

import os
import unittest

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget

from src.gui.config import _DEFAULTS
from src.gui.settings_tab import ConversionSettingsForm
from src.gui.ultrasinger_runner import UltraSingerRunner
from src.UltraSinger import init_settings

_app = QApplication.instance() or QApplication([])


class TestHeardWordStartsSwitch(unittest.TestCase):

    def test_on_by_default_and_no_option(self):
        self.assertIs(_DEFAULTS["disable_heard_word_starts"], False)
        form = ConversionSettingsForm({})
        self.assertTrue(form._heard_word_starts.isChecked())
        config = form.collect_config()
        self.assertIs(config["disable_heard_word_starts"], False)
        args = UltraSingerRunner().build_args(config, "test.mp3")
        self.assertNotIn("--disable_heard_word_starts", args)
        self.assertTrue(init_settings(args[args.index("-i"):]).heard_word_starts)

    def test_switched_off_reaches_the_conversion(self):
        form = ConversionSettingsForm({})
        form._heard_word_starts.setChecked(False)
        config = form.collect_config()
        self.assertIs(config["disable_heard_word_starts"], True)
        args = UltraSingerRunner().build_args(config, "test.mp3")
        self.assertIn("--disable_heard_word_starts", args)
        self.assertFalse(init_settings(args[args.index("-i"):]).heard_word_starts)
        init_settings(["-i", "test.mp3"])  # reset the shared settings

    def test_the_form_shows_a_saved_setting(self):
        self.assertFalse(ConversionSettingsForm({"disable_heard_word_starts": True})._heard_word_starts.isChecked())
        self.assertTrue(ConversionSettingsForm({"disable_heard_word_starts": False})._heard_word_starts.isChecked())

    def test_the_settings_of_one_song_have_the_switch_too(self):
        form = ConversionSettingsForm({}, per_song=True)
        form._heard_word_starts.setChecked(False)
        self.assertIs(form.collect_config()["disable_heard_word_starts"], True)

    def test_the_tooltip_tells_what_it_does(self):
        form = ConversionSettingsForm({})
        self.assertTrue(any("the transcription heard it" in widget.toolTip() for widget in form.findChildren(QWidget)))


if __name__ == "__main__":
    unittest.main()
