"""Round-trip tests for the "Pitch Context" settings and the places a setting has to be in.

ConversionSettingsForm widgets -> collect_config() -> UltraSingerRunner.build_args()
-> UltraSinger.py CLI option parsing (init_settings) -> Settings fields; then the
help text, the README, the settings info file and the queue file lists.
"""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton

import src.UltraSinger as us
from modules.Pitcher.pitch_context import STRENGTH_CHOICES, PitchContextResult
from src.gui import queue_file
from src.gui.config import _DEFAULTS
from src.gui.settings_tab import ConversionSettingsForm
from src.gui.ultrasinger_runner import UltraSingerRunner
from src.Settings import Settings
from src.UltraSinger import arg_options, init_settings

_app = QApplication.instance() or QApplication([])

REPO = Path(__file__).resolve().parents[2]


def _cli(args: list[str]) -> list[str]:
    return args[args.index("-i"):] if "-i" in args else ["-i", "test.mp3"] + args


class TestPitchContextSetting(unittest.TestCase):

    def tearDown(self):
        init_settings(["-i", "test.mp3"])  # reset the shared settings

    def _round_trip(self, config: dict):
        config = ConversionSettingsForm(config).collect_config()
        args = UltraSingerRunner().build_args(config, "test.mp3")
        settings = init_settings(_cli(args))
        return config, args, (settings.pitch_context, settings.pitch_context_model)

    def test_off_by_default_without_arguments(self):
        self.assertEqual((_DEFAULTS["pitch_context"], _DEFAULTS["pitch_context_model"]), ("off", ""))
        self.assertEqual((Settings().pitch_context, Settings().pitch_context_model), ("off", None))
        config, args, settings = self._round_trip({})
        self.assertEqual((config["pitch_context"], config["pitch_context_model"]), ("off", ""))
        self.assertNotIn("--pitch_context", args)
        self.assertNotIn("--pitch_context_model", args)
        self.assertEqual(settings, ("off", None))

    def test_strengths_round_trip_with_the_model_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "pitch context.npz")
            open(path, "wb").close()
            for strength in ("balanced", "strong"):
                with self.subTest(strength=strength):
                    form = ConversionSettingsForm({"pitch_context": strength, "pitch_context_model": path})
                    self.assertEqual(form._pitch_context.currentData(), strength)
                    self.assertEqual(form._pitch_context_model.text(), path)
                    config, args, settings = self._round_trip({"pitch_context": strength,
                                                               "pitch_context_model": path})
                    self.assertEqual(args[args.index("--pitch_context") + 1], strength)
                    self.assertEqual(args[args.index("--pitch_context_model") + 1], path)
                    self.assertEqual(settings, (strength, path))

    def test_the_choices_are_those_of_the_conversion(self):
        form = ConversionSettingsForm({})
        combo = form._pitch_context
        self.assertEqual(tuple(combo.itemData(i) for i in range(combo.count())), STRENGTH_CHOICES)
        self.assertEqual([combo.itemText(i) for i in range(combo.count())], ["Off", "Balanced", "Strong"])

    def test_a_strength_without_a_model_file_reaches_the_conversion(self):
        """It then says that the model is missing and goes on without the step."""
        config, args, settings = self._round_trip({"pitch_context": "strong"})
        self.assertEqual(args[args.index("--pitch_context") + 1], "strong")
        self.assertNotIn("--pitch_context_model", args)
        self.assertEqual(settings, ("strong", None))

    def test_a_model_file_alone_changes_nothing(self):
        config, args, settings = self._round_trip({"pitch_context_model": "D:/Models/pitch_context.npz"})
        self.assertEqual(config["pitch_context_model"], "D:/Models/pitch_context.npz")
        self.assertNotIn("--pitch_context", args)
        self.assertNotIn("--pitch_context_model", args)
        self.assertEqual(settings, ("off", None))

    def test_whitespace_is_trimmed(self):
        form = ConversionSettingsForm({"pitch_context": "balanced", "pitch_context_model": "  model.npz  "})
        config = form.collect_config()
        self.assertEqual(config["pitch_context_model"], "model.npz")
        args = UltraSingerRunner().build_args({"pitch_context": "balanced", "pitch_context_model": "  m.npz "},
                                              "test.mp3")
        self.assertEqual(args[args.index("--pitch_context_model") + 1], "m.npz")

    def test_a_value_the_setting_does_not_offer_is_off(self):
        # e.g. from a hand-edited settings or queue file: the CLI would stop instead of converting
        for value in ("loud", "", None, 3, True):
            with self.subTest(value=value):
                self.assertEqual(ConversionSettingsForm({"pitch_context": value}).collect_config()["pitch_context"],
                                 "off")
                with self.assertLogs("src.gui.ultrasinger_runner", level="WARNING") if value not in ("", None) \
                        else contextlib.nullcontext():
                    args = UltraSingerRunner().build_args({"pitch_context": value, "pitch_context_model": "m.npz"},
                                                          "test.mp3")
                self.assertNotIn("--pitch_context", args)
                self.assertNotIn("--pitch_context_model", args)

    def test_reset_buttons(self):
        form = ConversionSettingsForm({"pitch_context": "strong", "pitch_context_model": "m.npz"})
        form._set_pitch_context(_DEFAULTS["pitch_context"])
        self.assertEqual(form.collect_config()["pitch_context"], "off")

    def test_in_the_settings_of_one_song_too(self):
        form = ConversionSettingsForm({"pitch_context": "balanced", "pitch_context_model": "m.npz"}, per_song=True)
        config = form.collect_config()
        self.assertEqual((config["pitch_context"], config["pitch_context_model"]), ("balanced", "m.npz"))

    def test_tooltips_and_browse_button(self):
        form = ConversionSettingsForm({})
        self.assertIn("up to two semitones", form._pitch_context.toolTip())
        self.assertIn("Balanced", form._pitch_context.toolTip())
        self.assertIn("tools/train_pitch_context.py", form._pitch_context_model.toolTip())
        browse = [b for b in form._pitch_context_model.parent().findChildren(QPushButton) if b.text() == "Browse"]
        self.assertEqual(len(browse), 1)
        self.assertIn(".npz", browse[0].toolTip())
        with patch("src.gui.settings_tab.QFileDialog.getOpenFileName",
                   return_value=("D:/Models/pc.npz", "")) as dialog:
            browse[0].click()
        self.assertIn("*.npz", dialog.call_args.args[3])
        self.assertEqual(form._pitch_context_model.text(), "D:/Models/pc.npz")


class TestPitchContextCommandLine(unittest.TestCase):

    def tearDown(self):
        init_settings(["-i", "test.mp3"])  # reset the shared settings

    def test_options(self):
        long_options, _ = arg_options()
        self.assertIn("pitch_context=", long_options)
        self.assertIn("pitch_context_model=", long_options)
        settings = init_settings(["-i", "test.mp3", "--pitch_context", "Strong", "--pitch_context_model", "m.npz"])
        self.assertEqual((settings.pitch_context, settings.pitch_context_model), ("strong", "m.npz"))
        settings = init_settings(["-i", "test.mp3", "--pitch_context", "off", "--pitch_context_model", ""])
        self.assertEqual((settings.pitch_context, settings.pitch_context_model), ("off", None))
        # no option of an earlier run carries over
        settings = init_settings(["-i", "test.mp3"])
        self.assertEqual((settings.pitch_context, settings.pitch_context_model), ("off", None))

    def test_an_unknown_strength_is_an_error(self):
        out = io.StringIO()
        with self.assertRaises(SystemExit) as stop, contextlib.redirect_stdout(out):
            init_settings(["-i", "test.mp3", "--pitch_context", "loud"])
        self.assertEqual(stop.exception.code, 1)
        self.assertIn("Unknown pitch context strength 'loud'", out.getvalue())
        self.assertIn("off, balanced, strong", out.getvalue())

    def test_help_text(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            us.print_help()
        text = out.getvalue()
        self.assertIn("--pitch_context         off|balanced|strong, default off.", text)
        self.assertIn("--pitch_context_model   Path to a trained pitch context model (.npz).", text)

    def test_readme_and_docs(self):
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        self.assertIn("--pitch_context         off|balanced|strong, default off.", readme)
        self.assertIn("--pitch_context_model   Path to a trained pitch context model (.npz).", readme)
        self.assertIn("#### Note Pitches from Their Musical Context (`--pitch_context`)", readme)
        self.assertIn("(#note-pitches-from-their-musical-context---pitch_context)", readme)
        self.assertIn("docs/pitch-context-model.md", readme)
        doc = (REPO / "docs" / "pitch-context-model.md").read_text(encoding="utf-8")
        self.assertIn("tools/train_pitch_context.py", doc)
        self.assertIn("--pitch_context_model", doc)


class TestPitchContextInfoLine(unittest.TestCase):
    """The settings info file says whether the step ran and what it did."""

    def _info(self, strength, model=None, **kwargs):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(us.settings, "pitch_context", strength), \
                patch.object(us.settings, "pitch_context_model", model), \
                contextlib.redirect_stdout(io.StringIO()):
            us._write_settings_info_file(folder, None, None, **kwargs)
            with open(os.path.join(folder, "ultrasinger_parameter.info"), encoding="utf-8") as f:
                return f.read()

    def test_off(self):
        self.assertIn("  Pitch context:            off\n", self._info("off", "D:/Models/pc.npz"))

    def test_applied(self):
        info = self._info("balanced", "D:/Models/pc.npz", model_segmentation_used=True,
                          pitch_context_result=PitchContextResult(True, notes=498, moved=37))
        self.assertIn("  Pitch context:            balanced (pc.npz): 37 of 498 note pitches moved\n", info)
        info = self._info("balanced", "D:/Models/pc.npz", model_segmentation_used=True,
                          pitch_context_result=PitchContextResult(True, notes=498, moved=25, refined=41))
        self.assertIn("  Pitch context:            balanced (pc.npz): 25 of 498 note pitches moved "
                      "(41 set by the pitch refinement left as they are)\n", info)

    def test_not_applied(self):
        result = PitchContextResult(False, reason="the notes are not from a segmentation model")
        info = self._info("strong", "D:/Models/pc.npz", pitch_context_result=result)
        self.assertIn("  Pitch context:            strong (pc.npz): not applied (the notes are not from a "
                      "segmentation model)\n", info)
        info = self._info("strong", None,
                          pitch_context_result=PitchContextResult(False, reason="no model file is set"))
        self.assertIn("  Pitch context:            strong (no model file): not applied (no model file is set)\n",
                      info)


class TestPitchContextInQueueFiles(unittest.TestCase):

    def test_both_are_settings_of_one_song(self):
        self.assertIn("pitch_context", queue_file.PER_SONG_SETTINGS)
        self.assertIn("pitch_context_model", queue_file.PER_SONG_SETTINGS)

    def test_an_imported_file_sets_the_strength_but_no_model_file(self):
        self.assertEqual(queue_file._checked_setting("pitch_context_model", "C:/x.npz", imported=True)[0], None)
        self.assertEqual(queue_file._checked_setting("pitch_context_model", "C:/x.npz", imported=False),
                         ("C:/x.npz", None))
        for strength in STRENGTH_CHOICES:
            self.assertEqual(queue_file._checked_setting("pitch_context", strength, imported=True),
                             (strength, None))
        value, why = queue_file._checked_setting("pitch_context", "../x", imported=True)
        self.assertIsNone(value)
        self.assertIn("a value the per-song settings do not offer", why)


if __name__ == "__main__":
    unittest.main()
