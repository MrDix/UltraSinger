"""Round-trip tests for the freestyle detection threshold GUI widgets.

Covers the full chain: ConversionSettingsForm widgets -> collect_config()
-> UltraSingerRunner.build_args() -> UltraSinger.py CLI option parsing
(init_settings) -> Settings fields.

The internal Settings fields are named ``growl_*`` (not user-facing), while
the CLI flags and GUI config keys use the user-facing ``freestyle_*`` /
``--freestyle_*`` naming. This file exercises that mapping end to end.
"""

import os
import unittest

import pytest

# Skip the whole module (instead of failing collection) when PySide6 or its
# native Qt libraries are unavailable, e.g. on a runner without the gui extra.
pytest.importorskip("PySide6.QtWidgets")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.gui.config import _DEFAULTS
from src.gui.settings_tab import ConversionSettingsForm
from src.gui.ultrasinger_runner import UltraSingerRunner
from src.Settings import Settings
from src.UltraSinger import init_settings

_app = QApplication.instance() or QApplication([])


class TestFreestyleThresholdWidgetsDefaults(unittest.TestCase):
    """Widgets should reflect config values (or defaults when absent)."""

    def test_widgets_default_to_config_defaults(self):
        form = ConversionSettingsForm({})
        self.assertAlmostEqual(form._freestyle_harmonicity.value(),
                                _DEFAULTS["freestyle_harmonicity"])
        self.assertAlmostEqual(form._freestyle_energy.value(),
                                _DEFAULTS["freestyle_energy"])
        self.assertAlmostEqual(form._freestyle_confidence.value(),
                                _DEFAULTS["freestyle_confidence"])
        self.assertAlmostEqual(form._freestyle_pitch_stdev.value(),
                                _DEFAULTS["freestyle_pitch_stdev"])
        self.assertAlmostEqual(form._freestyle_spectral_flatness.value(),
                                _DEFAULTS["freestyle_spectral_flatness"])
        self.assertEqual(form._freestyle_use_spectral.isChecked(),
                          _DEFAULTS["freestyle_use_spectral"])

    def test_threshold_widgets_disabled_when_detection_off(self):
        form = ConversionSettingsForm({"detect_growl": False})
        self.assertFalse(form._detect_growl.isChecked())
        self.assertFalse(form._freestyle_harmonicity.isEnabled())
        self.assertFalse(form._freestyle_energy.isEnabled())
        self.assertFalse(form._freestyle_confidence.isEnabled())
        self.assertFalse(form._freestyle_pitch_stdev.isEnabled())
        self.assertFalse(form._freestyle_spectral_flatness.isEnabled())
        self.assertFalse(form._freestyle_use_spectral.isEnabled())

    def test_threshold_widgets_enabled_when_detection_on(self):
        form = ConversionSettingsForm({"detect_growl": True})
        self.assertTrue(form._detect_growl.isChecked())
        self.assertTrue(form._freestyle_harmonicity.isEnabled())
        self.assertTrue(form._freestyle_energy.isEnabled())
        self.assertTrue(form._freestyle_confidence.isEnabled())
        self.assertTrue(form._freestyle_pitch_stdev.isEnabled())
        self.assertTrue(form._freestyle_spectral_flatness.isEnabled())
        self.assertTrue(form._freestyle_use_spectral.isEnabled())

    def test_toggling_main_switch_live_updates_enabled_state(self):
        form = ConversionSettingsForm({"detect_growl": False})
        self.assertFalse(form._freestyle_harmonicity.isEnabled())
        form._detect_growl.setChecked(True)
        form._detect_growl.toggled_signal.emit(True)
        self.assertTrue(form._freestyle_harmonicity.isEnabled())


class TestCollectConfig(unittest.TestCase):
    """collect_config() must surface all six new fields."""

    def test_collect_config_includes_freestyle_fields(self):
        form = ConversionSettingsForm({"detect_growl": True})
        form._freestyle_harmonicity.setValue(0.55)
        form._freestyle_energy.setValue(0.02)
        form._freestyle_confidence.setValue(0.45)
        form._freestyle_pitch_stdev.setValue(6.5)
        form._freestyle_spectral_flatness.setValue(0.33)
        form._freestyle_use_spectral.setChecked(False)

        config = form.collect_config()

        self.assertTrue(config["detect_growl"])
        self.assertAlmostEqual(config["freestyle_harmonicity"], 0.55)
        self.assertAlmostEqual(config["freestyle_energy"], 0.02)
        self.assertAlmostEqual(config["freestyle_confidence"], 0.45)
        self.assertAlmostEqual(config["freestyle_pitch_stdev"], 6.5)
        self.assertAlmostEqual(config["freestyle_spectral_flatness"], 0.33)
        self.assertFalse(config["freestyle_use_spectral"])


class TestBuildArgs(unittest.TestCase):
    """UltraSingerRunner.build_args() must emit the correct --freestyle_* flags."""

    def setUp(self):
        self.runner = UltraSingerRunner()

    def test_no_freestyle_flags_when_detection_disabled(self):
        config = dict(_DEFAULTS)
        config["detect_growl"] = False
        config["freestyle_harmonicity"] = 0.9  # should be ignored entirely
        args = self.runner.build_args(config, "test.mp3")
        self.assertNotIn("--detect_freestyle", args)
        for flag in ("--freestyle_harmonicity", "--freestyle_energy",
                     "--freestyle_confidence", "--freestyle_pitch_stdev",
                     "--freestyle_spectral_flatness", "--no_freestyle_spectral"):
            self.assertNotIn(flag, args)

    def test_only_detect_flag_when_thresholds_are_default(self):
        config = dict(_DEFAULTS)
        config["detect_growl"] = True
        args = self.runner.build_args(config, "test.mp3")
        self.assertIn("--detect_freestyle", args)
        for flag in ("--freestyle_harmonicity", "--freestyle_energy",
                     "--freestyle_confidence", "--freestyle_pitch_stdev",
                     "--freestyle_spectral_flatness", "--no_freestyle_spectral"):
            self.assertNotIn(flag, args)

    def test_nondefault_thresholds_are_forwarded(self):
        config = dict(_DEFAULTS)
        config.update({
            "detect_growl": True,
            "freestyle_harmonicity": 0.55,
            "freestyle_energy": 0.02,
            "freestyle_confidence": 0.45,
            "freestyle_pitch_stdev": 6.5,
            "freestyle_spectral_flatness": 0.33,
            "freestyle_use_spectral": False,
        })
        args = self.runner.build_args(config, "test.mp3")

        self.assertIn("--detect_freestyle", args)
        self._assert_flag_value(args, "--freestyle_harmonicity", "0.55")
        self._assert_flag_value(args, "--freestyle_energy", "0.02")
        self._assert_flag_value(args, "--freestyle_confidence", "0.45")
        self._assert_flag_value(args, "--freestyle_pitch_stdev", "6.5")
        self._assert_flag_value(args, "--freestyle_spectral_flatness", "0.33")
        self.assertIn("--no_freestyle_spectral", args)

    def _assert_flag_value(self, args, flag, expected_value):
        self.assertIn(flag, args)
        idx = args.index(flag)
        self.assertEqual(args[idx + 1], expected_value)


class TestFullRoundTrip(unittest.TestCase):
    """GUI widget -> collect_config -> build_args -> CLI parsing -> Settings."""

    def test_gui_values_survive_the_full_round_trip(self):
        form = ConversionSettingsForm({"detect_growl": True})
        form._freestyle_harmonicity.setValue(0.55)
        form._freestyle_energy.setValue(0.02)
        form._freestyle_confidence.setValue(0.45)
        form._freestyle_pitch_stdev.setValue(6.5)
        form._freestyle_spectral_flatness.setValue(0.33)
        form._freestyle_use_spectral.setChecked(False)

        config = form.collect_config()
        args = UltraSingerRunner().build_args(config, "test.mp3")

        settings = init_settings(args)

        self.assertTrue(settings.detect_growl)
        self.assertAlmostEqual(settings.growl_harmonicity_threshold, 0.55)
        self.assertAlmostEqual(settings.growl_energy_threshold, 0.02)
        self.assertAlmostEqual(settings.growl_confidence_threshold, 0.45)
        self.assertAlmostEqual(settings.growl_pitch_stdev_threshold, 6.5)
        self.assertAlmostEqual(settings.growl_spectral_flatness_threshold, 0.33)
        self.assertFalse(settings.growl_use_spectral)

    def test_default_thresholds_round_trip_to_settings_defaults(self):
        form = ConversionSettingsForm({"detect_growl": True})
        # Leave all threshold widgets at their default values.

        config = form.collect_config()
        args = UltraSingerRunner().build_args(config, "test.mp3")
        settings = init_settings(args)

        fresh = Settings()
        self.assertTrue(settings.detect_growl)
        self.assertAlmostEqual(settings.growl_harmonicity_threshold,
                                fresh.growl_harmonicity_threshold)
        self.assertAlmostEqual(settings.growl_energy_threshold,
                                fresh.growl_energy_threshold)
        self.assertAlmostEqual(settings.growl_confidence_threshold,
                                fresh.growl_confidence_threshold)
        self.assertAlmostEqual(settings.growl_pitch_stdev_threshold,
                                fresh.growl_pitch_stdev_threshold)
        self.assertAlmostEqual(settings.growl_spectral_flatness_threshold,
                                fresh.growl_spectral_flatness_threshold)
        self.assertEqual(settings.growl_use_spectral, fresh.growl_use_spectral)


if __name__ == "__main__":
    unittest.main()
