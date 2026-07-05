"""Round-trip tests for the remote speech-to-text rate-limit retry GUI widgets.

Covers the full chain: ConversionSettingsForm widgets -> collect_config()
-> UltraSingerRunner.build_args() -> UltraSinger.py CLI option parsing
(init_settings) -> Settings fields.

Mirrors ``pytest/gui/test_freestyle_threshold_widgets.py`` in structure, but
for the three new remote-STT retry fields, which themselves mirror the
existing LLM lyric correction retry fields (``llm_retry_*`` / ``--llm_*``).
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


class TestRemoteSttRetryWidgetsDefaults(unittest.TestCase):
    """Widgets should reflect config values (or defaults when absent)."""

    def test_widgets_default_to_config_defaults(self):
        form = ConversionSettingsForm({})
        self.assertEqual(form._remote_stt_retry.isChecked(),
                          _DEFAULTS["remote_stt_retry_on_rate_limit"])
        self.assertEqual(form._remote_stt_retry_wait.value(),
                          _DEFAULTS["remote_stt_retry_wait"])
        self.assertEqual(form._remote_stt_retry_max.value(),
                          _DEFAULTS["remote_stt_retry_max"])

    def test_retry_widgets_disabled_when_remote_stt_off(self):
        form = ConversionSettingsForm({"remote_stt": False})
        self.assertFalse(form._remote_stt.isChecked())
        self.assertFalse(form._remote_stt_retry.isEnabled())
        self.assertFalse(form._remote_stt_retry_wait.isEnabled())
        self.assertFalse(form._remote_stt_retry_max.isEnabled())

    def test_retry_widgets_enabled_when_remote_stt_on(self):
        form = ConversionSettingsForm({"remote_stt": True})
        self.assertTrue(form._remote_stt.isChecked())
        self.assertTrue(form._remote_stt_retry.isEnabled())
        self.assertTrue(form._remote_stt_retry_wait.isEnabled())
        self.assertTrue(form._remote_stt_retry_max.isEnabled())

    def test_toggling_main_switch_live_updates_enabled_state(self):
        form = ConversionSettingsForm({"remote_stt": False})
        self.assertFalse(form._remote_stt_retry_wait.isEnabled())
        form._remote_stt.setChecked(True)
        form._remote_stt.toggled_signal.emit(True)
        self.assertTrue(form._remote_stt_retry_wait.isEnabled())


class TestCollectConfig(unittest.TestCase):
    """collect_config() must surface all three new retry fields."""

    def test_collect_config_includes_retry_fields(self):
        form = ConversionSettingsForm({"remote_stt": True})
        form._remote_stt_retry.setChecked(False)
        form._remote_stt_retry_wait.setValue(45)
        form._remote_stt_retry_max.setValue(7)

        config = form.collect_config()

        self.assertFalse(config["remote_stt_retry_on_rate_limit"])
        self.assertEqual(config["remote_stt_retry_wait"], 45)
        self.assertEqual(config["remote_stt_retry_max"], 7)


class TestBuildArgs(unittest.TestCase):
    """UltraSingerRunner.build_args() must emit the correct --remote_stt_* flags."""

    def setUp(self):
        self.runner = UltraSingerRunner()

    def test_no_retry_flags_when_remote_stt_disabled(self):
        config = dict(_DEFAULTS)
        config["remote_stt"] = False
        config["remote_stt_retry_on_rate_limit"] = False  # should be ignored entirely
        args = self.runner.build_args(config, "test.mp3")
        self.assertNotIn("--remote_stt", args)
        for flag in ("--remote_stt_no_retry", "--remote_stt_retry_wait", "--remote_stt_retry_max"):
            self.assertNotIn(flag, args)

    def test_no_retry_flags_when_retry_settings_are_default(self):
        config = dict(_DEFAULTS)
        config["remote_stt"] = True
        args = self.runner.build_args(config, "test.mp3")
        self.assertIn("--remote_stt", args)
        for flag in ("--remote_stt_no_retry", "--remote_stt_retry_wait", "--remote_stt_retry_max"):
            self.assertNotIn(flag, args)

    def test_no_retry_flag_emitted_when_retry_disabled(self):
        config = dict(_DEFAULTS)
        config.update({
            "remote_stt": True,
            "remote_stt_retry_on_rate_limit": False,
        })
        args = self.runner.build_args(config, "test.mp3")
        self.assertIn("--remote_stt_no_retry", args)
        # When retry is disabled, wait/max flags are irrelevant and omitted.
        self.assertNotIn("--remote_stt_retry_wait", args)
        self.assertNotIn("--remote_stt_retry_max", args)

    def test_nondefault_retry_settings_are_forwarded(self):
        config = dict(_DEFAULTS)
        config.update({
            "remote_stt": True,
            "remote_stt_retry_on_rate_limit": True,
            "remote_stt_retry_wait": 45,
            "remote_stt_retry_max": 7,
        })
        args = self.runner.build_args(config, "test.mp3")

        self.assertNotIn("--remote_stt_no_retry", args)
        self._assert_flag_value(args, "--remote_stt_retry_wait", "45")
        self._assert_flag_value(args, "--remote_stt_retry_max", "7")

    def _assert_flag_value(self, args, flag, expected_value):
        self.assertIn(flag, args)
        idx = args.index(flag)
        self.assertEqual(args[idx + 1], expected_value)


class TestFullRoundTrip(unittest.TestCase):
    """GUI widget -> collect_config -> build_args -> CLI parsing -> Settings."""

    def test_gui_values_survive_the_full_round_trip(self):
        form = ConversionSettingsForm({"remote_stt": True})
        form._remote_stt_retry.setChecked(True)
        form._remote_stt_retry_wait.setValue(45)
        form._remote_stt_retry_max.setValue(7)

        config = form.collect_config()
        args = UltraSingerRunner().build_args(config, "test.mp3")

        settings = init_settings(args)

        self.assertTrue(settings.remote_stt)
        self.assertTrue(settings.remote_stt_retry_on_rate_limit)
        self.assertEqual(settings.remote_stt_retry_wait, 45)
        self.assertEqual(settings.remote_stt_retry_max, 7)

    def test_retry_disabled_round_trips_to_settings(self):
        form = ConversionSettingsForm({"remote_stt": True})
        form._remote_stt_retry.setChecked(False)

        config = form.collect_config()
        args = UltraSingerRunner().build_args(config, "test.mp3")
        settings = init_settings(args)

        self.assertTrue(settings.remote_stt)
        self.assertFalse(settings.remote_stt_retry_on_rate_limit)

    def test_default_retry_settings_round_trip_to_settings_defaults(self):
        form = ConversionSettingsForm({"remote_stt": True})
        # Leave all retry widgets at their default values.

        config = form.collect_config()
        args = UltraSingerRunner().build_args(config, "test.mp3")
        settings = init_settings(args)

        fresh = Settings()
        self.assertTrue(settings.remote_stt)
        self.assertEqual(settings.remote_stt_retry_on_rate_limit,
                          fresh.remote_stt_retry_on_rate_limit)
        self.assertEqual(settings.remote_stt_retry_wait, fresh.remote_stt_retry_wait)
        self.assertEqual(settings.remote_stt_retry_max, fresh.remote_stt_retry_max)


if __name__ == "__main__":
    unittest.main()
