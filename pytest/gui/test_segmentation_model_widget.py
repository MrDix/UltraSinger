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


class TestSegmentationModelRepoSetting(unittest.TestCase):

    def test_defaults_empty(self):
        form = ConversionSettingsForm({})
        config = form.collect_config()
        self.assertEqual(config["segmentation_model_repo"], "")
        self.assertEqual(config["segmentation_model_token"], "")

    def test_repo_and_token_round_trip(self):
        form = ConversionSettingsForm({"segmentation_model_repo": "owner/repo",
                                       "segmentation_model_token": "tok"})
        args = UltraSingerRunner().build_args(form.collect_config(), "test.mp3")
        self.assertEqual(args[args.index("--segmentation_model_repo") + 1], "owner/repo")
        self.assertEqual(args[args.index("--segmentation_model_token") + 1], "tok")
        settings = init_settings(args[args.index("-i"):] if "-i" in args else ["-i", "test.mp3"] + args)
        self.assertEqual(settings.segmentation_model_repo, "owner/repo")

    def test_local_file_wins_over_repo(self):
        args = UltraSingerRunner().build_args({"segmentation_model": "m.pt",
                                               "segmentation_model_repo": "owner/repo",
                                               "segmentation_model_token": "tok"}, "test.mp3")
        self.assertIn("--segmentation_model", args)
        self.assertNotIn("--segmentation_model_repo", args)
        self.assertNotIn("--segmentation_model_token", args)

    def test_token_field_is_masked(self):
        from PySide6.QtWidgets import QLineEdit
        form = ConversionSettingsForm({})
        self.assertEqual(form._segmentation_model_token.echoMode(), QLineEdit.EchoMode.Password)

    def test_token_is_a_secret(self):
        from src.gui.config import _is_secret_key
        self.assertTrue(_is_secret_key("segmentation_model_token"))
        self.assertFalse(_is_secret_key("segmentation_model_repo"))


class TestSecretRedaction(unittest.TestCase):
    def test_secrets_redacted_in_logged_command(self):
        from src.gui.ultrasinger_runner import redact_secrets
        cmd = ["python", "UltraSinger.py", "--segmentation_model_token", "tok",
               "--llm_api_key", "k", "--remote_stt_api_key", "r", "-i", "x.mp3"]
        shown = redact_secrets(cmd)
        self.assertNotIn("tok", shown)
        self.assertNotIn("k", shown)
        self.assertNotIn("r", shown)
        self.assertEqual(shown[-2:], ["-i", "x.mp3"])
        self.assertEqual(cmd[3], "tok")  # original untouched

    def test_trailing_flag_without_value(self):
        from src.gui.ultrasinger_runner import redact_secrets
        self.assertEqual(redact_secrets(["--segmentation_model_token"]), ["--segmentation_model_token"])
