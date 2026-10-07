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
from src.gui.ultrasinger_runner import MODEL_TOKEN_ENV, UltraSingerRunner
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
        config = form.collect_config()
        runner = UltraSingerRunner()
        args = runner.build_args(config, "test.mp3")
        self.assertEqual(args[args.index("--segmentation_model_repo") + 1], "owner/repo")
        settings = init_settings(args[args.index("-i"):] if "-i" in args else ["-i", "test.mp3"] + args)
        self.assertEqual(settings.segmentation_model_repo, "owner/repo")
        # The token is handed over through the environment, never as an argument
        self.assertNotIn("--segmentation_model_token", args)
        self.assertNotIn("tok", args)
        self.assertEqual(runner.build_env(config), {MODEL_TOKEN_ENV: "tok"})

    def test_token_env_name_matches_cli(self):
        from modules.Segmentation.model_source import TOKEN_ENV
        self.assertEqual(MODEL_TOKEN_ENV, TOKEN_ENV)

    def test_no_token_no_env(self):
        self.assertEqual(UltraSingerRunner().build_env({"segmentation_model_repo": "owner/repo"}), {})

    def test_local_file_wins_over_repo(self):
        config = {"segmentation_model": "m.pt", "segmentation_model_repo": "owner/repo",
                  "segmentation_model_token": "tok"}
        runner = UltraSingerRunner()
        args = runner.build_args(config, "test.mp3")
        self.assertIn("--segmentation_model", args)
        self.assertNotIn("--segmentation_model_repo", args)
        self.assertNotIn("--segmentation_model_token", args)
        self.assertEqual(runner.build_env(config), {})

    def test_worker_passes_extra_env_to_child(self):
        from unittest.mock import MagicMock, patch
        from src.gui.ultrasinger_runner import ConversionWorker
        proc = MagicMock(stdout=iter([]), returncode=0)
        with patch("src.gui.ultrasinger_runner.subprocess.Popen", return_value=proc) as popen:
            ConversionWorker(["-i", "x.mp3"], extra_env={MODEL_TOKEN_ENV: "tok"}).run()
        cmd = popen.call_args.args[0]
        self.assertNotIn("tok", cmd)
        self.assertEqual(popen.call_args.kwargs["env"][MODEL_TOKEN_ENV], "tok")

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
class TestLeadVocalPitchSetting(unittest.TestCase):
    def test_default_on_and_no_flag(self):
        self.assertTrue(_DEFAULTS["lead_vocal_pitch"])
        form = ConversionSettingsForm({})
        self.assertTrue(form.collect_config()["lead_vocal_pitch"])
        args = UltraSingerRunner().build_args(form.collect_config(), "test.mp3")
        self.assertNotIn("--disable_lead_vocal_pitch", args)

    def test_off_round_trip(self):
        form = ConversionSettingsForm({"lead_vocal_pitch": False})
        args = UltraSingerRunner().build_args(form.collect_config(), "test.mp3")
        self.assertIn("--disable_lead_vocal_pitch", args)
        settings = init_settings(args[args.index("-i"):] if "-i" in args else ["-i", "test.mp3"] + args)
        self.assertFalse(settings.lead_vocal_pitch)
