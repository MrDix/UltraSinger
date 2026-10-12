"""What the GUI runner hands to the CLI: arguments, environment and cookie file.

ConversionSettingsForm / config -> UltraSingerRunner.build_args() / build_env()
-> UltraSinger.py CLI option parsing (init_settings).
"""

import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QLocale
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QApplication

from src.gui.config import _DEFAULTS
from src.gui.settings_tab import ConversionSettingsForm
from src.gui.ultrasinger_runner import (
    LLM_KEY_ENV,
    PO_TOKEN_ENV,
    REMOTE_STT_KEY_ENV,
    ConversionWorker,
    UltraSingerRunner,
    redact_secrets,
)
from src.UltraSinger import init_settings
from src.UltraSinger import settings as cli_settings

_app = QApplication.instance() or QApplication([])

_CLI_SOURCE = (Path(__file__).resolve().parents[2] / "src" / "UltraSinger.py").read_text(encoding="utf-8")


def _cli(args: list[str]):
    return init_settings(args[args.index("-i"):] if "-i" in args else ["-i", "test.mp3"] + args)


@pytest.fixture(autouse=True)
def _restore_cli_settings():
    # init_settings works on a shared settings object that later test modules use too
    saved = dict(vars(cli_settings))
    yield
    vars(cli_settings).clear()
    vars(cli_settings).update(saved)


class TestWhisperModels:
    def test_every_model_the_gui_offers_is_accepted_by_the_cli(self):
        form = ConversionSettingsForm(dict(_DEFAULTS))
        models = [form._whisper_model.itemText(i) for i in range(form._whisper_model.count())]
        assert "medium.en" in models
        for model in models:
            form._whisper_model.setCurrentText(model)
            args = UltraSingerRunner().build_args(form.collect_config(), "test.mp3")
            assert _cli(args).whisper_model.value == model


class TestRefinementSwitch:
    def test_off_reaches_the_cli(self):
        form = ConversionSettingsForm({**_DEFAULTS, "refine_from_vocal": False})
        args = UltraSingerRunner().build_args(form.collect_config(), "test.mp3")
        assert "--disable_refine" in args
        assert "--refine_from_vocal" not in args
        assert _cli(args).refine_from_vocal is False

    def test_on_reaches_the_cli(self):
        form = ConversionSettingsForm({**_DEFAULTS, "refine_from_vocal": True})
        args = UltraSingerRunner().build_args(form.collect_config(), "test.mp3")
        assert "--disable_refine" not in args
        assert _cli(args).refine_from_vocal is True


class TestBpmOverride:
    def test_decimal_comma_is_passed_with_a_point(self):
        args = UltraSingerRunner().build_args({"bpm_override": "120,5"}, "test.mp3")
        assert args[args.index("--bpm") + 1] == "120.5"
        assert _cli(args).bpm_override == 120.5

    def test_field_accepts_no_decimal_comma_in_a_comma_locale(self):
        previous = QLocale()
        QLocale.setDefault(QLocale(QLocale.Language.German, QLocale.Country.Germany))
        try:
            form = ConversionSettingsForm({})
        finally:
            QLocale.setDefault(previous)
        validator = form._bpm_override.validator()
        assert validator.validate("120,5", 0)[0] == QValidator.State.Invalid
        assert validator.validate("120.5", 0)[0] == QValidator.State.Acceptable


class TestApiKeysGoThroughTheEnvironment:
    def _config(self):
        return {
            "llm_correct": True,
            "llm_provider_id": "p1",
            "llm_providers": [{"id": "p1", "name": "P", "api_base_url": "https://llm.invalid/v1",
                               "default_model": "m", "is_default": True}],
            "llm_api_key_p1": "llm-secret",
            "remote_stt": True,
            "remote_stt_api_base_url": "https://stt.invalid/v1",
            "remote_stt_api_key": "stt-secret",
        }

    def test_keys_are_not_on_the_command_line(self):
        runner = UltraSingerRunner()
        args = runner.build_args(self._config(), "test.mp3")
        assert "llm-secret" not in args and "stt-secret" not in args
        assert "--llm_api_key" not in args and "--remote_stt_api_key" not in args
        assert args[args.index("--llm_api_base_url") + 1] == "https://llm.invalid/v1"
        assert args[args.index("--llm_model") + 1] == "m"
        env = runner.build_env(self._config())
        assert env[LLM_KEY_ENV] == "llm-secret"
        assert env[REMOTE_STT_KEY_ENV] == "stt-secret"

    def test_legacy_flat_key(self):
        env = UltraSingerRunner().build_env({"llm_correct": True, "llm_api_key": "flat-secret"})
        assert env[LLM_KEY_ENV] == "flat-secret"

    def test_no_keys_when_the_features_are_off(self):
        config = {**self._config(), "llm_correct": False, "remote_stt": False}
        env = UltraSingerRunner().build_env(config)
        assert LLM_KEY_ENV not in env and REMOTE_STT_KEY_ENV not in env

    def test_the_cli_reads_these_variables(self):
        assert re.search(r'os\.environ\.get\(\s*"%s"' % LLM_KEY_ENV, _CLI_SOURCE)
        assert re.search(r'os\.environ\.get\(\s*"%s"' % REMOTE_STT_KEY_ENV, _CLI_SOURCE)

    def test_child_process_gets_the_keys(self):
        runner = UltraSingerRunner()
        proc = MagicMock(stdout=iter([]), returncode=0)
        with patch("src.gui.ultrasinger_runner.subprocess.Popen", return_value=proc) as popen:
            ConversionWorker(runner.build_args(self._config(), "x.mp3"),
                             extra_env=runner.build_env(self._config())).run()
        assert "llm-secret" not in popen.call_args.args[0]
        child_env = popen.call_args.kwargs["env"]
        assert child_env[LLM_KEY_ENV] == "llm-secret"
        assert child_env[REMOTE_STT_KEY_ENV] == "stt-secret"


class TestPoTokenGoesThroughTheEnvironment:
    """Other local users can read a command line (e.g. /proc/<pid>/cmdline), not the environment."""

    def test_not_on_the_command_line(self):
        runner = UltraSingerRunner()
        args = runner.build_args({"yt_po_token": "pot-value"}, "https://video.invalid/watch?v=x")
        assert "--yt_po_token" not in args and "pot-value" not in args
        assert runner.build_env({"yt_po_token": "pot-value"})[PO_TOKEN_ENV] == "pot-value"
        assert PO_TOKEN_ENV not in runner.build_env({})

    def test_child_process_gets_it(self):
        runner = UltraSingerRunner()
        config = {"yt_po_token": "pot-value"}
        proc = MagicMock(stdout=iter([]), returncode=0)
        with patch("src.gui.ultrasinger_runner.subprocess.Popen", return_value=proc) as popen:
            ConversionWorker(runner.build_args(config, "x.mp3"), extra_env=runner.build_env(config)).run()
        assert "pot-value" not in popen.call_args.args[0]
        assert popen.call_args.kwargs["env"][PO_TOKEN_ENV] == "pot-value"

    def test_the_cli_reads_the_variable(self, monkeypatch):
        monkeypatch.setenv(PO_TOKEN_ENV, "pot-value")
        assert _cli([]).yt_po_token == "pot-value"
        assert _cli(["--yt_po_token", "given"]).yt_po_token == "given"  # the option wins
        monkeypatch.delenv(PO_TOKEN_ENV)
        assert _cli([]).yt_po_token is None


class TestPoTokenIsMasked:
    def test_redacted_in_the_logged_command(self):
        shown = redact_secrets(["python", "UltraSinger.py", "--yt_po_token", "pot-value", "-i", "x"])
        assert "pot-value" not in shown

    def test_not_in_the_console_line(self):
        lines = []
        worker = ConversionWorker(["-i", "x.mp3", "--yt_po_token", "pot-value"])
        worker.line_output.connect(lines.append)
        with patch("src.gui.ultrasinger_runner.subprocess.Popen",
                   return_value=MagicMock(stdout=iter([]), returncode=0)):
            worker.run()
        assert lines and not any("pot-value" in line for line in lines)


class TestPrivateCookieCopy:
    """yt-dlp writes its cookie jar back when it ends; the conversion gets its own copy."""

    def test_child_gets_a_copy_and_the_shared_file_stays_as_it_was(self, tmp_path):
        shared = tmp_path / "cookies.txt"
        shared.write_text("# Netscape HTTP Cookie File\nshared\n", encoding="utf-8")
        seen = {}

        def fake_popen(cmd, **kwargs):
            path = Path(cmd[cmd.index("--cookiefile") + 1])
            seen["path"] = path
            seen["content"] = path.read_text(encoding="utf-8")
            path.write_text("written back by yt-dlp", encoding="utf-8")
            return MagicMock(stdout=iter([]), returncode=0)

        with patch("src.gui.ultrasinger_runner.subprocess.Popen", side_effect=fake_popen):
            ConversionWorker(["-i", "x", "--cookiefile", str(shared)]).run()
        assert seen["path"] != shared
        assert seen["path"].parent == shared.parent
        assert seen["content"] == "# Netscape HTTP Cookie File\nshared\n"
        assert shared.read_text(encoding="utf-8") == "# Netscape HTTP Cookie File\nshared\n"
        assert not seen["path"].exists()  # removed after the run

    def test_a_deleted_cookie_file_is_not_passed(self, tmp_path):
        missing = tmp_path / "cookies.txt"
        proc = MagicMock(stdout=iter([]), returncode=0)
        with patch("src.gui.ultrasinger_runner.subprocess.Popen", return_value=proc) as popen:
            ConversionWorker(["-i", "x", "--cookiefile", str(missing)]).run()
        assert "--cookiefile" not in popen.call_args.args[0]
        assert not missing.exists()
