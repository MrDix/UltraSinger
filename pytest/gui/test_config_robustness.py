"""Loading and saving the GUI settings file (config.json).

A hand-edited or differently encoded file must not stop the GUI, and nothing
the user had in it may be lost silently.
"""

import json
import logging
import os
import sys
from unittest.mock import patch

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.gui import config as config_module
from src.gui import secrets as secrets_module
from src.gui.config import _DEFAULTS, load_config, save_config
from src.gui.settings_tab import ConversionSettingsForm

_app = QApplication.instance() or QApplication([])


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """config.json in a temporary directory, and no keyring (nothing reaches the real one)."""
    monkeypatch.setattr(config_module, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "_CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(secrets_module, "_keyring_available", False)
    for name in ("ULTRASINGER_LLM_API_KEY", "ULTRASINGER_REMOTE_STT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _backups(directory):
    return sorted(directory.glob("config.json.*.bak"))


class TestUnreadableFile:
    def test_other_encoding_falls_back_to_defaults_and_keeps_a_backup(self, config_dir, caplog):
        raw = '{"output_folder": "C:\\\\Songs\\\\Gr\xfc\xdfe", "whisper_model": "medium"}'.encode("cp1252")
        (config_dir / "config.json").write_bytes(raw)
        with caplog.at_level(logging.WARNING):
            config = load_config()
        assert config["whisper_model"] == _DEFAULTS["whisper_model"]
        assert config["output_folder"] == ""
        backups = _backups(config_dir)
        assert len(backups) == 1 and backups[0].read_bytes() == raw
        assert "cannot be read" in caplog.text
        # The form can be built from it
        ConversionSettingsForm(config)

    def test_invalid_json_falls_back_to_defaults_and_keeps_a_backup(self, config_dir):
        (config_dir / "config.json").write_text('{"whisper_model": "medium",', encoding="utf-8")
        config = load_config()
        assert config["whisper_model"] == _DEFAULTS["whisper_model"]
        assert len(_backups(config_dir)) == 1

    def test_byte_order_mark_is_accepted(self, config_dir):
        (config_dir / "config.json").write_bytes(
            b"\xef\xbb\xbf" + json.dumps({"whisper_model": "medium"}).encode("utf-8"))
        assert load_config()["whisper_model"] == "medium"
        assert _backups(config_dir) == []


class TestWrongValueTypes:
    def test_wrong_types_fall_back_per_setting(self, config_dir, caplog):
        stored = {
            "whisper_batch_size": "8",         # text for a number
            "hyphenation": "yes",               # text for a switch
            "output_folder": ["a"],             # list for text
            "remote_stt_api_key": ["key-in-a-list"],  # a secret: its value must not be logged
            "denoise_nr": 20.5,                 # a float for an int default is fine
            "bpm_override": 120,                # a number for text is kept as text
            "whisper_compute_type": None,       # older versions saved "auto" as null
            "vad_onset": 0.5,
        }
        (config_dir / "config.json").write_text(json.dumps(stored), encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            config = load_config()
        assert config["whisper_batch_size"] == _DEFAULTS["whisper_batch_size"]
        assert config["hyphenation"] is _DEFAULTS["hyphenation"]
        assert config["output_folder"] == _DEFAULTS["output_folder"]
        assert config["remote_stt_api_key"] == ""
        assert config["denoise_nr"] == 20.5
        assert config["bpm_override"] == "120"
        assert config["whisper_compute_type"] == ""
        assert config["vad_onset"] == 0.5
        assert "whisper_batch_size" in caplog.text
        assert "key-in-a-list" not in caplog.text
        assert "whisper_compute_type" not in caplog.text
        # The widgets accept every value now (a "8" made QSpinBox.setValue raise)
        form = ConversionSettingsForm(config)
        assert form._whisper_batch_size.value() == _DEFAULTS["whisper_batch_size"]
        assert form._bpm_override.text() == "120"

    def test_whole_numbers_outside_the_spin_box_range(self, config_dir, caplog):
        """QSpinBox takes a C int; a bigger number raised OverflowError and stopped the GUI start."""
        (config_dir / "config.json").write_text(
            json.dumps({"whisper_batch_size": 99999999999, "denoise_nf": -1e12,
                        "llm_retry_max": 2**31 - 1}), encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            config = load_config()
        assert config["whisper_batch_size"] == _DEFAULTS["whisper_batch_size"]
        assert config["denoise_nf"] == _DEFAULTS["denoise_nf"]
        assert config["llm_retry_max"] == 2**31 - 1  # in range: the spin box clamps it
        assert "whisper_batch_size" in caplog.text and "99999999999" not in caplog.text
        ConversionSettingsForm(config)

    def test_unknown_choices_select_the_default(self, caplog):
        with caplog.at_level(logging.WARNING):
            form = ConversionSettingsForm({"whisper_model": "huge", "format_version": "9.9.9",
                                           "audio_separator_model": "custom.ckpt"})
        # The first entries are "tiny", "0.3.0" and another model
        assert form._whisper_model.currentText() == _DEFAULTS["whisper_model"]
        assert form._format_version.currentText() == _DEFAULTS["format_version"]
        assert form._audio_separator_model.currentText() == _DEFAULTS["audio_separator_model"]
        assert "huge" in caplog.text

    def test_compute_type_auto_round_trip(self):
        form = ConversionSettingsForm({"whisper_compute_type": ""})
        assert form._whisper_compute.currentText() == "auto"
        assert form.collect_config()["whisper_compute_type"] == ""
        form = ConversionSettingsForm({"whisper_compute_type": "int8"})
        assert form.collect_config()["whisper_compute_type"] == "int8"

    def test_provider_with_wrong_field_types(self):
        from src.gui.models import LLMProvider
        provider = LLMProvider.from_dict({"id": "", "name": 5, "default_model": None, "is_default": "no"})
        assert provider.name == "5"
        assert provider.default_model == ""
        assert provider.is_default is False
        assert provider.id


class TestProviderMigration:
    def test_an_emptied_provider_list_stays_empty(self, config_dir):
        (config_dir / "config.json").write_text(
            json.dumps({"llm_providers": [], "llm_api_base_url": "https://example.invalid/v1"}),
            encoding="utf-8")
        assert load_config()["llm_providers"] == []

    def test_a_config_without_provider_list_is_migrated_once(self, config_dir):
        (config_dir / "config.json").write_text(
            json.dumps({"llm_api_base_url": "https://example.invalid/v1", "llm_model": "m"}),
            encoding="utf-8")
        config = load_config()
        assert len(config["llm_providers"]) == 1
        save_config(config)
        again = load_config()
        assert again["llm_providers"] == config["llm_providers"]


class TestLegacyPlainTextKey:
    """Without a keyring backend a key that config.json holds in plain text is kept."""

    def test_kept_when_the_keyring_cannot_take_it(self, config_dir):
        (config_dir / "config.json").write_text(
            json.dumps({"llm_api_key": "legacy-key", "llm_providers": []}), encoding="utf-8")
        config = load_config()
        assert config["llm_api_key"] == "legacy-key"
        save_config(config)
        stored = json.loads((config_dir / "config.json").read_text(encoding="utf-8"))
        assert stored["llm_api_key"] == "legacy-key"

    def test_a_new_key_is_not_written_to_the_file(self, config_dir):
        save_config({"llm_providers": [], "remote_stt_api_key": "new-key"})
        stored = json.loads((config_dir / "config.json").read_text(encoding="utf-8"))
        assert "remote_stt_api_key" not in stored

    def test_a_cleared_key_is_removed(self, config_dir):
        (config_dir / "config.json").write_text(
            json.dumps({"llm_api_key": "legacy-key", "llm_providers": []}), encoding="utf-8")
        config = load_config()
        config["llm_api_key"] = ""
        save_config(config)
        stored = json.loads((config_dir / "config.json").read_text(encoding="utf-8"))
        assert "llm_api_key" not in stored

    def test_migrated_provider_uses_the_kept_key(self, config_dir):
        """The provider made from the old settings uses the key the keyring could not take."""
        from src.gui.ultrasinger_runner import LLM_KEY_ENV, UltraSingerRunner

        (config_dir / "config.json").write_text(json.dumps({
            "llm_api_key": "legacy-key", "llm_api_base_url": "https://llm.invalid/v1",
            "llm_model": "m"}), encoding="utf-8")
        config = load_config()
        pid = config["llm_providers"][0]["id"]
        assert config[f"llm_api_key_{pid}"] == "legacy-key"
        env = UltraSingerRunner().build_env({**config, "llm_correct": True, "llm_provider_id": pid})
        assert env[LLM_KEY_ENV] == "legacy-key"

        # Other providers never get this key: one for another service, and one
        # added later for the same service without a key of its own
        config["llm_providers"].append({"id": "other", "name": "Other",
                                        "api_base_url": "https://other.invalid/v1",
                                        "default_model": "x", "is_default": False})
        config["llm_providers"].append({"id": "second", "name": "Second",
                                        "api_base_url": "https://llm.invalid/v1",
                                        "default_model": "y", "is_default": False})
        save_config(config)
        stored = json.loads((config_dir / "config.json").read_text(encoding="utf-8"))
        assert stored["llm_api_key"] == "legacy-key"
        assert stored["llm_legacy_provider_id"] == pid
        assert not any(k.startswith("llm_api_key_") for k in stored)  # no new key in the file
        again = load_config()  # the next start: no migration any more
        assert again[f"llm_api_key_{pid}"] == "legacy-key"
        assert not again.get("llm_api_key_other")
        assert not again.get("llm_api_key_second")

    @pytest.mark.parametrize("change", ["remove", "move", "keep"])
    def test_kept_key_goes_with_its_provider(self, config_dir, change):
        """Once the migrated provider is removed or points elsewhere, saving deletes the plain-text key."""
        from src.gui.preferences_tab import PreferencesTab
        from src.gui.widgets.llm_provider_list import LLMProviderRow

        (config_dir / "config.json").write_text(json.dumps({
            "llm_api_key": "legacy-key", "llm_api_base_url": "https://llm.invalid/v1"}),
            encoding="utf-8")
        config = load_config()
        pid = config["llm_legacy_provider_id"]
        with patch.object(LLMProviderRow, "fetch_models_if_ready", lambda self: None):
            tab = PreferencesTab(config)
        if change == "remove":
            tab._llm_provider_list._remove_provider(pid)
        elif change == "move":
            tab._llm_provider_list._rows[0]._url_edit.setText("https://moved.invalid/v1")
        config.update(tab.collect_all())
        save_config(config)
        stored = json.loads((config_dir / "config.json").read_text(encoding="utf-8"))
        assert ("llm_api_key" in stored) is (change == "keep")

    def test_kept_key_is_not_used_by_a_provider_with_another_url(self, config_dir):
        (config_dir / "config.json").write_text(json.dumps({
            "llm_api_key": "legacy-key", "llm_api_base_url": "https://llm.invalid/v1"}),
            encoding="utf-8")
        config = load_config()
        pid = config["llm_providers"][0]["id"]
        config["llm_providers"][0]["api_base_url"] = "https://moved.invalid/v1"  # edited by the user
        config[f"llm_api_key_{pid}"] = ""
        save_config(config)
        assert not load_config().get(f"llm_api_key_{pid}")

    def test_stripped_after_the_keyring_took_it(self, config_dir, monkeypatch):
        stored_keys = {}
        monkeypatch.setattr(secrets_module, "store_secret",
                            lambda key, value: stored_keys.__setitem__(key, value) or True)
        (config_dir / "config.json").write_text(
            json.dumps({"llm_api_key": "legacy-key", "llm_providers": []}), encoding="utf-8")
        save_config(load_config())
        stored = json.loads((config_dir / "config.json").read_text(encoding="utf-8"))
        assert "llm_api_key" not in stored
        assert stored_keys["llm_api_key"] == "legacy-key"


def test_load_creates_the_config_directory(tmp_path, monkeypatch):
    """Created on start through the owner-only path, before e.g. a cookie export could create it."""
    directory = tmp_path / "new_config_dir"
    monkeypatch.setattr(config_module, "_CONFIG_DIR", directory)
    monkeypatch.setattr(config_module, "_CONFIG_FILE", directory / "config.json")
    monkeypatch.setattr(secrets_module, "_keyring_available", False)
    secured = []
    monkeypatch.setattr(config_module, "_secure_directory", secured.append)
    load_config()
    assert directory.is_dir()
    assert secured == [directory]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
class TestPrivatePermissions:
    def test_existing_directory_and_files_are_restricted(self, config_dir):
        cookie_file = config_dir / "cookies.txt"
        cookie_file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
        (config_dir / "config.json").write_text(
            json.dumps({"cookie_file": str(cookie_file)}), encoding="utf-8")
        os.chmod(config_dir / "config.json", 0o644)
        os.chmod(cookie_file, 0o644)
        os.chmod(config_dir, 0o755)
        load_config()
        assert config_dir.stat().st_mode & 0o777 == 0o700
        assert cookie_file.stat().st_mode & 0o777 == 0o600
        assert (config_dir / "config.json").stat().st_mode & 0o777 == 0o600

    def test_saving_restricts_an_existing_directory(self, config_dir):
        os.chmod(config_dir, 0o755)
        save_config({"llm_providers": []})
        assert config_dir.stat().st_mode & 0o777 == 0o700
        assert (config_dir / "config.json").stat().st_mode & 0o777 == 0o600
