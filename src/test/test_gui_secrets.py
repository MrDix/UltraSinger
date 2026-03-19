"""Tests for GUI secrets module (keyring integration)."""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure no env vars leak between tests."""
    monkeypatch.delenv("ULTRASINGER_LLM_API_KEY", raising=False)


class TestGetSecret:
    """Tests for get_secret() priority chain."""

    def test_env_var_fallback(self, monkeypatch):
        """Environment variable is used when keyring is unavailable."""
        monkeypatch.setenv("ULTRASINGER_LLM_API_KEY", "env-key-123")

        from gui import secrets

        original = secrets._keyring_available
        secrets._keyring_available = False
        try:
            result = secrets.get_secret("llm_api_key")
            assert result == "env-key-123"
        finally:
            secrets._keyring_available = original

    def test_legacy_migration(self):
        """Legacy config value is returned and migration attempted."""
        from gui import secrets

        original = secrets._keyring_available
        secrets._keyring_available = False
        try:
            config = {"llm_api_key": "legacy-key-456"}
            result = secrets.get_secret("llm_api_key", config)
            assert result == "legacy-key-456"
        finally:
            secrets._keyring_available = original

    def test_empty_when_nothing_available(self):
        """Returns empty string when no source has the secret."""
        from gui import secrets

        original = secrets._keyring_available
        secrets._keyring_available = False
        try:
            result = secrets.get_secret("llm_api_key")
            assert result == ""
        finally:
            secrets._keyring_available = original

    def test_keyring_takes_priority_over_env(self, monkeypatch):
        """Keyring value wins over environment variable."""
        monkeypatch.setenv("ULTRASINGER_LLM_API_KEY", "env-key")

        from gui import secrets

        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = "keyring-key"

        original_available = secrets._keyring_available
        original_mod = secrets._keyring_mod
        secrets._keyring_available = True
        secrets._keyring_mod = mock_keyring
        try:
            result = secrets.get_secret("llm_api_key")
            assert result == "keyring-key"
            mock_keyring.get_password.assert_called_once_with("ultrasinger", "llm_api_key")
        finally:
            secrets._keyring_available = original_available
            secrets._keyring_mod = original_mod


class TestStoreSecret:
    """Tests for store_secret()."""

    def test_store_calls_keyring(self):
        """Storing a secret calls keyring.set_password."""
        from gui import secrets

        mock_keyring = MagicMock()

        original_available = secrets._keyring_available
        original_mod = secrets._keyring_mod
        secrets._keyring_available = True
        secrets._keyring_mod = mock_keyring
        try:
            result = secrets.store_secret("llm_api_key", "new-key")
            assert result is True
            mock_keyring.set_password.assert_called_once_with(
                "ultrasinger", "llm_api_key", "new-key"
            )
        finally:
            secrets._keyring_available = original_available
            secrets._keyring_mod = original_mod

    def test_store_empty_deletes(self):
        """Storing empty string triggers deletion."""
        from gui import secrets

        mock_keyring = MagicMock()

        original_available = secrets._keyring_available
        original_mod = secrets._keyring_mod
        secrets._keyring_available = True
        secrets._keyring_mod = mock_keyring
        try:
            result = secrets.store_secret("llm_api_key", "")
            assert result is True
            mock_keyring.delete_password.assert_called_once()
        finally:
            secrets._keyring_available = original_available
            secrets._keyring_mod = original_mod

    def test_store_fails_gracefully_without_keyring(self):
        """Returns False when no keyring is available."""
        from gui import secrets

        original = secrets._keyring_available
        secrets._keyring_available = False
        try:
            result = secrets.store_secret("llm_api_key", "some-key")
            assert result is False
        finally:
            secrets._keyring_available = original


class TestConfigSecretIntegration:
    """Tests for config.py secret stripping."""

    def test_secret_keys_not_in_json(self, tmp_path, monkeypatch):
        """API keys must not appear in saved config.json."""
        from gui import config

        monkeypatch.setattr(config, "_CONFIG_DIR", tmp_path)
        monkeypatch.setattr(config, "_CONFIG_FILE", tmp_path / "config.json")

        # Mock the secrets import inside save_config
        with patch("gui.secrets.store_secret", return_value=True):
            test_config = {
                "llm_api_key": "super-secret-key",
                "output_folder": "/some/path",
                "whisper_model": "large-v2",
            }
            config.save_config(test_config)

        # Read back the JSON file directly
        import json

        with open(tmp_path / "config.json", "r") as f:
            saved = json.load(f)

        assert "llm_api_key" not in saved
        assert saved["output_folder"] == "/some/path"
        assert saved["whisper_model"] == "large-v2"


class TestSecureDirectory:
    """Tests for directory permission setting."""

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions test")
    def test_posix_directory_permissions(self, tmp_path):
        """Directory gets 0o700 on POSIX systems."""
        from gui.config import _secure_directory

        test_dir = tmp_path / "test_secure"
        test_dir.mkdir()
        _secure_directory(test_dir)

        perms = oct(test_dir.stat().st_mode & 0o777)
        assert perms == "0o700"

    @pytest.mark.skipif(os.name != "nt", reason="Windows ACL test")
    def test_windows_acl_no_crash(self, tmp_path):
        """Windows ACL setting doesn't crash."""
        from gui.config import _secure_directory

        test_dir = tmp_path / "test_secure"
        test_dir.mkdir()
        # Should not raise
        _secure_directory(test_dir)
