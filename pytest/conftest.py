"""Keep every test away from the user's own settings.

The GUI saves its settings to ~/.ultrasinger/config.json and its secrets to the
system keyring. A test that saves without redirecting both (e.g. one that only
moves the settings folder) would overwrite the settings of whoever runs the
tests and delete their stored API keys. Each test gets its own settings folder
and an in-memory keyring; tests that redirect them themselves still do.
"""

import pytest


class _MemoryKeyring:
    """The part of the keyring module the GUI uses, kept in memory."""

    def __init__(self):
        self._store = {}

    def get_password(self, service, key):
        return self._store.get((service, key))

    def set_password(self, service, key, value):
        self._store[(service, key)] = value

    def delete_password(self, service, key):
        self._store.pop((service, key), None)

    def get_keyring(self):
        return self


@pytest.fixture(autouse=True)
def _private_settings(tmp_path_factory, monkeypatch):
    try:
        from src.gui import config as config_module
        from src.gui import secrets as secrets_module
    except ImportError:  # without the GUI dependencies
        return
    folder = tmp_path_factory.mktemp("settings")
    monkeypatch.setattr(config_module, "_CONFIG_DIR", folder)
    monkeypatch.setattr(config_module, "_CONFIG_FILE", folder / "config.json")
    monkeypatch.setattr(secrets_module, "_keyring_mod", _MemoryKeyring())
    monkeypatch.setattr(secrets_module, "_keyring_available", True)
