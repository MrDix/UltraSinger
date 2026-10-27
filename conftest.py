# Suppress noisy third-party warnings at import time.
import warnings

import pytest

warnings.filterwarnings("ignore", module="requests")
warnings.filterwarnings("ignore", module="pyannote")
warnings.filterwarnings("ignore", message="In 2\\.9.*torchaudio")

# Keep every test away from the user's own settings.
#
# The GUI saves its settings to ~/.ultrasinger/config.json and its secrets to the
# system keyring. A test that saves without redirecting both (e.g. one that only
# moves the settings folder) would overwrite the settings of whoever runs the
# tests and delete their stored API keys. Each test gets its own settings folder
# and an in-memory keyring; tests that redirect them themselves still do.


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
    import importlib

    folder = tmp_path_factory.mktemp("settings")
    keyring = _MemoryKeyring()
    # Tests import the GUI both as src.gui and (src is on the path) as gui:
    # two copies of each module, both get redirected
    for package in ("src.gui", "gui"):
        try:
            config_module = importlib.import_module(f"{package}.config")
            secrets_module = importlib.import_module(f"{package}.secrets")
        except ImportError:  # without the GUI dependencies
            continue
        monkeypatch.setattr(config_module, "_CONFIG_DIR", folder)
        monkeypatch.setattr(config_module, "_CONFIG_FILE", folder / "config.json")
        # The window deletes an exported cookie file while signing in is off
        monkeypatch.setitem(config_module._DEFAULTS, "cookie_file", str(folder / "cookies.txt"))
        monkeypatch.setattr(secrets_module, "_keyring_mod", keyring)
        monkeypatch.setattr(secrets_module, "_keyring_available", True)
