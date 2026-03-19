"""Secure credential storage for UltraSinger GUI.

Priority chain for reading secrets:
  1. System keyring (Windows Credential Manager, macOS Keychain, Linux Secret Service)
  2. Environment variable (ULTRASINGER_LLM_API_KEY)
  3. Legacy config.json fallback (one-time migration to keyring)

On write, secrets are stored in the system keyring only.
The API key is never written to config.json.
"""

import logging
import os

logger = logging.getLogger(__name__)

_SERVICE_NAME = "ultrasinger"

# Mapping of secret keys to environment variable names
_ENV_VAR_MAP = {
    "llm_api_key": "ULTRASINGER_LLM_API_KEY",
}

_keyring_available = False
_keyring_mod = None

try:
    import keyring as _keyring_mod
    from keyring.errors import KeyringError as _KeyringError
    from keyring.errors import NoKeyringError as _NoKeyringError

    # Probe whether a real backend is available (not the fail backend)
    _keyring_mod.get_password(_SERVICE_NAME, "__probe__")
    _keyring_available = True
    logger.info("Keyring backend: %s", type(_keyring_mod.get_keyring()).__name__)
except ImportError:
    logger.info(
        "keyring package not installed. "
        "Install GUI extras with: uv sync --extra gui"
    )
except Exception as exc:  # noqa: BLE001 — defensive, keyring backends can raise anything
    # Covers NoKeyringError, KeyringError, and unexpected backend errors
    logger.warning("Keyring not usable: %s", exc)


def store_secret(key: str, value: str) -> bool:
    """Store a secret in the system keyring.

    Returns True if stored successfully, False otherwise.
    """
    if not value:
        delete_secret(key)
        return True

    if not _keyring_available or _keyring_mod is None:
        logger.warning("Cannot store secret '%s': no keyring backend available", key)
        return False

    try:
        _keyring_mod.set_password(_SERVICE_NAME, key, value)
        logger.info("Secret '%s' stored in system keyring", key)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to store secret '%s' in keyring: %s", key, exc)
        return False


def get_secret(key: str, config: dict | None = None) -> str:
    """Retrieve a secret using the priority chain.

    1. System keyring
    2. Environment variable
    3. Legacy config.json value (triggers migration to keyring)

    Args:
        key: The secret identifier (e.g., "llm_api_key").
        config: Optional config dict for legacy migration.

    Returns:
        The secret value, or empty string if not found.
    """
    # 1. Try keyring
    if _keyring_available and _keyring_mod is not None:
        try:
            value = _keyring_mod.get_password(_SERVICE_NAME, key)
            if value:
                return value
        except Exception as exc:  # noqa: BLE001
            logger.warning("Keyring read failed for '%s': %s", key, exc)

    # 2. Try environment variable
    env_var = _ENV_VAR_MAP.get(key)
    if env_var:
        value = os.environ.get(env_var, "")
        if value:
            logger.debug("Secret '%s' loaded from environment variable %s", key, env_var)
            return value

    # 3. Legacy migration from config.json
    if config and key in config and config[key]:
        legacy_value = config[key]
        logger.info(
            "Migrating secret '%s' from config.json to system keyring", key
        )
        if store_secret(key, legacy_value):
            # Clear from config dict (caller should persist this)
            config[key] = ""
            logger.info("Secret '%s' migrated successfully, cleared from config", key)
        return legacy_value

    return ""


def delete_secret(key: str) -> bool:
    """Remove a secret from the system keyring.

    Returns True if deleted (or didn't exist), False on error.
    """
    if not _keyring_available or _keyring_mod is None:
        return True

    try:
        _keyring_mod.delete_password(_SERVICE_NAME, key)
        logger.info("Secret '%s' removed from keyring", key)
        return True
    except Exception as exc:  # noqa: BLE001
        # delete_password raises if the entry doesn't exist on some backends
        logger.debug("Could not delete secret '%s': %s", key, exc)
        return True


def get_keyring_backend_name() -> str:
    """Return human-readable name of the active keyring backend."""
    if _keyring_available and _keyring_mod is not None:
        backend = type(_keyring_mod.get_keyring()).__name__
        # Friendly names
        names = {
            "WinVaultKeyring": "Windows Credential Manager",
            "Keyring": "macOS Keychain",
            "SecretServiceKeyring": "GNOME/KDE Secret Service",
        }
        return names.get(backend, backend)
    return "Not available"


def is_keyring_available() -> bool:
    """Check whether a system keyring backend is available."""
    return _keyring_available
