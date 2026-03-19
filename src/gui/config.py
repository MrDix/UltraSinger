"""Application configuration persistence using JSON.

Security model:
  - Config directory (~/.ultrasinger/) has restricted permissions (0o700 / owner-only ACL)
  - API keys are stored in the system keyring, NOT in config.json
  - config.json contains only non-sensitive settings (permissions 0o600)
  - Legacy API keys in config.json are migrated to keyring on first load
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path.home() / ".ultrasinger"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

# Keys that must never be written to config.json (stored in keyring instead)
_SECRET_KEYS = frozenset({"llm_api_key"})

# Defaults for all settings
_DEFAULTS = {
    # Output
    "output_folder": "",
    # Whisper
    "whisper_model": "large-v2",
    "whisper_batch_size": 16,
    "whisper_compute_type": "",
    "whisper_align_model": "",
    # Language
    "language_mode": "auto",  # "auto" or "manual"
    "language": "en",
    # Post-processing
    "hyphenation": True,
    "disable_separation": False,
    "disable_quantization": False,
    "disable_vocal_center": False,
    "disable_onset_correction": False,
    "disable_denoise_track_noise": False,
    "denoise_nr": 20,
    "denoise_nf": -80,
    "syllable_split": False,
    "vocal_gap_fill": False,
    "keep_numbers": False,
    # LLM
    "llm_correct": False,
    "llm_api_base_url": "https://api.groq.com/openai/v1",
    "llm_model": "qwen/qwen3-32b",
    # Scoring
    "calculate_score": False,
    # Output options
    "format_version": "1.2.0",
    "create_plot": False,
    "create_midi": True,
    "create_audio_chunks": False,
    "create_karaoke": True,
    # Device
    "force_cpu": False,
    "force_whisper_cpu": False,
    # Cache / Misc
    "keep_cache": False,
    # BPM / Octave overrides
    "bpm_override": "",
    "octave_shift": "",
    # Demucs
    "demucs_model": "htdemucs",
    # Paths
    "musescore_path": "",
    "ffmpeg_path": "",
    # Cookie
    "cookie_file": str(_CONFIG_DIR / "cookies.txt"),
}


def _secure_directory(path: Path) -> None:
    """Restrict directory permissions to owner-only on all platforms."""
    if sys.platform == "win32":
        _secure_directory_windows(path)
    else:
        try:
            os.chmod(path, 0o700)
        except OSError as exc:
            logger.warning("Could not set permissions on %s: %s", path, exc)


def _secure_directory_windows(path: Path) -> None:
    """Restrict directory access to current user via icacls on Windows."""
    try:
        username = os.environ.get("USERNAME", "")
        if not username:
            logger.warning("Cannot determine Windows username for ACL")
            return

        # Disable inheritance and remove all inherited ACEs
        subprocess.run(
            ["icacls", str(path), "/inheritance:r"],
            capture_output=True, check=True, timeout=10,
        )
        # Grant full control to current user only
        subprocess.run(
            ["icacls", str(path), "/grant:r", f"{username}:(OI)(CI)F"],
            capture_output=True, check=True, timeout=10,
        )
        logger.info("Windows ACL set on %s for user %s", path, username)
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
        logger.warning("Could not set Windows ACL on %s: %s", path, exc)


def _ensure_config_dir() -> None:
    """Create config directory with restricted permissions if needed."""
    created = not _CONFIG_DIR.exists()
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if created:
        _secure_directory(_CONFIG_DIR)
        logger.info("Created secure config directory: %s", _CONFIG_DIR)


def load_config() -> dict:
    """Load configuration from disk, merged with defaults.

    API keys are loaded from the system keyring (via secrets module),
    not from config.json. Legacy keys in config.json are migrated
    automatically.
    """
    config = dict(_DEFAULTS)
    if _CONFIG_FILE.exists():
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                config.update(stored)
            else:
                logger.warning("Config file %s has unexpected format, ignoring", _CONFIG_FILE)
        except (json.JSONDecodeError, TypeError, OSError) as exc:
            logger.warning("Failed to load config from %s: %s", _CONFIG_FILE, exc)

    # Load secrets from keyring (with legacy migration from config.json)
    try:
        from .secrets import get_secret

        for key in _SECRET_KEYS:
            config[key] = get_secret(key, config)
    except ImportError:
        logger.debug("secrets module not available, skipping keyring integration")

    return config


def save_config(config: dict) -> None:
    """Save configuration to disk.

    Uses atomic write (write to temp file then rename) and restricts
    file permissions to owner-only. API keys are stored in the system
    keyring and stripped from the JSON file.
    """
    _ensure_config_dir()

    # Store secrets in keyring, strip from JSON data
    try:
        from .secrets import store_secret

        for key in _SECRET_KEYS:
            value = config.get(key, "")
            store_secret(key, value)
    except ImportError:
        logger.debug("secrets module not available, skipping keyring storage")

    # Build JSON-safe copy without secret keys
    json_config = {k: v for k, v in config.items() if k not in _SECRET_KEYS}

    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(_CONFIG_DIR), suffix=".tmp", prefix="config_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(json_config, f, indent=2, ensure_ascii=False)
            if sys.platform != "win32":
                os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, str(_CONFIG_FILE))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError:
        logger.warning("Failed to save config to %s", _CONFIG_FILE, exc_info=True)


def get_browser_profile_path() -> str:
    """Return path for persistent browser profile storage."""
    p = _CONFIG_DIR / "browser_profile"
    p.mkdir(parents=True, exist_ok=True)
    _secure_directory(p)
    return str(p)
