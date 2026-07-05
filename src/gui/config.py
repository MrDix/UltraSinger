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

# Defaults for all settings
_DEFAULTS = {
    # Output
    "output_folder": "",
    # Whisper
    "whisper_model": "large-v2",
    "whisper_batch_size": 16,
    "whisper_compute_type": "",
    "whisper_align_model": "",
    # VAD / ASR (tuned for singing)
    "vad_onset": 0.35,
    "vad_offset": 0.20,
    "no_speech_threshold": 0.4,
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
    "golden_notes": False,
    "pitch_change_split": True,
    "pitch_notes": False,
    "keep_numbers": False,
    # Lyrics Lookup
    "disable_lyrics_lookup": False,
    "disable_reference_lyrics": False,
    # Freestyle detection
    "detect_growl": False,
    "freestyle_harmonicity": 0.40,
    "freestyle_energy": 0.01,
    "freestyle_confidence": 0.35,
    "freestyle_pitch_stdev": 4.0,
    "freestyle_spectral_flatness": 0.25,
    "freestyle_use_spectral": True,
    # LLM
    "llm_correct": False,
    "llm_providers": [],  # list of LLMProvider dicts
    # Legacy single-LLM keys (kept for migration detection)
    "llm_api_base_url": "https://api.groq.com/openai/v1",
    "llm_model": "qwen/qwen3-32b",
    "llm_retry_on_rate_limit": True,
    "llm_retry_wait": 60,
    "llm_retry_max": 3,
    # Remote speech-to-text (text-only Whisper alternative; timing stays local)
    "remote_stt": False,
    "remote_stt_api_base_url": "https://api.groq.com/openai/v1",
    "remote_stt_api_key": "",  # stored in system keyring, never written to config.json
    "remote_stt_model": "whisper-large-v3",
    "remote_stt_timeout": 120,  # seconds
    "remote_stt_retry_on_rate_limit": True,
    "remote_stt_retry_wait": 60,
    "remote_stt_retry_max": 3,
    # Scoring
    "calculate_score": True,
    # Output options
    "format_version": "1.2.0",
    "create_plot": False,
    "create_midi": True,
    "create_audio_chunks": False,
    "create_karaoke": True,
    "write_metadata_tags": True,
    "keep_audio_in_video": False,
    "write_settings_info": False,
    # Device
    "force_cpu": False,
    "force_whisper_cpu": False,
    # Cache / Misc
    "keep_cache": False,
    # BPM / Octave overrides
    "bpm_override": "",
    "octave_shift": "",
    # Pitch detection
    "pitcher": "swiftf0",
    # Vocal separation
    "separator_backend": "audio_separator",
    "audio_separator_model": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
    "demucs_model": "htdemucs",
    # Paths
    "musescore_path": "",
    "ffmpeg_path": "",
    # Cookie
    "cookie_file": str(_CONFIG_DIR / "cookies.txt"),
    # Refinement
    "refine_from_vocal": True,
    "refine_pitch": True,
    "refine_timing": True,
    "refine_hit_ratio": 0.4,
    "refine_timing_threshold": 30.0,
    # ptAKF chart refit
    "ptakf_refit": True,
    "ptakf_refit_min_note_ms": 100.0,
    "ptakf_refit_fill": True,
    "ptakf_refit_fill_min_ms": 300.0,
    # bgutil PO-token provider (full-quality video downloads)
    "potoken_auto_start": True,          # check/start provider on GUI launch
    "potoken_auto_start_node": True,     # allow launching the local Node.js server
    "potoken_auto_start_docker": True,   # allow launching the provider via Docker
    "potoken_base_url": "",              # empty -> default http://127.0.0.1:4416
}


def _is_secret_key(key: str) -> bool:
    """Check whether a config key holds secret data (API keys)."""
    return (
        key == "llm_api_key"
        or key.startswith("llm_api_key_")
        or key == "remote_stt_api_key"
    )


def _get_secret_keys(config: dict) -> set[str]:
    """Collect all secret key names from config (legacy + per-provider)."""
    keys = set()
    if "llm_api_key" in config:
        keys.add("llm_api_key")
    for provider in config.get("llm_providers", []):
        pid = provider.get("id", "") if isinstance(provider, dict) else ""
        if pid:
            keys.add(f"llm_api_key_{pid}")
    # Also catch any llm_api_key_* / remote_stt_api_key that are directly
    # in the config dict
    for k in config:
        if _is_secret_key(k):
            keys.add(k)
    return keys


def _migrate_single_llm_to_provider(config: dict) -> None:
    """Auto-create an LLM provider from legacy single-LLM config fields.

    Called when ``llm_providers`` is empty but legacy fields exist.
    """
    from .models import LLMProvider

    url = config.get("llm_api_base_url", "")
    model = config.get("llm_model", "")
    if not url and not model:
        return

    # Derive a display name from the URL
    name = "Groq"
    if url and "groq" not in url.lower():
        from urllib.parse import urlparse
        try:
            name = urlparse(url).hostname or "Custom"
        except Exception:
            name = "Custom"

    provider = LLMProvider(
        name=name,
        api_base_url=url,
        default_model=model,
        is_default=True,
    )
    config["llm_providers"] = [provider.to_dict()]

    # Migrate API key in keyring: llm_api_key → llm_api_key_{provider.id}
    try:
        from .secrets import get_secret, store_secret

        legacy_key = get_secret("llm_api_key", config)
        if legacy_key:
            store_secret(f"llm_api_key_{provider.id}", legacy_key)
            logger.info(
                "Migrated LLM API key to provider '%s' (%s)",
                provider.name, provider.id,
            )
    except ImportError:
        pass

    logger.info(
        "Migrated legacy LLM config to provider: %s (%s, %s)",
        provider.name, url, model,
    )


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

    # Migrate legacy single-LLM to multi-provider
    providers = config.get("llm_providers", [])
    if not providers:
        _migrate_single_llm_to_provider(config)

    # Migrate legacy "youtube_url" key (pre-rename) to "video_url"
    if "youtube_url" in config:
        config.setdefault("video_url", config.pop("youtube_url"))

    # Load secrets from keyring (legacy key + per-provider keys)
    try:
        from .secrets import get_secret

        secret_keys = _get_secret_keys(config)
        for key in secret_keys:
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

        secret_keys = _get_secret_keys(config)
        for key in secret_keys:
            value = config.get(key, "")
            store_secret(key, value)
    except ImportError:
        logger.debug("secrets module not available, skipping keyring storage")

    # Build JSON-safe copy without secret keys
    json_config = {k: v for k, v in config.items() if not _is_secret_key(k)}

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
