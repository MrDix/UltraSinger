"""Application configuration persistence using JSON."""

import json
import logging
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
    "llm_api_key": "",
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


def load_config() -> dict:
    """Load configuration from disk, merged with defaults."""
    config = dict(_DEFAULTS)
    if _CONFIG_FILE.exists():
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                stored = json.load(f)
            config.update(stored)
        except Exception:
            logger.warning("Failed to load config from %s", _CONFIG_FILE, exc_info=True)
    return config


def save_config(config: dict):
    """Save configuration to disk."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception:
        logger.warning("Failed to save config to %s", _CONFIG_FILE, exc_info=True)


def get_browser_profile_path() -> str:
    """Return path for persistent browser profile storage."""
    p = _CONFIG_DIR / "browser_profile"
    p.mkdir(parents=True, exist_ok=True)
    return str(p)
