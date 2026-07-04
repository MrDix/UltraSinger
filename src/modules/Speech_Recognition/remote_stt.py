"""Remote (cloud) speech-to-text as a Whisper alternative.

Sends audio to an OpenAI-compatible ``/audio/transcriptions`` endpoint
(e.g. Groq's Whisper endpoint) and returns plain transcript text. This
is a *text-only* source: timing/alignment is always produced afterward
by the existing local wav2vec2 CTC forced-alignment model, exactly like
the Reference-Lyrics-First pipeline does for LRCLIB plain-text lyrics
(see ``create_midi_segments_from_plain_lyrics``). Remote responses are
never trusted for timestamps — see docs/remote-stt-design.md §2 for why.

Fail-open by design: every failure mode (network error, timeout,
non-200 response, oversized file, empty transcript) returns ``None``
instead of raising, so a misconfigured or unreachable provider never
aborts a conversion — callers fall through to local Whisper.
"""

from __future__ import annotations

import os

import requests

from modules.console_colors import ULTRASINGER_HEAD, red_highlighted, gold_highlighted
from modules.Speech_Recognition.llm_corrector import (
    OPENAI_COMPATIBLE_USER_AGENT,
    validate_url_scheme,
)

# Most OpenAI-compatible providers (Groq, OpenAI) cap uploads at 25 MB.
# Enforced client-side so we fail open before ever spending an upload
# instead of discovering the limit via an HTTP error.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Generous default: upload + remote inference of a multi-minute song is
# not instant.
DEFAULT_TIMEOUT_S = 120


def transcribe_remote(
        audio_path: str,
        api_base_url: str,
        api_key: str | None,
        model: str,
        language: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_S,
) -> str | None:
    """Transcribe *audio_path* via a remote OpenAI-compatible STT API.

    Returns the plain transcript text on success, or ``None`` on any
    failure (network error, timeout, HTTP error, oversized file, empty
    response) — this function never raises for expected failure modes,
    so callers can treat it as a fail-open text source.
    """
    if not api_key:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Remote STT skipped: no API key configured')}")
        return None

    if not os.path.isfile(audio_path):
        print(f"{ULTRASINGER_HEAD} {red_highlighted(f'Remote STT skipped: audio file not found: {audio_path}')}")
        return None

    try:
        file_size = os.path.getsize(audio_path)
    except OSError as e:
        print(f"{ULTRASINGER_HEAD} {red_highlighted(f'Remote STT skipped: could not stat audio file: {e}')}")
        return None

    if file_size > MAX_UPLOAD_BYTES:
        print(
            f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} Remote STT skipped: "
            f"audio file ({file_size / (1024 * 1024):.1f} MB) exceeds the "
            f"{MAX_UPLOAD_BYTES / (1024 * 1024):.0f} MB upload limit"
        )
        return None

    url = api_base_url.rstrip("/") + "/audio/transcriptions"

    try:
        validate_url_scheme(url)
    except ValueError as e:
        print(f"{ULTRASINGER_HEAD} {red_highlighted(f'Remote STT skipped: {e}')}")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": OPENAI_COMPATIBLE_USER_AGENT,
    }

    data = {
        "model": model,
        "response_format": "json",
        "temperature": "0",
    }
    if language:
        data["language"] = language

    try:
        with open(audio_path, "rb") as audio_file:
            files = {"file": (os.path.basename(audio_path), audio_file)}
            response = requests.post(
                url,
                headers=headers,
                data=data,
                files=files,
                timeout=timeout,
            )
    except requests.exceptions.Timeout:
        print(f"{ULTRASINGER_HEAD} {red_highlighted(f'Remote STT timed out after {timeout}s')}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"{ULTRASINGER_HEAD} {red_highlighted(f'Remote STT network error: {e}')}")
        return None
    except OSError as e:
        print(f"{ULTRASINGER_HEAD} {red_highlighted(f'Remote STT skipped: could not read audio file: {e}')}")
        return None

    if response.status_code != 200:
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted(f'Remote STT HTTP error {response.status_code}: {response.text[:200]}')}"
        )
        return None

    try:
        body = response.json()
    except ValueError as e:
        print(f"{ULTRASINGER_HEAD} {red_highlighted(f'Remote STT response parse error: {e}')}")
        return None

    text = body.get("text") if isinstance(body, dict) else None
    if not text or not text.strip():
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Remote STT returned empty transcript')}")
        return None

    return text.strip()
