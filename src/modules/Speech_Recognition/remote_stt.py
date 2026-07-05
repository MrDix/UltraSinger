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
import time

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

# Hard upper bound on how long a single 429 wait may take, even if a
# provider sends an absurd ``Retry-After`` value.
MAX_RETRY_WAIT_S = 300


def _resolve_retry_wait(response: requests.Response, fallback_wait: float) -> float:
    """Determine how long to wait before retrying a 429 response.

    Respects the ``Retry-After`` header when present and numeric,
    otherwise falls back to *fallback_wait*. Always capped at
    ``MAX_RETRY_WAIT_S``.
    """
    retry_after = response.headers.get("Retry-After") if response is not None else None
    wait_s = fallback_wait
    if retry_after is not None:
        try:
            wait_s = float(retry_after)
        except ValueError:
            wait_s = fallback_wait
    return min(max(wait_s, 0), MAX_RETRY_WAIT_S)


def transcribe_remote(
        audio_path: str,
        api_base_url: str,
        api_key: str | None,
        model: str,
        language: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_S,
        retry_on_rate_limit: bool = True,
        retry_wait: float = 60.0,
        retry_max: int = 3,
) -> str | None:
    """Transcribe *audio_path* via a remote OpenAI-compatible STT API.

    Returns the plain transcript text on success, or ``None`` on any
    failure (network error, timeout, HTTP error, oversized file, empty
    response) — this function never raises for expected failure modes,
    so callers can treat it as a fail-open text source.

    On an HTTP 429 (rate limited) response, and if *retry_on_rate_limit*
    is enabled, waits and retries up to *retry_max* times before falling
    back to fail-open ``None``. The wait honors the ``Retry-After``
    response header when present (capped at ``MAX_RETRY_WAIT_S``),
    otherwise uses *retry_wait* seconds. Every other HTTP error still
    fails open immediately, unchanged.
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

    # max(1, ...): a negative retry_max must not zero out the loop --
    # at least one attempt always runs so failure stays fail-open.
    max_attempts = max(1, 1 + (retry_max if retry_on_rate_limit else 0))
    response = None

    for attempt in range(max_attempts):
        try:
            # Re-open the file for every attempt: requests consumes the
            # file handle on the previous POST, so a retry needs a fresh one.
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

        if response.status_code == 429 and attempt < max_attempts - 1:
            wait_s = _resolve_retry_wait(response, retry_wait)
            print(
                f"{ULTRASINGER_HEAD} {gold_highlighted('Remote STT rate limited (429)')}, "
                f"waiting {wait_s:.0f}s, attempt {attempt + 1}/{max_attempts - 1}..."
            )
            time.sleep(wait_s)
            continue

        break

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
