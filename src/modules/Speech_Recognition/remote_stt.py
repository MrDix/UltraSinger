"""Remote (cloud) speech-to-text as a Whisper alternative.

Sends audio to an OpenAI-compatible ``/audio/transcriptions`` endpoint
(e.g. Groq's Whisper endpoint) and returns what it heard: the text and,
when the service returns them (``verbose_json``), its words with their
times and its segments. The notes never take these times: the words are placed
afterwards by the local wav2vec2 CTC forced-alignment model, exactly like
the Reference-Lyrics-First pipeline places LRCLIB plain-text lyrics (see
``create_midi_segments_from_plain_lyrics``). The remote times only tell the
alignment roughly where each line is sung: the remote words are the line
anchors of plain lyrics, and the remote segments are the lines of the remote
text. See docs/remote-stt-design.md §2 for why the remote times are not
precise enough for the notes themselves.

Fail-open by design: every failure mode (network error, timeout,
non-200 response, oversized file, empty transcript) returns ``None``
instead of raising, so a misconfigured or unreachable provider never
aborts a conversion — callers fall through to local Whisper.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

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

# The audio is uploaded as mono FLAC at the sample rate of the Whisper
# models: a few MB per song, where the 44.1 kHz WAV of the vocals passed the
# upload limit from about five minutes on.
UPLOAD_SAMPLE_RATE = 16000

# Generous default: upload + remote inference of a multi-minute song is
# not instant.
DEFAULT_TIMEOUT_S = 120

# Hard upper bound on how long a single 429 wait may take, even if a
# provider sends an absurd ``Retry-After`` value.
MAX_RETRY_WAIT_S = 300

# How a service turns down a request it does not support (here: the one for
# word times); 422 is the validation error of many self-hosted servers
_REJECTED = (400, 422)


@dataclass
class RemoteTranscription:
    """What remote speech-to-text heard."""
    text: str
    # (text, start s, end s) of the words, as the service placed them; empty
    # when it returned no times
    words: list[tuple[str, float, float]] = field(default_factory=list)
    # the text of each segment (a stretch the service heard at once), in order
    segments: list[str] = field(default_factory=list)

    def lines(self) -> str:
        """The text with each segment on a line of its own; without
        segments, the text as it came."""
        texts = [text.strip() for text in self.segments if text.strip()]
        return "\n".join(texts) if texts else self.text


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


def _upload_copy(audio_path: str) -> Optional[str]:
    """A temporary mono FLAC copy of the audio at ``UPLOAD_SAMPLE_RATE``, or
    None when the audio cannot be read that way (it is then sent as it is)."""
    path = None
    try:
        import librosa
        import numpy as np
        import soundfile as sf

        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        samples = samples.mean(axis=1)
        if sample_rate != UPLOAD_SAMPLE_RATE:
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=UPLOAD_SAMPLE_RATE)
        # (an integer format would wrap values past full scale around)
        samples = np.clip(samples, -1.0, 1.0)
        handle, path = tempfile.mkstemp(prefix="ultrasinger_stt_", suffix=".flac")
        os.close(handle)
        sf.write(path, samples, UPLOAD_SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        return path
    except Exception:  # noqa: BLE001 - any audio the copy cannot be made of is sent as it is
        _remove(path)
        return None


def _remove(path: Optional[str]) -> None:
    """Remove a temporary copy; one that is held open (e.g. by a virus
    scanner) stays, with a note."""
    if path is None or not os.path.exists(path):
        return
    try:
        os.remove(path)
    except OSError as e:
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Note:')} the temporary audio copy {path} "
              f"could not be removed ({e})")


def _is_time(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _timed_words(items) -> list[tuple[str, float, float]]:
    """(text, start s, end s) of the words of a response, those with a text
    and times that can be (0 <= start <= end)."""
    out = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        text, start, end = item.get("word"), item.get("start"), item.get("end")
        if isinstance(text, str) and _is_time(start) and _is_time(end) and 0 <= start <= end:
            out.append((text, float(start), float(end)))
    return out


def _segment_texts(items) -> list[str]:
    """The text of each segment of a response; none when one of them has
    no text (the lines would leave its words out)."""
    if not isinstance(items, list):
        return []
    texts = [item.get("text") if isinstance(item, dict) else None for item in items]
    return texts if all(isinstance(text, str) for text in texts) else []


def _post(url: str, headers: dict, data: dict, upload_path: str, timeout: int,
          max_attempts: int, retry_wait: float) -> Optional[requests.Response]:
    """POST the audio; on HTTP 429 wait and try again, up to ``max_attempts``
    times in all. The last response, or None on a network error."""
    response = None
    for attempt in range(max_attempts):
        try:
            # Re-open the file for every attempt: requests consumes the
            # file handle on the previous POST, so a retry needs a fresh one.
            with open(upload_path, "rb") as audio_file:
                files = {"file": (os.path.basename(upload_path), audio_file)}
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
    return response


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
) -> RemoteTranscription | None:
    """Transcribe *audio_path* via a remote OpenAI-compatible STT API.

    Returns the transcript on success, or ``None`` on any failure (network
    error, timeout, HTTP error, oversized file, empty response) — this
    function never raises for expected failure modes, so callers can treat
    it as a fail-open text source.

    The words come with their times, and the text in segments
    (``verbose_json``); a service that turns that request down (HTTP 400 or
    422) is asked once more for the text alone, and the transcript then has
    no words and segments. The
    audio is sent as a mono FLAC copy at ``UPLOAD_SAMPLE_RATE``, or as it is
    when no such copy can be made of it.

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

    url = api_base_url.rstrip("/") + "/audio/transcriptions"

    try:
        validate_url_scheme(url)
    except ValueError as e:
        print(f"{ULTRASINGER_HEAD} {red_highlighted(f'Remote STT skipped: {e}')}")
        return None

    copy_path = _upload_copy(audio_path)
    try:
        return _transcribe(copy_path or audio_path, url, api_key, model, language, timeout,
                           retry_on_rate_limit, retry_wait, retry_max)
    finally:
        _remove(copy_path)


def _transcribe(upload_path: str, url: str, api_key: str, model: str, language: str | None, timeout: int,
                retry_on_rate_limit: bool, retry_wait: float, retry_max: int) -> RemoteTranscription | None:
    """``transcribe_remote`` with the file to upload."""
    try:
        file_size = os.path.getsize(upload_path)
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

    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": OPENAI_COMPATIBLE_USER_AGENT,
    }

    text_only = {
        "model": model,
        "response_format": "json",
        "temperature": "0",
    }
    if language:
        text_only["language"] = language
    timed = {**text_only, "response_format": "verbose_json", "timestamp_granularities[]": ["word", "segment"]}

    # max(1, ...): a negative retry_max must not zero out the loop --
    # at least one attempt always runs so failure stays fail-open.
    max_attempts = max(1, 1 + (retry_max if retry_on_rate_limit else 0))
    response = _post(url, headers, timed, upload_path, timeout, max_attempts, retry_wait)
    if response is not None and response.status_code in _REJECTED:
        # Not every OpenAI-compatible service returns times
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Remote STT rejected the request for word times')} "
              f"(HTTP {response.status_code}: {response.text[:200]}) - asking for the text only")
        response = _post(url, headers, text_only, upload_path, timeout, max_attempts, retry_wait)
    if response is None:
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
    if not isinstance(text, str) or not text.strip():
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Remote STT returned empty transcript')}")
        return None

    return RemoteTranscription(text.strip(), _timed_words(body.get("words")), _segment_texts(body.get("segments")))
