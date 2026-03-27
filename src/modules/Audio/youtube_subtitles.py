"""Extract manually uploaded YouTube subtitles as lyrics fallback.

When LRCLIB has no synced lyrics for a song, YouTube videos sometimes
have manually uploaded (curated) subtitle tracks that contain the actual
lyrics — often marked with musical note symbols (♪).  These are NOT
auto-generated speech recognition captions.

This module extracts only manually uploaded subtitle tracks and converts
them to LRC format for use in the reference-lyrics-first pipeline.

Usage in the pipeline:
    1. LRCLIB synced lyrics (best quality)
    2. **YouTube manual subtitles** (this module) ← NEW
    3. LRCLIB plain lyrics + forced alignment
    4. Whisper ASR (last resort)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import shutil
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Musical note markers used in YouTube lyrics subtitles
_MUSIC_NOTE_RE = re.compile(r"[♪♫🎵🎶\u266a-\u266f]+\s*")

# Timing markers in VTT/SRT format
_VTT_TIMESTAMP_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})"
    r"\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})"
)


@dataclass
class YouTubeSubtitleInfo:
    """Result of a YouTube subtitle extraction attempt."""

    synced_lyrics: Optional[str] = None  # LRC format string
    language: Optional[str] = None  # ISO language code (e.g. "en", "de")
    source: str = "youtube_subtitles"  # always this value
    is_manual: bool = True  # always True (we never use auto-generated)


def extract_manual_subtitles(
    url: str,
    preferred_lang: Optional[str] = None,
) -> Optional[YouTubeSubtitleInfo]:
    """Extract manually uploaded subtitles from a YouTube video.

    Only returns subtitles if they are manually uploaded by the video
    creator (not auto-generated speech recognition).  This ensures
    lyrics quality comparable to LRCLIB.

    Args:
        url: YouTube video URL.
        preferred_lang: Preferred language code (e.g. "en", "de").
            If not available, falls back to English, then any available
            manual subtitle track.

    Returns:
        YouTubeSubtitleInfo with LRC-formatted synced lyrics, or None
        if no suitable manual subtitles are found.
    """
    yt_dlp_path = shutil.which("yt-dlp")
    if not yt_dlp_path:
        logger.warning("yt-dlp not found — cannot extract YouTube subtitles")
        return None

    try:
        # Fetch video metadata including subtitle track info
        info = _fetch_subtitle_info(yt_dlp_path, url)
        if info is None:
            return None

        # Only use manually uploaded subtitles, never auto-generated
        manual_subs = info.get("subtitles") or {}
        if not manual_subs:
            logger.debug("No manually uploaded subtitles found")
            return None

        # Pick the best language track
        lang_code = _pick_language(manual_subs, preferred_lang)
        if lang_code is None:
            logger.debug("No suitable subtitle language found")
            return None

        # Download the subtitle track in srv3 (timed XML) format
        subtitle_text = _download_subtitle_track(
            yt_dlp_path, url, lang_code
        )
        if not subtitle_text:
            return None

        # Parse and convert to LRC format
        lrc_text = _convert_to_lrc(subtitle_text)
        if not lrc_text:
            logger.debug("Subtitle conversion to LRC produced no lines")
            return None

        logger.info(
            "Extracted YouTube manual subtitles (%s) as synced lyrics",
            lang_code,
        )
        return YouTubeSubtitleInfo(
            synced_lyrics=lrc_text,
            language=lang_code,
        )

    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as e:
        logger.warning("YouTube subtitle extraction failed: %s", e)
        return None


def _fetch_subtitle_info(
    yt_dlp_path: str, url: str
) -> Optional[dict]:
    """Fetch video metadata including subtitle track availability.

    Uses ``yt-dlp --dump-json`` which only fetches metadata without
    downloading the video itself.
    """
    cmd = [
        yt_dlp_path, "--dump-json", "--no-download",
        "--no-playlist", "--skip-download", url,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        logger.debug("yt-dlp metadata fetch failed: %s", result.stderr[:200])
        return None

    return json.loads(result.stdout)


def _pick_language(
    manual_subs: dict, preferred_lang: Optional[str]
) -> Optional[str]:
    """Pick the best subtitle language from available manual tracks.

    Priority:
    1. Preferred language (if specified and available)
    2. English ("en" or "en-*" variants)
    3. First available track
    """
    available = set(manual_subs.keys())
    if not available:
        return None

    # 1. Preferred language (exact match or prefix match)
    if preferred_lang:
        if preferred_lang in available:
            return preferred_lang
        # Try prefix match (e.g. "en" matches "en-US")
        for code in available:
            if code.startswith(preferred_lang):
                return code

    # 2. English fallback
    if "en" in available:
        return "en"
    for code in available:
        if code.startswith("en"):
            return code

    # 3. Any available track
    return next(iter(sorted(available)))


def _download_subtitle_track(
    yt_dlp_path: str, url: str, lang_code: str
) -> Optional[str]:
    """Download a specific subtitle track in VTT format to stdout.

    Uses yt-dlp with ``--write-subs`` and prints to stdout via
    a pipe, avoiding temporary files.
    """
    cmd = [
        yt_dlp_path,
        "--skip-download",
        "--no-playlist",
        "--write-subs",
        "--no-write-auto-subs",  # explicitly exclude auto-generated
        "--sub-langs", lang_code,
        "--sub-format", "vtt",
        "--print-to-file", f"requested_subtitles.{lang_code}.filepath", "-",
        "-o", "-",
        url,
    ]

    # Simpler approach: use --dump-json which already has subtitle URLs,
    # then fetch the VTT content directly
    # Actually, let's use yt-dlp Python API for cleaner extraction
    try:
        import yt_dlp as yt_dlp_module

        ydl_opts = {
            "writesubtitles": True,
            "writeautomaticsub": False,
            "subtitleslangs": [lang_code],
            "subtitlesformat": "vtt",
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp_module.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subs = info.get("subtitles") or {}
            track = subs.get(lang_code, [])

            # Find VTT format URL
            vtt_url = None
            for fmt in track:
                if fmt.get("ext") == "vtt":
                    vtt_url = fmt.get("url")
                    break

            if not vtt_url:
                logger.debug("No VTT URL found for subtitle track %s", lang_code)
                return None

            # Download VTT content
            import urllib.request
            with urllib.request.urlopen(vtt_url, timeout=10) as resp:
                return resp.read().decode("utf-8")

    except Exception as e:
        logger.debug("Subtitle download via Python API failed: %s", e)
        return None


def _convert_to_lrc(vtt_text: str) -> Optional[str]:
    """Convert WebVTT subtitle text to LRC format.

    Strips musical note markers (♪) and filters out non-lyric cues
    like "[Music]", "[Applause]", etc.

    Args:
        vtt_text: Raw WebVTT subtitle content.

    Returns:
        LRC-formatted string, or None if no lyric lines found.
    """
    lines = vtt_text.split("\n")
    lrc_lines: list[str] = []
    seen_texts: set[str] = set()  # deduplicate repeated cues

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for timestamp lines
        m = _VTT_TIMESTAMP_RE.match(line)
        if m:
            # Parse start time
            hours = int(m.group(1))
            minutes = int(m.group(2))
            seconds = int(m.group(3))
            millis = int(m.group(4))

            total_minutes = hours * 60 + minutes
            total_seconds = seconds + millis / 1000.0

            # Collect text lines following the timestamp
            i += 1
            text_parts = []
            while i < len(lines) and lines[i].strip():
                text_parts.append(lines[i].strip())
                i += 1

            text = " ".join(text_parts)

            # Clean up the text
            text = _clean_subtitle_text(text)

            # Skip empty, non-lyric, or duplicate lines
            if not text or text in seen_texts:
                continue
            if _is_non_lyric(text):
                continue

            seen_texts.add(text)
            lrc_lines.append(
                f"[{total_minutes:02d}:{total_seconds:05.2f}] {text}"
            )
        else:
            i += 1

    if not lrc_lines:
        return None

    return "\n".join(lrc_lines)


def _clean_subtitle_text(text: str) -> str:
    """Clean subtitle text by removing musical notes, HTML tags, etc."""
    # Remove HTML tags (VTT can contain <b>, <i>, etc.)
    text = re.sub(r"<[^>]+>", "", text)
    # Remove musical note markers
    text = _MUSIC_NOTE_RE.sub("", text)
    # Remove trailing musical notes
    text = re.sub(r"\s*[♪♫🎵🎶\u266a-\u266f]+\s*$", "", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip()


def _is_non_lyric(text: str) -> bool:
    """Check if a subtitle line is a non-lyric descriptor.

    YouTube subtitles often contain descriptors like [Music],
    [Applause], (instrumental), etc. that are not actual lyrics.
    """
    lower = text.lower().strip("[]() ")
    non_lyric_markers = {
        "music", "instrumental", "applause", "cheering",
        "crowd cheering", "guitar solo", "drum solo",
        "solo", "intro", "outro", "interlude",
    }
    return lower in non_lyric_markers
