"""Read metadata tags from audio/video files.

Uses mutagen for cross-format tag reading. Returns a dict with
standardized keys (title, artist, album, year, genre).
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def read_media_metadata(file_path: Optional[str]) -> dict[str, Optional[str]]:
    """Read metadata tags from an audio or video file.

    Returns a dict with keys: title, artist, album, year, genre.
    Missing values are None. Returns empty dict on any error.
    """
    result = {
        "title": None,
        "artist": None,
        "album": None,
        "year": None,
        "genre": None,
    }

    if not file_path or not Path(file_path).exists():
        return result

    try:
        from mutagen import File as MutagenFile
    except ImportError:
        logger.debug("mutagen not installed, cannot read metadata")
        return result

    try:
        audio = MutagenFile(file_path, easy=True)
        if audio is None or audio.tags is None:
            return result

        result["title"] = _first(audio.tags.get("title"))
        result["artist"] = _first(audio.tags.get("artist"))
        result["album"] = _first(audio.tags.get("album"))
        result["year"] = _first(audio.tags.get("date"))
        result["genre"] = _first(audio.tags.get("genre"))

    except Exception as e:
        logger.debug("Failed to read metadata from %s: %s", file_path, e)

    return result


def format_display_title(metadata: dict[str, Optional[str]], fallback: str = "") -> str:
    """Format metadata into a display title like 'Artist - Title'.

    Falls back to the provided fallback string if no useful metadata.
    """
    artist = metadata.get("artist")
    title = metadata.get("title")

    if artist and title:
        return f"{artist} - {title}"
    elif title:
        return title
    elif artist:
        return artist
    return fallback


def _first(value) -> Optional[str]:
    """Extract first string from a tag value (may be list or str)."""
    if value is None:
        return None
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value) if value else None
