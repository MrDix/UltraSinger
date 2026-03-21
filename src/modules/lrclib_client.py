"""LRCLIB API client for fetching verified song lyrics.

Uses the free LRCLIB API (https://lrclib.net) to search for song lyrics
by artist and title. No API key required.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

from Settings import Settings
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted

_BASE_URL = "https://lrclib.net/api"
_REQUEST_TIMEOUT_S = 15
_MAX_RETRIES = 3


@dataclass
class LyricsInfo:
    """Lyrics data returned from LRCLIB."""
    track_name: str
    artist_name: str
    plain_lyrics: Optional[str] = None
    synced_lyrics: Optional[str] = None
    duration: Optional[float] = None
    instrumental: bool = False


def search_lyrics(artist: str, title: str) -> Optional[LyricsInfo]:
    """Search LRCLIB for lyrics matching the given artist and title.

    Tries an artist+title search first, then falls back to a combined
    query string search if no results are found.

    Returns ``LyricsInfo`` on success, or ``None`` if no lyrics found.
    """
    print(f"{ULTRASINGER_HEAD} Searching LRCLIB for lyrics: "
          f"{blue_highlighted(artist)} - {blue_highlighted(title)}")

    # Try precise search first
    result = _search_by_fields(artist, title)

    # Fall back to combined query
    if result is None:
        result = _search_by_query(f"{artist} {title}")

    if result is None:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('No lyrics found on LRCLIB')}")
        return None

    has_plain = result.plain_lyrics is not None
    has_synced = result.synced_lyrics is not None
    parts = []
    if has_plain:
        word_count = len(result.plain_lyrics.split())
        parts.append(f"plain ({word_count} words)")
    if has_synced:
        parts.append("synced")
    if result.instrumental:
        parts.append("instrumental")

    info_str = ", ".join(parts) if parts else "no lyrics content"
    print(f"{ULTRASINGER_HEAD} Found lyrics on LRCLIB: "
          f"{blue_highlighted(result.artist_name)} - {blue_highlighted(result.track_name)} "
          f"[{info_str}]")

    return result


def _search_by_fields(artist: str, title: str) -> Optional[LyricsInfo]:
    """Search LRCLIB using separate artist_name and track_name parameters."""
    params = urllib.parse.urlencode({
        "artist_name": artist,
        "track_name": title,
    })
    return _do_search(f"{_BASE_URL}/search?{params}")


def _search_by_query(query: str) -> Optional[LyricsInfo]:
    """Search LRCLIB using a combined query string."""
    params = urllib.parse.urlencode({"q": query})
    return _do_search(f"{_BASE_URL}/search?{params}")


def _do_search(url: str) -> Optional[LyricsInfo]:
    """Execute a search request and return the best result with plain lyrics."""
    headers = {
        "User-Agent": f"UltraSinger/{Settings.APP_VERSION}",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < _MAX_RETRIES - 1:
                time.sleep(1)
                continue
            return None
        except (json.JSONDecodeError, ValueError):
            return None
    else:
        return None

    if not isinstance(data, list) or len(data) == 0:
        return None

    # Prefer results that have plain lyrics (non-instrumental)
    best = None
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if entry.get("instrumental", False):
            continue
        plain = entry.get("plainLyrics")
        if plain and isinstance(plain, str) and plain.strip():
            best = entry
            break

    # Fall back to first result if no plain lyrics found
    if best is None:
        best = data[0] if isinstance(data[0], dict) else None

    if best is None:
        return None

    return LyricsInfo(
        track_name=best.get("trackName", ""),
        artist_name=best.get("artistName", ""),
        plain_lyrics=best.get("plainLyrics"),
        synced_lyrics=best.get("syncedLyrics"),
        duration=best.get("duration"),
        instrumental=best.get("instrumental", False),
    )
