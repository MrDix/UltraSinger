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
from modules.name_matching import MIN_MATCH_SIMILARITY, normalize_name, similarity, strip_title_additions

_BASE_URL = "https://lrclib.net/api"
_REQUEST_TIMEOUT_S = 15
_MAX_RETRIES = 3
# Answers of a busy or briefly unavailable server; the request is tried again
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
# Found versions of a song at most this much further off the length of the
# audio than the closest one count as just as close (see _best_version)
_SAME_LENGTH_S = 5.0


@dataclass
class LyricsInfo:
    """Lyrics data returned from LRCLIB."""
    track_name: str
    artist_name: str
    plain_lyrics: Optional[str] = None
    synced_lyrics: Optional[str] = None
    duration: Optional[float] = None
    instrumental: bool = False


def search_lyrics(artist: str, title: str, duration: Optional[float] = None) -> Optional[LyricsInfo]:
    """Search LRCLIB for lyrics matching the given artist and title.

    Searches as ``find_lyrics`` does and reports the search and its result.
    ``duration`` is the length of the audio in seconds, if known.

    Returns ``LyricsInfo`` on success, or ``None`` if no lyrics found.
    """
    print(f"{ULTRASINGER_HEAD} Searching LRCLIB for lyrics: "
          f"{blue_highlighted(artist)} - {blue_highlighted(title)}")

    result = find_lyrics(artist, title, duration, verbose=True)

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
    length = ""
    if result.duration and duration:
        length = f", {result.duration:.0f} s long (the audio: {duration:.0f} s)"
    print(f"{ULTRASINGER_HEAD} Found lyrics on LRCLIB: "
          f"{blue_highlighted(result.artist_name)} - {blue_highlighted(result.track_name)} "
          f"[{info_str}]{length}")

    return result


def find_lyrics(artist: str, title: str, duration: Optional[float] = None,
                verbose: bool = False) -> Optional[LyricsInfo]:
    """Find the lyrics of a song on LRCLIB.

    Tries an artist+title search first, then falls back to a combined
    query string search if no results are found. When the title carries
    additions from a video title ("(Official Video)", "- US Version",
    "[HD]", "feat. Guest") and nothing is found with them, both searches are
    repeated without (``verbose`` reports that).

    Only results whose artist and title match are taken. Of several
    versions of the song, the one closest in length to the audio
    (``duration``, in seconds) wins, preferably one with synced lyrics,
    see ``_best_version``. The length never rules a version out.
    """
    stripped = strip_title_additions(title)
    titles = [title] if stripped.casefold() == title.strip().casefold() else [title, stripped]
    found = None
    for i, wanted in enumerate(titles):
        if i and verbose:
            print(f"{ULTRASINGER_HEAD} Searching LRCLIB again without the additions to the title: "
                  f"{blue_highlighted(wanted)}")
        for result in _searches(artist, wanted, duration):
            if result is not None and result.plain_lyrics:
                return result
            # Only an instrumental entry so far: lyrics may still come with the next search
            found = found or result
    return found


def _searches(artist: str, title: str, duration: Optional[float]):
    """The results of the artist+title search, then of the combined query (run only when asked for)."""
    yield _search_by_fields(artist, title, duration)
    yield _search_by_query(f"{artist} {title}", artist, title, duration)


def _search_by_fields(artist: str, title: str, duration: Optional[float] = None) -> Optional[LyricsInfo]:
    """Search LRCLIB using separate artist_name and track_name parameters."""
    params = urllib.parse.urlencode({
        "artist_name": artist,
        "track_name": title,
    })
    return _do_search(f"{_BASE_URL}/search?{params}", artist=artist, title=title, duration=duration)


def _search_by_query(query: str, artist: str = "", title: str = "",
                     duration: Optional[float] = None) -> Optional[LyricsInfo]:
    """Search LRCLIB using a combined query string."""
    params = urllib.parse.urlencode({"q": query})
    return _do_search(f"{_BASE_URL}/search?{params}", artist=artist, title=title, duration=duration)


def _do_search(url: str, artist: str = "", title: str = "",
               duration: Optional[float] = None) -> Optional[LyricsInfo]:
    """Execute a search request and return the best result with plain lyrics.

    With ``artist`` and ``title`` given, only results of that song count
    (see ``_names_match``). A song that LRCLIB only knows as instrumental is
    returned as such, without lyrics.
    """
    data = _get_json(url)
    if not isinstance(data, list):
        return None

    matching = [
        entry for entry in data
        if isinstance(entry, dict) and (not (artist and title) or _names_match(entry, artist, title))
    ]
    with_lyrics = [entry for entry in matching if _has_lyrics(entry)]
    if with_lyrics:
        best = _best_version(with_lyrics, duration)
    else:
        instrumental = [entry for entry in matching if entry.get("instrumental", False)]
        if not instrumental:
            return None
        best = instrumental[0]

    return LyricsInfo(
        track_name=best.get("trackName", ""),
        artist_name=best.get("artistName", ""),
        plain_lyrics=best.get("plainLyrics"),
        synced_lyrics=best.get("syncedLyrics"),
        duration=best.get("duration"),
        instrumental=best.get("instrumental", False),
    )


def _get_json(url: str):
    """GET ``url`` and return its parsed JSON answer, or None if that fails."""
    headers = {
        "User-Agent": f"UltraSinger/{Settings.APP_VERSION}",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in _RETRY_STATUS and attempt < _MAX_RETRIES - 1:
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
    return None


def _has_lyrics(entry: dict) -> bool:
    plain = entry.get("plainLyrics")
    return not entry.get("instrumental", False) and isinstance(plain, str) and bool(plain.strip())


def _names_match(entry: dict, artist: str, title: str) -> bool:
    """Whether a search result is the wanted song: its artist and its title
    match ours, titles compared without their additions. Also accepted the
    other way round, for video titles that name the song before the artist.
    """
    found_artist = entry.get("artistName") or ""
    found_title = strip_title_additions(entry.get("trackName") or "")
    wanted_title = strip_title_additions(title)

    def match(wanted_artist: str, wanted_track: str) -> bool:
        return (_artist_matches(found_artist, wanted_artist)
                and similarity(found_title, wanted_track) >= MIN_MATCH_SIMILARITY)

    return match(artist, wanted_title) or match(wanted_title, artist)


def _artist_matches(found: str, wanted: str) -> bool:
    """The same artist, also when one credit names more than the other
    ("Artist" and "Artist & Guest", "Band" and "The Band")."""
    if similarity(found, wanted, artist=True) >= MIN_MATCH_SIMILARITY:
        return True
    shorter, longer = sorted((normalize_name(found, artist=True).split(),
                              normalize_name(wanted, artist=True).split()), key=len)
    return bool(shorter) and all(word in longer for word in shorter)


def _best_version(entries: list[dict], duration: Optional[float]) -> dict:
    """The found version of the song that fits the audio best.

    Album, single, video and live versions of a song differ in length, and
    often in their lyrics too (a repeated chorus, an extra verse). So the
    versions closest in length to the audio (``duration``, in seconds) are
    taken: the closest one and all at most ``_SAME_LENGTH_S`` further off.
    Of these, one with synced lyrics comes first, then the closest, then
    LRCLIB's order. Without a known ``duration`` all versions are taken.
    """
    offs = [_length_off(entry, duration) for entry in entries]
    closest = min(offs)
    ranked = [(not entry.get("syncedLyrics"), off, index)
              for index, (entry, off) in enumerate(zip(entries, offs))
              if off <= closest + _SAME_LENGTH_S]
    return entries[min(ranked)[2]]


def _length_off(entry: dict, duration: Optional[float]) -> float:
    """How many seconds the length of a found version is off the audio."""
    if not duration:
        return 0.0
    length = entry.get("duration")
    if isinstance(length, (int, float)) and length > 0:
        return abs(length - duration)
    return float("inf")
