"""LRCLIB API client for fetching verified song lyrics.

Uses the free LRCLIB API (https://lrclib.net) to search for song lyrics
by artist and title. No API key required.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

from Settings import Settings
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted
from modules.name_matching import (
    MIN_MATCH_SIMILARITY,
    SAME_RECORDING_S,
    normalize_name,
    similarity,
    strip_title_additions,
    title_before_dash,
    title_without_part,
    titles_match,
)

_BASE_URL = "https://lrclib.net/api"
_REQUEST_TIMEOUT_S = 15
_MAX_RETRIES = 3
# Answers of a busy or briefly unavailable server; the request is tried again
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
# Found versions of a song at most this much further off the length of the
# audio than the closest one count as just as close (see _best_version)
_SAME_LENGTH_S = 5.0
# What separates the artists of one credit: "A & B", "A, B", "A / B", "A and B",
# "A feat. B" - not a slash or a comma inside a name ("A/B", "A,B")
_CREDIT_SEPARATOR_RE = re.compile(
    r"\s*[&+]\s*|\s+/\s+|,\s+|\s+(?:and|und|with|vs\.?|feat\.?|ft\.?|featuring)\s+", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"(?:the|a|an|der|die|das|le|la|les|los|las|el)\s", re.IGNORECASE)


@dataclass
class LyricsInfo:
    """Lyrics data returned from LRCLIB."""
    track_name: str
    artist_name: str
    plain_lyrics: Optional[str] = None
    synced_lyrics: Optional[str] = None
    duration: Optional[float] = None
    instrumental: bool = False

    @property
    def has_lyrics(self) -> bool:
        """Whether plain or synced lyrics with text came along."""
        return _has_text(self.plain_lyrics) or _has_text(self.synced_lyrics)


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

    has_plain = _has_text(result.plain_lyrics)
    has_synced = _has_text(result.synced_lyrics)
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
    if isinstance(result.duration, (int, float)) and result.duration > 0 and duration:
        length = f", {result.duration:.0f} s long (the audio: {duration:.0f} s)"
    print(f"{ULTRASINGER_HEAD} Found lyrics on LRCLIB: "
          f"{blue_highlighted(result.artist_name)} - {blue_highlighted(result.track_name)} "
          f"[{info_str}]{length}")

    return result


def find_lyrics(artist: str, title: str, duration: Optional[float] = None,
                verbose: bool = False) -> Optional[LyricsInfo]:
    """Find the lyrics of a song on LRCLIB.

    Runs an artist+title search and a combined query string search; they
    can list different versions of the song. When the title carries
    additions from a video title ("(Official Video)", "- US Version",
    "[HD]", "feat. Guest") and nothing is found with them, both searches are
    repeated without, and then with the title before a dash that a
    description follows ("Title - Singer Name live in the studio", see
    ``title_before_dash``); ``verbose`` reports that. A title that ends
    with a part number ("Title - I", "Title (Pt. 1)") is then searched
    once (by artist and title) without it, for the same part under another
    name ("Title, Part I", see ``title_without_part``). Last, with a known
    ``duration``, the title before a dash whatever follows it ("Title -
    Chapter 2") is searched, taking only versions of about the length of
    the audio (see ``SAME_RECORDING_S``).

    Only results whose artist and title match are taken. Of several
    versions of the song, the one closest in length to the audio
    (``duration``, in seconds) wins, preferably one with synced lyrics,
    see ``_best_version``. Apart from that last search, the length never
    rules a version out.
    """
    if not (artist or "").strip() or not (title or "").strip():
        return None
    # (title searched, title of the song, how it was shortened, how far a version may be off the length)
    titles, seen = [], set()
    for wanted, song, how, max_off in (
            (title, title, "", None),
            (strip_title_additions(title), None, "without the additions to the title", None),
            (title_before_dash(title), None, "with the title before the dash", None),
            # "Title - I" is listed as "Title, Part I": all parts of "Title", for the one of ours
            (title_without_part(title), title, "with the title before its part number", None),
            (title_before_dash(title, any_rest=True) if duration else "", None,
             "with the title before the dash, for a version of the length of the audio", SAME_RECORDING_S)):
        song = song or wanted
        key = (" ".join(wanted.split()).casefold(), " ".join(song.split()).casefold())
        if key[0] and key not in seen:
            seen.add(key)
            titles.append((wanted, song, how, max_off))
    found = None
    for wanted, song, how, max_off in titles:
        if max_off is not None and found is not None:
            # The song was found, as instrumental: the title before the dash
            # can be another song
            break
        if how and verbose:
            print(f"{ULTRASINGER_HEAD} Searching LRCLIB again {how}: {blue_highlighted(wanted)}")
        searches = [_search_by_fields(artist, wanted, duration, max_off, song=song)]
        if song == wanted:
            # The parts of a title are found by its name alone: one search is enough
            searches.append(_search_by_query(f"{artist} {wanted}", artist, wanted, duration, max_off))
        results = [result for result in searches if result is not None]
        with_lyrics = [result for result in results if result.has_lyrics]
        if with_lyrics:
            return _best_result(with_lyrics, duration)
        # Only instrumental entries so far: lyrics may still come with the next title
        found = found or next(iter(results), None)
    return found


def _best_result(results: list[LyricsInfo], duration: Optional[float]) -> LyricsInfo:
    """The best of the results of both searches, chosen like ``_best_version``."""
    versions = [{"duration": result.duration, "syncedLyrics": result.synced_lyrics} for result in results]
    best = _best_version(versions, duration)
    return next(result for result, version in zip(results, versions) if version is best)


def _search_by_fields(artist: str, title: str, duration: Optional[float] = None,
                      max_off: Optional[float] = None, song: Optional[str] = None) -> Optional[LyricsInfo]:
    """Search LRCLIB using separate artist_name and track_name parameters;
    the results must be of the title ``song``, if given, else of ``title``."""
    params = urllib.parse.urlencode({
        "artist_name": artist,
        "track_name": title,
    })
    return _do_search(f"{_BASE_URL}/search?{params}", artist=artist, title=song or title, duration=duration,
                      max_off=max_off)


def _search_by_query(query: str, artist: str = "", title: str = "",
                     duration: Optional[float] = None, max_off: Optional[float] = None) -> Optional[LyricsInfo]:
    """Search LRCLIB using a combined query string."""
    params = urllib.parse.urlencode({"q": query})
    return _do_search(f"{_BASE_URL}/search?{params}", artist=artist, title=title, duration=duration,
                      max_off=max_off)


def _do_search(url: str, artist: str = "", title: str = "",
               duration: Optional[float] = None, max_off: Optional[float] = None) -> Optional[LyricsInfo]:
    """Execute a search request and return the best result with lyrics.

    With ``artist`` and ``title`` given, only results of that song count
    (see ``_names_match``); with ``max_off`` and a known ``duration``, only
    those at most that many seconds off the length of the audio. A song that
    LRCLIB only knows as instrumental is returned as such, without lyrics.
    """
    data = _get_json(url)
    if not isinstance(data, list):
        return None

    matching = [
        entry for entry in data
        if isinstance(entry, dict) and (not (artist and title) or _names_match(entry, artist, title))
        and (max_off is None or _length_off(entry, duration) <= max_off)
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


def _has_text(lyrics) -> bool:
    return isinstance(lyrics, str) and bool(lyrics.strip())


def _has_lyrics(entry: dict) -> bool:
    """Not instrumental, with plain or synced lyrics (synced ones alone are
    enough for the reference-lyrics pipeline)."""
    return not entry.get("instrumental", False) and (
        _has_text(entry.get("plainLyrics")) or _has_text(entry.get("syncedLyrics")))


def _names_match(entry: dict, artist: str, title: str) -> bool:
    """Whether a search result is the wanted song: its artist and its title
    match ours, titles compared without their additions. Also accepted the
    other way round, for video titles that name the song before the artist.
    """
    found_artist = entry.get("artistName") or ""
    found_title = entry.get("trackName") or ""

    def match(wanted_artist: str, wanted_track: str) -> bool:
        return _artist_matches(found_artist, wanted_artist) and titles_match(found_title, wanted_track)

    return match(artist, title) or match(strip_title_additions(title), artist)


def _artist_matches(found: str, wanted: str) -> bool:
    """The same artist, also when one credit names more artists than the
    other ("Artist" and "Artist & Guest" or "Artist, Guest") or a leading
    "The". Not a longer name: "Band" is not "Band Tribute"."""
    if similarity(found, wanted, artist=True) >= MIN_MATCH_SIMILARITY:
        return True
    return bool(_credited_artists(found) & _credited_artists(wanted))


def _credited_artists(credit: str) -> set[str]:
    """The artists one credit names, each normalized (see ``normalize_name``).

    A separator followed by an article is part of a band name, not a list:
    "Artist and the Band" or "Artist & the Band" is one artist.
    """
    credit = credit or ""
    parts, start = [], 0
    for separator in _CREDIT_SEPARATOR_RE.finditer(credit):
        if _ARTICLE_RE.match(credit, separator.end()):
            continue
        parts.append(credit[start:separator.start()])
        start = separator.end()
    parts.append(credit[start:])
    return {name for name in (normalize_name(part, artist=True) for part in parts) if name}


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
    ranked = [(not _has_text(entry.get("syncedLyrics")), off, index)
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
