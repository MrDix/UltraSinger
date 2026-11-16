"""LRCLIB API client for fetching verified song lyrics.

Uses the free LRCLIB API (https://lrclib.net) to search for song lyrics
by artist and title. No API key required. When LRCLIB has no lyrics of a
song, far too few for its length, or only ones without times, the lyrics
database of the music service at music.163.com is asked as well (see
``find_lyrics``).
"""

from __future__ import annotations

import html
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
from modules.lookalike_letters import fold_lookalike_letters
from modules.name_matching import (
    MIN_MATCH_SIMILARITY,
    SAME_RECORDING_S,
    cover_of,
    normalize_name,
    similarity,
    strip_title_additions,
    title_before_dash,
    title_without_part,
    title_without_stripped,
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
# A found version at least this much longer or shorter than the audio, in
# seconds and as a share of the audio's length, may be of another recording
# (a live version with an extra part, a radio edit): see other_recording
OTHER_RECORDING_S = 20.0
OTHER_RECORDING_SHARE = 0.10
# Time tags of LRC lines ("[01:23.45]") and of enhanced LRC words ("<01:23.45>")
_LRC_TIME_TAG_RE = re.compile(r"\[\d+:\d+(?:\.\d+)?\]|<\d+:\d+(?:\.\d+)?>")
# The time tags at the start of an LRC line, and one of them
_LRC_LINE_TAGS_RE = re.compile(r"\s*(?:\[\d+:\d+(?:\.\d+)?\]\s*)+")
_LRC_LINE_TAG_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
# What separates the artists of one credit: "A & B", "A, B", "A / B", "A and B",
# "A feat. B" - not a slash or a comma inside a name ("A/B", "A,B")
_CREDIT_SEPARATOR_RE = re.compile(
    r"\s*[&+]\s*|\s+/\s+|,\s+|\s+(?:and|und|with|vs\.?|feat\.?|ft\.?|featuring)\s+", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"(?:the|a|an|der|die|das|le|la|les|los|las|el)\s", re.IGNORECASE)
# The dash after the artist in a title that names both: "Artist - Title"
_ARTIST_DASH_RE = re.compile(r"\s+[-–—]\s+")

# The second lyrics database, asked when LRCLIB has no lyrics of a song, far
# too few, or only ones without times (see find_lyrics)
SECOND_DB_NAME = "music.163.com"
_SECOND_DB_URL = "https://music.163.com/api"
# Search results looked at, and lyrics fetched at most (of the versions
# closest in length to the audio)
_SECOND_DB_RESULTS = 10
_SECOND_DB_LYRICS = 3
# Lyrics with fewer words than this per minute of the audio miss most of
# the song (a few lines of each chorus only); sung lyrics have 30 or more
SPARSE_WORDS_PER_MINUTE = 20.0
# Shorter audio is not judged by its words per minute
_SPARSE_MIN_LENGTH_S = 60.0
# Lyrics of the second database replace too few ones of LRCLIB only with
# at least this many times their words
FULLER_SHARE = 1.5
# Its synced lyrics replace plain ones of LRCLIB only when they have at
# least this share of their words (see _words_shared): the same text, and
# not much less of it. Of 30 songs that both databases have, its text had at
# least 78 % of the words of LRCLIB's in all but one (60 %, of a text a third
# shorter; median: 95 %); the text of another song never had more than 6 %
# (870 pairs)
SAME_TEXT_SHARE = 0.75
# Its time tags may separate the hundredths by a colon: "[01:23:45]"
_COLON_TIME_TAG_RE = re.compile(r"\[(\d+):(\d+):(\d+)\]")
# ID tags of an LRC file ("[ar:Artist]", "[by:Someone]")
_LRC_ID_TAG_RE = re.compile(r"\s*\[[A-Za-z]+:[^\]]*\]\s*$")
# A credit line of its lyrics: "<role> : <names>", the role in Chinese
# ("lyrics", "music", "producer", "mixing", an instrument) or in English
_CREDIT_RE = re.compile(r"\s*([^:\uff1a]{1,30}?)\s*[:\uff1a]\s*(\S.*)")
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\uf900-\ufaff]")
# The last character of a role in Chinese: of "lyrics", "music", "original
# singer", "director", "producer", "sound", the instruments, "band", "engineer"...
_CREDIT_ROLE_ENDS = frozenset("词曲唱监人音琴他斯鼓盘乐号"
                              "笛导筹品行师制录混声编手长")
# An English credit without a colon: "Mastered at <studio>", "Published by <publisher>"
# (not "Written in ...", which may be sung)
_CREDIT_PHRASE_RE = re.compile(
    r"(?:Mastered|Mixed|Recorded|Produced|Published|Engineered|Arranged|Composed|Programmed|"
    r"Distributed|Licensed|Written)\s+(?:at|by|for|under|to)\s+[A-Z0-9]")
# Among other credits, a role in Chinese of at most this many characters
# is one too, whatever its end ("录音助理", "合声编写")
_LOOSE_ROLE_CHARS = 8
# Title and artist lines at most this many seconds into the lyrics are a heading
_HEADER_S = 2.0
# An English role: one to four capitalized words ("Guitar", "Mixing Engineer", "Lyrics by",
# "Lead & Background Vocals")
_ENGLISH_ROLE_RE = re.compile(r"[A-Z][\w&'./-]*(?:\s+(?:[A-Z][\w&'./-]*|by|of|and|&)){0,3}")
# The text its lyrics of an instrumental hold ("pure music, please enjoy")
_INSTRUMENTAL_MARK = "\u7eaf\u97f3\u4e50"


@dataclass
class LyricsInfo:
    """Lyrics data returned from LRCLIB."""
    track_name: str
    artist_name: str
    plain_lyrics: Optional[str] = None
    synced_lyrics: Optional[str] = None
    duration: Optional[float] = None
    instrumental: bool = False
    # Where the lyrics come from: "LRCLIB" or SECOND_DB_NAME
    source: str = "LRCLIB"

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
        print(f"{ULTRASINGER_HEAD} {red_highlighted(f'No lyrics found on LRCLIB or {SECOND_DB_NAME}')}")
        return None
    report_found_lyrics(result, duration)
    return result


def report_found_lyrics(result: LyricsInfo, duration: Optional[float] = None) -> None:
    """Report found lyrics: where from, of which version, what kinds (the GUI
    queue reads this line, see queue_manager)."""
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
        length = f", {result.duration:.1f} s long (the audio: {duration:.1f} s)"
    print(f"{ULTRASINGER_HEAD} Found lyrics on {result.source}: "
          f"{blue_highlighted(result.artist_name)} - {blue_highlighted(result.track_name)} "
          f"[{info_str}]{length}")


def find_on_second_db(artist: str, title: str, duration: Optional[float] = None) -> Optional[LyricsInfo]:
    """The lyrics of a song in the second lyrics database alone (see
    ``_find_on_second_db``): for lyrics of LRCLIB that the singing does not
    have, also when they have words enough. None when it has none."""
    if not (artist or "").strip() or not (title or "").strip():
        return None
    return _find_on_second_db(artist, title, duration)


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

    A cover ("Title (Somebody Cover)", see ``cover_of``) has the lyrics of
    the song itself. When a search lists no lyrics under the title of the
    cover, the song it covers counts as a result of that search (see
    ``_is_covered_song``); a result with the title of the cover always
    comes first. When nothing is found with the title of the cover (also
    without its additions), the title without the cover mark is searched:
    by our artist and by the one the mark names (the artist of the song
    and the covering one, whichever is which). A stripped-down recording
    ("Title (Stripped)", see ``title_without_stripped``) is then searched
    without that mark as well.

    When LRCLIB has no lyrics of the song (and does not know it as
    instrumental), far too few for the length of the audio (see
    ``too_few_words``), or only plain ones, the second lyrics database is
    asked (see ``_find_on_second_db``). Its lyrics are taken when LRCLIB
    has none; instead of too few, only with at least ``FULLER_SHARE`` times
    their words and of a recording of about the length of the audio (see
    ``other_recording``); instead of plain ones, only synced lyrics of the
    same text and of about the same length (see ``_synced_for_plain``).

    Look-alike letters of other scripts in found lyrics are replaced by the
    Latin ones they stand for (see ``fold_lookalike_letters``).
    """
    if not (artist or "").strip() or not (title or "").strip():
        return None
    found = _find_on_lrclib(artist, title, duration, verbose)
    if found is not None and not found.has_lyrics:
        # Known as instrumental
        return found
    sparse = found is not None and too_few_words(found, duration)
    plain_only = found is not None and not sparse and not _has_text(found.synced_lyrics)
    if found is not None and not sparse and not plain_only:
        return found
    if verbose:
        reason = ""
        if sparse:
            reason = (f" (the lyrics on LRCLIB have {lyric_words(found)} words, "
                      f"for {_minutes(duration or found.duration)} of audio)")
        elif plain_only:
            reason = " (the lyrics on LRCLIB have no times)"
        print(f"{ULTRASINGER_HEAD} Searching {SECOND_DB_NAME} for {'synced ' if plain_only else ''}lyrics{reason}: "
              f"{blue_highlighted(artist)} - {blue_highlighted(title)}")
    other = _find_on_second_db(artist, title, duration, synced_only=plain_only)
    if other is None:
        if plain_only and verbose:
            print(f"{ULTRASINGER_HEAD} No synced lyrics found on {SECOND_DB_NAME}: "
                  f"keeping the plain ones of LRCLIB")
        return found
    if found is None:
        return other
    if plain_only:
        return _synced_for_plain(found, other, duration, verbose)
    if lyric_words(other) >= FULLER_SHARE * lyric_words(found) and not other_recording(other.duration, duration):
        return other
    if verbose:
        print(f"{ULTRASINGER_HEAD} The lyrics on {SECOND_DB_NAME} are not fuller "
              f"({lyric_words(other)} words, {_minutes(other.duration)} long): keeping the ones of LRCLIB")
    return found


def _synced_for_plain(found: LyricsInfo, other: LyricsInfo, duration: Optional[float],
                      verbose: bool) -> LyricsInfo:
    """The synced lyrics of the second database (``other``) when they can
    take the place of the plain ones of LRCLIB (``found``), else these.

    They are compared as two versions on LRCLIB are (see ``_best_version``):
    the synced ones win when their version is at most ``_SAME_LENGTH_S``
    further off the length of the audio (``duration``) than the plain ones.
    They must not be of another recording (see ``other_recording``), and
    they must be the same text: with at least ``SAME_TEXT_SHARE`` of the
    words of the plain ones (see ``_words_shared``).
    """
    if _best_result([found, other], duration) is not other or other_recording(other.duration, duration):
        why = (f"are of a version of another length ({_minutes(other.duration)}, "
               f"the plain ones: {_minutes(found.duration)})")
    else:
        shared = _words_shared(found.plain_lyrics, lrc_text(other.synced_lyrics))
        if shared >= SAME_TEXT_SHARE:
            if verbose:
                print(f"{ULTRASINGER_HEAD} The synced lyrics on {SECOND_DB_NAME} have {shared:.0%} of the "
                      f"words of the plain ones of LRCLIB: taking them")
            return other
        why = f"have only {shared:.0%} of the words of the plain ones"
    if verbose:
        print(f"{ULTRASINGER_HEAD} The synced lyrics on {SECOND_DB_NAME} {why}: "
              f"keeping the plain ones of LRCLIB")
    return found


def _words_shared(text: str, other: str) -> float:
    """The share of the words of ``text`` that ``other`` has too, in the
    same order: those in a sure pair with a word of it (two words in a row
    that both texts have, compared without case, accents and punctuation;
    see ``heard_share``)."""
    from modules.Speech_Recognition.lyrics_alignment import heard_share
    return heard_share(other.split(), [(word, 0.0, 0.0) for word in text.split()])[0]


def _find_on_lrclib(artist: str, title: str, duration: Optional[float], verbose: bool) -> Optional[LyricsInfo]:
    """The lyrics of a song on LRCLIB, searched as ``find_lyrics`` describes."""
    stripped = strip_title_additions(title)
    cover = cover_of(title)
    covered = _covered_song(artist, cover)
    # A cover is searched without its mark for the artist that the mark names, or for a mark that
    # is no addition to the title (else the title without its additions is the one without the mark)
    without_mark = cover[0] if cover and (cover[1] or cover[0].casefold() != stripped.casefold()) else ""
    # (title searched, title of the song, how it was shortened, how far a version may be off the length,
    # the artist its mark names ("" for none) when the title of a cover is searched without the mark)
    titles, seen = [], set()
    for wanted, song, how, max_off, named in (
            (title, title, "", None, None),
            (stripped, None, "without the additions to the title", None, None),
            # "Title (Somebody Cover)" has the lyrics of "Title": listed under our artist, or under Somebody
            (without_mark, title, "without the cover mark", None, cover[1] if cover else None),
            (title_without_stripped(title), None, "without the stripped mark", None, None),
            (title_before_dash(title), None, "with the title before the dash", None, None),
            # "Title - I" is listed as "Title, Part I": all parts of "Title", for the one of ours
            (title_without_part(title), title, "with the title before its part number", None, None),
            (title_before_dash(title, any_rest=True) if duration else "", None,
             "with the title before the dash, for a version of the length of the audio", SAME_RECORDING_S, None)):
        song = song or wanted
        key = (" ".join(wanted.split()).casefold(), " ".join(song.split()).casefold())
        if key[0] and key not in seen:
            seen.add(key)
            titles.append((wanted, song, how, max_off, named))
    found = None
    answers = {}  # LRCLIB's answer to each search URL: a later step may search the same again
    for wanted, song, how, max_off, named in titles:
        if max_off is not None and found is not None:
            # The song was found, as instrumental: the title before the dash
            # can be another song
            break
        if how and verbose:
            print(f"{ULTRASINGER_HEAD} Searching LRCLIB again {how}: {blue_highlighted(wanted)}")
        results = _search_title(artist, wanted, song, duration, max_off, named, answers)
        with_lyrics = [result for result in results if result.has_lyrics]
        if not with_lyrics and covered is not None and song in (title, stripped):
            # Nothing is listed under the title of the cover: the song it covers
            # counts, in the same answers (they are kept: nothing is asked again)
            with_lyrics = [result for result in _search_title(artist, wanted, song, duration, max_off, named,
                                                              answers, covered=covered) if result.has_lyrics]
        if with_lyrics:
            return _best_result(with_lyrics, duration)
        # Only instrumental entries so far: lyrics may still come with the next title
        found = found or next(iter(results), None)
    return found


def _covered_song(artist: str, cover: Optional[tuple[str, str]]) -> Optional[tuple[str, list[str]]]:
    """The song that the cover ``cover`` (see ``cover_of``) of our ``artist``
    covers, and the artists it may be listed under (see ``_is_covered_song``):
    ours and the one the cover mark names. None for no cover."""
    if cover is None:
        return None
    return cover[0], [name for name in (artist, cover[1]) if name]


def _search_title(artist: str, wanted: str, song: str, duration: Optional[float], max_off: Optional[float],
                  named: Optional[str], answers: dict,
                  covered: Optional[tuple[str, list[str]]] = None) -> list[LyricsInfo]:
    """The results of the searches of LRCLIB for the title ``wanted`` (see
    ``_find_on_lrclib``): by artist and title, by the combined query, and for
    a cover without its mark by the artist that the mark names (``named``).
    They must be of the title ``song``; with ``covered``, of the song a cover
    covers instead (see ``_do_search``)."""
    searches = [_search_by_fields(artist, wanted, duration, max_off, song=song, answers=answers, covered=covered)]
    if song == wanted or named is not None:
        # (not for the parts of a title: they are found by its name alone, one search is enough)
        searches.append(_search_by_query(f"{artist} {wanted}", artist, song, duration, max_off,
                                         answers=answers, covered=covered))
    if named:
        # The song, or its cover, listed under the artist that the cover mark names
        searches.append(_search_by_fields(artist, wanted, duration, max_off, song=song, answers=answers,
                                          searched_artist=named, covered=covered))
    return [result for result in searches if result is not None]


def _best_result(results: list[LyricsInfo], duration: Optional[float]) -> LyricsInfo:
    """The best of the results of both searches, chosen like ``_best_version``."""
    versions = [{"duration": result.duration, "syncedLyrics": result.synced_lyrics} for result in results]
    best = _best_version(versions, duration)
    return next(result for result, version in zip(results, versions) if version is best)


def _search_by_fields(artist: str, title: str, duration: Optional[float] = None,
                      max_off: Optional[float] = None, song: Optional[str] = None,
                      answers: Optional[dict] = None, searched_artist: Optional[str] = None,
                      covered: Optional[tuple[str, list[str]]] = None) -> Optional[LyricsInfo]:
    """Search LRCLIB using separate artist_name and track_name parameters;
    the results must be of the title ``song``, if given, else of ``title``
    (with ``covered``: see ``_do_search``). ``searched_artist``, if given,
    is searched in place of ``artist`` (the artist that the cover mark of
    ``song`` names)."""
    params = urllib.parse.urlencode({
        "artist_name": searched_artist or artist,
        "track_name": title,
    })
    return _do_search(f"{_BASE_URL}/search?{params}", artist=artist, title=song or title, duration=duration,
                      max_off=max_off, answers=answers, covered=covered)


def _search_by_query(query: str, artist: str = "", title: str = "",
                     duration: Optional[float] = None, max_off: Optional[float] = None,
                     answers: Optional[dict] = None,
                     covered: Optional[tuple[str, list[str]]] = None) -> Optional[LyricsInfo]:
    """Search LRCLIB using a combined query string."""
    params = urllib.parse.urlencode({"q": query})
    return _do_search(f"{_BASE_URL}/search?{params}", artist=artist, title=title, duration=duration,
                      max_off=max_off, answers=answers, covered=covered)


def _do_search(url: str, artist: str = "", title: str = "",
               duration: Optional[float] = None, max_off: Optional[float] = None,
               answers: Optional[dict] = None,
               covered: Optional[tuple[str, list[str]]] = None) -> Optional[LyricsInfo]:
    """Execute a search request and return the best result with lyrics.

    With ``artist`` and ``title`` given, only results of that song count
    (see ``_names_match``); with ``max_off`` and a known ``duration``, only
    those at most that many seconds off the length of the audio. A song that
    LRCLIB only knows as instrumental is returned as such, without lyrics.
    ``answers`` keeps the answer to each URL, so that one is not fetched twice.
    Look-alike letters in the lyrics are folded (see ``fold_lookalike_letters``).

    With ``covered`` (a song and the artists it may be listed under), the
    results that count are instead those with the lyrics of that song, for a
    cover of it (see ``_is_covered_song``): asked for when no result has the
    title of the cover itself.
    """
    if answers is not None and url in answers:
        data = answers[url]
    else:
        data = _get_json(url)
        if answers is not None:
            answers[url] = data
    if not isinstance(data, list):
        return None

    matching = [
        entry for entry in data
        if isinstance(entry, dict)
        and (_is_covered_song(entry, *covered) if covered is not None
             else not (artist and title) or _names_match(entry, artist, title))
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
        plain_lyrics=_folded(best.get("plainLyrics")),
        synced_lyrics=_folded(best.get("syncedLyrics")),
        duration=best.get("duration"),
        instrumental=best.get("instrumental", False),
    )


def _folded(lyrics):
    """Lyrics with their look-alike letters folded (see ``fold_lookalike_letters``);
    what is no text (a missing field) stays as it is."""
    return fold_lookalike_letters(lyrics) if isinstance(lyrics, str) else lyrics


def _get_json(url: str, attempts: int = _MAX_RETRIES):
    """GET ``url`` and return its parsed JSON answer, or None if that fails
    (after at most ``attempts`` tries)."""
    headers = {
        "User-Agent": f"UltraSinger/{Settings.APP_VERSION}",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in _RETRY_STATUS and attempt < attempts - 1:
                time.sleep(2 ** attempt)
                continue
            return None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < attempts - 1:
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


def _is_covered_song(entry: dict, song: str, artists: list[str]) -> bool:
    """Whether a search result has the lyrics of a cover of ``song``: it is
    the song itself or a cover of it, by one of ``artists``. Such results
    count for a cover ("Title (Somebody Cover)", see ``cover_of``) only when
    none has the title of the cover itself (see ``_names_match``): that
    title may be a song of its own ("Title (Take Cover)").

    The artists are the one of our title and the one its cover mark names
    (see ``_covered_song``): the artist of the song and the covering one,
    whichever is which ("Artist - Title (Somebody Cover)" is written for
    both). The title of the result may carry a cover mark of its own
    ("Title (Somebody Vocal Cover)") and may name one of them before the
    title ("Artist - Title"). A cover mark that names somebody else, when
    ours names an artist, tells of another cover or another song: such a
    result is not taken.
    """
    found_artist = entry.get("artistName") or ""
    found_title = entry.get("trackName") or ""

    def one_of_ours(name: str) -> bool:
        return any(_artist_matches(name, artist) for artist in artists)

    found_cover = cover_of(found_title)
    if found_cover is not None:
        found_title, named = found_cover
        if named and len(artists) > 1 and not one_of_ours(named):
            return False
    parts = _ARTIST_DASH_RE.split(found_title, maxsplit=1)
    if len(parts) == 2 and one_of_ours(parts[0]):
        found_title = parts[1]
    return one_of_ours(found_artist) and titles_match(found_title, song)


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


def other_recording(length: Optional[float], duration: Optional[float]) -> bool:
    """Whether found lyrics of a version ``length`` seconds long may be of
    another recording than the audio (``duration`` seconds): the lengths
    differ by more than ``OTHER_RECORDING_S`` and by more than
    ``OTHER_RECORDING_SHARE`` of the audio. False when a length is unknown.

    The LRC times of such lyrics only fit the audio when the recordings
    differ at the end (an audio that is cut short, a longer fade-out); a
    live version with an extra part moves all lines after it.
    """
    if not (isinstance(length, (int, float)) and length > 0 and duration):
        return False
    off = abs(length - duration)
    return off > OTHER_RECORDING_S and off > OTHER_RECORDING_SHARE * duration


def lrc_text(synced_lyrics: str) -> str:
    """The text of synced lyrics without their time tags, in the order it is
    sung: a line with several time tags ("[00:04.00][00:20.00] Again") comes
    once per tag. Lines without a time tag (ID tags such as "[ar:Artist]")
    are left out."""
    timed = []
    for n, line in enumerate(synced_lyrics.splitlines()):
        tags = _LRC_LINE_TAGS_RE.match(line)
        text = " ".join(_LRC_TIME_TAG_RE.sub(" ", line).split())
        if not tags or not text:
            continue
        for minutes, seconds in _LRC_LINE_TAG_RE.findall(tags.group(0)):
            timed.append((int(minutes) * 60 + float(seconds), n, text))
    return "\n".join(text for _, _, text in sorted(timed))


def _length_off(entry: dict, duration: Optional[float]) -> float:
    """How many seconds the length of a found version is off the audio."""
    if not duration:
        return 0.0
    length = entry.get("duration")
    if isinstance(length, (int, float)) and length > 0:
        return abs(length - duration)
    return float("inf")


def lyric_words(info: Optional[LyricsInfo]) -> int:
    """How many words found lyrics have: of their synced or of their plain
    text, whichever has more."""
    if info is None:
        return 0
    synced = len(lrc_text(info.synced_lyrics).split()) if _has_text(info.synced_lyrics) else 0
    plain = len(info.plain_lyrics.split()) if _has_text(info.plain_lyrics) else 0
    return max(synced, plain)


def too_few_words(info: LyricsInfo, duration: Optional[float]) -> bool:
    """Whether found lyrics have far too few words for the audio
    (``duration`` seconds; else the length of their version): fewer than
    ``SPARSE_WORDS_PER_MINUTE`` per minute. Such lyrics hold a few lines of
    the song only, and their times often crowd together. False for audio
    shorter than ``_SPARSE_MIN_LENGTH_S`` or of an unknown length."""
    length = duration or info.duration
    if not (isinstance(length, (int, float)) and length >= _SPARSE_MIN_LENGTH_S):
        return False
    return lyric_words(info) < SPARSE_WORDS_PER_MINUTE * length / 60


def _minutes(seconds: Optional[float]) -> str:
    """A length as "m:ss", or "an unknown length"."""
    if not (isinstance(seconds, (int, float)) and seconds > 0):
        return "an unknown length"
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _find_on_second_db(artist: str, title: str, duration: Optional[float],
                       synced_only: bool = False) -> Optional[LyricsInfo]:
    """The lyrics of a song in the second lyrics database (``SECOND_DB_NAME``).

    Searched by artist and title, then by the title without its additions
    (see ``strip_title_additions``) when that finds nothing. Then a cover
    is searched without its mark, by our artist and by the one the mark
    names (see ``cover_of``), and a stripped-down recording without that
    mark (see ``title_without_stripped``). Only results whose artist and
    title match are taken (see ``_names_match``), for a cover after them
    also those of the song it covers (see ``_is_covered_song``); their
    lyrics are fetched in the order of their length's distance to the audio
    (``duration``), at most ``_SECOND_DB_LYRICS`` per search and each
    version once; the first with lyrics wins, with ``synced_only`` the
    first with synced ones.
    """
    stripped = strip_title_additions(title)
    cover = cover_of(title)
    covered = _covered_song(artist, cover)
    plain_title = title_without_stripped(title)
    # (what is searched, title of the song)
    searches, seen = [], set()
    for searched, song in ((f"{artist} {title}", title), (f"{artist} {stripped}", stripped),
                           (f"{artist} {cover[0]}" if cover else "", title),
                           (f"{cover[1]} {cover[0]}" if cover and cover[1] else "", title),
                           (f"{artist} {plain_title}" if plain_title else "", plain_title)):
        if searched and searched.casefold() not in seen:
            seen.add(searched.casefold())
            searches.append((searched, song))
    asked = set()  # the versions whose lyrics were fetched: a later search may list them again
    for searched, song in searches:
        params = urllib.parse.urlencode({"s": searched, "type": 1, "limit": _SECOND_DB_RESULTS})
        # One try: an unreachable second database must not hold up the lookup
        data = _get_json(f"{_SECOND_DB_URL}/search/get?{params}", attempts=1)
        if data is None:
            # Unreachable, or no answer: not asked again
            return None
        result = data.get("result") if isinstance(data, dict) else None
        songs = result.get("songs") if isinstance(result, dict) else None
        if not isinstance(songs, list):
            continue
        listed = [version for version in map(_second_db_version, songs)
                  if version is not None and version["id"] not in asked]
        listed.sort(key=lambda version: _length_off(version, duration))
        versions = [version for version in listed if _names_match(version, artist, song)]
        if covered is not None and song in (title, stripped):
            # After the versions under the title of the cover: the song it covers
            versions += [version for version in listed
                         if version not in versions and _is_covered_song(version, *covered)]
        for version in versions[:_SECOND_DB_LYRICS]:
            asked.add(version["id"])
            synced, plain = _second_db_lyrics(version)
            if synced or (plain and not synced_only):
                return LyricsInfo(
                    track_name=version["trackName"],
                    artist_name=version["artistName"],
                    plain_lyrics=plain,
                    synced_lyrics=synced,
                    duration=version["duration"],
                    source=SECOND_DB_NAME,
                )
    return None


def _second_db_version(song) -> Optional[dict]:
    """A search result of the second database as LRCLIB lists a version
    (trackName, artistName, duration in seconds), with its id."""
    if not isinstance(song, dict) or not isinstance(song.get("id"), int):
        return None
    artists = [artist.get("name") for artist in song.get("artists") or ()
               if isinstance(artist, dict) and isinstance(artist.get("name"), str)]
    length = song.get("duration")
    return {
        "id": song["id"],
        "trackName": song.get("name") if isinstance(song.get("name"), str) else "",
        "artistName": ", ".join(artists),
        "duration": length / 1000 if isinstance(length, (int, float)) and length > 0 else None,
    }


def _second_db_lyrics(version: dict) -> tuple[Optional[str], Optional[str]]:
    """The synced and the plain lyrics of a version of a song in the second
    database (see ``_second_db_version``). (None, None) when it has none or
    knows the song as instrumental; lyrics without time tags are plain
    lyrics only.

    Its lyrics often begin with credit lines (writers, producers, players:
    see ``_is_credit``) and with the title and the artist of the song, and
    may end with more credits; these lines are left out. Only at the start
    and at the end: a sung line may look like a credit too. Look-alike
    letters in the lyrics are folded (see ``fold_lookalike_letters``).
    """
    params = urllib.parse.urlencode({"id": version["id"], "lv": 1})
    data = _get_json(f"{_SECOND_DB_URL}/song/lyric?{params}", attempts=1)
    if not isinstance(data, dict) or data.get("nolyric") or data.get("uncollected"):
        return None, None
    lrc = data.get("lrc")
    text = lrc.get("lyric") if isinstance(lrc, dict) else None
    if not _has_text(text):
        return None, None
    lines = []  # (the line, its text)
    # Its texts may hold HTML entities ("let&apos;s")
    for line in html.unescape(text).splitlines():
        line = _COLON_TIME_TAG_RE.sub(r"[\1:\2.\3]", line).strip()
        words = " ".join(_LRC_TIME_TAG_RE.sub(" ", line).split())
        if _INSTRUMENTAL_MARK in words:
            return None, None
        if words and not _LRC_ID_TAG_RE.match(words):
            lines.append((line, words))

    # Credits, and the title and the artist, before the lyrics: these at
    # their very start, or the artist right after the title ("Title" at
    # 0:01, "Artist" at 0:15). A first sung line that is just the title
    # stays. When the lines of the shape of a credit there hold a credit,
    # all of them are credits.
    run = 0
    while run < len(lines) and (_is_credit(lines[run][1], loose=True)
                                or _names_of_song(lines[run][1], version) is not None):
        run += 1
    credits = any(_is_credit(words) for _, words in lines[:run])
    named = None
    while lines:
        line, words = lines[0]
        if not _is_credit(words, loose=credits):
            names = _names_of_song(words, version)
            time_s = _line_time(line)
            if names is None or not ((time_s is not None and time_s <= _HEADER_S)
                                     or (named, names) == ("title", "artist")):
                break
            named = names if named is None else "both"
        lines.pop(0)
    # Credits after the lyrics: the lines of the shape of one at their end,
    # when one of them is a credit
    tail = 0
    while tail < len(lines) and _is_credit(lines[-1 - tail][1], loose=True):
        tail += 1
    if any(_is_credit(words) for _, words in lines[len(lines) - tail:]):
        del lines[len(lines) - tail:]
    synced = fold_lookalike_letters("\n".join(line for line, _ in lines))
    if _has_text(lrc_text(synced)):
        return synced, lrc_text(synced)
    plain = fold_lookalike_letters("\n".join(words for _, words in lines))
    return None, plain if _has_text(plain) else None


def _is_credit(text: str, loose: bool = False) -> bool:
    """Whether a line of lyrics of the second database looks like a credit
    of the writers, producers or players: "<role> : <names>", the role in
    Chinese, or one to four capitalized English words ("Guitar", "Mixing
    Engineer", "Lyrics by") before names that do not start in lower case
    (unlike a sung "Listen: this is ..."). A role in Chinese must end like
    one (see ``_CREDIT_ROLE_ENDS``); ``loose``, among other credits, it
    only needs to be short. Also an English credit without a colon:
    "Mastered at ...", "Published by ..." (see ``_CREDIT_PHRASE_RE``)."""
    if _CREDIT_PHRASE_RE.match(text):
        return True
    match = _CREDIT_RE.match(text)
    if match is None:
        return False
    role, names = match.group(1).strip(), match.group(2)
    if _CJK_RE.search(role):
        # "作词", "制作人", "吉他", "次中音萨克斯 / 长笛" - not a sung "我说：..." ("I said: ...")
        return ((loose and len(role) <= _LOOSE_ROLE_CHARS)
                or any(part.strip()[-1:] in _CREDIT_ROLE_ENDS for part in role.split("/")))
    return bool(_ENGLISH_ROLE_RE.fullmatch(role)) and not names[0].islower()


def _names_of_song(words: str, version: dict) -> Optional[str]:
    """Whether a line of lyrics names the song of ``version``: "title", "artist",
    "both" ("Title - Artist", "Artist - Title", also with the original
    artist of a cover), else None."""
    title, artist = version["trackName"], version["artistName"]
    if titles_match(words, title):
        return "title"
    if _artist_matches(words, artist):
        return "artist"
    halves = words.split(" - ", 1)
    if len(halves) == 2 and any(titles_match(half, title) for half in halves):
        return "both"
    return None


def _line_time(line: str) -> Optional[float]:
    """The time of an LRC line (of its first time tag), None without one."""
    tag = _LRC_LINE_TAG_RE.match(line)
    return int(tag.group(1)) * 60 + float(tag.group(2)) if tag else None
