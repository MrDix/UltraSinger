"""Lyrics from elsewhere than the lyrics databases.

**Your own lyrics.** A text file with the lyrics of the song: plain lines, or
LRC lines with times (``[01:23.45] line``). They are used instead of a
lookup, for a song no database knows or one whose found lyrics are wrong.
The file is given with ``--lyrics_file``, or lies next to the audio file
under its name (``Song.mp3`` and ``Song.lrc`` or ``Song.txt``; a ``.txt``
file has to look like lyrics, see ``sidecar_lyrics_file``).

**Lyrics sources of your own.** A folder of small Python files, each with a
function::

    def find_lyrics(artist: str, title: str, duration: float | None):
        ...
        return lyrics  # text (plain or LRC), a dict, or None

They are asked for the lyrics the lyrics databases do not have, or, with
``BEFORE_DATABASES = True`` in the file, before them. See ``LyricsSource``
for what a source may return.

A lyrics source is a program of the user's own choice that runs with the
user's rights: the folder is only ever the one the user names (by default
one in the settings folder), never a folder of a song.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, gold_highlighted
from modules.lookalike_letters import fold_lookalike_letters
from modules import lrclib_client
from modules.lrclib_client import (
    SAME_TEXT_SHARE,
    LyricsInfo,
    artist_matches,
    lrc_text,
    other_recording,
    with_dot_time_tags,
    words_shared,
)
from modules.name_matching import titles_match

# The source of lyrics that the user gave (``LyricsInfo.source``)
OWN_LYRICS_SOURCE = "your lyrics file"
# A lyrics file is text: a larger file is something else
MAX_LYRICS_FILE_BYTES = 1_000_000
# The file names of lyrics next to an audio file, after its name
SIDECAR_SUFFIXES = (".lrc", ".txt")
# A text file next to an audio file is taken for its lyrics when it looks like
# lyrics: some lines, none of them a paragraph (it could be notes on the song)
MIN_SIDECAR_LINES = 4
MAX_SIDECAR_LINE_CHARS = 300
# How long a lyrics source may take to answer (it asks a server, as a rule)
SOURCE_TIMEOUT_S = 20.0
# A lyrics source in the file sets this to be asked before the lyrics databases
BEFORE_DATABASES = "BEFORE_DATABASES"

# The time tag of an LRC line: "[01:23.45]", "[1:23]" ("[01:23:45]" is written with a dot first)
_TIME_TAG_RE = re.compile(r"^\s*\[\d+:\d+(?:\.\d+)?\]")
# A line that is only an ID tag of an LRC file: "[ar:Artist]", "[ti:Title]", "[by:Tool]"
_ID_TAG_LINE_RE = re.compile(r"^\s*\[[A-Za-z]+:[^\]]*\]\s*$")
# Lines that only an UltraStar chart has: its header tags and its notes
_CHART_HEADER_RE = re.compile(r"^#(?:TITLE|ARTIST|BPM|GAP|MP3|AUDIO)\s*:", re.IGNORECASE)
_CHART_NOTE_RE = re.compile(r"^[:*FRG]\s+-?\d+\s+\d+\s+-?\d+(?:\s|$)")


# The settings folder of the user (as the GUI's)
_SETTINGS_DIR = Path.home() / ".ultrasinger"


def default_sources_dir() -> Path:
    """The folder of the user's lyrics sources, in the settings folder."""
    return _SETTINGS_DIR / "lyrics_sources"


def is_ultrastar_chart(text: str) -> bool:
    """Whether a text is an UltraStar chart (its lyrics lie in its note lines)."""
    lines = text.splitlines()[:200]
    return (any(_CHART_HEADER_RE.match(line) for line in lines)
            or sum(1 for line in lines if _CHART_NOTE_RE.match(line)) >= 3)


def lyrics_from_text(text: str, name: str, source: str = OWN_LYRICS_SOURCE, trusted: bool = True) -> LyricsInfo:
    """The lyrics that a text holds.

    They are synced lyrics when most of its lines start with a time tag
    (LRC), else plain ones; the ID tags of an LRC file (``[ar:Artist]``) are
    no lines of the lyrics. Time tags with a colon before the hundredths
    (``[01:23:45]``) are written with a dot. ``name`` is shown where found
    lyrics are named. Look-alike letters of other scripts are replaced as
    in found lyrics.

    Raises:
        ValueError: The text has no lyrics, or it is an UltraStar chart.
    """
    if is_ultrastar_chart(text):
        raise ValueError("it is an UltraStar chart, not a lyrics text")
    text = fold_lookalike_letters(text.replace("\r\n", "\n").replace("\r", "\n")).strip("\n")
    text = with_dot_time_tags(text)
    kept = [line for line in text.split("\n") if not _ID_TAG_LINE_RE.match(line)]
    lines = [line for line in kept if line.strip()]
    if not lines:
        raise ValueError("it has no text")
    timed = sum(1 for line in lines if _TIME_TAG_RE.match(line))
    if 2 * timed >= len(lines):
        plain = lrc_text(text)
        if not plain.strip():
            raise ValueError("its lines have times, but no text")
        return LyricsInfo(track_name=name, artist_name="", plain_lyrics=plain, synced_lyrics=text,
                          source=source, trusted=trusted)
    return LyricsInfo(track_name=name, artist_name="", plain_lyrics="\n".join(kept).strip("\n"),
                      source=source, trusted=trusted)


def read_lyrics_file(path: str | os.PathLike) -> LyricsInfo:
    """The lyrics of a file of the user (see ``lyrics_from_text``).

    The text is read as UTF-8 (also UTF-16 with a byte order mark), else in
    the Western European encoding of Windows.

    Raises:
        OSError: The file cannot be read.
        ValueError: It is too large, has no lyrics, or is an UltraStar chart.
    """
    path = Path(path)
    if path.stat().st_size > MAX_LYRICS_FILE_BYTES:
        raise ValueError(f"it is larger than {MAX_LYRICS_FILE_BYTES // 1000} kB")
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16", errors="replace")
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1252", errors="replace")
    return lyrics_from_text(text, path.name)


def sidecar_lyrics_file(media_path: str | os.PathLike) -> Optional[str]:
    """The path of the lyrics file next to an audio or video file (see
    ``sidecar_lyrics``), or None."""
    found = sidecar_lyrics(media_path)
    return found[0] if found else None


def sidecar_lyrics(media_path: str | os.PathLike) -> Optional[tuple[str, LyricsInfo]]:
    """The lyrics file that lies next to an audio or video file under its
    name (``Song.lrc``, then ``Song.txt``) and its lyrics, or None. A text
    file that is an UltraStar chart is none, nor is the file itself, nor a
    ``.txt`` file that does not look like lyrics (fewer than
    ``MIN_SIDECAR_LINES`` lines, or one longer than
    ``MAX_SIDECAR_LINE_CHARS``)."""
    media = Path(media_path)
    for suffix in SIDECAR_SUFFIXES:
        candidate = media.with_suffix(suffix)
        if candidate == media or not candidate.is_file():
            continue
        try:
            lyrics = read_lyrics_file(candidate)
        except (OSError, ValueError):
            continue
        lines = [line for line in (lyrics.plain_lyrics or "").splitlines() if line.strip()]
        if suffix == ".txt" and (len(lines) < MIN_SIDECAR_LINES
                                 or any(len(line) > MAX_SIDECAR_LINE_CHARS for line in lines)):
            continue
        return str(candidate), lyrics
    return None


@dataclass
class LyricsSource:
    """A lyrics source of the user: a Python file with ``find_lyrics``.

    ``find_lyrics(artist, title, duration)`` gets the artist and the title
    of the song and the length of its audio in seconds (None when it is not
    known). It returns None when it has no lyrics, the lyrics as text (plain
    lines or LRC lines with times), or a dict with ``"lyrics"`` and, when
    the source knows them, ``"artist"``, ``"title"`` (lyrics of a song with
    another name are not taken) and ``"duration"`` (the length in seconds of
    the version the lyrics are of).

    ``NAME`` in the file names the source where found lyrics are reported
    (default: the file name), ``BEFORE_DATABASES = True`` lets it be asked
    before the lyrics databases instead of for what they do not have.

    ``failed`` is set when the source failed or did not answer in time: it
    is not asked again in this run (a conversion looks lyrics up twice, and
    would wait for it once more).
    """
    name: str
    find: Callable
    before_databases: bool
    path: str
    failed: bool = False


def _within_time(work: Callable, timeout_text: str, error_text: str):
    """``(True, work())`` when the work of a lyrics source is done within
    ``SOURCE_TIMEOUT_S``, else ``(False, None)``: it failed (reported with
    ``error_text``) or takes too long (``timeout_text``; it is left behind,
    in a thread that does not keep the program from ending)."""
    result: dict = {}

    def run():
        try:
            result["value"] = work()
        # (also a source that ends the program, sys.exit(): it only ends this thread)
        except BaseException as error:  # noqa: BLE001 - a broken source of the user must not stop the conversion
            result["error"] = error

    thread = threading.Thread(target=run, name="lyrics source", daemon=True)
    thread.start()
    thread.join(SOURCE_TIMEOUT_S)
    if thread.is_alive():
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} {timeout_text} within {SOURCE_TIMEOUT_S:.0f} s")
        return False, None
    if "error" in result:
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} {error_text} ({result['error']!r})")
        return False, None
    return True, result.get("value")


def load_lyrics_sources(folder: str | os.PathLike | None) -> list[LyricsSource]:
    """The lyrics sources in a folder: its ``*.py`` files with a
    ``find_lyrics`` function, by name (not those that start with ``_``). A
    file that cannot be loaded, or takes longer than ``SOURCE_TIMEOUT_S`` to
    load, is reported and left out."""
    if not folder or not os.path.isdir(folder):
        return []
    sources = []
    for path in sorted(Path(folder).glob("*.py")):
        if path.name.startswith("_"):
            continue
        # A module of its own name, known to Python while it is loaded and after (dataclasses
        # and type hints look their module up); the same file name in two folders is two modules
        module_name = (f"ultrasinger_lyrics_source_{path.stem}_"
                       f"{hashlib.sha1(str(path.resolve()).encode('utf-8')).hexdigest()[:8]}")
        def load(path=path, module_name=module_name):
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        loaded, module = _within_time(
            load, f"the lyrics source {blue_highlighted(path.name)} did not load",
            f"the lyrics source {blue_highlighted(path.name)} could not be loaded")
        if not loaded:
            sys.modules.pop(module_name, None)
            continue
        find = getattr(module, "find_lyrics", None)
        if not callable(find):
            print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} the lyrics source "
                  f"{blue_highlighted(path.name)} has no find_lyrics function")
            continue
        name = getattr(module, "NAME", None)
        sources.append(LyricsSource(name=name.strip() if isinstance(name, str) and name.strip() else path.stem,
                                    find=find, before_databases=getattr(module, BEFORE_DATABASES, False) is True,
                                    path=str(path)))
    return sources


def _answer(source: LyricsSource, artist: str, title: str, duration: Optional[float]):
    """What a source answers, None when it fails or takes too long: it is
    marked as failed then (see ``LyricsSource``)."""
    answered, answer = _within_time(lambda: source.find(artist, title, duration),
                                    f"the lyrics source {blue_highlighted(source.name)} did not answer",
                                    f"the lyrics source {blue_highlighted(source.name)} failed")
    if not answered:
        source.failed = True
    return answer


def _names_fit(answer: dict, artist: str, title: str) -> bool:
    """Whether the names a source gives with its lyrics are those of the song
    (a source that gives none is taken at its word)."""
    found_artist, found_title = answer.get("artist"), answer.get("title")
    if isinstance(found_title, str) and found_title.strip() and not titles_match(found_title, title):
        return False
    return not (isinstance(found_artist, str) and found_artist.strip()) or artist_matches(found_artist, artist)


def lyrics_of_source(source: LyricsSource, artist: str, title: str,
                     duration: Optional[float]) -> Optional[LyricsInfo]:
    """The lyrics a source has of a song, or None (see ``LyricsSource``)."""
    answer = _answer(source, artist, title, duration)
    if isinstance(answer, str):
        answer = {"lyrics": answer}
    if not isinstance(answer, dict) or not isinstance(answer.get("lyrics"), str):
        if answer is not None:
            print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} the lyrics source "
                  f"{blue_highlighted(source.name)} answered neither lyrics nor None")
        return None
    if not _names_fit(answer, artist, title):
        print(f"{ULTRASINGER_HEAD} The lyrics of {blue_highlighted(source.name)} are of another song "
              f"({answer.get('artist') or '?'} - {answer.get('title') or '?'}): not taken")
        return None
    try:
        lyrics = lyrics_from_text(answer["lyrics"], title, source=source.name, trusted=False)
    except ValueError:
        return None
    lyrics.artist_name = artist
    length = answer.get("duration")
    if isinstance(length, (int, float)) and not isinstance(length, bool) and length > 0:
        lyrics.duration = float(length)
    return lyrics


def lyrics_of_sources(sources: list[LyricsSource], artist: str, title: str, duration: Optional[float],
                      instead_of: Optional[LyricsInfo] = None) -> Optional[LyricsInfo]:
    """The lyrics of the first of the sources that has some. A source that
    failed before in this run is not asked (see ``LyricsSource``).

    With ``instead_of``, the plain lyrics a lyrics database has: only synced
    lyrics of the same text (``SAME_TEXT_SHARE`` of the words of the plain
    ones) that are not of another recording (see ``other_recording``): than
    the audio, or, when its length is not known, than the version of the
    plain lyrics.
    """
    recording = duration
    if not recording and instead_of is not None and isinstance(instead_of.duration, (int, float)):
        recording = instead_of.duration
    for source in sources:
        if source.failed:
            continue
        print(f"{ULTRASINGER_HEAD} Asking the lyrics source {blue_highlighted(source.name)}")
        lyrics = lyrics_of_source(source, artist, title, duration)
        if lyrics is None:
            continue
        if instead_of is not None:
            if not lyrics.synced_lyrics or other_recording(lyrics.duration, recording):
                continue
            if words_shared(instead_of.plain_lyrics or "", lyrics.plain_lyrics or "") < SAME_TEXT_SHARE:
                print(f"{ULTRASINGER_HEAD} The synced lyrics of {blue_highlighted(source.name)} are another "
                      f"text than the plain ones: not taken")
                continue
        return lyrics
    return None


def report_own_lyrics(lyrics: LyricsInfo, path: str | os.PathLike) -> None:
    """Report the lyrics of the user's file (the GUI queue reads this line,
    as the one of found lyrics: see queue_manager)."""
    kinds = f"plain ({len((lyrics.plain_lyrics or '').split())} words)"
    if lyrics.synced_lyrics:
        kinds += ", synced"
    print(f"{ULTRASINGER_HEAD} Found lyrics in your file {blue_highlighted(os.path.basename(path))}: [{kinds}]")


_loaded_sources: dict[str, list[LyricsSource]] = {}


def lyrics_sources(folder: str | os.PathLike | None) -> list[LyricsSource]:
    """The lyrics sources of a folder, loaded once (see ``load_lyrics_sources``)."""
    key = os.path.abspath(folder) if folder else ""
    if key not in _loaded_sources:
        _loaded_sources[key] = load_lyrics_sources(folder)
    return _loaded_sources[key]


def find_song_lyrics(artist: str, title: str, duration: Optional[float] = None,
                     own_lyrics: Optional[LyricsInfo] = None, own_lyrics_file: str | os.PathLike | None = None,
                     sources_dir: str | os.PathLike | None = None, lookup: bool = True) -> Optional[LyricsInfo]:
    """The lyrics of a song, from wherever they are to be had.

    1. The user's ``own_lyrics`` (read from ``own_lyrics_file`` before, see
       ``read_lyrics_file``): they are the lyrics of the song, and nothing is
       looked up.
    2. With ``lookup``: the lyrics sources in ``sources_dir`` that are asked
       before the lyrics databases, the lyrics databases (see
       ``lrclib_client.search_lyrics``), then the other lyrics sources: for
       lyrics when the databases have none, for synced lyrics when they have
       only plain ones (see ``lyrics_of_sources``).

    ``duration`` is the length of the audio in seconds, if known. Returns
    None when there are no lyrics.
    """
    if own_lyrics is not None:
        report_own_lyrics(own_lyrics, own_lyrics_file or own_lyrics.track_name)
        return own_lyrics
    if not lookup or not (artist or "").strip() or not (title or "").strip():
        return None
    sources = lyrics_sources(sources_dir)
    found = lyrics_of_sources([source for source in sources if source.before_databases], artist, title, duration)
    if found is not None:
        lrclib_client.report_found_lyrics(found, duration)
        return found
    found = lrclib_client.search_lyrics(artist, title, duration=duration)
    plain_only = found is not None and not found.synced_lyrics and bool((found.plain_lyrics or "").strip())
    if found is not None and not plain_only:
        # synced lyrics, or a song the databases know as instrumental
        return found
    other = lyrics_of_sources([source for source in sources if not source.before_databases], artist, title,
                              duration, instead_of=found)
    if other is not None:
        lrclib_client.report_found_lyrics(other, duration)
        return other
    return found
