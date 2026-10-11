"""Comparing song titles and artist names from different sources.

A video title, MusicBrainz and LRCLIB often write the same song differently:
other case and punctuation, additions like "(Official Video)" or
"- US Version", featured guests.
"""

import re
import string

from Levenshtein import ratio

# A found song is only trusted when both its artist and its title are at
# least this similar to what we searched for. Below that, the search hit is
# a different song.
MIN_MATCH_SIMILARITY = 0.8
# Similarity assigned when one name is the other plus only qualifier words
# (e.g. "Title" vs. "Title Full HD"): a match, but ranked below exact or
# near-exact matches.
CONTAINED_SIMILARITY = 0.85
# Words that only describe a release/upload and never distinguish songs or
# artists. Extra words outside this set (e.g. a second name) prevent a match.
QUALIFIER_WORDS = frozenset(
    "official video music audio lyric lyrics visualizer hd hq 4k full live version edit radio single "
    "album remaster remastered mix remix explicit clean acoustic original extended mono stereo".split()
)
_BRACKETS_RE = re.compile(r"[(\[{][^)\]}]*[)\]}]")
_FEATURING_RE = re.compile(r"\s+(?:feat\.?|ft\.?|featuring)\s+.*$", re.IGNORECASE)
# Video titles put more after the song title: "Title - US Version",
# "Title | Label Name", "Title // Live"
_SEPARATOR_RE = re.compile(r"\s+(?:-|–|—|\||//)\s+")
# A trailing "Official Video", "Official Music Video", ... without brackets
_OFFICIAL_RE = re.compile(
    r"\s+official(?:\s+(?:music|lyrics?|hd|hq|4k|video|audio|visuali[sz]er|clip|videoclip))*\s*$",
    re.IGNORECASE,
)
# A trailing quality mark: "HD", "HQ", "4K", "1080p"
_QUALITY_RE = re.compile(r"(?:\s+(?:hd|hq|4k|8k|\d{3,4}p))+\s*$", re.IGNORECASE)


def _clean(s: str) -> str:
    return s.translate(str.maketrans('', '', string.punctuation)).lower().strip()


def normalize_name(s: str, artist: bool) -> str:
    """Drop bracketed additions and, for artists, a featured-guest suffix."""
    s = _BRACKETS_RE.sub(" ", s or "")
    if artist:
        s = _FEATURING_RE.sub("", s)
    return " ".join(_clean(s).split())


def similarity(a: str, b: str, artist: bool = False) -> float:
    """Similarity of two titles (or artist names) in [0, 1].

    Case, punctuation, bracketed additions ("(Official Video)", "(Live 2023)")
    and, for artists, "feat. X" are ignored. If one name is the other plus
    only qualifier words (see QUALIFIER_WORDS, plus numbers), it still counts
    as a match; any other extra word does not - "Nova" is not "Nova Lights".
    """
    a, b = normalize_name(a, artist), normalize_name(b, artist)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    words_a, words_b = a.split(), b.split()
    # Different numbers on both sides mean different songs ("Part 1" vs.
    # "Part 2"), however similar the rest; a number on one side only (e.g. a
    # year) is fine and handled below.
    numbers_a = sorted(w for w in words_a if w.isdigit())
    numbers_b = sorted(w for w in words_b if w.isdigit())
    if numbers_a and numbers_b and numbers_a != numbers_b:
        return 0.0
    score = ratio(a, b)
    shorter, longer = (words_a, words_b) if len(words_a) <= len(words_b) else (words_b, words_a)
    extra = list(longer)
    for w in shorter:
        if w not in extra:
            return score
        extra.remove(w)
    if all(w in QUALIFIER_WORDS or w.isdigit() for w in extra):
        score = max(score, CONTAINED_SIMILARITY)
    return score


def strip_title_additions(title: str) -> str:
    """The song title without the additions video titles often carry.

    Drops bracketed parts ("(Official Video)", "[HD]", "(feat. Guest)"),
    everything after a separator (" - US Version", " | Label Name"), a
    featured guest ("feat. Guest"), a trailing "Official Video" and a
    trailing quality mark ("HD", "4K"). A title that would be left empty
    is returned unchanged.
    """
    s = _BRACKETS_RE.sub(" ", title or "")
    s = _SEPARATOR_RE.split(s, maxsplit=1)[0]
    s = _FEATURING_RE.sub("", s)
    s = _OFFICIAL_RE.sub("", s)
    s = _QUALITY_RE.sub("", s)
    return " ".join(s.split()) or (title or "").strip()
