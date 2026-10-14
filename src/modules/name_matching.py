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
_LEADING_BRACKETS_RE = re.compile(r"^\s*[(\[{][^)\]}]*[)\]}]\s*")
# A leading article is often left out: "Some Band" is "The Some Band". In
# titles only the English ones: "Die" or "Las" may be a word of the title.
_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+(?=\S)")
_LEADING_ARTIST_ARTICLE_RE = re.compile(r"^(?:the|a|an|der|die|das|le|la|les|los|las|el)\s+(?=\S)")
_FEATURING_RE = re.compile(r"\s+(?:feat\.?|ft\.?|featuring)\s+.*$", re.IGNORECASE)
_AMPERSAND_RE = re.compile(r"\s*[&+]\s*")
# What video titles and releases add to a song title, in brackets or after a
# dash: the kind of upload, a version, a featured guest. Such a part starts
# with one of _ADDITION_START (whatever follows), is one of _AMBIGUOUS_START -
# words that are common in song titles too - alone or followed by one of
# _AMBIGUOUS_FOLLOW ("Live at ...", "Radio Edit", but not "Live Forever"), or
# ends with one of _ADDITION_END. Any other part ("(Part 2)", "(Reprise)",
# "(I Can't Get No)") can tell two songs apart and is kept.
_OFFICIAL = r"official|offizielles?|oficial|officiel(?:le)?"
_ADDITION_START = (
    rf"{_OFFICIAL}|lyrics?|letra|legendad[oa]|subtitulad[oa]|visuali[sz]er|videoclip|hd|hq|4k|8k|\d{{3,4}}p"
    r"|mv|m/v|explicit|acoustic|unplugged|mono|stereo|remaster(?:ed)?|\d{4}|feat\.?|ft\.?|featuring"
    r"|recorded|bonus|deluxe|demo|karaoke|slowed|sped|nightcore"
)
_AMBIGUOUS_START = r"live|us|uk|radio|single|album|extended|original|clean|vid[eé]o|music|audio|clip|from"
_AMBIGUOUS_FOLLOW = (
    rf"version|edit|mix|cut|vid[eé]o|single|release|recording|performance|sessions?|{_OFFICIAL}"
    r"|at|in|on|from|@|only|acoustic|unplugged|studio|concert|tour|lounge|\d{4}"
    r"|(?:the\s+)?(?:album|movie|film|motion\s+picture|soundtrack|series|musical|show|game)"
)
_ADDITION_END = (
    r"version|edit|mix|remix|remaster(?:ed)?|cut|vid[eé]o|audio|lyrics?|visuali[sz]er|hd|4k"
    r"|edition|demo|karaoke|instrumental|sessions?|soundtrack|recording|acoustic|performance|playthrough"
)


def _addition_re(start: str) -> re.Pattern:
    return re.compile(
        rf"(?:{start})(?:\W.*)?"
        rf"|(?:{_AMBIGUOUS_START})(?:\s+(?:{_AMBIGUOUS_FOLLOW})(?:\W.*)?|\s*[\"“'‘].*)?"
        rf"|(?:.*\W)?(?:{_ADDITION_END})",
        re.IGNORECASE | re.DOTALL)


_ADDITION_RE = _addition_re(_ADDITION_START)
# In brackets, a guest also comes with "with": "(with Guest)"
_BRACKET_ADDITION_RE = _addition_re(_ADDITION_START + r"|with|w/")
# What channel names add to an artist's name: "ArtistVEVO", "Artist - Topic"
_CHANNEL_ADDITION_RE = re.compile(r"(?:\s*-\s*topic|\s*vevo|\s+official(?:\s+(?:channel|music))?)\s*$",
                                  re.IGNORECASE)
# Video titles put more after the song title: "Title - US Version",
# "Title | Label Name", "Title // Live"
_SEPARATOR_RE = re.compile(r"\s+(-|–|—|\||//)\s+")
# A trailing "Official Video", "Official Music Video", ... without brackets
_OFFICIAL_RE = re.compile(
    r"\s+official(?:\s+(?:music|lyrics?|hd|hq|4k|video|audio|visuali[sz]er|clip|videoclip))*\s*$",
    re.IGNORECASE,
)
# A trailing quality mark: "HD", "HQ", "4K", "1080p"
_QUALITY_RE = re.compile(r"(?:\s+(?:hd|hq|4k|8k|\d{3,4}p))+\s*$", re.IGNORECASE)
# What a video title puts after a dash is a description of the video, not a
# part of the song title, from this many words on (see title_before_dash)
DESCRIPTION_WORDS = 3
# A Roman number up to ten ("I" to "X")
_ROMAN = r"(?:i{1,3}|iv|vi{0,3}|ix|x)"
# A word that names a part of a work ("Part Two", "Vol. 3", "Chapter II",
# "The Second Movement", "Part2", "No. V")
_PART_WORD_RE = re.compile(
    rf"\b(?:part|pt|vol|volume|chapter|act|movement|episode|book)\b|\b(?:part|pt|vol|no)\.?\s*\d{{1,3}}\b"
    # a Roman number after "No" only with the period: "No I won't" is a sentence
    rf"|\b(?:part|pt|vol)\.?\s*{_ROMAN}\b|\bno\.\s*{_ROMAN}\b",
    re.IGNORECASE)
# A part or volume number standing alone ("3", "II"), not a year. Of the
# Roman numbers, "I", "V" and "X" only count at the start: elsewhere they are
# words ("I"), or join names ("Singer x Rapper").
_PART_NUMBER_RE = re.compile(rf"^\s*{_ROMAN}\b|\b(?:\d{{1,3}}|ii|iii|iv|vi|vii|viii|ix)\b|" + _PART_WORD_RE.pattern,
                             re.IGNORECASE)
# A dashed part that is only a part number ("3", "I", "IV")
_BARE_PART_NUMBER_RE = re.compile(rf"\s*(?:\d{{1,3}}|{_ROMAN})\s*", re.IGNORECASE)


def _clean(s: str) -> str:
    return s.translate(str.maketrans('', '', string.punctuation)).lower().strip()


def normalize_name(s: str, artist: bool) -> str:
    """Drop bracketed additions, a leading article and, for artists, a
    featured-guest suffix."""
    s = _BRACKETS_RE.sub(" ", s or "")
    if artist:
        s = _FEATURING_RE.sub("", s)
        # "A & B", "A + B" and "A and B" are the same name
        s = _AMPERSAND_RE.sub(" and ", s)
    s = " ".join(_clean(s).split())
    return (_LEADING_ARTIST_ARTICLE_RE if artist else _LEADING_ARTICLE_RE).sub("", s)


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
        return max(score, CONTAINED_SIMILARITY)
    # One name is the other plus a word that is no qualifier ("Nova Lights",
    # "Some Long Title Reprise"): another artist or song, however long the
    # shared part (the character similarity alone would pass long names)
    return min(score, MIN_MATCH_SIMILARITY / 2)


def _is_addition(part: str, in_brackets: bool = False) -> bool:
    part = part.strip()
    return not part or (_BRACKET_ADDITION_RE if in_brackets else _ADDITION_RE).fullmatch(part) is not None


def strip_title_additions(title: str) -> str:
    """The song title without the additions video titles often carry.

    Drops bracketed additions ("(Official Video)", "[HD]", "(US Version)",
    "(feat. Guest)", "(with Guest)", "(2011 Remaster)") and empty brackets,
    an addition after a dash (" - US Version", " - Live at Some Place"),
    everything after " | " or " // " (" | Label Name"), a featured guest
    ("feat. Guest"), a trailing "Official Video" and a trailing quality mark
    ("HD", "4K"). Other bracketed or dashed parts ("(Part 2)", "- Reprise",
    "(Live Forever)") stay, they can tell two songs apart. A title that would
    be left empty is returned unchanged.
    """
    s = _BRACKETS_RE.sub(lambda m: " " if _is_addition(m.group(0)[1:-1], in_brackets=True) else m.group(0),
                         title or "")
    for separator in _SEPARATOR_RE.finditer(s):
        rest = s[separator.end():]
        if separator.group(1) not in ("|", "//"):
            # A dashed part that names a part of the work stays, even when an
            # addition follows it: "Title - Part 2 - Live Performance"
            part = _SEPARATOR_RE.split(rest, maxsplit=1)[0]
            if not _is_addition(part) and (_PART_WORD_RE.search(part) or _BARE_PART_NUMBER_RE.fullmatch(part)):
                continue
            # An addition, alone ("- Live - Some Festival") or with what follows
            if not (_is_addition(part) or _is_addition(rest)):
                continue
        s = s[:separator.start()]
        break
    s = _FEATURING_RE.sub("", s)
    s = _OFFICIAL_RE.sub("", s)
    s = _QUALITY_RE.sub("", s)
    return " ".join(s.split()) or (title or "").strip()


def title_before_dash(title: str) -> str:
    """The title before the dash that a video title puts a description
    after ("Title - Singer Name live in the studio"), to search with when
    nothing was found with the whole title; "" when there is none.

    What follows the first dash (after strip_title_additions) is taken for
    a description when it has at least DESCRIPTION_WORDS words, no part or
    volume number and no further dash: "Title - Part Two" or "Title -
    Reprise" can name another song, "Title - Part 2 of the Story" too, and
    "Song A - Song B - Song C" is a medley.
    """
    s = strip_title_additions(title)
    separator = _SEPARATOR_RE.search(s)
    if separator is None or separator.group(1) in ("|", "//"):
        return ""
    before, rest = s[:separator.start()].strip(), s[separator.end():]
    if (not before or len(rest.split()) < DESCRIPTION_WORDS or _PART_NUMBER_RE.search(rest)
            or _SEPARATOR_RE.search(rest)):
        return ""
    return strip_title_additions(before)


def _title_words(title: str) -> str:
    """The title for comparing songs: without its additions and without a
    bracketed part it starts with (often left out: "(Some Words) Title" is
    also called "Title"); other bracketed parts stay, as plain words, since
    "Title (Part 2)" or "Title (Reprise)" is another song than "Title"."""
    s = strip_title_additions(title)
    s = _LEADING_BRACKETS_RE.sub("", s) or s
    return " ".join(re.sub(r"[()\[\]{}]", " ", s).split())


def titles_match(a: str, b: str) -> bool:
    """Whether two song titles name the same song.

    They are compared without their additions (see strip_title_additions)
    by ``similarity``; a bracketed part that is not an addition counts like
    any other word, except at the start of a title. Numbers left in both
    titles must be the same: "Intro (Part 1)" is not "Intro (Part 2)".
    """
    a, b = _title_words(a), _title_words(b)
    numbers_a, numbers_b = sorted(re.findall(r"\d+", a)), sorted(re.findall(r"\d+", b))
    if numbers_a and numbers_b and numbers_a != numbers_b:
        return False
    return similarity(a, b) >= MIN_MATCH_SIMILARITY


def title_key(title: str) -> str:
    """The title as ``titles_match`` compares it, to tell equal titles: its
    additions, a leading bracketed part and a leading article dropped, case
    and punctuation ignored; "Title (Live)" and "Title" are equal, "Title
    (Part 2)" is not."""
    return normalize_name(_title_words(title), artist=False)


def bracketed_parts(title: str) -> list[str]:
    """The contents of the bracketed parts of a title."""
    return [m.group(0)[1:-1] for m in _BRACKETS_RE.finditer(title or "")]


def without_brackets(title: str) -> str:
    """The title without any bracketed part."""
    return " ".join(_BRACKETS_RE.sub(" ", title or "").split())


def artist_from_channel(channel: str) -> str:
    """The artist's name in a channel name: without "VEVO", " - Topic" or
    " Official" at its end. Unchanged when nothing would be left."""
    return _CHANNEL_ADDITION_RE.sub("", channel or "").strip() or (channel or "").strip()
