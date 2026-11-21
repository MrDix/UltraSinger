"""Syllables of a written word, as many as it is spoken with.

A hyphenation dictionary knows where a word may be broken at the end of a
printed line. That is not where it is sung: it leaves short words whole
("only", "baby", "never", "city") and splits off no single letter ("a-way",
"o-ver"), so a word sung on two notes came as one syllable, and its second
note showed "~" instead of the syllable.

The number of syllables a word is spoken with is known from its phonemes
(eSpeak NG, as for the word starts): one per vowel sound. The written word
is cut between its vowel groups until it has that many parts. The spoken
count is the limit: a silent letter ("lovely", "sometimes") makes no
syllable, and a word is never cut inside a vowel group unless it has more
vowel sounds than vowel groups ("ra-di-o", "go-ing").
"""

from __future__ import annotations

import re

# Vowel letters of the Latin script (y: see _vowel_letters)
_VOWELS = frozenset("aeiouyäöüàáâãåæèéêëìíîïòóôõøœùúûýÿāēīōūăĕĭŏŭąęįųėěůőű")
# Vowels with a written stress that keeps them apart from the vowel beside them ("dí-a", "pa-ís")
_STRESSED_CLOSE = frozenset("íúìù")
_OPEN_VOWELS = frozenset("aeoáéóàèò")
# Letters that are one consonant sound together: never cut between them
_CONSONANT_UNITS = ("sch", "tch", "ch", "sh", "th", "ph", "wh", "gh", "ck", "qu")
# The end of a word whose "e" is no vowel sound of its own: its consonant goes with the "l" ("lit-tle")
_LE_ENDINGS = ("le", "les", "led")
# Two consonants a syllable starts with: of three or more consonants they stay together
_ONSETS = frozenset(("bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "sc", "sk", "sl", "sm", "sn",
                     "sp", "st", "sw", "tr", "tw"))
# Vowel pairs spoken as two sounds where the word has more sounds than vowel groups
_SPLIT_PAIRS = frozenset(("eo", "oe", "ao"))
# The sounds that carry a syllable in the phonemes of eSpeak NG; the mark of a syllabic consonant ("button")
_VOWEL_SOUNDS = frozenset("aeiouyɑɐɒæəɚɛɜɝɪɔøœʊʌʉɨɵɤɯᵻ")
_SYLLABIC = "̩"
# Words longer than this are not looked at (no word; and the phonemes of a long text cost time)
MAX_WORD_LETTERS = 40


class SpokenSyllables:
    """The number of syllables words are spoken with, in one language."""

    def __init__(self, voice: str):
        from modules.Speech_Recognition.phoneme_word_starts import _espeak_backend

        self._backend = _espeak_backend(voice)
        self._counts: dict[str, int | None] = {}
        self._tails: dict[str, int | None] = {}
        self._ng: dict[str, bool | None] = {}

    def count_many(self, words) -> None:
        """Look up the words that are not known yet, in one go."""
        from phonemizer.separator import Separator

        new = sorted({word for word in words if word not in self._counts})
        asked = [word for word in new if 0 < len(word) <= MAX_WORD_LETTERS and re.search(r"[^\W\d_]", word)]
        for word in new:
            self._counts[word] = self._tails[word] = self._ng[word] = None
        if not asked:
            return
        # (in small letters: a short word in capitals is read as an abbreviation, "IT" as "I T")
        answers = self._backend.phonemize([word.lower() for word in asked],
                                          separator=Separator(phone=" ", word=" | "), strip=True)
        if len(answers) != len(asked):  # (an empty answer was dropped: no count for these words)
            return
        for word, phonemes in zip(asked, answers):
            sounds = phonemes.replace("|", " ").split()
            vowel = [_SYLLABIC in sound or any(letter in _VOWEL_SOUNDS for letter in sound) for sound in sounds]
            if any(vowel):
                self._counts[word] = sum(vowel)
                self._tails[word] = vowel[::-1].index(True)
                self._ng[word] = any("ŋ" in sound for sound in sounds[:len(sounds) - 1 - self._tails[word]])

    def count(self, word: str) -> int | None:
        """Syllables the word is spoken with, None when that is not known."""
        if word not in self._counts:
            self.count_many([word])
        return self._counts[word]

    def tail(self, word: str) -> int | None:
        """Consonant sounds the word ends with, after its last vowel sound (None: not known)."""
        if word not in self._tails:
            self.count_many([word])
        return self._tails[word]

    def ng(self, word: str) -> bool | None:
        """Whether the word has the one sound of "ng" before its last vowel sound: "singing" has,
        "plunging" (n and a soft g) has not (None: not known)."""
        if word not in self._ng:
            self.count_many([word])
        return self._ng[word]


def spoken_syllables(language: str | None) -> SpokenSyllables | None:
    """The syllable counts of a language, None when eSpeak NG has no voice
    for it or cannot be used (the hyphenation alone splits the words then)."""
    try:
        from modules.Speech_Recognition.phoneme_word_starts import espeak_voice

        voice = espeak_voice((language or "").lower()[:2])
        return SpokenSyllables(voice) if voice else None
    except Exception:  # noqa: BLE001 - the syllable count is an improvement, never a condition
        return None


def _vowel_groups(low: str) -> list[tuple[int, int]]:
    """The vowel groups of a word in small letters, as (start, end): each may be a syllable."""
    vowels = [letter in _VOWELS for letter in low]
    ends_group = [False] * len(low)  # a vowel that the vowel after it does not belong to
    for i, letter in enumerate(low):
        before_vowel = i + 1 < len(low) and low[i + 1] in _VOWELS and low[i + 1] != "y"
        if letter == "y" and before_vowel:
            if low[i + 1:i + 4] in ("ing", "in'", "in’") or (i > 0 and low[i - 1] not in _VOWELS):
                ends_group[i] = i > 0  # a vowel of its own: "say-ing", "cry-ing", "an-y-one"
                vowels[i] = i > 0
            else:
                vowels[i] = False  # a consonant: "yeah", "be-yond"
        elif letter == "u" and i > 0 and low[i - 1] == "q":
            vowels[i] = False
        elif letter == "e" and low[i + 1:i + 2] == "o" and i > 1 and vowels[i - 2] and low[i - 1] not in _VOWELS:
            ends_group[i] = True  # the silent "e" of the word before, in one word with it: "some-one"
    groups: list[tuple[int, int]] = []
    for i, vowel in enumerate(vowels):
        if vowel and groups and groups[-1][1] == i and not ends_group[i - 1]:
            groups[-1] = (groups[-1][0], i + 1)
        elif vowel:
            groups.append((i, i + 1))
    return groups


def _consonant_cut(low: str, start: int, end: int, ng: bool | None = None) -> int:
    """Where the consonants low[start:end] between two vowel groups are cut
    (``ng``: see ``split_word``)."""
    if end == start:
        return start
    if end - start >= 2 and low[end - 1:] in _LE_ENDINGS:
        return end - 2  # "peo-ple", "lit-tle"
    if ng is not False and low[start:end] == "ng" and low[end:] in ("ing", "ings", "in", "in'", "in’"):
        return end  # "sing-ing"
    units: list[int] = []  # start of each consonant sound
    i = start
    while i < end:
        units.append(i)
        i += next((len(unit) for unit in _CONSONANT_UNITS if low.startswith(unit, i) and i + len(unit) <= end), 1)
    if len(units) == 1:
        return units[0]  # "ba-by"
    if len(units) == 2:
        return units[1]  # "gon-na"
    return units[-2] if low[units[-2]:end] in _ONSETS else units[-1]  # "hun-dred", "pump-kin"


def _inner_cuts(low: str, start: int, end: int) -> list[int]:
    """Where the vowel group low[start:end] may be cut, the surest place first."""
    cuts: list[int] = []
    if low[end - 1] == "i" and low[end:] in ("ng", "n", "n'", "n’", "ngs"):
        cuts.append(end - 1)  # "go-ing", "go-in'"
    for i in range(start + 1, end):
        if ((low[i] in _STRESSED_CLOSE and low[i - 1] in _OPEN_VOWELS)
                or (low[i - 1] in _STRESSED_CLOSE and low[i] in _OPEN_VOWELS)):
            cuts.append(i)  # "pa-ís", "dí-a"
    for i in range(start + 1, end):
        if low[i - 1] in "iuy" or low[i - 1:i + 1] in _SPLIT_PAIRS:
            cuts.append(i)  # "ra-di-o", "cru-el", "vi-de-o"
    return list(dict.fromkeys(cut for cut in cuts if start < cut < end))


def split_word(word: str, count: int | None, parts: list[str] | None = None, count_of=None,
               inner: bool = True, tail: int | None = None, ng: bool | None = None) -> list[str]:
    """The word in syllables, up to ``count`` of them.

    ``parts`` are its syllables by the hyphenation (None or one part when it
    does not split the word): they are kept when they are ``count`` or more,
    or when the word cannot be cut into more parts than they are. Too few of
    them are cut further, each as a word of its own with the count that
    ``count_of(part)`` gives ("every-one" is "eve-ry-one"), when that makes
    ``count`` syllables; else the whole word is cut anew. More of them than
    ``count`` stay, but for a part that starts at an apostrophe ("you-'re"
    is spoken as one syllable).

    Without ``inner`` no vowel group is cut ("ra-dio" stays two parts).
    ``tail`` is the number of consonant sounds the word ends with: with more
    of them than there are letters after a last "e", that "e" is not spoken
    and makes no syllable ("rea-lize", not "rea-li-ze"). ``ng`` tells whether
    an "ng" before the ending "-ing" is one sound, which stays together
    ("sing-ing"); False for two sounds ("plun-ging"), None when not known
    (taken for one sound).
    """
    parts = parts if parts and "".join(parts) == word else [word]
    if not count or len(word) > MAX_WORD_LETTERS:
        return parts
    while len(parts) > count and any(part[:1] in "'’" for part in parts[1:]):
        k = next(k for k, part in enumerate(parts) if k and part[:1] in "'’")
        parts = parts[:k - 1] + [parts[k - 1] + parts[k]] + parts[k + 1:]
    if count <= len(parts):
        return parts
    if len(parts) > 1 and count_of is not None:
        finer = [piece for part in parts for piece in _cut(part, count_of(part) or 1, inner)]
        if len(finer) == count:
            return finer
    split = _cut(word, count, inner, tail, ng)
    return split if len(split) > len(parts) else parts


def _cut(word: str, count: int, inner: bool = True, tail: int | None = None, ng: bool | None = None) -> list[str]:
    """The word cut between its vowel groups into up to ``count`` parts."""
    if count < 2:
        return [word]
    low = word.lower()
    if len(low) != len(word):  # (a letter that is two in small letters: the places would not fit)
        return [word]
    groups = _vowel_groups(low)
    if not groups:
        return [word]
    runs = list(groups)  # the vowels side by side, for the cuts inside them
    a, b = groups[-1]
    if tail is not None and len(groups) > 1 and low[a:b] == "e" and low[b:] in ("", "d", "s") and tail > len(low) - b:
        # The last "e" is not spoken: it belongs to the syllable before it
        runs.pop()
        groups[-2:] = [(groups[-2][0], b)]
    # One cut between each two vowel groups; cuts[k] lies before group k + 1
    cuts = [_consonant_cut(low, a_end, b_start, ng) for (_, a_end), (b_start, _) in zip(groups, groups[1:])]
    # More vowel groups than syllables: a silent "e" is no syllable ("love-ly", "some-times", "a-lone")
    while len(cuts) + 1 > count:
        lone_e = [k for k, (a, b) in enumerate(groups) if low[a:b] == "e"]
        after_one_consonant = [k for k in lone_e if k > 0 and _single_consonant(low, groups[k - 1][1], groups[k][0])]
        k = (after_one_consonant or lone_e or [len(groups) - 1])[-1]
        drop = max(k - 1, 0)  # the group joins the one before it (the first one the one after it)
        del cuts[drop]
        groups[drop:drop + 2] = [(groups[drop][0], groups[drop + 1][1])]
    # Fewer vowel groups than syllables: two vowel sounds written side by side
    if inner and len(cuts) + 1 < count:
        more = [cut for a, b in runs for cut in _inner_cuts(low, a, b)]
        cuts = sorted(cuts + more[:count - len(cuts) - 1])
    edges = [0] + cuts + [len(word)]
    split = [word[a:b] for a, b in zip(edges, edges[1:])]
    return split if all(re.search(r"[^\W\d_]", part) for part in split) else [word]


def _single_consonant(low: str, start: int, end: int) -> bool:
    """Whether low[start:end] is one consonant sound ("m" in "some", "th" in "bathe")."""
    return end > start and (end - start == 1 or low[start:end] in _CONSONANT_UNITS)
