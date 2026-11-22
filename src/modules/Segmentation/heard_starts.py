"""Word starts where a transcription heard the words.

The words of found lyrics get their times from a forced alignment: one path
of the letters of the lyrics over the vocal track. The path puts every word
somewhere, and on a held vowel the letters fit about as well anywhere, so a
word often starts late there. A transcription hears each word by itself, and
the notes of the segmentation model tell where a sound begins.

Where the alignment starts a word at a time at which no note of the model
begins, and the transcription heard the word start at a time at which one
does, the word starts there: the lyrics placement then puts it onto that
note.
"""

from __future__ import annotations

import unicodedata
from dataclasses import replace

import numpy as np

# A heard start counts as another time than the aligned one from this difference on
MIN_DIFFERENCE_MS = 100.0
# A heard word this far from the aligned start is another occurrence of the word, or a slip of the
# transcription
MAX_DIFFERENCE_MS = 3000.0
# A note of the model begins "there": within this of a start
NOTE_AT_START_MS = 60.0
# Equal words are paired with the occurrence near in time: a pair this far apart counts little
PAIR_TIME_SCALE_MS = 3000.0
# More words than this (lyrics x heard) are not paired: the table of the pairing would not fit
MAX_PAIRING_CELLS = 16_000_000
# The shortest time a moved word keeps
MIN_WORD_MS = 50.0


def word_key(text: str) -> str:
    """The word without case, accents, apostrophes and punctuation ("Don't" -> "dont")."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if c.isalnum() and not unicodedata.combining(c)).casefold()


def pair_words(keys: list[str], starts_ms: list[float],
               heard_keys: list[str], heard_starts_ms: list[float]) -> list[tuple[int, int]]:
    """Pairs (word, heard word) of equal words, in order: a common subsequence
    of the two texts in which every pair counts, and a pair near in time up to
    half as much again. So a word of a repeated chorus is paired with the
    occurrence beside it, and two pairs always count more than one."""
    n, m = len(keys), len(heard_keys)
    if not n or not m or n * m > MAX_PAIRING_CELLS:
        return []
    heard = np.array(heard_keys, dtype=object)
    heard_at = np.asarray(heard_starts_ms, dtype=float)
    # (a pair counts 1000 to 1500, and there are at most 4000 of them: far inside 32 bits)
    score = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i, (key, start) in enumerate(zip(keys, starts_ms), 1):
        near = np.round(500 * np.exp(-np.abs(heard_at - start) / PAIR_TIME_SCALE_MS)).astype(np.int32)
        gain = np.where((heard == key) & bool(key), 1000 + near, -1).astype(np.int32)
        # score[i, j] = max(score[i - 1, j], score[i, j - 1], score[i - 1, j - 1] + gain[j - 1])
        score[i, 1:] = np.maximum.accumulate(np.maximum(score[i - 1, 1:], score[i - 1, :-1] + gain))
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        if score[i, j] == score[i - 1, j]:
            i -= 1
        elif score[i, j] == score[i, j - 1]:
            j -= 1
        else:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
    return pairs[::-1]


def words_of(syllables: list) -> list[tuple[int, int]]:
    """The words of the syllables as (first syllable, one past the last): a
    syllable starts a word at the start of a line and after a syllable that
    ends with a space."""
    words: list[tuple[int, int]] = []
    for i, syllable in enumerate(syllables):
        if i == 0 or syllable.line_start or syllables[i - 1].text[-1:].isspace():
            words.append((i, i + 1))
        else:
            words[-1] = (words[-1][0], i + 1)
    return words


def heard_starts(syllables: list, heard_words: list[tuple[str, float, float]],
                 note_starts_ms) -> dict[int, float]:
    """The new start (ms) of the words that start where they were heard, by the
    index of their first syllable.

    ``heard_words`` are the words of a transcription as (text, start s,
    end s), ``note_starts_ms`` the starts of the notes of the model. A word
    is paired with the transcribed word of the same text (``pair_words``); it
    starts where that was heard when

    - the two starts differ by more than ``MIN_DIFFERENCE_MS`` and by no more
      than ``MAX_DIFFERENCE_MS``,
    - a note begins within ``NOTE_AT_START_MS`` of the heard start, and none
      within it of the aligned start,
    - and the syllables stay in their order: the heard start lies after the
      start of the last syllable of the word before it (as it is then), and
      the last syllable of the word still starts before the word after it.

    With a note at both starts the alignment is right about as often as the
    transcription, so the word stays; without a note at the aligned start the
    transcription is right far more often.
    """
    words = words_of(syllables)
    if not words or not heard_words:
        return {}
    keys = [word_key("".join(s.text for s in syllables[a:b])) for a, b in words]
    starts = [syllables[a].start_ms for a, _ in words]
    heard = [(word_key(text), start * 1000.0) for text, start, _ in heard_words]
    heard = [(key, start) for key, start in heard if key]
    notes = np.sort(np.asarray(note_starts_ms, dtype=float))
    if not heard or not len(notes):
        return {}

    def note_at(time: float) -> bool:
        k = int(np.searchsorted(notes, time))
        return min(abs(notes[min(k, len(notes) - 1)] - time), abs(notes[max(k - 1, 0)] - time)) <= NOTE_AT_START_MS

    moved: dict[int, float] = {}
    last_syllable_starts = [syllables[b - 1].start_ms for _, b in words]
    for i, j in pair_words(keys, starts, [key for key, _ in heard], [start for _, start in heard]):
        at = heard[j][1]
        if not MIN_DIFFERENCE_MS < abs(at - starts[i]) <= MAX_DIFFERENCE_MS:
            continue
        if not note_at(at) or note_at(starts[i]):
            continue
        if i > 0 and at <= last_syllable_starts[i - 1]:
            continue
        first, last = words[i]
        last_syllable_start = _times_from(syllables[first:last], at)[-1][0]
        if i + 1 < len(words) and last_syllable_start >= starts[i + 1]:
            continue
        last_syllable_starts[i] = last_syllable_start
        moved[first] = at
    return moved


def _times_from(word: list, start_ms: float) -> list[tuple[float, float]]:
    """The times of the syllables of a word that starts at ``start_ms``: they
    share the time from there to the end of the word by their letters, as they
    did from its old start. A word heard after its end keeps a short time."""
    end = max(word[-1].end_ms, start_ms + MIN_WORD_MS)
    letters = [max(len(syllable.text.strip()), 1) for syllable in word]
    total, done, times = sum(letters), 0, []
    for count in letters:
        a = start_ms + (end - start_ms) * done / total
        done += count
        times.append((a, start_ms + (end - start_ms) * done / total))
    return times


def start_where_heard(syllables: list, heard_words: list[tuple[str, float, float]] | None,
                      note_starts_ms) -> tuple[list, int]:
    """The syllables with the words that a transcription heard elsewhere
    started there (see ``heard_starts``), and the number of these words.

    The syllables of such a word share the time from its new start to its
    end by their letters (``_times_from``). The words around it keep their
    times.
    """
    if not heard_words:
        return syllables, 0
    moved = heard_starts(syllables, heard_words, note_starts_ms)
    if not moved:
        return syllables, 0
    out = list(syllables)
    for first, last in words_of(syllables):
        if first not in moved:
            continue
        for k, (start, end) in enumerate(_times_from(syllables[first:last], moved[first])):
            out[first + k] = replace(syllables[first + k], start_ms=start, end_ms=end)
    return out, len(moved)
