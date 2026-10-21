"""Forced alignment of lyrics lines to the emission of a wav2vec2 CTC model.

whisperx aligns the text of a segment with one CTC path over the segment's
audio. Aligned as one segment over the whole vocal track, a line that the path
puts in the wrong place pulls the lines around it along, and the times of
synced lyrics are not used. Here the same path (the same tokens, trellis and
backtrack as whisperx) is checked against these LRC times:

- Four alignments are made: with the emission of the whole song and with one
  computed in pieces of 30 s, each with the path ending at its best frame (as
  whisperx) or with a capped end, which makes it run on over sung frames
  after the last word. The one with the most lines at their LRC time (after a
  common offset) is used.
- A line that the path put in a poor place moves to a clearly better place
  near its LRC time. When most lines agree on the offset, poorly placed lines
  at the end that the LRC puts after the end of the audio (lyrics of a longer
  version) are left out.

On a benchmark of 89 songs, the share of words that start more than 1 s off
fell from 6.5 % to 4.9 % (more than 3 s off: from 4.6 % to 3.1 %).

Plain lyrics have no times; the words of a transcription give their lines
times (line_times_from_words), and with enough of them the lines are aligned
as above. On the same benchmark, with the LRC lines without their times as
plain lyrics, this put 4.8 % of the words more than 1 s off, against 6.5 % for
one path over all lines.

With a transcription, the LRC times are checked by its words
(checked_line_times): a line the words place far from its LRC time moves
there, and the passages it hears that the lyrics do not have
(passages_without_lyrics) are sung with its words.

The functions work on numpy arrays; the caller runs the model.
"""

from __future__ import annotations

import statistics
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

SAMPLE_RATE = 16000
# Samples per frame of the emission of the wav2vec2 models
HOP = 320
# The emission in pieces: their length and the audio context on both sides (s)
PIECE_S = 30.0
CONTEXT_S = 5.0
# Languages whose text whisperx does not split into words at spaces
# (whisperx.alignment.LANGUAGES_WITHOUT_SPACES)
LANGUAGES_WITHOUT_SPACES = ("ja", "zh")
# Capped path end: each frame after the last word costs at most this much
# (about log(0.2)); after the free end of whisperx, frames cost nothing
END_COST = -1.6
# A line is at its LRC time when it starts this close to it, after the offset (s)
FIT_S = 1.5
# The offset of the LRC times needs this many lines placed at least as well as
# the median line, else the lines are not checked. It is reliable from
# RELIABLE_LINES of them on, when they deviate from it by at most RELIABLE_S
# (median); only then are lines left out.
MIN_OFFSET_LINES = 5
RELIABLE_LINES = 8
RELIABLE_S = 1.0
# LRC window of a line: from its time to the next line's time (the last line:
# LAST_LINE_S after its time), shifted by the offset and widened by MARGIN_S (s)
MARGIN_S = 2.0
LAST_LINE_S = 10.0
# A line is placed poorly below this mean log-probability of its tokens
POOR_SCORE = -4.0
# Poorly placed lines at the end are left out when the LRC puts them after
# the end of the audio minus this (s)
TAIL_S = 2.0
# A line moves to its LRC window when it lands more than MOVE_S away there
# and the mean log-probability of its tokens rises by more than MOVE_GAIN
MOVE_S = 1.0
MOVE_GAIN = 0.3
# Consecutive lines may overlap this much (s)
OVERLAP_S = 0.3
# Log-probability of the tokens on a frame without a usable model output
UNUSABLE_TOKEN = -1e4
# Line times from the words of a transcription: the sung seconds per word are
# the median step between paired words, limited to WORD_S_RANGE, or WORD_S
# when there are fewer than WORD_STEPS such steps
WORD_S = 0.35
WORD_S_RANGE = (0.15, 1.0)
WORD_STEPS = 10
# A transcribed word this long (s) gives no line time: its start is often the
# start of the piece it was transcribed in
LONG_WORD_S = 3.0
# Plain lyrics are aligned with these line times when at least this share of
# their lines, and at least RELIABLE_LINES lines, got one from a word
ANCHORED_SHARE = 0.5
# Passages of a transcription that the lyrics do not have: a pair of a word of
# the lyrics and a transcribed word is sure when the next or the previous
# words of both are paired too (a single word pairs by chance: "you", "I").
# Transcribed words more than SURE_REACH words from a sure pair are not sung
# from the lyrics; PASSAGE_MIN_WORDS of them in a row make a passage, unless
# the lyrics have more than PASSAGE_LYRICS_SHARE as many words between the sure
# pairs around them: then they are lines of the lyrics heard badly.
SURE_REACH = 2
PASSAGE_MIN_WORDS = 6
PASSAGE_LYRICS_SHARE = 0.4
# LRC times checked by the words of a transcription: a line is confirmed by
# a sure pair (see above), at the time of its word counted back by the words
# before it in the line. With at least MIN_OFFSET_LINES confirmed lines
# (taken in time order) the common offset of the LRC times is their median
# offset; a confirmed line that is more than CHECKED_MOVE_S off its LRC time
# plus this offset moves to the confirmed time. A line without a sure pair
# whose LRC time no longer lies between the confirmed lines around it (give
# or take FIT_S; after the one before it has had half its words) is spread
# anew between them by its words (before the first or after the last one:
# counted from it by its words).
CHECKED_MOVE_S = 5.0


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

@dataclass
class LyricsTokens:
    """The characters of the lyrics that the path places, as whisperx.align
    cleans the text of a segment."""
    ids: np.ndarray          # dictionary index per token, -1 for the wildcard
    word: np.ndarray         # index of the token's word
    letter: np.ndarray       # True for a character of the word, False for the space after it
    words: list[str]         # in languages without spaces each character is a word, as in whisperx
    word_line: np.ndarray    # line of each word
    word_source: np.ndarray  # index of the word of `lines` each word comes from


def lyrics_tokens(lines: list[list[str]], dictionary: dict, language: str) -> LyricsTokens:
    """Tokens of the lyrics, as whisperx.align makes them of the text of one segment.

    ``lines`` holds the words of each line; the text is all words joined by
    single spaces. Each character is lower-cased; a space becomes "|" unless
    ``language`` (the model's) has no spaces. Characters in the model's
    ``dictionary`` are kept, others as the wildcard (-1), except spaces and
    "|", which are left out. The space after a word belongs to that word.
    """
    spaces = language not in LANGUAGES_WITHOUT_SPACES
    separator = "|" if spaces else " "
    ids, tok_word, letter = [], [], []
    words, word_line, word_source = [], [], []
    source = 0
    for i, line in enumerate(lines):
        for text in line:
            if words and separator in dictionary:
                ids.append(dictionary[separator])
                tok_word.append(len(words) - 1)
                letter.append(False)
            for char in text:
                c = char.lower()
                if c not in dictionary and c in (" ", "|"):
                    continue
                if not spaces or not words or word_source[-1] != source:
                    words.append(text if spaces else char)
                    word_line.append(i)
                    word_source.append(source)
                # lower() may give more than one character, each a token
                for ch in c:
                    ids.append(dictionary.get(ch, -1))
                    tok_word.append(len(words) - 1)
                    letter.append(True)
            source += 1
    return LyricsTokens(
        np.array(ids, dtype=np.int64), np.array(tok_word, dtype=np.int64), np.array(letter, dtype=bool),
        words, np.array(word_line, dtype=np.int64), np.array(word_source, dtype=np.int64),
    )


def blank_index(dictionary: dict) -> int:
    """Column of the CTC blank, as whisperx finds it: "[pad]" or "<pad>", else 0."""
    blank_id = 0
    for char, code in dictionary.items():
        if char == "[pad]" or char == "<pad>":
            blank_id = code
    return blank_id


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def emission_in_pieces(
    run_model: Callable[[np.ndarray], np.ndarray],
    audio: np.ndarray,
    n_frames: Optional[int] = None,
    piece_s: float = PIECE_S,
    context_s: float = CONTEXT_S,
) -> np.ndarray:
    """Log-probabilities per frame of 16-kHz ``audio``, computed piece by piece.

    ``run_model(samples)`` returns the log-probabilities (frames x vocabulary)
    of the model for any number of samples. Each piece gets up to
    ``context_s`` of audio on both sides; its rows without the context go to
    the frames of the piece. The result has ``n_frames`` rows, by default as
    many as the model gives for the whole audio (the last piece runs to its
    end). ``audio`` must not be empty.
    """
    piece = int(piece_s * SAMPLE_RATE) // HOP * HOP
    context = int(context_s * SAMPLE_RATE) // HOP * HOP
    rows = []  # first frame of each piece and its rows from there on
    for start in range(0, len(audio), piece):
        a = max(0, start - context)
        b = min(len(audio), start + piece + context)
        rows.append((start // HOP, run_model(audio[a:b])[(start - a) // HOP:]))
    if n_frames is None:
        n_frames = max(f0 + len(em) for f0, em in rows)
    out = np.zeros((n_frames, rows[0][1].shape[1]), dtype=np.float32)
    for f0, em in rows:
        n = min(piece // HOP, n_frames - f0, len(em))
        if n > 0:
            out[f0:f0 + n] = em[:n]
    return out


def with_wildcard(log_probs: np.ndarray, ids: np.ndarray, blank_id: int) -> tuple[np.ndarray, np.ndarray]:
    """The emission and the token columns, with a wildcard column when a token
    needs one, as whisperx adds it: per frame the best non-blank log-probability."""
    emission = np.asarray(log_probs, dtype=np.float32)
    wildcard = ids < 0
    if not wildcard.any():
        return emission, ids
    non_blank = np.ones(emission.shape[1], dtype=bool)
    non_blank[blank_id] = False
    emission = np.concatenate([emission, emission[:, non_blank].max(axis=1, keepdims=True)], axis=1)
    return emission, np.where(wildcard, emission.shape[1] - 1, ids)


# ---------------------------------------------------------------------------
# CTC path, as whisperx.alignment searches it
# ---------------------------------------------------------------------------

def trellis(emission: np.ndarray, tokens: np.ndarray, blank_id: int) -> np.ndarray:
    """whisperx.alignment.get_trellis in numpy, in float32 as there."""
    n_frames, n_tokens = emission.shape[0], len(tokens)
    blank = emission[:, blank_id]
    tr = np.empty((n_frames + 1, n_tokens + 1), dtype=np.float32)
    tr[0, 0] = 0
    # torch adds up a float32 column in double precision
    tr[1:, 0] = np.cumsum(blank, dtype=np.float64)
    tr[0, 1:] = -np.inf
    tr[max(0, n_frames + 1 - n_tokens):, 0] = np.inf
    for t in range(n_frames):
        np.maximum(tr[t, 1:] + blank[t], tr[t, :-1] + emission[t, tokens], out=tr[t + 1, 1:])
    return tr


def end_row(tr: np.ndarray, blank: np.ndarray, end_cost: Optional[float] = None) -> int:
    """Trellis row where the path ends.

    Free end (``end_cost`` None, as whisperx): the best row. Otherwise the best
    row when each frame after it costs its blank log-probability, but at most
    ``end_cost`` (0 gives the free end; with minus infinity the path pays for
    all frames, as one that runs to the last frame).
    """
    n_tokens = tr.shape[1] - 1
    if end_cost is None:
        return int(np.argmax(tr[:, n_tokens]))
    # cost of the frames from each row on
    tail = np.concatenate([np.cumsum(np.maximum(blank, end_cost)[::-1])[::-1], [0.0]])
    return int(np.argmax(tr[:, n_tokens] + tail))


def backtrack(
    tr: np.ndarray, emission: np.ndarray, tokens: np.ndarray, blank_id: int, t_end: int,
) -> Optional[np.ndarray]:
    """Emission frame of each token on the path that ends at trellis row
    ``t_end``, or None when there is none (whisperx.alignment.backtrack)."""
    j = len(tokens)
    frames = np.full(j, -1)
    for t in range(t_end, 0, -1):
        if tr[t - 1, j - 1] + emission[t - 1, tokens[j - 1]] > tr[t - 1, j] + emission[t - 1, blank_id]:
            frames[j - 1] = t - 1
            j -= 1
            if j == 0:
                return frames
    return None


def align_path(
    emission: np.ndarray, tokens: np.ndarray, blank_id: int, end_cost: Optional[float] = None,
) -> Optional[tuple[np.ndarray, int]]:
    """(emission frame of each token, frame where the path ends), or None
    when the tokens do not fit."""
    if len(tokens) == 0 or emission.shape[0] < len(tokens):
        return None
    tr = trellis(emission, tokens, blank_id)
    t_end = end_row(tr, emission[:, blank_id], end_cost)
    frames = backtrack(tr, emission, tokens, blank_id, t_end)
    return None if frames is None else (frames, t_end)


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------

@dataclass
class _Path:
    tokens: np.ndarray  # token indices on the path, in order
    frames: np.ndarray  # emission frame of each


@dataclass
class _Placement:
    start: float  # s
    end: float    # s
    score: float  # mean log-probability of the line's tokens at their frames
    path: _Path


class _Emission:
    """An emission with the columns of the lyrics tokens."""

    def __init__(self, log_probs: np.ndarray, ids: np.ndarray, blank_id: int, frame_s: float):
        self.em, self.ids = with_wildcard(log_probs, ids, blank_id)
        unusable = np.isnan(self.em).any(axis=1) | np.isposinf(self.em).any(axis=1)
        if unusable.any():
            # A frame without a usable model output (NaN) counts as silence: in
            # the trellis it would make every later frame NaN
            self.em = self.em.copy()
            self.em[unusable] = UNUSABLE_TOKEN
            self.em[unusable, blank_id] = 0.0
        self.blank_id = blank_id
        self.frame_s = frame_s
        self.n_frames = self.em.shape[0]

    def path(self, tokens: np.ndarray, f0: int = 0, f1: Optional[int] = None,
             end_cost: Optional[float] = None) -> Optional[_Path]:
        """The path of the tokens (indices, in order) in frames f0 to f1."""
        found = align_path(self.em[f0:f1], self.ids[tokens], self.blank_id, end_cost)
        if found is None:
            return None
        return _Path(tokens, found[0] + f0)

    def placement(self, path: _Path, p0: int, p1: int) -> _Placement:
        """Placement of the tokens at positions p0 to p1 of the path."""
        frames = path.frames[p0:p1]
        scores = self.em[frames, self.ids[path.tokens[p0:p1]]]
        return _Placement(frames[0] * self.frame_s, (frames[-1] + 1) * self.frame_s,
                          float(scores.mean()), path)

    def lines(self, ranges: dict, lines: list[int], end_cost: Optional[float]) -> Optional[dict]:
        """Placement of each line on one path over all frames."""
        path = self.path(np.concatenate([np.arange(*ranges[i]) for i in lines]), end_cost=end_cost)
        if path is None:
            return None
        out, pos = {}, 0
        for i in lines:
            n = ranges[i][1] - ranges[i][0]
            out[i] = self.placement(path, pos, pos + n)
            pos += n
        return out


def _anchors(placed: dict, lines: list[int], line_times: list[float]) -> list[float]:
    """Line start minus LRC time, of the lines placed at least as well as the median line."""
    scored = [(placed[i].score, placed[i].start - line_times[i]) for i in lines]
    cut = np.percentile([x for x, _ in scored], 50)
    return [d for x, d in scored if x >= cut]


def _fit(placed: dict, lines: list[int], line_times: list[float]) -> int:
    """Lines that start within FIT_S of their LRC time plus the offset."""
    offset = statistics.median(_anchors(placed, lines, line_times))
    return sum(abs(placed[i].start - line_times[i] - offset) <= FIT_S for i in lines)


def _local(em: _Emission, i: int, rng: tuple[int, int], offset: float,
           line_times: list[float]) -> Optional[_Placement]:
    """The line alone, aligned in its LRC window, or None when the window is too short."""
    k0, k1 = rng
    a = line_times[i] + offset - MARGIN_S
    b = (line_times[i + 1] if i + 1 < len(line_times) else line_times[i] + LAST_LINE_S) + offset + MARGIN_S
    f0, f1 = max(0, int(a / em.frame_s)), min(em.n_frames, int(b / em.frame_s))
    if f1 - f0 <= k1 - k0:
        return None
    path = em.path(np.arange(k0, k1), f0, f1)
    return None if path is None else em.placement(path, 0, k1 - k0)


def _tail_drops(lines: list[int], placed: dict, local: dict, line_times: list[float],
                offset: float, audio_end: float) -> list[int]:
    """Lines at the end that are poorly placed, also in their LRC window, and
    that the LRC puts after the end of the audio."""
    dropped = []
    for i in reversed(lines):
        g, l = placed[i], local[i]
        if not (g.score < POOR_SCORE and (l is None or l.score < POOR_SCORE)):
            break
        if line_times[i] + offset > audio_end - TAIL_S:
            dropped.append(i)
    return dropped[::-1]


def _moves(lines: list[int], placed: dict, local: dict) -> dict:
    """Per line 1 when it moves to its LRC window, else 0: the most gain in
    mean log-probability with the lines in order (each starting at most
    OVERLAP_S before the previous one ends)."""
    gains = {}
    for i in lines:
        g, l = placed[i], local[i]
        if l is None or abs(l.start - g.start) <= MOVE_S:
            gains[i] = None
        else:
            gains[i] = l.score - g.score - MOVE_GAIN
    # per state of the last line (0 = path, 1 = moved): total gain, its end, the states so far
    states = {0: (0.0, -np.inf, [])}
    for i in lines:
        options = {0: placed[i]}
        if gains[i] is not None and gains[i] > 0:
            options[1] = local[i]
        new = {}
        for state, p in options.items():
            ok = [(v, c) for v, e, c in states.values() if p.start >= e - OVERLAP_S]
            if ok:
                v, c = max(ok, key=lambda x: x[0])
                new[state] = (v + (gains[i] if state == 1 else 0.0), p.end, c + [state])
        if not new:  # cannot happen: the placements on the path are in order
            v, e, c = max(states.values(), key=lambda x: x[0])
            new[0] = (v, placed[i].end, c + [0])
        states = new
    return dict(zip(lines, max(states.values(), key=lambda x: x[0])[2]))


def _check_lines(em: _Emission, ranges: dict, lines: list[int], line_times: list[float],
                 placed: dict, end_cost: Optional[float]) -> tuple[dict, list[int], int]:
    """(placement per kept line, lines left out, lines moved) after the line check."""
    anchors = _anchors(placed, lines, line_times)
    if len(anchors) < MIN_OFFSET_LINES:
        return placed, [], 0
    offset = statistics.median(anchors)
    reliable = len(anchors) >= RELIABLE_LINES and statistics.median(
        [abs(d - offset) for d in anchors]) <= RELIABLE_S
    local = {i: _local(em, i, ranges[i], offset, line_times) for i in lines}
    keep, dropped = lines, []
    if reliable:
        dropped = _tail_drops(lines, placed, local, line_times, offset, em.n_frames * em.frame_s)
    if dropped and len(dropped) < len(lines):
        kept = [i for i in lines if i not in dropped]
        again = em.lines(ranges, kept, end_cost)
        if again is not None:
            keep, placed = kept, again
        else:
            dropped = []
    else:
        dropped = []
    choice = _moves(keep, placed, local)
    return {i: local[i] if choice[i] else placed[i] for i in keep}, dropped, sum(choice.values())


def _word_times(tokens: LyricsTokens, ranges: dict, placements: dict,
                frame_s: float) -> list[tuple[int, float, float]]:
    """(word, start s, end s) of the words of the placed lines, as whisperx
    forms them: from the frame of the word's first character to the frame of
    the token after its last character; the last token of a path ends one
    frame after its own, where the path ends (also with a capped end: going
    on after it never costs less than ending there)."""
    out = []
    for i in sorted(placements):
        path = placements[i].path
        k0, k1 = ranges[i]
        p0 = int(np.searchsorted(path.tokens, k0))
        first, last = {}, {}
        for p in range(p0, p0 + k1 - k0):
            k = path.tokens[p]
            if tokens.letter[k]:
                first.setdefault(int(tokens.word[k]), p)
                last[int(tokens.word[k])] = p
        for w in sorted(first):
            q = last[w] + 1
            end = path.frames[q] if q < len(path.tokens) else path.frames[-1] + 1
            out.append((w, round(float(path.frames[first[w]] * frame_s), 3), round(float(end * frame_s), 3)))
    return out


@dataclass
class LyricsAlignment:
    """What align_lyrics found."""
    words: list[tuple[int, float, float]]  # (word, start s, end s) of the placed words, in lyrics order
    emission: str            # label of the emission that was used
    capped: bool             # whether the path end was capped
    lines: int               # lines with tokens
    fit: Optional[int] = None  # of these, the lines at their LRC time (None without LRC times)
    moved: int = 0           # lines moved to their LRC window
    dropped: list[int] = field(default_factory=list)  # lines left out


def align_lyrics(
    tokens: LyricsTokens,
    emissions: list[tuple[str, np.ndarray]],
    blank_id: int,
    frame_s: float,
    line_times: Optional[list[float]] = None,
) -> Optional[LyricsAlignment]:
    """Place the words of the lyrics; None when the tokens fit no emission.

    ``emissions`` are (label, log-probabilities) pairs, the one to keep on a
    tie first. Without ``line_times`` (the LRC time of each line) the first
    emission is used with the free end, as whisperx aligns one segment. With
    them, of each emission with the free and the capped end, the alignment
    with the most lines at their LRC time is used (the earlier one on a tie),
    and its lines are checked.
    """
    if not len(tokens.ids) or not emissions:
        return None
    tok_line = tokens.word_line[tokens.word]
    ranges = {}
    for i in np.unique(tok_line):
        ks = np.flatnonzero(tok_line == i)
        ranges[int(i)] = (int(ks[0]), int(ks[-1]) + 1)
    lines = sorted(ranges)
    ems = [(label, _Emission(log_probs, tokens.ids, blank_id, frame_s)) for label, log_probs in emissions]
    if line_times is None:
        label, em = ems[0]
        placed = em.lines(ranges, lines, None)
        if placed is None:
            return None
        return LyricsAlignment(_word_times(tokens, ranges, placed, frame_s), label, False, len(lines))
    best = None
    for label, em in ems:
        for end_cost in (None, END_COST):
            placed = em.lines(ranges, lines, end_cost)
            if placed is None:
                continue
            fit = _fit(placed, lines, line_times)
            if best is None or fit > best[0]:
                best = (fit, label, em, end_cost, placed)
    if best is None:
        return None
    fit, label, em, end_cost, placed = best
    placements, dropped, moved = _check_lines(em, ranges, lines, line_times, placed, end_cost)
    return LyricsAlignment(_word_times(tokens, ranges, placements, frame_s), label, end_cost is not None,
                           len(lines), fit, moved, dropped)


# ---------------------------------------------------------------------------
# Line times of plain lyrics from the words of a transcription
# ---------------------------------------------------------------------------

def _word_key(text: str) -> str:
    """The word without case, accents, apostrophes and punctuation ("Don't" -> "dont")."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if c.isalnum() and not unicodedata.combining(c)).casefold()


def _common_words(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    """Pairs (index in a, index in b) of equal words, in order: a longest
    common subsequence (of the equally long ones, the one the backtrack from
    the end meets first)."""
    if not a or not b:
        return []
    b_words = np.array(b, dtype=object)
    score = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    for i, word in enumerate(a, 1):
        paired = np.where(b_words == word, score[i - 1, :-1] + 1, 0)
        score[i, 1:] = np.maximum.accumulate(np.maximum(score[i - 1, 1:], paired))
    pairs, i, j = [], len(a), len(b)
    while i > 0 and j > 0:
        if score[i, j] == score[i - 1, j]:
            i -= 1
        elif score[i, j] == score[i, j - 1]:
            j -= 1
        else:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
    return pairs[::-1]


def _longest_in_order(values: list[float]) -> list[int]:
    """Indices of a longest non-decreasing subsequence of the values (on a
    tie, the one that ends first, each element after the first one that fits)."""
    n = len(values)
    if not n:
        return []
    best, prev = [1] * n, [-1] * n
    for i in range(n):
        for j in range(i):
            if values[j] <= values[i] and best[j] + 1 > best[i]:
                best[i], prev[i] = best[j] + 1, j
    i = max(range(n), key=lambda k: best[k])
    out = []
    while i >= 0:
        out.append(i)
        i = prev[i]
    return out[::-1]


def line_times_from_words(
    lines: list[list[str]],
    words: list[tuple[str, float, float]],
) -> tuple[Optional[list[float]], int]:
    """Times (s) of the lines of plain lyrics from the words of a transcription.

    ``lines`` holds the words of each line, ``words`` the transcribed words
    as (text, start s, end s), in the order of the transcription.

    1. The words of the lyrics are paired in order with transcribed words
       that are the same without case, accents and punctuation (a longest
       common subsequence).
    2. The sung seconds per word: the median step from start to start of two
       consecutive words of a line paired with two consecutive transcribed
       words (limited to WORD_S_RANGE; WORD_S with fewer than WORD_STEPS such
       steps).
    3. A line is anchored at the start of its first paired word, minus the
       seconds per word for each word before it in the line (not before 0).
       Transcribed words longer than LONG_WORD_S anchor nothing: their start
       is often the start of the piece they were transcribed in.
    4. Of these anchors, a longest run in time order is kept.
    5. The other lines are spread by their words between the anchored lines
       around them; before the first and after the last one, at the seconds
       per word.

    Returns the time of each line (None without any anchor) and the number of
    anchored lines.
    """
    lyric = [(i, p, _word_key(w)) for i, line in enumerate(lines) for p, w in enumerate(line)]
    keyed = [k for k, (_, _, key) in enumerate(lyric) if key]
    heard = [(key, start, end) for key, start, end in ((_word_key(w), s, e) for w, s, e in words)
             if key and start is not None and end is not None]
    pair = {keyed[a]: b for a, b in _common_words([lyric[k][2] for k in keyed], [h[0] for h in heard])}
    steps = [heard[pair[k + 1]][1] - heard[pair[k]][1] for k in pair
             if k + 1 in pair and lyric[k + 1][0] == lyric[k][0] and pair[k + 1] == pair[k] + 1]
    word_s = float(np.clip(statistics.median(steps), *WORD_S_RANGE)) if len(steps) >= WORD_STEPS else WORD_S
    anchor = {}
    for k in sorted(pair):
        _, start, end = heard[pair[k]]
        i, p, _ = lyric[k]
        if i not in anchor and end - start <= LONG_WORD_S:
            anchor[i] = max(0.0, start - p * word_s)
    order = sorted(anchor)
    anchor = {order[k]: anchor[order[k]] for k in _longest_in_order([anchor[i] for i in order])}
    if not anchor:
        return None, 0
    # index of the first word of each line
    first_word = np.concatenate([[0], np.cumsum([len(line) for line in lines])])[:-1]
    anchored = sorted(anchor)
    times = []
    for i in range(len(lines)):
        if i in anchor:
            t = anchor[i]
        else:
            prev = max((a for a in anchored if a < i), default=None)
            nxt = min((a for a in anchored if a > i), default=None)
            if prev is not None and nxt is not None:
                t = anchor[prev] + (anchor[nxt] - anchor[prev]) * (
                    first_word[i] - first_word[prev]) / (first_word[nxt] - first_word[prev])
            elif nxt is not None:
                t = max(0.0, anchor[nxt] - (first_word[nxt] - first_word[i]) * word_s)
            else:
                t = anchor[prev] + (first_word[i] - first_word[prev]) * word_s
        times.append(float(t))
    return times, len(anchor)


def _sure_pairs(a: list[str], b: list[str]) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """The sure pairs of equal words of a and b (see SURE_REACH), and all pairs."""
    pairs = set(_common_words(a, b))
    return sorted((i, j) for i, j in pairs if (i + 1, j + 1) in pairs or (i - 1, j - 1) in pairs), pairs


@dataclass
class CheckedLineTimes:
    """LRC times checked by the words of a transcription (checked_line_times)."""
    times: list[float]  # the time of each line, in the LRC times (before their common offset)
    offset: float       # the common offset of the LRC times (s): the audio time is LRC time + offset
    confirmed: int      # lines confirmed by the transcription's words
    moved: int          # of these, the lines moved to the time the words give
    respread: int       # lines without such words spread anew between the confirmed ones


def checked_line_times(
    lines: list[list[str]],
    line_times: list[float],
    words: list[tuple[str, float, float]],
) -> Optional[CheckedLineTimes]:
    """The LRC times of the lines of synced lyrics, checked by the words of a transcription.

    ``lines`` holds the words of each line, ``line_times`` their LRC times,
    ``words`` the transcribed words as (text, start s, end s), in order.

    Synced lyrics may have lines at the wrong time (a chorus sung earlier
    than written) or lines that are not sung where they stand; the forced
    alignment would put their words onto whatever is sung there. The
    transcription's words give a time to each line they confirm (a sure
    pair, see CHECKED_MOVE_S): a confirmed line far off its LRC time moves
    to this time, and lines without confirmation that then stand out of
    order are spread anew between the confirmed ones around them.

    Returns None with fewer than MIN_OFFSET_LINES confirmed lines: the
    transcription then cannot check the lyrics.
    """
    lyric = [(i, p, _word_key(w)) for i, line in enumerate(lines) for p, w in enumerate(line)]
    keyed = [k for k, (_, _, key) in enumerate(lyric) if key]
    heard = [(key, start, end) for key, start, end in ((_word_key(w), s, e) for w, s, e in words)
             if key and start is not None and end is not None]
    sure, pairs = _sure_pairs([lyric[k][2] for k in keyed], [h[0] for h in heard])
    # The sung seconds per word: steps between sure pairs in the same line
    steps = [heard[b + 1][1] - heard[b][1] for a, b in sure
             if (a + 1, b + 1) in pairs and lyric[keyed[a + 1]][0] == lyric[keyed[a]][0]]
    word_s = float(np.clip(statistics.median(steps), *WORD_S_RANGE)) if len(steps) >= WORD_STEPS else WORD_S
    confirmed = {}
    for a, b in sure:
        i, p, _ = lyric[keyed[a]]
        _, start, end = heard[b]
        if i not in confirmed and end - start <= LONG_WORD_S:
            confirmed[i] = max(0.0, start - p * word_s)
    order = sorted(confirmed)
    confirmed = {order[k]: confirmed[order[k]] for k in _longest_in_order([confirmed[i] for i in order])}
    if len(confirmed) < MIN_OFFSET_LINES:
        return None
    offset = statistics.median(confirmed[i] - line_times[i] for i in confirmed)
    times = list(line_times)
    moved = 0
    for i, t in confirmed.items():
        if abs(t - offset - line_times[i]) > CHECKED_MOVE_S:
            times[i] = t - offset
            moved += 1
    first_word = np.concatenate([[0], np.cumsum([len(line) for line in lines])])[:-1]
    anchored = sorted(confirmed)
    respread = 0
    for i in range(len(lines)):
        if i in confirmed:
            continue
        prev = max((a for a in anchored if a < i), default=None)
        nxt = min((a for a in anchored if a > i), default=None)
        earliest = times[prev] + len(lines[prev]) * word_s / 2 - FIT_S if prev is not None else -np.inf
        latest = times[nxt] + FIT_S if nxt is not None else np.inf
        if earliest <= times[i] <= latest:
            continue
        if prev is not None and nxt is not None:
            times[i] = times[prev] + (times[nxt] - times[prev]) * (
                first_word[i] - first_word[prev]) / (first_word[nxt] - first_word[prev])
        elif nxt is not None:
            # before the first confirmed line: counted back from it by the words between
            times[i] = times[nxt] - (first_word[nxt] - first_word[i]) * word_s
        else:
            # after the last confirmed line: counted on from it
            times[i] = times[prev] + (first_word[i] - first_word[prev]) * word_s
        respread += 1
    return CheckedLineTimes([float(t) for t in times], float(offset), len(confirmed), moved, respread)


def passages_without_lyrics(
    lines: list[list[str]],
    words: list[tuple[str, float, float]],
    min_words: int = PASSAGE_MIN_WORDS,
) -> tuple[list[tuple[int, int]], set[int]]:
    """The passages of a transcription that the lyrics do not have (an extra
    part of a live version, a chorus sung more often than written), and the
    lines of the lyrics that it has.

    ``lines`` holds the words of each line, ``words`` the transcribed words
    as (text, start s, end s), in the order of the transcription. The words
    are paired as in line_times_from_words. A pair is sure when the next or
    the previous words of both are paired with each other too. A transcribed
    word more than SURE_REACH words away from a sure pair is not sung from
    the lyrics, and ``min_words`` such words in a row make a passage, which
    reaches up to the words in sure pairs around it. Where the lyrics have
    more than PASSAGE_LYRICS_SHARE as many words as the passage between these
    pairs, the transcriber heard their lines badly, and it is no passage.

    Returns the passages as (first, end) indices into ``words`` (without
    words that have no text or no times), and the lines with a sure pair.
    """
    lyric = [(i, _word_key(w)) for i, line in enumerate(lines) for w in line]
    keyed = [k for k, (_, key) in enumerate(lyric) if key]
    heard = [j for j, (w, s, e) in enumerate(words) if _word_key(w) and s is not None and e is not None]
    sure, _ = _sure_pairs([lyric[k][1] for k in keyed], [_word_key(words[j][0]) for j in heard])
    sung_lines = {lyric[keyed[a]][0] for a, _ in sure}
    from_lyrics = np.zeros(len(heard), dtype=bool)
    for _, b in sure:
        from_lyrics[max(0, b - SURE_REACH):b + SURE_REACH + 1] = True
    sure_heard = {b: a for a, b in sure}
    passages, start = [], None
    for b in range(len(heard) + 1):
        if b < len(heard) and not from_lyrics[b]:
            start = b if start is None else start
            continue
        if start is not None and b - start >= min_words:
            # The passage reaches up to the sure pairs around it
            first, end = start, b
            while first > 0 and first - 1 not in sure_heard:
                first -= 1
            while end < len(heard) and end not in sure_heard:
                end += 1
            lyrics_between = ((sure_heard[end] if end < len(heard) else len(keyed))
                              - (sure_heard[first - 1] + 1 if first > 0 else 0))
            if lyrics_between <= PASSAGE_LYRICS_SHARE * (end - first):
                passages.append((heard[first], heard[end - 1] + 1))
        start = None
    return passages, sung_lines
