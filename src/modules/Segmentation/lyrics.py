"""Place lyrics onto predicted notes.

The existing word segments (lyrics source + forced alignment) provide the
text and its timing. Words are split into syllables, then a monotonic dynamic
programme gives every note either a new syllable or a "~" continuation of the
previous one. Syllables without a matching note are merged, in order, into the
note before or after them, never across the start of a line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import librosa
import numpy as np

from modules.Midi.MidiSegment import MidiSegment
from modules.Segmentation.decode import PredictedNote

DROP_COST_MS = 600.0      # cost of a syllable that gets no note of its own
CONTINUE_COST_MS = 150.0  # base cost of a "~" continuation note
MAX_START_COST_MS = 2500.0
MAX_SKIP = 3              # syllables that may be merged in one step
BAND_MS = 8000.0          # only syllables this close to a note are candidates

_WORD_RE = re.compile(r"[\w'’-]+", re.UNICODE)


@dataclass
class Syllable:
    text: str        # with a trailing space when it ends a word
    start_ms: float
    end_ms: float
    line_start: bool = False


def _hyphenator(language: str | None):
    try:
        from hyphen import Hyphenator
        from modules.Speech_Recognition.hyphenation import language_check

        region = language_check((language or "en").lower()[:2])
        return Hyphenator(region) if region else None
    except Exception:  # noqa: BLE001 - hyphenation is optional
        return None


def _hyphen_chain(core: str) -> list[str]:
    """Pieces of a hyphen chain, sung one after another: "ooh-ooh-oh," -> ["ooh-", "ooh-", "oh,"].

    The hyphen stays with the piece before it; a piece without letters (a lone
    or doubled hyphen) joins its neighbour, so it never becomes a syllable.
    """
    pieces: list[str] = []
    for piece in re.split(r"(?<=-)", core):
        if not piece:
            continue
        if pieces and (not re.search(r"\w", piece) or not re.search(r"\w", pieces[-1])):
            pieces[-1] += piece
        else:
            pieces.append(piece)
    return pieces


def _syllable_parts(core: str, hyph) -> list[str]:
    """Syllables of a word: hyphen chains split at their hyphens, then each piece hyphenated."""
    parts: list[str] = []
    for piece in _hyphen_chain(core):
        word = piece.rstrip("-")
        sub = None
        if hyph and len(word) > 3 and "-" not in word and _WORD_RE.fullmatch(word):
            try:
                sub = hyph.syllables(word)
            except Exception:  # noqa: BLE001
                sub = None
        if sub and len(sub) >= 2 and "".join(sub) == word:
            parts.extend(sub[:-1] + [sub[-1] + piece[len(word):]])
        else:
            parts.append(piece)
    return parts


def syllables_from_segments(segments: list[MidiSegment], language: str | None) -> list[Syllable]:
    """Word/syllable tokens with times from word segments, hyphenated further."""
    tokens: list[list] = []  # [text, start_ms, end_ms, line_start]
    line_start = True
    for seg in segments:
        raw = seg.word or ""
        core = raw.strip()
        if (core == "" or core.startswith("~")) and tokens:
            tokens[-1][2] = seg.end * 1000
            if raw.endswith(" ") and not tokens[-1][0].endswith(" "):
                tokens[-1][0] += " "
        elif core:
            tokens.append([raw.lstrip("~"), seg.start * 1000, seg.end * 1000, line_start])
            line_start = False
        if getattr(seg, "line_break_after", False):
            line_start = True

    hyph = _hyphenator(language)
    out: list[Syllable] = []
    for text, a, b, ls in tokens:
        core = text.strip()
        parts = _syllable_parts(core, hyph)
        if len(parts) < 2 or "".join(parts) != core:
            out.append(Syllable(text, a, b, ls))
            continue
        lead = text[: len(text) - len(text.lstrip())]
        trail = " " if text.endswith(" ") else ""
        total = sum(len(p) for p in parts)
        t0 = a
        for k, part in enumerate(parts):
            t1 = a + (b - a) * sum(len(q) for q in parts[: k + 1]) / total
            out.append(Syllable((lead if k == 0 else "") + part + (trail if k == len(parts) - 1 else ""),
                                t0, t1, ls and k == 0))
            t0 = t1
    return out


def align_syllables(note_starts_ms: np.ndarray, syllables: list[Syllable]) -> list[tuple[int, bool]]:
    """Monotonic DP: for each note (syllable index, starts_new_syllable)."""
    k_notes, m = len(note_starts_ms), len(syllables)
    if k_notes == 0 or m == 0:
        return []
    inf = np.inf  # must be non-finite: reachability is checked with np.isfinite
    sa = np.array([s.start_ms for s in syllables])
    sb = np.array([s.end_ms for s in syllables])
    dp = np.full((k_notes, m), inf)
    back = np.full((k_notes, m, 2), -1, dtype=np.int64)

    def start_cost(i: int, j: int) -> float:
        return min(abs(note_starts_ms[i] - sa[j]), MAX_START_COST_MS)

    for j in range(min(m, MAX_SKIP + 1)):
        dp[0, j] = start_cost(0, j) + j * DROP_COST_MS
        back[0, j] = (-1, 1)
    def fill_row(i: int, j_from: int, j_to: int) -> None:
        for j in range(max(j_from, 0), min(j_to, m)):
            best = dp[i - 1, j] + CONTINUE_COST_MS + max(0.0, note_starts_ms[i] - sb[j])
            arg = (j, 0)
            for skip in range(MAX_SKIP + 1):
                pj = j - 1 - skip
                if pj < 0:
                    break
                c = dp[i - 1, pj] + start_cost(i, j) + skip * DROP_COST_MS
                if c < best:
                    best, arg = c, (pj, 1)
            dp[i, j] = best
            back[i, j] = arg

    for i in range(1, k_notes):
        lo = int(np.searchsorted(sa, note_starts_ms[i] - BAND_MS))
        hi = int(np.searchsorted(sa, note_starts_ms[i] + BAND_MS))
        fill_row(i, lo - MAX_SKIP, hi + 1)
        if not np.isfinite(dp[i]).any():
            # Note far from every syllable (e.g. a long untexted ad-lib): widen
            # to all syllables reachable from the previous note so the path
            # continues instead of becoming impossible.
            reach = np.flatnonzero(np.isfinite(dp[i - 1]))
            if len(reach):
                fill_row(i, int(reach[0]), int(reach[-1]) + MAX_SKIP + 2)
    final = dp[k_notes - 1] + (m - 1 - np.arange(m)) * DROP_COST_MS
    if not np.isfinite(final).any():
        return []  # no valid alignment; caller keeps the word-based notes
    j = int(np.argmin(final))
    path: list[tuple[int, bool]] = [(0, True)] * k_notes
    for i in range(k_notes - 1, -1, -1):
        pj, is_start = back[i, j]
        path[i] = (j, bool(is_start))
        if pj >= 0:
            j = int(pj)
    return path


def _gap_split(gap: list[Syllable], prev_end_ms: float | None, next_start_ms: float,
               next_starts_line: bool) -> int:
    """How many of the syllables without a note of their own (in sung order) join the
    note before them; the rest join the note after them.

    A line start among them, or at the next syllable, fixes the split there, so no
    syllable moves into another line. Otherwise the split with the least distance
    in time wins.
    """
    if prev_end_ms is None:
        return 0
    for k, s in enumerate(gap):
        if s.line_start:
            return k
    if next_starts_line:
        return len(gap)
    best, best_cost = 0, float("inf")
    for split in range(len(gap) + 1):
        cost = (sum(abs(s.start_ms - prev_end_ms) for s in gap[:split])
                + sum(abs(next_start_ms - s.end_ms) for s in gap[split:]))
        if cost < best_cost:
            best, best_cost = split, cost
    return best


def place_lyrics(notes: list[PredictedNote], syllables: list[Syllable]) -> list[MidiSegment]:
    """MidiSegments with syllable text, "~" continuations and line breaks."""
    if not notes:
        return []
    path = align_syllables(np.array([n.start * 1000 for n in notes]), syllables)
    if not path:
        return []
    texts: list[str] = []
    line_starts: set[int] = set()
    prev_j = -1
    prev_head = -1        # index of the note that carries syllable prev_j
    prev_end_ms = 0.0     # end of the last note carrying prev_j (incl. continuations)
    for i, (j, is_start) in enumerate(path):
        if is_start and j != prev_j:
            # Syllables without a note of their own join the note before or after
            # them, keeping their order and their line.
            gap = syllables[prev_j + 1:j]
            split = _gap_split(gap, prev_end_ms if prev_head >= 0 else None, notes[i].start * 1000,
                               syllables[j].line_start)
            for s in gap[:split]:
                texts[prev_head] += s.text
            head = "".join(s.text for s in gap[split:])
            tail_line = any(s.line_start for s in gap[split:])
            texts.append(head + syllables[j].text)
            if tail_line or syllables[j].line_start:
                line_starts.add(i)
            prev_j, prev_head = j, i
        else:
            texts.append("~")
        prev_end_ms = notes[i].end * 1000
    if prev_j < len(syllables) - 1:
        tail = "".join(s.text for s in syllables[prev_j + 1:])
        last = max((i for i, t in enumerate(texts) if t != "~"), default=len(texts) - 1)
        texts[last] = (texts[last].rstrip() + " " + tail.lstrip()) if texts[last] != "~" else tail
    # The trailing space marks a word boundary and belongs to the LAST note of
    # the syllable group: "mind" "~" "~ " rather than "mind " "~" "~".
    for i, text in enumerate(texts):
        if text == "~" or not text.endswith(" "):
            continue
        last = i
        while last + 1 < len(texts) and texts[last + 1] == "~":
            last += 1
        if last != i:
            texts[i] = text.rstrip(" ")
            texts[last] = "~ "
    segments = [MidiSegment(librosa.midi_to_note(n.midi), n.start, n.end, text, "F" if n.freestyle else ":")
                for n, text in zip(notes, texts)]
    for i in line_starts:
        if i > 0:
            segments[i - 1].line_break_after = True
    return segments
