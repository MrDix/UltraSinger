"""Place lyrics onto predicted notes.

The existing word segments (lyrics source + forced alignment) provide the
text and its timing. Words are split into syllables, then a monotonic dynamic
programme gives every note either a new syllable or a "~" continuation of the
previous one. Syllables without a matching note are merged into the next one.
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
        parts = None
        if hyph and len(core) > 3 and _WORD_RE.fullmatch(core):
            try:
                parts = hyph.syllables(core)
            except Exception:  # noqa: BLE001
                parts = None
        if not parts or len(parts) < 2 or "".join(parts) != core:
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
    inf = 1e18
    sa = np.array([s.start_ms for s in syllables])
    sb = np.array([s.end_ms for s in syllables])
    dp = np.full((k_notes, m), inf)
    back = np.full((k_notes, m, 2), -1, dtype=np.int64)

    def start_cost(i: int, j: int) -> float:
        return min(abs(note_starts_ms[i] - sa[j]), MAX_START_COST_MS)

    for j in range(min(m, MAX_SKIP + 1)):
        dp[0, j] = start_cost(0, j) + j * DROP_COST_MS
        back[0, j] = (-1, 1)
    for i in range(1, k_notes):
        lo = int(np.searchsorted(sa, note_starts_ms[i] - BAND_MS))
        hi = int(np.searchsorted(sa, note_starts_ms[i] + BAND_MS))
        for j in range(max(lo - MAX_SKIP, 0), min(hi + 1, m)):
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
    final = dp[k_notes - 1] + (m - 1 - np.arange(m)) * DROP_COST_MS
    j = int(np.argmin(final))
    path: list[tuple[int, bool]] = [(0, True)] * k_notes
    for i in range(k_notes - 1, -1, -1):
        pj, is_start = back[i, j]
        path[i] = (j, bool(is_start))
        if pj >= 0:
            j = int(pj)
    return path


def place_lyrics(notes: list[PredictedNote], syllables: list[Syllable]) -> list[MidiSegment]:
    """MidiSegments with syllable text, "~" continuations and line breaks."""
    if not notes:
        return []
    path = align_syllables(np.array([n.start * 1000 for n in notes]), syllables)
    if not path:
        return [MidiSegment(librosa.midi_to_note(n.midi), n.start, n.end, "~ ", "F" if n.freestyle else ":")
                for n in notes]
    texts: list[str] = []
    line_starts: set[int] = set()
    prev_j = -1
    prev_head = -1        # index of the note that carries syllable prev_j
    prev_end_ms = 0.0     # end of the last note carrying prev_j (incl. continuations)
    for i, (j, is_start) in enumerate(path):
        if is_start and j != prev_j:
            # Syllables without a note of their own join the closer neighbour in time.
            head, tail_line = "", False
            for k in range(prev_j + 1, j):
                s = syllables[k]
                to_prev = abs(s.start_ms - prev_end_ms) if prev_head >= 0 else float("inf")
                to_next = abs(notes[i].start * 1000 - s.end_ms)
                if to_prev < to_next and not s.line_start:
                    texts[prev_head] += s.text
                else:
                    head += s.text
                    tail_line = tail_line or s.line_start
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
