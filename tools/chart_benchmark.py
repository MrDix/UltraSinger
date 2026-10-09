"""Chart benchmark: measure generated charts against hand-made reference charts.

The game score (``tools/regression_benchmark.py``) only tells how well a
singer hits a chart. It cannot tell a good chart from a bad one: a chart
that slavishly traces the vocal stem scores just as high as a carefully
charted one, while being much harder to read and sing. This tool instead
compares UltraSinger's output with an existing, hand-made UltraStar chart of
the same song, so pipeline changes can be judged without anyone singing.

Workflow (all data lives in a WORKDIR outside this repository)::

    # 1. pick a reproducible sample of songs from your UltraStar library
    python tools/chart_benchmark.py sample D:/Songs D:/bench --count 100

    # 2. convert every sampled song with the current pipeline
    python tools/chart_benchmark.py convert D:/bench --label baseline

    # 3. measure the generated charts against the reference charts
    python tools/chart_benchmark.py evaluate D:/bench --label baseline --plots

    # 4. after a code change: convert + evaluate under a new label, then compare
    python tools/chart_benchmark.py compare D:/bench baseline my-change

A library song qualifies when its folder holds exactly one solo UltraStar
TXT (no duets) with enough pitched notes and the audio (or video) file it
references. Inputs are copied into the WORKDIR first, so the reference TXT
next to the original audio can never leak into the conversion.

Reference charts are only as good as their timing. ``evaluate`` therefore
fits the reference to the sung pitch of the separated vocal (searching a
time offset) and flags songs whose reference does not follow the vocal well
(``ref_fit_pct`` below ``--min-ref-fit``); those are left out of the
summary. Song IDs in all reports are anonymous (``song_001`` ...); the
mapping to real folders is only kept in ``WORKDIR/songs.json``.

Some charts carry a second voice appended after the end of the song (a duet
flattened into one track: the notes go on for about another song length
after the audio ends). ``evaluate`` folds such a part back onto the song by
fitting its own offset to the sung pitch and then accepts a generated note
that matches either voice. A part that does not fit the vocal well enough is
dropped, as notes after the end of the audio cannot be measured.

The WORKDIR contains file names of your library and copies of your audio.
Never check it into git.

Metrics (per song; the summary reports the median over reliable songs):

``chart_agreement_pct`` (primary)
    Share of the reference's pitched note time on which the generated chart
    also has a note within +-1 semitone (octave folded, like the games on
    Medium). Equivalent to "a singer who sings the generated chart perfectly
    scores this on the reference chart". Generated notes where the reference
    has none do not lower it - see the precision below.
``chart_precision_pct`` / ``chart_f1_pct``
    Share of the generated pitched note time that agrees with the reference
    (within +-1 semitone, folded): what a singer who sings the reference
    perfectly scores on the generated chart, as the games divide by the
    chart's own note time. Notes that run past the reference notes or chart
    backing vocals lower the precision but not the agreement. The F1 is the
    harmonic mean of agreement and precision.
``onset_hit_50_pct`` / ``onset_hit_100_pct``
    Reference note onsets with a generated onset within 50 / 100 ms.
``onset_precision_100_pct``
    Generated onsets with a reference onset within 100 ms (low = notes
    that do not exist in the reference, e.g. over-segmentation).
``note_count_ratio`` / ``median_note_ms`` / ``short_notes_pct``
    Generated note profile (ratio 1.0 = same number of notes as the
    reference; short = under 150 ms).
``pitch_agree_pct``
    Where both charts have a note: pitch within +-1 semitone, folded.
``ref_coverage_pct`` / ``extra_time_pct``
    Reference note time covered by any generated note / generated note
    time where the reference has no pitched note (backing vocals, ad-libs).
``freestyle_charted_pct``
    Reference freestyle/rap time that the generated chart covers with
    pitched notes (spoken passages charted as melody).
``oracle_pitch_pct``
    Pitch accuracy the pitch tracker would reach with the reference's note
    boundaries (median of confident frames per reference note). Shows how
    much is lost in segmentation rather than in pitch detection.
``vocal_hits_ref_pct`` / ``vocal_hits_gen_pct``
    Share of confident sung frames inside notes that hit the reference /
    generated chart (+-1 semitone, folded).

Lyrics are compared word by word, so different hyphenation does not count as
an error; case, accents, apostrophes and punctuation are ignored:

``lyrics_agreement_pct``
    Share of the reference's sung time (notes with text) on which the
    generated chart sings the same word: the text is under the right notes.
``lyrics_agree_pct``
    The same, counted only where both charts have a note (placement without
    the time the generated chart leaves out).
``lyrics_words_found_pct``
    Reference words that occur in the generated lyrics, in order.
``word_start_100_pct`` / ``word_start_250_pct``
    Found words whose generated start lies within 100 / 250 ms of the
    reference start.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_ultrastar import _beat_to_ms, parse_ultrastar  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

FRAME_MS = 10
CONFIDENCE = 0.7
MIN_PITCHED_NOTES = 100
OFFSET_SEARCH_MS = 1200
OFFSET_STEP_MS = 10
SHORT_NOTE_MS = 150
WORD_TIME_SCALE_MS = 3000.0  # matching repeated words: prefer occurrences this close in time
# A chart continues after the song when at least this share of its pitched
# note time (and this many pitched notes) starts after the end of the audio.
APPENDED_MIN_SHARE = 0.1
APPENDED_MIN_NOTES = 20
APPENDED_SEARCH_MARGIN_MS = 10000
DEFAULT_TIMEOUT_S = 3600
MAX_INPUT_STEM = 100

AUDIO_EXTENSIONS = {".mp3", ".ogg", ".m4a", ".wav", ".flac", ".opus", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".webm", ".mov", ".mpg", ".mpeg", ".m4v", ".divx"}

PITCHED_TYPES = {":", "*"}
FREESTYLE_TYPES = {"F", "R", "G"}

PRIMARY_METRIC = "chart_agreement_pct"
SUMMARY_METRICS = [
    "chart_agreement_pct", "chart_precision_pct", "chart_f1_pct",
    "onset_hit_50_pct", "onset_hit_100_pct", "onset_precision_100_pct",
    "note_count_ratio", "median_note_ms", "short_notes_pct", "pitch_agree_pct",
    "ref_coverage_pct", "extra_time_pct", "freestyle_charted_pct", "oracle_pitch_pct",
    "vocal_hits_ref_pct", "vocal_hits_gen_pct",
    "lyrics_agreement_pct", "lyrics_agree_pct", "lyrics_words_found_pct",
    "word_start_100_pct", "word_start_250_pct",
]


# ---------------------------------------------------------------------------
# Notes and frame grids
# ---------------------------------------------------------------------------

@dataclass
class ChartNote:
    start_ms: float
    end_ms: float
    midi: int
    kind: str
    word: str = ""
    voice: int = 0  # > 0: an extra voice of the reference (see split_appended_voice)

    @property
    def pitched(self) -> bool:
        return self.kind in PITCHED_TYPES

    @property
    def freestyle(self) -> bool:
        return self.kind in FREESTYLE_TYPES


def load_chart(path: str | Path, shift_ms: float = 0.0) -> list[ChartNote]:
    """Parse an UltraStar TXT into notes with absolute times (ms) and MIDI pitch."""
    parsed = parse_ultrastar(path)
    return [ChartNote(n.start_ms + shift_ms, n.end_ms + shift_ms, n.midi, n.note_type, n.word.strip())
            for n in parsed.notes]


def shift_notes(notes: list[ChartNote], shift_ms: float) -> list[ChartNote]:
    return [ChartNote(n.start_ms + shift_ms, n.end_ms + shift_ms, n.midi, n.kind, n.word, n.voice)
            for n in notes]


def fold(diff):
    """Fold a semitone difference into [-6, 6) the way the games ignore octaves."""
    return (np.asarray(diff) + 6) % 12 - 6


def frame_grid(notes: list[ChartNote], n_frames: int, kinds: set[str] = PITCHED_TYPES) -> np.ndarray:
    """Per-frame MIDI pitch of the notes of the given kinds (NaN = no note)."""
    grid = np.full(n_frames, np.nan)
    for n in notes:
        if n.kind not in kinds:
            continue
        a = max(int(n.start_ms // FRAME_MS), 0)
        b = min(max(int(n.end_ms // FRAME_MS), 0), n_frames)
        if b > a:
            grid[a:b] = n.midi
    return grid


@dataclass
class SungPitch:
    """Confident pitch frames of the separated vocal (seconds, MIDI)."""
    times: np.ndarray
    midi: np.ndarray
    duration_s: float | None = None  # length of the analysed audio, if known

    @classmethod
    def from_pitch_json(cls, path: str | Path, confidence: float = CONFIDENCE) -> "SungPitch":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        t = np.asarray(data["times"], dtype=float)
        f = np.asarray(data["frequencies"], dtype=float)
        c = np.asarray(data["confidence"], dtype=float)
        keep = (c >= confidence) & (f > 40.0)
        duration = float(t[-1]) if len(t) else None  # the pitch track covers the whole audio
        return cls(t[keep], 69.0 + 12.0 * np.log2(f[keep] / 440.0), duration)

    def frame_indices(self) -> np.ndarray:
        return (self.times * 1000.0 // FRAME_MS).astype(int)


def _grid_at(grid: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.full(len(idx), np.nan)
    ok = (idx >= 0) & (idx < len(grid))
    out[ok] = grid[idx[ok]]
    return out


def hit_rate(grid: np.ndarray | list[np.ndarray], sung: SungPitch) -> float | None:
    """Share of sung frames inside notes that are within +-1 semitone (folded).

    With several grids (voices), a frame inside any voice's note counts and
    hits when it matches any of them.
    """
    grids = grid if isinstance(grid, list) else [grid]
    idx = sung.frame_indices()
    inside = np.zeros(len(idx), bool)
    hit = np.zeros(len(idx), bool)
    for g in grids:
        target = _grid_at(g, idx)
        m = ~np.isnan(target)
        inside |= m
        hit[m] |= np.abs(fold(sung.midi[m] - target[m])) <= 1
    if not inside.any():
        return None
    return float(np.mean(hit[inside]))


def fit_reference_offset(ref: list[ChartNote], sung: SungPitch,
                         search_ms: int = OFFSET_SEARCH_MS,
                         step_ms: int = OFFSET_STEP_MS,
                         center_ms: int = 0) -> tuple[float, float]:
    """Find the time shift that lays the reference notes best onto the sung pitch.

    Only the reference and the vocal are used, never the generated chart, so
    the fit cannot favour the chart under test. Shifts within
    ``center_ms +- search_ms`` are tried. Returns ``(offset_ms, fit)`` where
    ``fit`` is the hit rate of the sung pitch on the shifted reference.
    """
    idx = sung.frame_indices()
    end = max((n.end_ms for n in ref), default=0.0) + max(center_ms, 0) + search_ms + 1000
    n_frames = int(end // FRAME_MS) + 1
    best = (0.0, -1.0, 0.0)  # (offset, weighted score, fit)
    for shift in range(center_ms - search_ms, center_ms + search_ms + 1, step_ms):
        target = _grid_at(frame_grid(shift_notes(ref, shift), n_frames), idx)
        m = ~np.isnan(target)
        if m.sum() < 50:
            continue
        fit = float(np.mean(np.abs(fold(sung.midi[m] - target[m])) <= 1))
        score = fit * m.sum()  # prefer shifts that also cover more sung frames
        if score > best[1]:
            best = (float(shift), score, fit)
    return best[0], best[2]


def split_appended_voice(notes: list[ChartNote],
                         duration_ms: float | None) -> tuple[list[ChartNote], list[ChartNote]]:
    """Split off a second voice that a chart appends after the end of the song.

    A duet flattened into one track sometimes lists the second voice after
    the first one, so its notes start after the end of the audio. Returns
    ``(song, appended)``; ``appended`` is empty unless the notes starting
    after ``duration_ms`` make up a real part of the chart.
    """
    if not duration_ms:
        return notes, []
    late = [n for n in notes if n.start_ms >= duration_ms]
    late_pitched = [n for n in late if n.pitched]
    total = sum(n.end_ms - n.start_ms for n in notes if n.pitched)
    late_time = sum(n.end_ms - n.start_ms for n in late_pitched)
    if len(late_pitched) < APPENDED_MIN_NOTES or total <= 0 or late_time < APPENDED_MIN_SHARE * total:
        return notes, []
    return [n for n in notes if n.start_ms < duration_ms], late


def fit_appended_offset(part: list[ChartNote], sung: SungPitch, duration_ms: float) -> tuple[float, float]:
    """Offset that lays an appended voice onto the song (coarse search over the
    whole song, then the usual fine search). Returns ``(offset_ms, fit)``."""
    first = min(n.start_ms for n in part)
    last = max(n.end_ms for n in part)
    lo = int(-first - APPENDED_SEARCH_MARGIN_MS)
    hi = int(duration_ms - last + APPENDED_SEARCH_MARGIN_MS)
    hi = max(hi, lo)
    center = (lo + hi) // 2
    coarse, _ = fit_reference_offset(part, sung, search_ms=(hi - lo) // 2 + 50, step_ms=50, center_ms=center)
    return fit_reference_offset(part, sung, search_ms=60, step_ms=OFFSET_STEP_MS, center_ms=int(coarse))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _pct(x: float | None) -> float | None:
    return None if x is None else round(100.0 * x, 1)


def _f1(recall: float | None, precision: float | None) -> float | None:
    """Harmonic mean of agreement (recall) and precision."""
    if recall is None or precision is None:
        return None
    return 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)


def _onset_hits(onsets: np.ndarray, others: np.ndarray, tol_ms: float) -> float | None:
    if len(onsets) == 0:
        return None
    if len(others) == 0:
        return 0.0
    others = np.sort(others)
    pos = np.searchsorted(others, onsets)
    left = np.abs(onsets - others[np.clip(pos - 1, 0, len(others) - 1)])
    right = np.abs(others[np.clip(pos, 0, len(others) - 1)] - onsets)
    return float(np.mean(np.minimum(left, right) <= tol_ms))


def oracle_pitch(ref: list[ChartNote], sung: SungPitch) -> float | None:
    """Pitch accuracy with the reference's own note boundaries."""
    hits = []
    for n in ref:
        if not n.pitched:
            continue
        sel = (sung.times * 1000 >= n.start_ms) & (sung.times * 1000 <= n.end_ms)
        if sel.sum() < 2:
            continue
        hits.append(abs(float(fold(np.round(np.median(sung.midi[sel])) - n.midi))) <= 1)
    return float(np.mean(hits)) if hits else None


def voice_grids(notes: list[ChartNote], n_frames: int, kinds: set[str] = PITCHED_TYPES) -> list[np.ndarray]:
    """One frame grid per voice of the chart (see ``frame_grid``)."""
    voices = sorted({n.voice for n in notes}) or [0]
    return [frame_grid([n for n in notes if n.voice == v], n_frames, kinds) for v in voices]


def chart_metrics(ref: list[ChartNote], gen: list[ChartNote], sung: SungPitch | None = None) -> dict:
    """Compare a generated chart with a (time-aligned) reference chart.

    A reference with several voices (``ChartNote.voice``) counts a frame as
    charted when any voice has a note there, and a generated note agrees when
    it matches any of them.
    """
    end = max([n.end_ms for n in ref + gen] or [0.0]) + 1000
    n_frames = int(end // FRAME_MS) + 1
    r_grids = voice_grids(ref, n_frames)
    g = frame_grid(gen, n_frames)
    g_on = ~np.isnan(g)
    r_on = np.zeros(n_frames, bool)
    agree_at = np.zeros(n_frames, bool)
    for r in r_grids:
        on = ~np.isnan(r) & g_on
        r_on |= ~np.isnan(r)
        agree_at[on] |= np.abs(fold(g[on] - r[on])) <= 1
    r_free = np.zeros(n_frames, bool)
    for r in voice_grids(ref, n_frames, FREESTYLE_TYPES):
        r_free |= ~np.isnan(r)
    both = r_on & g_on
    agree = agree_at[both]
    recall = agree.sum() / r_on.sum() if r_on.any() else None
    precision = agree.sum() / g_on.sum() if g_on.any() else None

    ref_p = [n for n in ref if n.pitched]
    gen_p = [n for n in gen if n.pitched]
    ref_on = np.array([n.start_ms for n in ref_p])
    gen_on = np.array([n.start_ms for n in gen_p])
    gen_dur = [n.end_ms - n.start_ms for n in gen_p]

    m = {
        "ref_notes": len(ref_p),
        "gen_notes": len(gen_p),
        "chart_agreement_pct": _pct(recall),
        "chart_precision_pct": _pct(precision),
        "chart_f1_pct": _pct(_f1(recall, precision)),
        "onset_hit_50_pct": _pct(_onset_hits(ref_on, gen_on, 50)),
        "onset_hit_100_pct": _pct(_onset_hits(ref_on, gen_on, 100)),
        "onset_precision_100_pct": _pct(_onset_hits(gen_on, ref_on, 100)),
        "note_count_ratio": round(len(gen_p) / len(ref_p), 2) if ref_p else None,
        "median_note_ms": round(float(np.median(gen_dur))) if gen_dur else None,
        "short_notes_pct": _pct(float(np.mean(np.array(gen_dur) < SHORT_NOTE_MS))) if gen_dur else None,
        "pitch_agree_pct": _pct(float(agree.mean())) if both.any() else None,
        "ref_coverage_pct": _pct(both.sum() / r_on.sum()) if r_on.any() else None,
        "extra_time_pct": _pct((g_on & ~r_on).sum() / g_on.sum()) if g_on.any() else None,
        "freestyle_charted_pct": _pct((g_on & r_free).sum() / r_free.sum()) if r_free.any() else None,
    }
    if sung is not None:
        m["oracle_pitch_pct"] = _pct(oracle_pitch(ref, sung))
        m["vocal_hits_ref_pct"] = _pct(hit_rate(r_grids, sung))
        m["vocal_hits_gen_pct"] = _pct(hit_rate(g, sung))
    return m


# ---------------------------------------------------------------------------
# Lyrics
# ---------------------------------------------------------------------------

@dataclass
class ChartWord:
    """A word as the games display it, with the time span of every note sung on it."""
    text: str  # normalized, see normalize_word()
    spans: list[tuple[float, float]]  # (start_ms, end_ms) per note

    @property
    def start_ms(self) -> float:
        return self.spans[0][0]


def normalize_word(text: str) -> str:
    """Word key that ignores case, accents, apostrophes and punctuation ("Don't" -> "dont")."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if c.isalnum() and not unicodedata.combining(c)).casefold()


# The text of a note line starts after exactly one separator; further spaces belong to it.
_NOTE_TEXT_RE = re.compile(r"^\s*\S+\s+\S+\s+\S+\s+\S+(?:[ \t](.*))?$")


def load_words(path: str | Path, shift_ms: float = 0.0) -> list[ChartWord]:
    """Words of an UltraStar TXT, joined from its syllables the way the games show them.

    A syllable starts a new word when it begins with a space, when the previous
    syllable ended with one, or after a line break. "~" notes and notes without
    text extend the current word. A note that carries several words shares its
    time among them by their length. Words that are empty after normalization
    (punctuation only) are dropped.
    """
    parsed = parse_ultrastar(path)
    words: list[list] = []  # [raw text, spans]
    new_word = True
    for raw in _read_text(Path(path)).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper() == "E":
            break
        if line.startswith("-"):
            new_word = True
            continue
        parts = line.split(None, 4)
        if len(parts) < 4 or parts[0] not in PITCHED_TYPES | FREESTYLE_TYPES:
            continue
        try:
            beat, length = int(parts[1]), int(parts[2])
            int(parts[3])
        except ValueError:
            continue
        match = _NOTE_TEXT_RE.match(raw)
        text = (match.group(1) or "") if match else ""
        start = _beat_to_ms(beat, parsed.bpm, parsed.gap) + shift_ms
        end = _beat_to_ms(beat + length, parsed.bpm, parsed.gap) + shift_ms
        tokens = text.split()
        if words and (not tokens or tokens[0].startswith("~")):
            words[-1][0] += "".join(tokens)
            words[-1][1].append((start, end))
            new_word = new_word or text[-1:].isspace()
            continue
        total, done = sum(len(t) for t in tokens), 0
        for k, token in enumerate(tokens):
            a = start + (end - start) * done / total
            done += len(token)
            span = (a, start + (end - start) * done / total)
            if k == 0 and words and not new_word and not text[:1].isspace():
                words[-1][0] += token
                words[-1][1].append(span)
            else:
                words.append([token, [span]])
        new_word = text[-1:].isspace()
    out = []
    for raw_text, spans in words:
        key = normalize_word(raw_text)
        if key:
            out.append(ChartWord(key, spans))
    return out


def _word_grid(words: list[ChartWord], n_frames: int, vocab: dict[str, int]) -> np.ndarray:
    """Per-frame word id (-1 = no note with text)."""
    grid = np.full(n_frames, -1, dtype=np.int64)
    for w in words:
        wid = vocab.setdefault(w.text, len(vocab))
        for start, end in w.spans:
            a = max(int(start // FRAME_MS), 0)
            b = min(max(int(end // FRAME_MS), 0), n_frames)
            if b > a:
                grid[a:b] = wid
    return grid


def align_words(ref: list[ChartWord], gen: list[ChartWord]) -> list[tuple[int, int]]:
    """Order-preserving pairs ``(ref index, gen index)`` of equal words.

    Finds the most pairs (longest common subsequence); among equally long
    matchings, close start times win, so a repeated chorus word is paired with
    the nearby occurrence rather than with one a verse away.
    """
    n, m = len(ref), len(gen)
    if not n or not m:
        return []
    gen_text = np.array([w.text for w in gen], dtype=object)
    gen_start = np.array([w.start_ms for w in gen])
    score = np.zeros((n + 1, m + 1), dtype=np.int64)
    for i, w in enumerate(ref, 1):
        closeness = np.round(500 * np.exp(-np.abs(gen_start - w.start_ms) / WORD_TIME_SCALE_MS))
        gain = np.where(gen_text == w.text, 1000 + closeness.astype(np.int64), -1)
        # score[i, j] = max(score[i-1, j], score[i, j-1], score[i-1, j-1] + gain[j-1])
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


LYRICS_METRICS = ("lyrics_agreement_pct", "lyrics_agree_pct", "lyrics_words_found_pct",
                  "word_start_100_pct", "word_start_250_pct")


def lyrics_metrics(ref: list[ChartWord], gen: list[ChartWord]) -> dict:
    """Compare the generated lyrics and their placement with a (time-aligned) reference."""
    counts = {"ref_words": len(ref), "gen_words": len(gen)}
    if not ref:
        return {**counts, **dict.fromkeys(LYRICS_METRICS)}
    end = max(span[1] for w in ref + gen for span in w.spans) + 1000
    n_frames = int(end // FRAME_MS) + 1
    vocab: dict[str, int] = {}
    r = _word_grid(ref, n_frames, vocab)
    g = _word_grid(gen, n_frames, vocab)
    r_on = r >= 0
    same = r_on & (g == r)
    both = r_on & (g >= 0)
    pairs = align_words(ref, gen)
    dt = np.array([abs(gen[j].start_ms - ref[i].start_ms) for i, j in pairs])
    return {
        **counts,
        "lyrics_agreement_pct": _pct(float(same.sum() / r_on.sum())) if r_on.any() else None,
        "lyrics_agree_pct": _pct(float(same.sum() / both.sum())) if both.any() else None,
        "lyrics_words_found_pct": _pct(len(pairs) / len(ref)),
        "word_start_100_pct": _pct(float(np.mean(dt <= 100))) if len(dt) else None,
        "word_start_250_pct": _pct(float(np.mean(dt <= 250))) if len(dt) else None,
    }


# ---------------------------------------------------------------------------
# Library discovery and sampling
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^#([A-Za-z0-9_]+):(.*)$")


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def inspect_song_folder(folder: Path, prefer_video: bool = False) -> dict | None:
    """Return a candidate entry if the folder holds a usable solo chart + media."""
    txts = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    if len(txts) != 1:
        return None
    text = _read_text(txts[0])
    headers: dict[str, str] = {}
    pitched = 0
    for line in text.splitlines():
        line = line.strip()
        hm = _HEADER_RE.match(line)
        if hm:
            headers[hm.group(1).upper()] = hm.group(2).strip()
        elif line[:2] in (": ", "* "):
            pitched += 1
        elif re.match(r"^P\s*\d", line):
            return None  # duet
    if pitched < MIN_PITCHED_NOTES or "BPM" not in headers:
        return None

    def existing(name: str | None, exts: set[str]) -> Path | None:
        if not name:
            return None
        p = folder / name
        return p if p.is_file() and p.suffix.lower() in exts else None

    audio = existing(headers.get("AUDIO"), AUDIO_EXTENSIONS) or existing(headers.get("MP3"), AUDIO_EXTENSIONS)
    video = existing(headers.get("VIDEO"), VIDEO_EXTENSIONS)
    media = (video or audio) if prefer_video else (audio or video)
    if media is None:
        return None
    return {"folder": str(folder), "txt": str(txts[0]), "media": str(media),
            "audio": str(audio) if audio else None, "pitched_notes": pitched}


def find_candidates(library: Path, prefer_video: bool = False) -> list[dict]:
    """Walk the library and collect every qualifying song folder."""
    out = []
    for folder in sorted({p.parent for p in library.rglob("*.txt")}):
        try:
            entry = inspect_song_folder(folder, prefer_video)
        except OSError:
            continue
        if entry:
            out.append(entry)
    return out


def sample_songs(candidates: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    picked = rng.sample(candidates, min(count, len(candidates)))
    return [{"id": f"song_{i + 1:03d}", **entry} for i, entry in enumerate(picked)]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _clean_input_name(media: Path) -> str:
    """Drop bracketed tags like ``[CO]`` from a file name (keeps 'Artist - Title')."""
    stem = re.sub(r"\s*\[[^\]]*\]", "", media.stem).strip() or "song"
    return stem + media.suffix.lower()


_INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def input_name(song: dict, media: Path) -> str:
    """File name for the conversion input: 'Artist - Title.ext' from the reference chart.

    Library media files are not always named after the song, but UltraSinger
    derives its metadata and lyrics lookup from the input file name. Using the
    chart's #ARTIST/#TITLE mimics a properly named download; the media file
    name is the fallback.
    """
    try:
        headers = {}
        for line in _read_text(Path(song["txt"])).splitlines():
            hm = _HEADER_RE.match(line.strip())
            if hm:
                headers[hm.group(1).upper()] = hm.group(2).strip()
    except (OSError, KeyError):
        headers = {}
    # Strip separator characters at both ends first, so they can neither eat
    # into the length budget nor leave an empty field after truncation.
    artist = _INVALID_PATH_CHARS.sub("", headers.get("ARTIST", "")).strip(" .-")
    title = _INVALID_PATH_CHARS.sub("", headers.get("TITLE", "")).strip(" .-")
    if artist and title:
        # Keep well below file name / path length limits (the run directory
        # and UltraSinger's output folder add to the full path length).
        # Both fields must survive: UltraSinger splits the name at " - ".
        budget = MAX_INPUT_STEM - 3
        if len(artist) + len(title) > budget:
            artist = artist[:max(budget - len(title), budget // 2)].rstrip(" .-")
            title = title[:budget - len(artist)].rstrip(" .-")
        if artist and title:
            return f"{artist} - {title}{media.suffix.lower()}"
    return _clean_input_name(media)


def has_audio_stream(path: Path) -> bool:
    """True if ffprobe finds an audio stream (or ffprobe is unavailable)."""
    try:
        proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                               "stream=index", "-of", "csv=p=0", str(path)],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return True
    return proc.returncode != 0 or bool(proc.stdout.strip())


def choose_input(song: dict) -> Path:
    """The file to convert: the sampled media, unless it is a video without sound."""
    media = Path(song["media"])
    audio = song.get("audio")
    if media.suffix.lower() in VIDEO_EXTENSIONS and audio and not has_audio_stream(media):
        return Path(audio)
    return media


def _song_output_txt(out_dir: Path) -> Path | None:
    txts = [p for p in out_dir.rglob("*.txt")
            if "cache" not in p.parts and not p.name.endswith("_info.txt")]
    return max(txts, key=lambda p: p.stat().st_mtime) if txts else None


_KEEP_SUFFIXES = {".txt", ".json"}


def prune_run_output(out_dir: Path) -> int:
    """Delete everything but TXT/JSON files (audio, video, stems, PDFs) to save disk.

    The evaluation only needs the generated TXT and the cached pitch JSON.
    Returns the number of bytes freed.
    """
    freed = 0
    for p in sorted(out_dir.rglob("*"), reverse=True):
        if p.is_file() and p.suffix.lower() not in _KEEP_SUFFIXES:
            freed += p.stat().st_size
            p.unlink()
        elif p.is_dir() and not any(p.iterdir()):
            p.rmdir()
    return freed


def _as_text(data) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def convert_song(song: dict, run_dir: Path, extra_args: list[str], python: str,
                 keep_audio: bool = False, timeout_s: float | None = DEFAULT_TIMEOUT_S) -> dict:
    """Convert one song into ``run_dir`` (resumable via ``result.json``).

    A conversion that exceeds ``timeout_s`` is recorded as failed
    (``returncode`` -1, ``timed_out``) so the batch can continue.
    """
    marker = run_dir / "result.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    media = choose_input(song)
    inp = run_dir / "input" / input_name(song, media)
    inp.parent.mkdir(parents=True, exist_ok=True)
    if not inp.exists():
        shutil.copy2(media, inp)
    out = run_dir / "out"
    t0 = time.time()
    cmd = [python, str(REPO / "src" / "UltraSinger.py"), "-i", str(inp), "-o", str(out), "--keep_cache",
           *extra_args]
    timed_out = False
    try:
        proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                              cwd=str(REPO), timeout=timeout_s)
        returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode, stdout = -1, _as_text(exc.stdout)
        stderr = _as_text(exc.stderr) + f"\nTIMEOUT after {timeout_s} s\n"
    log = re.sub(r"\x1b\[[0-9;]*m", "", _as_text(stdout) + _as_text(stderr))
    (run_dir / "log.txt").write_text(log, encoding="utf-8")
    txt = _song_output_txt(out) if out.exists() and not timed_out else None
    result = {"returncode": returncode, "seconds": round(time.time() - t0),
              "txt": str(txt) if txt else None, "timed_out": timed_out}
    shutil.rmtree(inp.parent, ignore_errors=True)
    if not keep_audio and out.exists():
        prune_run_output(out)
    marker.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def find_pitch_cache(song_out_dir: Path) -> Path | None:
    """Locate UltraSinger's cached pitch JSON (``cache/<pitcher>_<flag>.json``)."""
    for p in sorted(song_out_dir.glob("cache/*.json")):
        try:
            head = p.read_text(encoding="utf-8")[:200]
        except OSError:
            continue
        if '"times"' in head:
            return p
    return None


def evaluate_song(song: dict, run_dir: Path, min_ref_fit: float, plots: bool) -> dict:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if not result.get("txt") or not Path(result["txt"]).exists():
        return {"id": song["id"], "status": "no_output"}
    gen_txt = Path(result["txt"])
    gen = load_chart(gen_txt)
    ref = load_chart(song["txt"])
    pitch_json = find_pitch_cache(gen_txt.parent)
    sung = SungPitch.from_pitch_json(pitch_json) if pitch_json else None

    offset, fit = (0.0, None)
    appended_info = {}
    appended_start_ms = None
    if sung is not None and len(sung.times):
        duration_ms = sung.duration_s * 1000 if sung.duration_s else None
        ref, appended = split_appended_voice(ref, duration_ms)
        offset, fit = fit_reference_offset(ref, sung)
        ref = shift_notes(ref, offset)
        if appended:
            appended_start_ms = min(n.start_ms for n in appended) + offset
            a_offset, a_fit = fit_appended_offset(appended, sung, duration_ms)
            folded = a_fit * 100 >= min_ref_fit
            if folded:
                ref += [replace(n, voice=1) for n in shift_notes(appended, a_offset)]
            appended_info = {"appended_voice": "folded" if folded else "dropped",
                             "appended_offset_ms": a_offset, "appended_fit_pct": _pct(a_fit)}
    metrics = chart_metrics(ref, gen, sung)
    ref_words = load_words(song["txt"], offset)
    if appended_start_ms is not None:
        # the lyrics are compared for the first voice only (the chart has one text track)
        ref_words = [replace(w, spans=[s for s in w.spans if s[0] < appended_start_ms]) for w in ref_words]
        ref_words = [w for w in ref_words if w.spans]
    metrics.update(lyrics_metrics(ref_words, load_words(gen_txt)))
    reliable = fit is not None and fit * 100 >= min_ref_fit
    row = {"id": song["id"], "status": "ok" if reliable else "unreliable_reference",
           "ref_offset_ms": offset, "ref_fit_pct": _pct(fit), **appended_info,
           "seconds": result.get("seconds"), **metrics}
    if plots:
        render_piano_roll(ref, gen, sung, run_dir / "plots", song["id"])
    return row


def _median(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.median(vals), 2) if vals else None


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    return {
        "songs_total": len(rows),
        "songs_reliable": len(ok),
        "songs_unreliable_reference": sum(r.get("status") == "unreliable_reference" for r in rows),
        "songs_failed": sum(r.get("status") == "no_output" for r in rows),
        "median": {k: _median(ok, k) for k in SUMMARY_METRICS},
    }


def format_summary_md(label: str, summary: dict, rows: list[dict]) -> str:
    lines = [f"# Chart benchmark: {label}", "",
             f"Reliable songs: {summary['songs_reliable']} of {summary['songs_total']} "
             f"(unreliable reference: {summary['songs_unreliable_reference']}, "
             f"failed: {summary['songs_failed']})", "",
             "| Metric | Median |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["median"].items()]
    cols = ["chart_agreement_pct", "chart_precision_pct", "chart_f1_pct", "onset_hit_100_pct",
            "note_count_ratio", "pitch_agree_pct", "extra_time_pct", "oracle_pitch_pct",
            "lyrics_agreement_pct", "ref_fit_pct"]
    lines += ["", "| Song | status | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r.get('status')} | " + " | ".join(str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def compare_summaries(a: dict, b: dict) -> list[tuple[str, float | None, float | None, float | None]]:
    out = []
    for k in SUMMARY_METRICS:
        va, vb = a["median"].get(k), b["median"].get(k)
        delta = round(vb - va, 2) if va is not None and vb is not None else None
        out.append((k, va, vb, delta))
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def render_piano_roll(ref: list[ChartNote], gen: list[ChartNote], sung: SungPitch | None,
                      out_dir: Path, title: str, window_s: float = 12.0, rows: int = 4) -> list[Path]:
    """Game-like piano roll: reference (filled green), generated (red outline), sung pitch (black).

    Everything is folded into one octave band around the reference melody, as
    the games ignore octaves when scoring. A small ``+12``/``-12`` label marks
    generated notes that were folded.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    # Charts use different pitch conventions (absolute MIDI vs. relative values)
    # and often sit a whole octave off. Move each chart by whole octaves next to
    # the sung pitch first, so the +-12 labels only mark notes that jump octaves.
    anchor = float(np.median(sung.midi)) if sung is not None and len(sung.midi) else None

    def to_anchor(notes: list[ChartNote], target: float | None) -> list[ChartNote]:
        pitched = [n.midi for n in notes if n.pitched]
        if target is None or not pitched:
            return notes
        k = 12 * round((target - float(np.median(pitched))) / 12)
        return [replace(n, midi=n.midi + k) for n in notes]

    ref = to_anchor(ref, anchor)
    ref_pitched = [n.midi for n in ref if n.pitched]
    gen = to_anchor(gen, anchor if anchor is not None else
                    (float(np.median(ref_pitched)) if ref_pitched else None))
    pitched_ref = [n for n in ref if not n.freestyle] or ref
    if not pitched_ref:
        return []
    first = max(min(n.start_ms for n in ref) / 1000 - 1, 0)
    last = max(n.end_ms for n in ref) / 1000 + 1

    def fold_to(midi, center):
        return center + fold(np.asarray(midi, dtype=float) - center)

    def draw(ax, notes, a, b, color, filled, lyric_y, center):
        for n in notes:
            if n.end_ms < a * 1000 or n.start_ms > b * 1000:
                continue
            x, w = n.start_ms / 1000, (n.end_ms - n.start_ms) / 1000
            y = float(fold_to(n.midi, center))
            if y != n.midi:
                ax.text(x + w, y + 0.45, f"{int(n.midi - y):+d}", color=color, fontsize=5)
            ax.add_patch(Rectangle((x, y - 0.4), w, 0.8, facecolor=color if filled else "none",
                                   alpha=0.35 if filled else 1.0, edgecolor=color, lw=1.4,
                                   ls=":" if n.freestyle else "-"))
            if n.word:
                ax.text(x, lyric_y, n.word, color=color, fontsize=6.5, va="center", clip_on=True)

    pages, page, start = [], 0, first
    while start < last:
        fig, axes = plt.subplots(rows, 1, figsize=(18, 3.2 * rows), squeeze=False)
        for ax in axes[:, 0]:
            a, b = start, start + window_s
            inside = [n.midi for n in pitched_ref if a * 1000 <= n.start_ms <= b * 1000]
            center = float(np.median(inside)) if inside else float(np.median([n.midi for n in pitched_ref]))
            lo, hi = center - 8, center + 8
            if sung is not None:
                sel = (sung.times >= a) & (sung.times <= b)
                ax.scatter(sung.times[sel], fold_to(sung.midi[sel], center), s=2, c="k", alpha=0.6, zorder=3)
            draw(ax, ref, a, b, "tab:green", True, hi - 0.6, center)
            draw(ax, gen, a, b, "tab:red", False, lo + 0.6, center)
            ax.set_xlim(a, b)
            ax.set_ylim(lo, hi)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)
            start = b
        axes[0, 0].set_title(f"{title}   green = reference   red = generated (dotted = freestyle)   "
                             f"black = sung pitch (octave folded)", fontsize=9)
        fig.tight_layout()
        path = out_dir / f"page_{page:02d}.png"
        fig.savefig(path, dpi=80)
        plt.close(fig)
        pages.append(path)
        page += 1
    return pages


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _require_outside_repo(workdir: Path) -> Path:
    """Refuse work directories inside the repository (they hold library data)."""
    try:
        workdir.resolve().relative_to(REPO.resolve())
    except ValueError:
        return workdir
    raise SystemExit("the workdir must be outside the repository (it holds library data)")


def _load_songs(workdir: Path) -> list[dict]:
    _require_outside_repo(workdir)
    path = workdir / "songs.json"
    if not path.exists():
        raise SystemExit(f"{path} not found - run 'sample' first")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_label(label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", label):
        raise SystemExit(f"invalid label {label!r} (use letters, digits, '.', '_', '-')")
    return label


def cmd_sample(args) -> int:
    library, workdir = Path(args.library), _require_outside_repo(Path(args.workdir))
    if not library.is_dir():
        raise SystemExit(f"library folder not found: {library}")
    target = workdir / "songs.json"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} exists - use --force to resample")
    candidates = find_candidates(library, args.prefer_video)
    songs = sample_songs(candidates, args.count, args.seed)
    workdir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(songs, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(candidates)} qualifying songs, sampled {len(songs)} -> {target}")
    return 0


def cmd_convert(args) -> int:
    workdir = Path(args.workdir)
    label = _check_label(args.label)
    songs = _load_songs(workdir)
    extra = shlex.split(args.args or "", posix=True)
    for i, song in enumerate(songs, 1):
        if args.only and song["id"] not in args.only:
            continue
        res = convert_song(song, workdir / "runs" / label / song["id"], extra, args.python,
                           args.keep_audio, args.timeout or None)
        if res.get("timed_out"):
            print(f"  {song['id']}: timed out after {args.timeout} s", flush=True)
        print(f"[{i}/{len(songs)}] {song['id']}: rc={res['returncode']} {res['seconds']}s "
              f"{'ok' if res['txt'] else 'NO OUTPUT'}", flush=True)
    return 0


def _reports_dir(workdir: Path, override: str | None) -> Path:
    return Path(override) if override else workdir / "reports"


def cmd_evaluate(args) -> int:
    workdir = Path(args.workdir)
    label = _check_label(args.label)
    rows = []
    for song in _load_songs(workdir):
        run_dir = workdir / "runs" / label / song["id"]
        if not (run_dir / "result.json").exists():
            continue
        try:
            row = evaluate_song(song, run_dir, args.min_ref_fit, args.plots)
        except Exception as exc:  # noqa: BLE001 - one broken song must not stop the report
            row = {"id": song["id"], "status": "error", "error": repr(exc)}
        rows.append(row)
        print(f"{row['id']}: {row.get('status')} agreement={row.get(PRIMARY_METRIC)} "
              f"onset100={row.get('onset_hit_100_pct')} ref_fit={row.get('ref_fit_pct')}", flush=True)
    summary = summarize(rows)
    out = _reports_dir(workdir, args.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{label}.json").write_text(json.dumps({"label": label, "summary": summary, "songs": rows},
                                                  indent=1), encoding="utf-8")
    md = format_summary_md(label, summary, rows)
    (out / f"{label}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def cmd_compare(args) -> int:
    reports = _reports_dir(_require_outside_repo(Path(args.workdir)), args.reports_dir)
    if args.metric not in SUMMARY_METRICS:
        raise SystemExit(f"unknown metric {args.metric!r} (choose from: {', '.join(SUMMARY_METRICS)})")
    a = json.loads((reports / f"{_check_label(args.label_a)}.json").read_text(encoding="utf-8"))
    b = json.loads((reports / f"{_check_label(args.label_b)}.json").read_text(encoding="utf-8"))
    print(f"{'metric':<26}{args.label_a:>14}{args.label_b:>14}{'delta':>10}")
    for k, va, vb, d in compare_summaries(a["summary"], b["summary"]):
        print(f"{k:<26}{str(va):>14}{str(vb):>14}{(f'{d:+.2f}' if d is not None else '-'):>10}")
    pa = {r["id"]: r.get(args.metric) for r in a["songs"] if r.get("status") == "ok"}
    pb = {r["id"]: r.get(args.metric) for r in b["songs"] if r.get("status") == "ok"}
    common = sorted(set(pa) & set(pb))
    diffs = sorted(((pb[s] - pa[s], s) for s in common if pa[s] is not None and pb[s] is not None))
    if diffs:
        # Only the primary metric has a known "good" direction; report the others neutrally.
        up, down, low = (("better", "worse", "worst") if args.metric == PRIMARY_METRIC
                         else ("higher", "lower", "lowest"))
        print(f"\n{args.metric} per song: {sum(d > 0 for d, _ in diffs)} {up}, "
              f"{sum(d < 0 for d, _ in diffs)} {down} of {len(diffs)}")
        for d, s in diffs[:5]:
            print(f"  {low} {s}: {d:+.1f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="pick a reproducible song sample from a library")
    s.add_argument("library")
    s.add_argument("workdir")
    s.add_argument("--count", type=int, default=100)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--prefer-video", action="store_true", help="convert from the video file when present")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_sample)

    c = sub.add_parser("convert", help="run UltraSinger on every sampled song")
    c.add_argument("workdir")
    c.add_argument("--label", default="baseline")
    c.add_argument("--args", default="", help="extra UltraSinger arguments, e.g. \"--chart_style score\"")
    c.add_argument("--only", nargs="*", help="limit to these song IDs")
    c.add_argument("--python", default=sys.executable, help="Python interpreter to run UltraSinger with")
    c.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                   help="seconds per song before a conversion is recorded as failed (0 = no limit)")
    c.add_argument("--keep-audio", action="store_true",
                   help="keep audio/video/stems of the output (default: keep only TXT and JSON)")
    c.set_defaults(func=cmd_convert)

    e = sub.add_parser("evaluate", help="measure generated charts against the reference charts")
    e.add_argument("workdir")
    e.add_argument("--label", default="baseline")
    e.add_argument("--min-ref-fit", type=float, default=50.0,
                   help="leave songs out of the summary whose reference fits the vocal worse (percent)")
    e.add_argument("--plots", action="store_true", help="write game-like piano-roll PNGs per song")
    e.add_argument("--reports-dir", help="write the reports here instead of WORKDIR/reports")
    e.set_defaults(func=cmd_evaluate)

    k = sub.add_parser("compare", help="compare two evaluated labels")
    k.add_argument("workdir")
    k.add_argument("label_a")
    k.add_argument("label_b")
    k.add_argument("--reports-dir", help="read the reports from here instead of WORKDIR/reports")
    k.add_argument("--metric", default=PRIMARY_METRIC,
                   help=f"metric for the per-song better/worse count (default {PRIMARY_METRIC})")
    k.set_defaults(func=cmd_compare)
    return p


def _benchmark_options() -> set[str]:
    """All option strings the benchmark itself understands (any subcommand)."""
    parser = build_parser()
    opts = set(parser._option_string_actions)
    for action in parser._subparsers._group_actions:
        for sub in action.choices.values():
            opts |= set(sub._option_string_actions)
    return opts


def _join_args_value(argv: list[str]) -> list[str]:
    """Turn ``--args --some_flag`` into ``--args=--some_flag``.

    argparse would otherwise read a value that starts with ``-`` as an option
    of its own and fail with "expected one argument". A following option the
    benchmark understands itself (e.g. ``--keep-audio``) is left alone; to pass
    such a name on to UltraSinger, use ``--args=...``.
    """
    own = _benchmark_options()
    out, i = [], 0
    while i < len(argv):
        if (argv[i] == "--args" and i + 1 < len(argv)
                and argv[i + 1].split("=", 1)[0] not in own):
            out.append(f"--args={argv[i + 1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(_join_args_value(list(argv)))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
