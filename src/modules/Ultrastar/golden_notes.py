"""Mark held notes as UltraStar "golden" (``*``) bonus notes.

Golden notes double the score a player gets for hitting them (see
``ultrastar_score.parser.Note.score_factor``: 1 for normal/rap, 2 for
golden/rap-golden). UltraSinger never emits any on its own, so this
optional pass adds a bonus selection the way hand-made charts do.

Hand-made professional charts mark only a handful of golden notes per
song - typically about ten single notes, whatever the song's length -
favour long held notes, and spread them over the whole song. The pass
follows that:

1. Only normal notes (``":"``) held for at least ``min_duration_ms`` are
   eligible; freestyle (``"F"``) and rap (``"R"``/``"G"``) notes are never
   touched. Tilde continuations (``"~"``) are eligible too: a long one is
   a held vowel, a typical golden note.
2. ``count`` notes are marked (default 10), but never more than
   ``max_fraction`` of all scorable notes (``":"``, ``"*"``, ``"R"``,
   ``"G"``), so short songs stay mostly normal.
3. Notes are ranked by duration, weighted by how well the singing stays
   on the note's pitch (the share of the note's pitch frames that are
   confident and within one semitone of it, octaves ignored) - a golden
   note that cannot be hit is worse than none - and with a bonus for the
   highest note of a phrase (notes without a pause longer than
   ``PHRASE_PAUSE_MS``). Without a pitch track only duration and the
   phrase peak count.
4. At most ``max_per_part`` golden notes in each tenth of the sung time,
   so they do not pile up in one section (long notes tend to gather at
   the end of a song).
"""

from __future__ import annotations

import librosa
import numpy as np
from librosa.util.exceptions import ParameterError

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher.pitched_data import PitchedData
from modules.Ultrastar.ultrastar_txt import UltrastarTxtNoteTypeTag

DEFAULT_GOLDEN_COUNT = 10
MAX_GOLDEN_FRACTION = 0.15
MIN_GOLDEN_DURATION_MS = 200.0
SPREAD_PARTS = 10
MAX_GOLDEN_PER_PART = 3
PITCH_LOCK_CONFIDENCE = 0.7  # pitch frames at least this confident count as sung
PITCH_LOCK_FLOOR = 0.5  # rank weight = PITCH_LOCK_FLOOR + share of on-pitch frames
PHRASE_PAUSE_MS = 400.0  # a longer pause between two notes starts a new phrase
PHRASE_PEAK_BONUS = 1.35  # rank weight of the highest note of a phrase

# Everything that contributes score points (i.e. everything but freestyle).
_SCORABLE_TYPES = (
    UltrastarTxtNoteTypeTag.NORMAL.value,
    UltrastarTxtNoteTypeTag.GOLDEN.value,
    UltrastarTxtNoteTypeTag.RAP.value,
    UltrastarTxtNoteTypeTag.RAP_GOLDEN.value,
)


def _note_midi(seg: MidiSegment) -> float | None:
    try:
        return float(librosa.note_to_midi(seg.note))
    except (ValueError, TypeError, ParameterError):
        return None


def _pitch_lock(midi_segments: list[MidiSegment], indices: list[int],
                pitched_data: PitchedData | None) -> dict[int, float]:
    """Share of each note's pitch frames that are confident and within one
    semitone of the note (octaves ignored); 1.0 for every note without a
    pitch track, so the weighting is neutral then."""
    if pitched_data is None or not len(pitched_data.times):
        return {i: 1.0 for i in indices}
    times = np.asarray(pitched_data.times, dtype=float)
    freqs = np.asarray(pitched_data.frequencies, dtype=float)
    conf = np.asarray(pitched_data.confidence, dtype=float)
    sung = (conf >= PITCH_LOCK_CONFIDENCE) & (freqs > 40.0)
    track = 69.0 + 12.0 * np.log2(np.maximum(freqs, 1.0) / 440.0)
    lock = {}
    for i in indices:
        seg = midi_segments[i]
        midi = _note_midi(seg)
        lo, hi = np.searchsorted(times, seg.start), np.searchsorted(times, seg.end)
        if midi is None or hi <= lo:
            lock[i] = 0.0
            continue
        folded = (track[lo:hi] - midi + 6.0) % 12.0 - 6.0
        lock[i] = float(np.mean(sung[lo:hi] & (np.abs(folded) <= 1.0)))
    return lock


def _phrase_peaks(midi_segments: list[MidiSegment]) -> set[int]:
    """Indices of the highest pitched note(s) of each phrase."""
    pitched = [(i, _note_midi(seg)) for i, seg in enumerate(midi_segments)
               if seg.note_type in (UltrastarTxtNoteTypeTag.NORMAL.value, UltrastarTxtNoteTypeTag.GOLDEN.value)]
    pitched = [(i, m) for i, m in pitched if m is not None]
    peaks: set[int] = set()
    phrase: list[tuple[int, float]] = []

    def close_phrase() -> None:
        if phrase:
            top = max(m for _, m in phrase)
            peaks.update(j for j, m in phrase if m == top)
            phrase.clear()

    for i, m in pitched:
        if phrase and (midi_segments[i].start - midi_segments[phrase[-1][0]].end) * 1000.0 > PHRASE_PAUSE_MS:
            close_phrase()
        phrase.append((i, m))
    close_phrase()
    return peaks


def mark_golden_notes(
    midi_segments: list[MidiSegment],
    bpm: float,
    *,
    pitched_data: PitchedData | None = None,
    count: int = DEFAULT_GOLDEN_COUNT,
    max_fraction: float = MAX_GOLDEN_FRACTION,
    min_duration_ms: float = MIN_GOLDEN_DURATION_MS,
    max_per_part: int = MAX_GOLDEN_PER_PART,
) -> list[MidiSegment]:
    """Mark long held notes the singing stays on, spread over the song, as golden.

    Args:
        midi_segments: Notes to mark. Mutated in place (matching the
            convention used by ``growl_detector.detect_growl_segments``)
            and also returned for convenient chaining.
        bpm: Real BPM. Not used by the time-based selection but kept for
            API symmetry with the other post-processing passes.
        pitched_data: Pitch track of the vocal; notes the singing does not
            stay on rank lower. Optional.
        count: Number of golden notes to mark (default 10).
        max_fraction: Upper bound on the golden share of all scorable
            notes (default 0.15), which only matters for short songs.
        min_duration_ms: Minimum note duration (in ms) to be eligible
            as golden (default 200ms).
        max_per_part: At most this many golden notes in each tenth of
            the sung time (default 3).

    Returns:
        The same list, with up to ``count`` normal notes switched from
        ``":"`` to ``"*"``.
    """
    del bpm  # not used by the time-based selection; see docstring

    if not midi_segments:
        return midi_segments

    scorable_count = sum(
        1 for seg in midi_segments if seg.note_type in _SCORABLE_TYPES
    )
    golden_slots = min(count, int(scorable_count * max_fraction))
    if golden_slots <= 0:
        return midi_segments

    candidates = [
        i
        for i, seg in enumerate(midi_segments)
        if seg.note_type == UltrastarTxtNoteTypeTag.NORMAL.value
        and (seg.end - seg.start) * 1000.0 >= min_duration_ms
    ]
    if not candidates:
        return midi_segments

    # Best rank first (earlier note on ties), at most max_per_part per tenth
    # of the sung time.
    lock = _pitch_lock(midi_segments, candidates, pitched_data)
    peaks = _phrase_peaks(midi_segments)
    sung_start = min(seg.start for seg in midi_segments)
    sung_span = max(max(seg.end for seg in midi_segments) - sung_start, 1e-9)
    per_part: dict[int, int] = {}
    chosen: list[int] = []

    def best_first(i: int) -> tuple[float, int]:
        duration_ms = round((midi_segments[i].end - midi_segments[i].start) * 1000.0)
        weight = (PITCH_LOCK_FLOOR + lock[i]) * (PHRASE_PEAK_BONUS if i in peaks else 1.0)
        return -round(duration_ms * weight, 3), i

    for i in sorted(candidates, key=best_first):
        if len(chosen) >= golden_slots:
            break
        part = min(int((midi_segments[i].start - sung_start) / sung_span * SPREAD_PARTS), SPREAD_PARTS - 1)
        if per_part.get(part, 0) >= max_per_part:
            continue
        per_part[part] = per_part.get(part, 0) + 1
        chosen.append(i)

    for i in chosen:
        midi_segments[i].note_type = UltrastarTxtNoteTypeTag.GOLDEN.value

    if chosen:
        print(
            f"{ULTRASINGER_HEAD} Golden notes: "
            f"{blue_highlighted(str(len(chosen)))} of "
            f"{blue_highlighted(str(scorable_count))} scorable notes marked golden"
        )

    return midi_segments
