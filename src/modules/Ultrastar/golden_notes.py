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
3. The longest eligible notes are chosen, with at most
   ``max_per_part`` of them in each tenth of the sung time, so golden
   notes do not pile up in one section (long notes tend to gather at
   the end of a song).
"""

from __future__ import annotations

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.Midi.MidiSegment import MidiSegment
from modules.Ultrastar.ultrastar_txt import UltrastarTxtNoteTypeTag

DEFAULT_GOLDEN_COUNT = 10
MAX_GOLDEN_FRACTION = 0.15
MIN_GOLDEN_DURATION_MS = 200.0
SPREAD_PARTS = 10
MAX_GOLDEN_PER_PART = 3

# Everything that contributes score points (i.e. everything but freestyle).
_SCORABLE_TYPES = (
    UltrastarTxtNoteTypeTag.NORMAL.value,
    UltrastarTxtNoteTypeTag.GOLDEN.value,
    UltrastarTxtNoteTypeTag.RAP.value,
    UltrastarTxtNoteTypeTag.RAP_GOLDEN.value,
)


def mark_golden_notes(
    midi_segments: list[MidiSegment],
    bpm: float,
    *,
    count: int = DEFAULT_GOLDEN_COUNT,
    max_fraction: float = MAX_GOLDEN_FRACTION,
    min_duration_ms: float = MIN_GOLDEN_DURATION_MS,
    max_per_part: int = MAX_GOLDEN_PER_PART,
) -> list[MidiSegment]:
    """Mark the longest held notes, spread over the song, as golden.

    Args:
        midi_segments: Notes to mark. Mutated in place (matching the
            convention used by ``growl_detector.detect_growl_segments``)
            and also returned for convenient chaining.
        bpm: Real BPM. Not used by the time-based selection but kept for
            API symmetry with the other post-processing passes.
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

    # Longest first (whole milliseconds, earlier note on ties), at most
    # max_per_part per tenth of the sung time.
    sung_start = min(seg.start for seg in midi_segments)
    sung_span = max(max(seg.end for seg in midi_segments) - sung_start, 1e-9)
    per_part: dict[int, int] = {}
    chosen: list[int] = []

    def longest_first(i: int) -> tuple[int, int]:
        return -round((midi_segments[i].end - midi_segments[i].start) * 1000.0), i

    for i in sorted(candidates, key=longest_first):
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
