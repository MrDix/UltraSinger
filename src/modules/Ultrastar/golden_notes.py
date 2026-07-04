"""Mark held notes as UltraStar "golden" (``*``) bonus notes.

Golden notes double the score a player gets for hitting them (see
``ultrastar_score.parser.Note.score_factor``: 1 for normal/rap, 2 for
golden/rap-golden). UltraSinger currently never emits any, so charts
never contain a bonus section the way commercial/manually-authored
songs do.

Heuristic (conservative, USDX-conformant):

1. Only real syllable notes are eligible: ``note_type == ":"`` and the
   word is not a tilde continuation (``"~"``/``"~ "``) produced by
   melisma/pitch-change splitting. Freestyle (``"F"``) and rap
   (``"R"``/``"G"``) notes are never touched.
2. Only notes held for at least ``min_duration_ms`` are eligible —
   short notes rarely stay on-pitch long enough to be reliably hit, so
   making them golden would risk losing the bonus rather than gaining
   it.
3. The total number of golden notes is capped at ``max_fraction`` of
   all scorable notes (``":"``, ``"*"``, ``"R"``, ``"G"``), mirroring
   how USDX-style games keep golden sections a minority of the chart
   rather than the whole song.
4. Chosen notes are spread across the song instead of clustering in
   one high-energy section: eligible candidates are split (in song
   order) into as many contiguous chunks as there are golden slots,
   and the single longest note of each chunk is picked. This keeps the
   "pick the most reliable notes" property while guaranteeing golden
   notes appear throughout the track, not just at the start.
"""

from __future__ import annotations

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.Midi.MidiSegment import MidiSegment
from modules.Ultrastar.ultrastar_txt import UltrastarTxtNoteTypeTag

# Everything that contributes score points (i.e. everything but freestyle).
_SCORABLE_TYPES = (
    UltrastarTxtNoteTypeTag.NORMAL.value,
    UltrastarTxtNoteTypeTag.GOLDEN.value,
    UltrastarTxtNoteTypeTag.RAP.value,
    UltrastarTxtNoteTypeTag.RAP_GOLDEN.value,
)


def _is_continuation(word: str) -> bool:
    """True for tilde-continuation notes ("~" / "~ ") produced by melisma
    and pitch-change splitting — these are not independent syllables and
    should never be promoted to golden on their own."""
    return word.strip() == "~"


def mark_golden_notes(
    midi_segments: list[MidiSegment],
    bpm: float,
    *,
    max_fraction: float = 0.15,
    min_duration_ms: float = 350.0,
) -> list[MidiSegment]:
    """Mark a bounded, evenly-spread subset of held notes as golden.

    Args:
        midi_segments: Notes to mark. Mutated in place (matching the
            convention used by ``growl_detector.detect_growl_segments``)
            and also returned for convenient chaining.
        bpm: Real BPM. Not used by the current time-based heuristic
            (durations are measured directly from segment start/end
            seconds) but kept for API symmetry with the other
            post-processing passes (e.g. ``refit_notes_ptakf``) and to
            leave room for a future beat-based minimum spacing.
        max_fraction: Upper bound on the golden share of all scorable
            notes (default 0.15, i.e. at most 15%).
        min_duration_ms: Minimum note duration (in ms) to be eligible
            as golden (default 350ms).

    Returns:
        The same list, with up to ``max_fraction`` of scorable notes
        switched from ``":"`` to ``"*"``.
    """
    del bpm  # not used by the current heuristic; see docstring

    if not midi_segments:
        return midi_segments

    scorable_count = sum(
        1 for seg in midi_segments if seg.note_type in _SCORABLE_TYPES
    )
    if scorable_count == 0:
        return midi_segments

    max_golden = int(scorable_count * max_fraction)
    if max_golden <= 0:
        return midi_segments

    candidates = [
        (i, seg)
        for i, seg in enumerate(midi_segments)
        if seg.note_type == UltrastarTxtNoteTypeTag.NORMAL.value
        and not _is_continuation(seg.word)
        and (seg.end - seg.start) * 1000.0 >= min_duration_ms
    ]
    if not candidates:
        return midi_segments

    # Split candidates (already in song order) into as many contiguous
    # chunks as golden slots and keep the longest note per chunk. This
    # spreads golden notes across the whole song while still favouring
    # the most reliably-held notes within each region.
    chunk_count = min(max_golden, len(candidates))
    chunk_size = len(candidates) / chunk_count
    chosen: list[int] = []
    for c in range(chunk_count):
        lo = int(c * chunk_size)
        hi = len(candidates) if c == chunk_count - 1 else int((c + 1) * chunk_size)
        if lo >= hi:
            continue
        chunk = candidates[lo:hi]
        best_i, _ = max(chunk, key=lambda pair: pair[1].end - pair[1].start)
        chosen.append(best_i)

    for i in chosen:
        midi_segments[i].note_type = UltrastarTxtNoteTypeTag.GOLDEN.value

    if chosen:
        print(
            f"{ULTRASINGER_HEAD} Golden notes: "
            f"{blue_highlighted(str(len(chosen)))} of "
            f"{blue_highlighted(str(scorable_count))} scorable notes marked golden"
        )

    return midi_segments
