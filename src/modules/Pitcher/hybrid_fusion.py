"""Hybrid fusion: combine pitch-contour notes with lyrics word timings.

The hybrid pipeline generates notes from two independent tracks:
1. **Pitch track** — notes segmented by pitch contour (accurate note
   boundaries, but words are all placeholders).
2. **Lyrics track** — word-level timing from WhisperX CTC alignment
   or Whisper transcription (accurate text, but one word = one note).

This module fuses them: pitch-derived note boundaries with lyrics
assigned by time-overlap alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modules.Midi.MidiSegment import MidiSegment
from modules.Midi.midi_creator import confidence_weighted_median_note
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitched_data_helper import get_frequencies_with_high_confidence
from modules.Pitcher.pitch_based_note_generator import fill_lyrics_from_reference
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted


@dataclass
class WordTiming:
    """A word with its time boundaries from alignment."""
    word: str
    start: float
    end: float


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Compute the overlap duration between two intervals."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _find_overlapping_words(
    note: MidiSegment,
    words: list[WordTiming],
    min_overlap_ratio: float,
) -> list[WordTiming]:
    """Find words that overlap significantly with a pitch note.

    A word is considered overlapping when the overlap covers at least
    *min_overlap_ratio* of the word's duration.
    """
    result: list[WordTiming] = []
    for w in words:
        w_dur = w.end - w.start
        if w_dur <= 0:
            continue
        ovl = _overlap(note.start, note.end, w.start, w.end)
        if ovl / w_dur >= min_overlap_ratio:
            result.append(w)
    return result


def _split_note_at_word_boundaries(
    note: MidiSegment,
    words: list[WordTiming],
) -> list[MidiSegment]:
    """Split a single pitch note at word boundaries when multiple words overlap.

    The first sub-note gets the first word, subsequent sub-notes get
    their respective words.  The last sub-note extends to the original
    note end.
    """
    if len(words) <= 1:
        if words:
            note.word = words[0].word + " "
        return [note]

    # Sort words by start time
    words = sorted(words, key=lambda w: w.start)

    sub_notes: list[MidiSegment] = []
    for i, w in enumerate(words):
        if i == 0:
            sub_start = note.start
        else:
            # Split at word start (clamped to note boundaries)
            sub_start = max(w.start, note.start)

        if i == len(words) - 1:
            sub_end = note.end
        else:
            # Split at next word start
            sub_end = max(words[i + 1].start, sub_start)

        sub_end = min(sub_end, note.end)
        if sub_end <= sub_start:
            continue

        word_text = w.word
        if i < len(words) - 1:
            word_text = word_text.rstrip()

        sub = MidiSegment(note.note, sub_start, sub_end, word_text + " ")
        sub_notes.append(sub)

    return sub_notes if sub_notes else [note]


def _create_gap_note(
    word: WordTiming,
    pitched_data: PitchedData,
    low_conf_threshold: float = 0.1,
) -> MidiSegment:
    """Create a note for a word that has no overlapping pitch note.

    Tries to compute a pitch from low-confidence frames.  Falls back
    to a freestyle note (``note_type="F"``) if no pitch data is usable.
    """
    from modules.Midi.midi_creator import find_nearest_index

    start_idx = find_nearest_index(pitched_data.times, word.start)
    end_idx = find_nearest_index(pitched_data.times, word.end)
    start_idx = min(start_idx, len(pitched_data.times) - 1) if pitched_data.times else 0
    if end_idx <= start_idx:
        end_idx = start_idx + 1
    end_idx = min(end_idx, len(pitched_data.times))

    freqs = pitched_data.frequencies[start_idx:end_idx]
    confs = pitched_data.confidence[start_idx:end_idx]

    # Try to get a pitch from low-confidence frames
    valid_freqs = []
    valid_confs = []
    for f, c in zip(freqs, confs, strict=True):
        if f > 0 and c > low_conf_threshold:
            valid_freqs.append(f)
            valid_confs.append(c)

    if valid_freqs:
        note = confidence_weighted_median_note(valid_freqs, valid_confs)
    else:
        note = "C4"  # fallback

    seg = MidiSegment(note, word.start, word.end, word.word + " ")

    if not valid_freqs:
        seg.note_type = "F"

    return seg


def fuse_pitch_notes_with_lyrics(
    pitch_notes: list[MidiSegment],
    word_timings: list[WordTiming],
    pitched_data: PitchedData | None = None,
    plain_lyrics: str | None = None,
    min_overlap_ratio: float = 0.3,
) -> list[MidiSegment]:
    """Fuse pitch-contour notes with lyrics word timings.

    For each pitch note, finds overlapping words and assigns lyrics.
    Words that don't overlap any pitch note get their own gap notes
    (with pitch from low-confidence frames or as freestyle).

    Args:
        pitch_notes: Notes from ``create_pitch_notes_only()`` (word="~ ").
        word_timings: Word-level timing from alignment.
        pitched_data: Raw pitch data (needed for gap note creation).
        plain_lyrics: Optional full lyrics for reference fill.
        min_overlap_ratio: Minimum overlap fraction for word assignment.

    Returns:
        Fused list of MidiSegments with lyrics assigned.
    """
    if not pitch_notes:
        return []

    # Track which words get assigned to a note
    assigned_words: set[int] = set()

    # Step 1: Assign words to pitch notes
    fused: list[MidiSegment] = []

    for note in pitch_notes:
        overlapping = _find_overlapping_words(note, word_timings, min_overlap_ratio)

        if not overlapping:
            # No word overlaps — keep as placeholder
            fused.append(MidiSegment(note.note, note.start, note.end, "~ "))
            continue

        # Mark words as assigned
        for w in overlapping:
            for j, wt in enumerate(word_timings):
                if wt is w:
                    assigned_words.add(j)

        if len(overlapping) == 1:
            # Single word — assign directly
            seg = MidiSegment(note.note, note.start, note.end, overlapping[0].word + " ")
            fused.append(seg)
        else:
            # Multiple words — split note at word boundaries
            sub_notes = _split_note_at_word_boundaries(note, overlapping)
            fused.extend(sub_notes)

    # Step 2: Create gap notes for unassigned words
    if pitched_data is not None:
        gap_notes: list[MidiSegment] = []
        for j, wt in enumerate(word_timings):
            if j not in assigned_words:
                gap = _create_gap_note(wt, pitched_data)
                gap_notes.append(gap)

        if gap_notes:
            fused.extend(gap_notes)
            # Sort by start time after adding gaps
            fused.sort(key=lambda s: s.start)

    # Step 3: Reference lyrics fill for remaining placeholders
    if plain_lyrics:
        # Build mock transcribed_data for fill_lyrics_from_reference
        from modules.Speech_Recognition.TranscribedData import TranscribedData
        mock_td = [
            TranscribedData(
                word=s.word, start=s.start, end=s.end,
                confidence=1.0, is_word_end=True,
            )
            for s in fused
        ]
        fused = fill_lyrics_from_reference(fused, mock_td, plain_lyrics)

    # Step 4: Ensure proper word spacing
    for seg in fused:
        if seg.word and not seg.word.endswith(" "):
            seg.word += " "

    total = len(fused)
    with_lyrics = sum(1 for s in fused if s.word.strip() not in ("~", ""))
    placeholders = total - with_lyrics

    print(
        f"{ULTRASINGER_HEAD} Hybrid fusion: "
        f"{blue_highlighted(str(total))} notes "
        f"({with_lyrics} with lyrics, {placeholders} continuations)"
    )

    return fused


def transcribed_data_to_word_timings(transcribed_data: list) -> list[WordTiming]:
    """Convert TranscribedData list to WordTiming list for fusion."""
    return [
        WordTiming(word=td.word, start=td.start, end=td.end)
        for td in transcribed_data
        if hasattr(td, "word") and hasattr(td, "start") and hasattr(td, "end")
    ]
