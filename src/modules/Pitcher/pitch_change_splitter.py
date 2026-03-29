"""Pitch-change splitter: split notes at pitch change boundaries.

When a singer changes pitch within a single note (melismas, runs),
this module splits the note into sub-notes at the pitch change
boundaries instead of averaging to a single flat note.
"""

from typing import Optional

import librosa
import numpy as np

from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitched_data_helper import get_frequencies_with_high_confidence
from modules.Midi.midi_creator import find_nearest_index, confidence_weighted_median_note
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted


def _freq_to_midi_safe(freq: float) -> Optional[float]:
    """Convert frequency to MIDI note number, returning None for invalid values."""
    if freq <= 0 or not np.isfinite(freq):
        return None
    try:
        return float(librosa.hz_to_midi(freq))
    except Exception:
        return None


def _midi_to_note_safe(midi_val: float) -> str:
    """Convert MIDI number to note name, clamping to valid range."""
    clamped = max(0, min(127, round(midi_val)))
    return librosa.midi_to_note(clamped)


def _get_frames_for_segment(
    segment: MidiSegment,
    pitched_data: PitchedData,
) -> tuple[list[float], list[float], list[float]]:
    """Extract pitch frames within a segment's time range.

    Returns:
        Tuple of (times, frequencies, confidences) for frames in range.
    """
    start_idx = find_nearest_index(pitched_data.times, segment.start)
    end_idx = find_nearest_index(pitched_data.times, segment.end)

    # Include the end frame if it falls within/near the segment boundary
    if end_idx < len(pitched_data.times):
        end_idx += 1

    # Ensure we get at least one frame
    if start_idx >= end_idx:
        end_idx = start_idx + 1

    end_idx = min(end_idx, len(pitched_data.times))
    start_idx = min(start_idx, end_idx)

    times = pitched_data.times[start_idx:end_idx]
    freqs = pitched_data.frequencies[start_idx:end_idx]
    confs = pitched_data.confidence[start_idx:end_idx]

    return times, freqs, confs


def _median_smooth(values: list[Optional[float]], window: int = 9) -> list[Optional[float]]:
    """Apply median filter over valid (non-None) values to smooth vibrato.

    Vibrato typically oscillates at 4-8 Hz.  At a ~16 ms frame interval
    that is 8-16 frames per cycle.  A window of 9 covers roughly one
    half-cycle, enough to pull the median toward the centre pitch and
    prevent vibrato from triggering false pitch-change splits.
    """
    n = len(values)
    smoothed: list[Optional[float]] = [None] * n
    half_w = window // 2

    for i in range(n):
        if values[i] is None:
            continue
        vals = [
            values[j]
            for j in range(max(0, i - half_w), min(n, i + half_w + 1))
            if values[j] is not None
        ]
        if vals:
            smoothed[i] = float(np.median(vals))

    return smoothed


def _detect_pitch_change_points(
    times: list[float],
    frequencies: list[float],
    confidences: list[float],
    min_semitone_change: float,
    min_note_duration_ms: float,
    median_filter_window: int = 9,
) -> list[int]:
    """Detect frame indices where significant pitch changes occur.

    A pitch change is detected when:
    1. The **region median** of the current stable pitch differs from
       the candidate new pitch by at least *min_semitone_change*
       semitones (vibrato-resistant — comparing region medians instead
       of adjacent frames prevents periodic oscillation from triggering
       false splits).
    2. The new pitch is sustained for at least *min_note_duration_ms*.

    Returns:
        List of frame indices where splits should occur (the first frame
        of each new pitch region).
    """
    if len(times) < 2:
        return []

    min_duration_s = min_note_duration_ms / 1000.0

    # Convert frequencies to MIDI values, filtering by confidence
    midi_raw: list[Optional[float]] = []
    for freq, conf in zip(frequencies, confidences, strict=True):
        if conf > 0.3 and freq > 0:
            midi_raw.append(_freq_to_midi_safe(freq))
        else:
            midi_raw.append(None)

    # Smooth with a wide median filter to suppress vibrato oscillation
    midi_values = _median_smooth(midi_raw, window=median_filter_window)

    # Single-pass scan: detect candidate change points and immediately
    # verify sustain in one forward sweep.  This keeps region_center
    # up-to-date so later real changes are not missed.
    region_values: list[float] = []
    confirmed: list[int] = []

    i = 0
    while i < len(midi_values):
        midi_val = midi_values[i]
        if midi_val is None:
            i += 1
            continue

        if not region_values:
            region_values.append(midi_val)
            i += 1
            continue

        region_center = float(np.median(region_values))
        diff = abs(midi_val - region_center)

        if diff >= min_semitone_change:
            # Immediately check if this new pitch is sustained
            sustained_values: list[float] = [midi_val]
            sustained_until = times[i]
            last_sustained = i
            for j in range(i + 1, len(times)):
                if midi_values[j] is None:
                    break  # unvoiced gap ends sustain
                if abs(midi_values[j] - midi_val) < min_semitone_change:
                    sustained_values.append(midi_values[j])
                    sustained_until = times[j]
                    last_sustained = j
                else:
                    break

            duration = sustained_until - times[i]
            if duration >= min_duration_s:
                confirmed.append(i)
                region_values = sustained_values
                # Skip past the sustained region to avoid reprocessing
                i = last_sustained + 1
                continue
            else:
                # Not sustained — noise, absorb into current region
                region_values.append(midi_val)
        else:
            region_values.append(midi_val)

        i += 1

    return confirmed


def _compute_segment_note(
    frequencies: list[float],
    confidences: list[float],
) -> str:
    """Compute the note for a sub-segment using weighted median."""
    conf_f, conf_weights = get_frequencies_with_high_confidence(
        frequencies, confidences
    )
    if not conf_f or sum(conf_weights) == 0:
        return "C4"
    return confidence_weighted_median_note(conf_f, conf_weights)


def _split_single_segment(
    segment: MidiSegment,
    pitched_data: PitchedData,
    min_semitone_change: float,
    min_note_duration_ms: float,
    median_filter_window: int = 9,
) -> list[MidiSegment]:
    """Split a single MidiSegment at pitch change boundaries.

    Returns:
        List of MidiSegments (1 if no split needed, >1 if splits found).
    """
    times, freqs, confs = _get_frames_for_segment(segment, pitched_data)

    if len(times) < 2:
        return [segment]

    change_points = _detect_pitch_change_points(
        times, freqs, confs, min_semitone_change, min_note_duration_ms,
        median_filter_window,
    )

    if not change_points:
        return [segment]

    # Build sub-segments from the change points
    # Boundaries: [0, cp1, cp2, ..., len(times)]
    boundaries = [0, *change_points, len(times)]

    sub_segments: list[MidiSegment] = []
    for b_idx in range(len(boundaries) - 1):
        start_frame = boundaries[b_idx]
        end_frame = boundaries[b_idx + 1]

        if start_frame >= end_frame:
            continue

        sub_freqs = freqs[start_frame:end_frame]
        sub_confs = confs[start_frame:end_frame]

        # Determine time boundaries
        if b_idx == 0:
            sub_start = segment.start
        else:
            sub_start = times[start_frame]

        if b_idx == len(boundaries) - 2:
            sub_end = segment.end
        else:
            sub_end = times[end_frame]

        # Compute note for this sub-segment
        note = _compute_segment_note(sub_freqs, sub_confs)

        # First sub-note gets the original text, rest get "~"
        if b_idx == 0:
            word = segment.word
            # Strip trailing space from first sub-note if there are continuations
            if len(boundaries) > 2 and word.endswith(" "):
                word = word.rstrip()
        else:
            word = "~"

        sub_seg = MidiSegment(note, sub_start, sub_end, word)
        # Propagate LRCLIB metadata from original segment
        sub_seg.note_type = getattr(segment, "note_type", ":")
        sub_segments.append(sub_seg)

    # The last sub-segment inherits line_break_after from the original
    if sub_segments:
        sub_segments[-1].line_break_after = getattr(segment, "line_break_after", False)

    # Merge adjacent sub-segments that resolved to the same note
    # (e.g. a single-frame glitch splits into two C4 regions)
    merged: list[MidiSegment] = []
    for seg in sub_segments:
        if merged and merged[-1].note == seg.note:
            merged[-1].end = seg.end
            # Transfer trailing space if present
            if seg.word.endswith(" ") and not merged[-1].word.endswith(" "):
                merged[-1].word += " "
            # Transfer line_break_after from later segment
            if seg.line_break_after:
                merged[-1].line_break_after = True
        else:
            merged.append(seg)
    sub_segments = merged

    # Restore trailing space on the last sub-segment if original had it
    if sub_segments and segment.word.endswith(" "):
        if not sub_segments[-1].word.endswith(" "):
            sub_segments[-1].word += " "

    return sub_segments if sub_segments else [segment]


def _merge_short_fragments(
    segments: list[MidiSegment],
    min_note_duration_ms: float,
) -> list[MidiSegment]:
    """Merge fragments shorter than min_note_duration_ms with nearest neighbor.

    Short fragments are absorbed by the adjacent segment that has the
    closest pitch (in semitones). This prevents micro-notes that would
    be unsingable in karaoke games.
    """
    if len(segments) <= 1:
        return segments

    min_duration_s = min_note_duration_ms / 1000.0
    result = list(segments)

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result):
            duration = result[i].end - result[i].start
            if duration >= min_duration_s or len(result) <= 1:
                i += 1
                continue

            # This segment is too short — merge with nearest neighbor
            current_midi = _freq_to_midi_safe(
                librosa.note_to_hz(result[i].note)
            )

            best_neighbor = None
            best_diff = float("inf")

            # Check previous neighbor
            if i > 0:
                prev_midi = _freq_to_midi_safe(
                    librosa.note_to_hz(result[i - 1].note)
                )
                if prev_midi is not None and current_midi is not None:
                    diff = abs(current_midi - prev_midi)
                    if diff < best_diff:
                        best_diff = diff
                        best_neighbor = i - 1

            # Check next neighbor
            if i < len(result) - 1:
                next_midi = _freq_to_midi_safe(
                    librosa.note_to_hz(result[i + 1].note)
                )
                if next_midi is not None and current_midi is not None:
                    diff = abs(current_midi - next_midi)
                    if diff < best_diff:
                        best_diff = diff
                        best_neighbor = i + 1

            if best_neighbor is None:
                # No valid neighbor found, merge with adjacent
                best_neighbor = i - 1 if i > 0 else i + 1

            if best_neighbor < i:
                # Merge into previous: extend previous end
                result[best_neighbor].end = result[i].end
                # If current segment had trailing space, transfer it
                if result[i].word.endswith(" ") and not result[best_neighbor].word.endswith(" "):
                    result[best_neighbor].word += " "
                result.pop(i)
            else:
                # Merge into next: extend next start, keep next's word
                result[best_neighbor].start = result[i].start
                # If this was the first segment with actual text, transfer text
                if result[i].word != "~":
                    result[best_neighbor].word = result[i].word
                result.pop(i)

            changed = True
            # Don't increment i, re-check the same position

    return result


def split_notes_at_pitch_changes(
    midi_segments: list[MidiSegment],
    pitched_data: PitchedData,
    min_semitone_change: float = 2.0,
    min_note_duration_ms: float = 80.0,
    median_filter_window: int = 9,
) -> list[MidiSegment]:
    """Split MIDI segments at pitch change boundaries.

    When a singer changes pitch within a single note (melismas, runs,
    ornaments), this function detects the pitch transitions and splits
    the note into sub-notes, each with its own pitch.

    Args:
        midi_segments: List of MIDI segments from pitch detection.
        pitched_data: Raw pitch data with per-frame frequencies.
        min_semitone_change: Minimum pitch difference in semitones to
            trigger a split (default 2.0).
        min_note_duration_ms: Minimum duration in milliseconds for a
            sub-note to survive (shorter fragments are merged back).
        median_filter_window: Window size for the median filter used
            to suppress vibrato oscillation (default 9).

    Returns:
        New list of MidiSegments, potentially longer than the input
        due to splits.
    """
    if not midi_segments:
        return midi_segments

    total_splits = 0
    result: list[MidiSegment] = []

    for segment in midi_segments:
        sub_segments = _split_single_segment(
            segment, pitched_data, min_semitone_change, min_note_duration_ms,
            median_filter_window,
        )

        # Merge back short fragments
        sub_segments = _merge_short_fragments(sub_segments, min_note_duration_ms)

        if len(sub_segments) > 1:
            total_splits += len(sub_segments) - 1

        result.extend(sub_segments)

    if total_splits > 0:
        print(
            f"{ULTRASINGER_HEAD} Pitch-change split: "
            f"split {blue_highlighted(str(total_splits))} "
            f"note{'s' if total_splits != 1 else ''} at pitch change boundaries"
        )

    return result
