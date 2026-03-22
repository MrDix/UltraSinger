"""Pitch-based note generation: create notes from pitch contour.

Instead of using Whisper word timing as note boundaries (one word = one note),
this module generates notes directly from the pitch contour detected by
SwiftF0.  Lyrics from Whisper are overlaid on the pitch-derived notes by
time alignment.

This produces far more accurate results for melismatic songs (runs, slides,
ornaments) where a single word can span many different pitches.
"""

from __future__ import annotations

from typing import Optional

import librosa
import numpy as np

from modules.Midi.MidiSegment import MidiSegment
from modules.Midi.midi_creator import (
    confidence_weighted_median_note,
    find_nearest_index,
)
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitched_data_helper import get_frequencies_with_high_confidence
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _freq_to_midi_safe(freq: float) -> Optional[float]:
    """Convert frequency to MIDI note number, returning None for invalid values."""
    if freq <= 0 or not np.isfinite(freq):
        return None
    try:
        return float(librosa.hz_to_midi(freq))
    except Exception:
        return None


def _find_voiced_regions(
    pitched_data: PitchedData,
    confidence_threshold: float = 0.3,
    bridge_gap_frames: int = 3,
    min_region_frames: int = 5,
) -> list[tuple[int, int]]:
    """Find contiguous voiced regions in the pitch data.

    A frame is "voiced" if its confidence exceeds the threshold and
    frequency is positive.  Small unvoiced gaps (up to *bridge_gap_frames*)
    between voiced regions are bridged.

    Returns:
        List of (start_idx, end_idx) tuples (exclusive end).
    """
    n = len(pitched_data.times)
    if n == 0:
        return []

    voiced = [
        pitched_data.confidence[i] > confidence_threshold
        and pitched_data.frequencies[i] > 0
        for i in range(n)
    ]

    # Bridge small gaps
    i = 0
    while i < n:
        if not voiced[i]:
            # Count gap length
            gap_start = i
            while i < n and not voiced[i]:
                i += 1
            gap_len = i - gap_start
            if gap_len <= bridge_gap_frames and gap_start > 0 and i < n:
                # Bridge: mark gap frames as voiced
                for j in range(gap_start, i):
                    voiced[j] = True
        else:
            i += 1

    # Extract contiguous regions
    regions: list[tuple[int, int]] = []
    in_region = False
    region_start = 0

    for i in range(n):
        if voiced[i] and not in_region:
            region_start = i
            in_region = True
        elif not voiced[i] and in_region:
            if i - region_start >= min_region_frames:
                regions.append((region_start, i))
            in_region = False

    if in_region and n - region_start >= min_region_frames:
        regions.append((region_start, n))

    return regions


def _median_filter_midi(
    frequencies: list[float],
    confidences: list[float],
    confidence_threshold: float = 0.3,
    window: int = 5,
) -> list[Optional[float]]:
    """Convert frequencies to MIDI and apply median filter to smooth vibrato.

    Args:
        frequencies: Hz values.
        confidences: Per-frame confidence.
        confidence_threshold: Below this → None.
        window: Median filter window size (frames).

    Returns:
        List of smoothed MIDI values (None where unvoiced).
    """
    midi_raw: list[Optional[float]] = []
    for freq, conf in zip(frequencies, confidences):
        if conf > confidence_threshold and freq > 0:
            midi_raw.append(_freq_to_midi_safe(freq))
        else:
            midi_raw.append(None)

    if not midi_raw:
        return midi_raw

    # Apply median filter only across valid values
    n = len(midi_raw)
    smoothed: list[Optional[float]] = [None] * n
    half_w = window // 2

    for i in range(n):
        if midi_raw[i] is None:
            continue

        # Collect values in window
        vals = []
        for j in range(max(0, i - half_w), min(n, i + half_w + 1)):
            if midi_raw[j] is not None:
                vals.append(midi_raw[j])

        if vals:
            smoothed[i] = float(np.median(vals))

    return smoothed


def _segment_voiced_region(
    times: list[float],
    frequencies: list[float],
    confidences: list[float],
    smoothed_midi: list[Optional[float]],
    min_semitone_change: float,
    min_note_duration_ms: float,
) -> list[tuple[int, int]]:
    """Segment a voiced region into stable pitch sub-regions.

    Returns:
        List of (start_idx, end_idx) within the region (local indices).
    """
    if not times:
        return []

    min_duration_s = min_note_duration_ms / 1000.0

    # Find the first valid MIDI value
    current_midi: Optional[float] = None
    current_start = 0

    for i, m in enumerate(smoothed_midi):
        if m is not None:
            current_midi = m
            current_start = i
            break

    if current_midi is None:
        return [(0, len(times))]

    segments: list[tuple[int, int]] = []

    i = current_start + 1
    while i < len(times):
        m = smoothed_midi[i]
        if m is None:
            i += 1
            continue

        diff = abs(m - current_midi)
        if diff >= min_semitone_change:
            # Check if the new pitch is sustained
            sustained_until = times[i]
            for j in range(i, len(times)):
                if smoothed_midi[j] is None:
                    continue
                if abs(smoothed_midi[j] - m) < min_semitone_change:
                    sustained_until = times[j]
                else:
                    break

            duration = sustained_until - times[i]
            if duration >= min_duration_s:
                # Confirmed pitch change — close current segment, start new
                if i > current_start:
                    segments.append((current_start, i))
                current_start = i
                current_midi = m

        i += 1

    # Close final segment
    if current_start < len(times):
        segments.append((current_start, len(times)))

    return segments if segments else [(0, len(times))]


def _compute_note_for_frames(
    frequencies: list[float],
    confidences: list[float],
) -> str:
    """Compute the note for a set of pitch frames using weighted median."""
    conf_f, conf_weights = get_frequencies_with_high_confidence(
        frequencies, confidences
    )
    if not conf_f:
        return "C4"
    return confidence_weighted_median_note(conf_f, conf_weights)


def _merge_short_notes(
    segments: list[MidiSegment],
    min_note_duration_ms: float,
) -> list[MidiSegment]:
    """Merge notes shorter than min_note_duration_ms with their neighbor.

    Short notes are absorbed by the adjacent note with the closest pitch.
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

            # Find best neighbor (closest pitch)
            current_midi = _freq_to_midi_safe(librosa.note_to_hz(result[i].note))
            best_idx = None
            best_diff = float("inf")

            for neighbor_idx in (i - 1, i + 1):
                if 0 <= neighbor_idx < len(result):
                    n_midi = _freq_to_midi_safe(
                        librosa.note_to_hz(result[neighbor_idx].note)
                    )
                    if n_midi is not None and current_midi is not None:
                        d = abs(current_midi - n_midi)
                        if d < best_diff:
                            best_diff = d
                            best_idx = neighbor_idx

            if best_idx is None:
                best_idx = i - 1 if i > 0 else i + 1 if i + 1 < len(result) else None

            if best_idx is None:
                i += 1
                continue

            if best_idx < i:
                result[best_idx].end = result[i].end
            else:
                result[best_idx].start = result[i].start

            result.pop(i)
            changed = True

    return result


def _merge_same_pitch_neighbors(
    segments: list[MidiSegment],
) -> list[MidiSegment]:
    """Merge adjacent notes with the same pitch."""
    if not segments:
        return segments

    merged: list[MidiSegment] = [segments[0]]
    for seg in segments[1:]:
        if merged[-1].note == seg.note:
            merged[-1].end = seg.end
            # Keep word from the one that has actual text
            if seg.word not in ("~", "~ ") and merged[-1].word in ("~", "~ "):
                merged[-1].word = seg.word
        else:
            merged.append(seg)

    return merged


def _overlay_lyrics(
    midi_segments: list[MidiSegment],
    transcribed_data: list[TranscribedData],
) -> list[MidiSegment]:
    """Assign lyrics from Whisper transcription to pitch-derived notes.

    Each Whisper word is assigned to the pitch-note with the most time
    overlap.  The first overlapping note gets the word text; subsequent
    notes within the same word get "~" continuation markers.

    Notes with no overlapping word get "~".
    """
    if not transcribed_data or not midi_segments:
        return midi_segments

    # Initialize all notes as "~" (continuation/placeholder)
    for seg in midi_segments:
        seg.word = "~ "

    # For each Whisper word, find the best-overlapping note
    for td in transcribed_data:
        word_start = td.start
        word_end = td.end
        word_text = td.word

        best_overlap = 0.0
        best_idx = -1

        for i, seg in enumerate(midi_segments):
            # Compute time overlap
            overlap_start = max(word_start, seg.start)
            overlap_end = min(word_end, seg.end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        if best_idx >= 0 and best_overlap > 0:
            # Assign word to the note with most overlap
            if midi_segments[best_idx].word in ("~", "~ "):
                midi_segments[best_idx].word = word_text
            else:
                # Multiple words on same note — concatenate
                existing = midi_segments[best_idx].word.rstrip()
                new_word = word_text.strip()
                midi_segments[best_idx].word = existing + new_word + " "

    # Ensure proper trailing space convention:
    # All notes get trailing space except possibly the last of a word group
    for seg in midi_segments:
        if seg.word and not seg.word.endswith(" "):
            seg.word += " "

    return midi_segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_midi_segments_from_pitch(
    pitched_data: PitchedData,
    transcribed_data: list[TranscribedData],
    allowed_notes: set[str] | None = None,
    confidence_threshold: float = 0.3,
    min_semitone_change: float = 2.0,
    min_note_duration_ms: float = 80.0,
    median_filter_window: int = 5,
) -> list[MidiSegment]:
    """Generate MIDI segments from pitch contour with lyrics overlay.

    Instead of using Whisper word boundaries as note boundaries, this
    function segments the pitch contour directly and creates notes from
    stable pitch regions.  Lyrics from Whisper are then overlaid by
    time alignment.

    Args:
        pitched_data: Raw pitch data from SwiftF0.
        transcribed_data: Whisper transcription (used for lyrics only).
        allowed_notes: Optional key-quantization note set.
        confidence_threshold: Minimum confidence for a frame to be voiced.
        min_semitone_change: Minimum pitch change to trigger a new note.
        min_note_duration_ms: Minimum note duration in ms.
        median_filter_window: Window size for vibrato smoothing.

    Returns:
        List of MidiSegments with pitch-derived timing and lyrics overlay.
    """
    if not pitched_data or not pitched_data.times:
        return []

    print(
        f"{ULTRASINGER_HEAD} Generating notes from pitch contour "
        f"(threshold={min_semitone_change} ST, "
        f"min_duration={min_note_duration_ms}ms)"
    )

    # Stage 1: Find voiced regions
    voiced_regions = _find_voiced_regions(
        pitched_data,
        confidence_threshold=confidence_threshold,
    )

    if not voiced_regions:
        print(f"{ULTRASINGER_HEAD} No voiced regions found in pitch data")
        return []

    # Stage 2: Smooth pitch contour and segment each voiced region
    all_segments: list[MidiSegment] = []

    for region_start, region_end in voiced_regions:
        region_times = pitched_data.times[region_start:region_end]
        region_freqs = pitched_data.frequencies[region_start:region_end]
        region_confs = pitched_data.confidence[region_start:region_end]

        # Median-filter the MIDI values to smooth vibrato
        smoothed = _median_filter_midi(
            region_freqs, region_confs,
            confidence_threshold=confidence_threshold,
            window=median_filter_window,
        )

        # Segment by pitch stability
        sub_regions = _segment_voiced_region(
            region_times, region_freqs, region_confs, smoothed,
            min_semitone_change=min_semitone_change,
            min_note_duration_ms=min_note_duration_ms,
        )

        # Create MidiSegments for each sub-region
        for sub_start, sub_end in sub_regions:
            if sub_start >= sub_end:
                continue

            sub_freqs = region_freqs[sub_start:sub_end]
            sub_confs = region_confs[sub_start:sub_end]

            note = _compute_note_for_frames(sub_freqs, sub_confs)

            if allowed_notes is not None:
                from modules.Audio.key_detector import quantize_note_to_key
                note = quantize_note_to_key(note, allowed_notes)

            seg_start_time = region_times[sub_start]
            seg_end_time = region_times[min(sub_end - 1, len(region_times) - 1)]

            # Ensure minimum duration
            if seg_end_time - seg_start_time < min_note_duration_ms / 1000.0:
                seg_end_time = seg_start_time + min_note_duration_ms / 1000.0

            all_segments.append(MidiSegment(note, seg_start_time, seg_end_time, "~ "))

    # Stage 3: Merge short fragments and same-pitch neighbors
    all_segments = _merge_short_notes(all_segments, min_note_duration_ms)
    all_segments = _merge_same_pitch_neighbors(all_segments)

    # Stage 4: Overlay lyrics from Whisper transcription
    all_segments = _overlay_lyrics(all_segments, transcribed_data)

    total_notes = len(all_segments)
    lyrics_notes = sum(1 for s in all_segments if s.word not in ("~", "~ "))
    placeholder_notes = total_notes - lyrics_notes

    print(
        f"{ULTRASINGER_HEAD} Pitch-based generation: "
        f"{blue_highlighted(str(total_notes))} notes "
        f"({lyrics_notes} with lyrics, {placeholder_notes} continuations)"
    )

    return all_segments
