"""Pitch-based note generation: create notes from pitch contour.

Instead of using Whisper word timing as note boundaries (one word = one note),
this module generates notes directly from the pitch contour detected by
SwiftF0.  Lyrics from Whisper are overlaid on the pitch-derived notes by
time alignment.

This produces far more accurate results for melismatic songs (runs, slides,
ornaments) where a single word can span many different pitches.
"""

from __future__ import annotations

import difflib
import re
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
    if not conf_f or sum(conf_weights) == 0:
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

    For each pitch-note, all overlapping Whisper words are collected.
    If multiple words fall on a single note, the note is **split at word
    boundaries** so each word gets its own note (same pitch).  Notes with
    no overlapping word keep the "~" continuation marker.
    """
    if not transcribed_data or not midi_segments:
        return midi_segments

    # Initialize all notes as "~" (continuation/placeholder)
    for seg in midi_segments:
        seg.word = "~ "

    # For each note, collect all overlapping words with their time ranges
    # word_assignments[seg_idx] = [(word_text, word_start, word_end), ...]
    word_assignments: dict[int, list[tuple[str, float, float]]] = {}

    for td in transcribed_data:
        best_overlap = 0.0
        best_idx = -1

        for i, seg in enumerate(midi_segments):
            overlap_start = max(td.start, seg.start)
            overlap_end = min(td.end, seg.end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        if best_idx >= 0 and best_overlap > 0:
            if best_idx not in word_assignments:
                word_assignments[best_idx] = []
            word_assignments[best_idx].append((td.word, td.start, td.end))

    # Build the result list, splitting notes that have multiple words
    result: list[MidiSegment] = []

    for i, seg in enumerate(midi_segments):
        words = word_assignments.get(i)

        if not words:
            # No words — keep as "~" placeholder
            result.append(seg)
        elif len(words) == 1:
            # Exactly one word — simple assignment
            seg.word = words[0][0]
            if not seg.word.endswith(" "):
                seg.word += " "
            result.append(seg)
        else:
            # Multiple words on one note — split at word boundaries
            words.sort(key=lambda w: w[1])  # sort by word start time
            note_start = seg.start
            note_end = seg.end

            for wi, (word_text, w_start, w_end) in enumerate(words):
                # Determine split boundaries:
                # - First sub-note starts at the original note start
                # - Last sub-note ends at the original note end
                # - Middle boundaries are midpoints between consecutive words
                if wi == 0:
                    sub_start = note_start
                else:
                    prev_end = words[wi - 1][2]
                    sub_start = (prev_end + w_start) / 2.0

                if wi == len(words) - 1:
                    sub_end = note_end
                else:
                    next_start = words[wi + 1][1]
                    sub_end = (w_end + next_start) / 2.0

                # Clamp to note boundaries
                sub_start = max(sub_start, note_start)
                sub_end = min(sub_end, note_end)

                if sub_end <= sub_start:
                    continue

                word_with_space = word_text if word_text.endswith(" ") else word_text + " "
                result.append(MidiSegment(seg.note, sub_start, sub_end, word_with_space))

    # Ensure proper trailing space convention
    for seg in result:
        if seg.word and not seg.word.endswith(" "):
            seg.word += " "

    return result


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


# ---------------------------------------------------------------------------
# Reference lyrics fill (LRCLIB integration)
# ---------------------------------------------------------------------------

def _normalize_word(word: str) -> str:
    """Normalize a word for comparison (lowercase, strip punctuation)."""
    word = word.strip().lower()
    word = re.sub(r"[^\w']", "", word)
    return word.strip("'")


def _normalize_lyrics_to_words(plain_lyrics: str) -> list[str]:
    """Normalize plain lyrics text into a flat list of lowercase words."""
    text = plain_lyrics.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"[^\w\s'\-]", " ", text)
    words = [_normalize_word(w) for w in text.split() if w.strip()]
    return [w for w in words if w]


def fill_lyrics_from_reference(
    midi_segments: list[MidiSegment],
    transcribed_data: list[TranscribedData],
    plain_lyrics: str,
) -> list[MidiSegment]:
    """Fill ``~`` placeholder notes with words from reference lyrics.

    After ``create_midi_segments_from_pitch`` overlays Whisper words onto
    pitch-derived notes, many notes may still have ``~`` placeholders —
    especially for melismatic passages that Whisper couldn't transcribe.

    This function uses the full reference lyrics (from LRCLIB) to fill
    those gaps.  It aligns the already-assigned Whisper words against the
    reference text to find which reference words are missing, then
    distributes them chronologically onto the ``~`` placeholder notes.

    Args:
        midi_segments: Segments from ``create_midi_segments_from_pitch``.
        transcribed_data: Whisper transcription (for anchor identification).
        plain_lyrics: Full plain-text lyrics from LRCLIB.

    Returns:
        The same segment list with ``~`` placeholders replaced where possible.
    """
    if not midi_segments or not plain_lyrics or not plain_lyrics.strip():
        return midi_segments

    ref_words = _normalize_lyrics_to_words(plain_lyrics)
    if not ref_words:
        return midi_segments

    # Build list of (segment_index, normalized_word) for assigned notes
    assigned: list[tuple[int, str]] = []
    for i, seg in enumerate(midi_segments):
        if seg.word not in ("~", "~ "):
            norm = _normalize_word(seg.word)
            if norm:
                assigned.append((i, norm))

    if not assigned:
        # No anchor words at all — distribute reference words sequentially
        # to all placeholder notes
        placeholder_indices = list(range(len(midi_segments)))
        _assign_words_to_segments(
            midi_segments, placeholder_indices, ref_words
        )
        return midi_segments

    # Align assigned words against reference to find gaps
    assigned_words = [w for _, w in assigned]
    assigned_indices = [idx for idx, _ in assigned]

    matcher = difflib.SequenceMatcher(None, assigned_words, ref_words, autojunk=False)

    # Build a map: ref_word_index → assigned_segment_index (for matched words)
    ref_to_seg: dict[int, int] = {}
    for op, a_start, a_end, r_start, r_end in matcher.get_opcodes():
        if op == "equal" or op == "replace":
            # Map each matched/replaced pair
            for offset in range(min(a_end - a_start, r_end - r_start)):
                ref_to_seg[r_start + offset] = assigned_indices[a_start + offset]

    # Find reference words that have NO match (the gaps to fill)
    unmatched_ref: list[tuple[int, str]] = []
    for ri, rw in enumerate(ref_words):
        if ri not in ref_to_seg:
            unmatched_ref.append((ri, rw))

    if not unmatched_ref:
        print(f"{ULTRASINGER_HEAD} Reference lyrics: all words already assigned")
        return midi_segments

    # Group unmatched reference words into runs (consecutive ref indices)
    # and find the best ~ placeholder notes to assign them to
    runs: list[list[tuple[int, str]]] = []
    current_run: list[tuple[int, str]] = [unmatched_ref[0]]

    for ri, rw in unmatched_ref[1:]:
        if ri == current_run[-1][0] + 1:
            current_run.append((ri, rw))
        else:
            runs.append(current_run)
            current_run = [(ri, rw)]
    runs.append(current_run)

    filled_count = 0
    overflow_words: list[str] = []  # Words that couldn't be placed in their region

    for run in runs:
        run_ref_start = run[0][0]
        run_ref_end = run[-1][0]
        run_words = [rw for _, rw in run]

        # Find the bounding segment indices from matched neighbors
        # Left bound: the segment of the last matched ref word before this run
        # Right bound: the segment of the first matched ref word after this run
        left_seg_idx = -1
        right_seg_idx = len(midi_segments)

        for ri in range(run_ref_start - 1, -1, -1):
            if ri in ref_to_seg:
                left_seg_idx = ref_to_seg[ri]
                break

        for ri in range(run_ref_end + 1, len(ref_words)):
            if ri in ref_to_seg:
                right_seg_idx = ref_to_seg[ri]
                break

        # Collect ~ placeholder notes between the bounds
        placeholders = []
        for si in range(left_seg_idx + 1, right_seg_idx):
            if si < 0 or si >= len(midi_segments):
                continue
            if midi_segments[si].word in ("~", "~ "):
                placeholders.append(si)

        if placeholders:
            filled_count += _assign_words_to_segments(
                midi_segments, placeholders, run_words
            )
        else:
            # No placeholders in this region — save for overflow pass
            overflow_words.extend(run_words)

    # Overflow pass: distribute remaining words onto any ~ notes globally
    if overflow_words:
        global_placeholders = [
            i for i, seg in enumerate(midi_segments)
            if seg.word in ("~", "~ ")
        ]
        if global_placeholders:
            filled_count += _assign_words_to_segments(
                midi_segments, global_placeholders, overflow_words
            )
        else:
            print(
                f"{ULTRASINGER_HEAD} Warning: {len(overflow_words)} reference "
                f"word(s) could not be placed (no placeholder notes available)"
            )

    total_ref = len(ref_words)
    total_assigned_after = sum(
        1 for s in midi_segments if s.word not in ("~", "~ ")
    )
    placeholder_remaining = len(midi_segments) - total_assigned_after

    print(
        f"{ULTRASINGER_HEAD} Reference lyrics fill: "
        f"{blue_highlighted(str(filled_count))} words assigned to placeholder notes "
        f"({total_ref} reference words, "
        f"{total_assigned_after} notes with text, "
        f"{placeholder_remaining} placeholders remaining)"
    )

    return midi_segments


def _assign_words_to_segments(
    midi_segments: list[MidiSegment],
    placeholder_indices: list[int],
    words: list[str],
) -> int:
    """Assign words to placeholder segments, distributing evenly.

    If there are more placeholders than words, words are spread out evenly.
    If there are more words than placeholders, multiple words are concatenated
    into single notes.

    Returns:
        Number of words actually assigned.
    """
    if not placeholder_indices or not words:
        return 0

    n_placeholders = len(placeholder_indices)
    n_words = len(words)
    assigned = 0

    if n_words <= n_placeholders:
        # Distribute words evenly across placeholders
        # Use proportional spacing: word i goes to placeholder at position
        # i * n_placeholders / n_words
        for wi, word in enumerate(words):
            pi = int(wi * n_placeholders / n_words)
            pi = min(pi, n_placeholders - 1)
            seg_idx = placeholder_indices[pi]
            midi_segments[seg_idx].word = word + " "
            assigned += 1
    else:
        # More words than placeholders — group words into placeholder slots
        words_per_slot = n_words / n_placeholders
        for pi_offset, seg_idx in enumerate(placeholder_indices):
            start_w = int(pi_offset * words_per_slot)
            end_w = int((pi_offset + 1) * words_per_slot)
            end_w = min(end_w, n_words)
            if start_w < end_w:
                combined = "".join(words[start_w:end_w])
                midi_segments[seg_idx].word = combined + " "
                assigned += end_w - start_w

    return assigned
