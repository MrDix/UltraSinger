"""Reverse-scoring refinement — uses the vocal audio as ground truth to
correct pitch values and note timings in an already-generated note list.

This runs as a post-processing pass *after* the initial pitch detection,
octave correction, and syllable merging, but *before* the UltraStar TXT
is written to disk.  It re-analyses the SwiftF0 pitched data for each note
window and corrects values that deviate beyond configurable thresholds.
"""

from __future__ import annotations

import math

import librosa
import numpy as np

from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitched_data_helper import get_frequencies_with_high_confidence
from modules.Midi.midi_creator import (
    confidence_weighted_median_note,
    find_nearest_index,
)
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted


# Difficulty presets: how many semitones off a note must be before correcting.
# Easy = lenient (only fix gross errors), Hard = strict (fix everything).
DIFFICULTY_TOLERANCE = {
    "easy": 2.0,
    "medium": 1.0,
    "hard": 0.0,
}


def damp_vibrato(
    frequencies: list[float],
    confidence: list[float],
    smoothing_window: int = 5,
    vibrato_threshold_cents: float = 50.0,
) -> tuple[list[float], list[float]]:
    """Smooth pitch oscillations caused by singer vibrato.

    If the standard deviation of detected pitches (in cents relative to
    the median) exceeds *vibrato_threshold_cents*, a moving-average filter
    is applied to stabilise the pitch contour before note detection.

    Args:
        frequencies: Detected frequencies in Hz.
        confidence: Confidence weights for each frequency.
        smoothing_window: Number of frames for the moving-average kernel.
        vibrato_threshold_cents: Minimum pitch spread (in cents) to
            trigger smoothing.  50 cents = half a semitone.

    Returns:
        Tuple of (smoothed_frequencies, unchanged confidence).
    """
    if len(frequencies) < smoothing_window or not frequencies:
        return frequencies, confidence

    freqs = np.asarray(frequencies, dtype=float)

    # Compute median and deviation in cents
    median_freq = np.median(freqs[freqs > 0]) if np.any(freqs > 0) else 0.0
    if median_freq <= 0:
        return frequencies, confidence

    cents = 1200.0 * np.log2(np.where(freqs > 0, freqs, median_freq) / median_freq)
    spread = float(np.std(cents))

    if spread < vibrato_threshold_cents:
        return frequencies, confidence

    # Apply moving-average smoothing
    kernel = np.ones(smoothing_window) / smoothing_window
    smoothed = np.convolve(freqs, kernel, mode="same")

    # Preserve edges (don't smooth the first/last half-window)
    half = smoothing_window // 2
    smoothed[:half] = freqs[:half]
    smoothed[-half:] = freqs[-half:]

    return smoothed.tolist(), confidence


def _get_note_pitch_from_window(
    pitched_data: PitchedData,
    start_time: float,
    end_time: float,
    vibrato_window: int = 5,
    vibrato_threshold_cents: float = 50.0,
) -> str | None:
    """Extract the detected note for a time window from pitched data.

    Returns None if no confident frequencies are found.
    """
    start_idx = find_nearest_index(pitched_data.times, start_time)
    end_idx = find_nearest_index(pitched_data.times, end_time)

    if start_idx == end_idx:
        freqs = [pitched_data.frequencies[start_idx]]
        confs = [pitched_data.confidence[start_idx]]
    else:
        freqs = list(pitched_data.frequencies[start_idx:end_idx])
        confs = list(pitched_data.confidence[start_idx:end_idx])

    if not freqs:
        return None

    # Apply vibrato damping before note extraction
    freqs, confs = damp_vibrato(
        freqs, confs,
        smoothing_window=vibrato_window,
        vibrato_threshold_cents=vibrato_threshold_cents,
    )

    # Filter by confidence
    conf_f, conf_w = get_frequencies_with_high_confidence(freqs, confs)
    if not conf_f:
        return None

    return confidence_weighted_median_note(conf_f, conf_w)


def refine_pitch(
    midi_segments: list[MidiSegment],
    pitched_data: PitchedData,
    pitch_threshold_ht: float = 1.0,
    difficulty: str = "easy",
    vibrato_window: int = 5,
    vibrato_threshold_cents: float = 50.0,
) -> tuple[list[MidiSegment], int]:
    """Refine note pitches by comparing against the vocal audio.

    For each note, the detected pitch from the vocal audio is compared
    against the current MIDI value.  If the deviation exceeds the
    threshold (adjusted by difficulty tolerance), the note is corrected.

    Args:
        midi_segments: Notes to refine (modified in-place).
        pitched_data: SwiftF0 pitch contour from the vocal audio.
        pitch_threshold_ht: Base threshold in semitones before correcting.
        difficulty: Tolerance preset — ``"easy"`` (lenient), ``"medium"``,
            or ``"hard"`` (strict).
        vibrato_window: Smoothing window for vibrato damping.
        vibrato_threshold_cents: Vibrato detection threshold in cents.

    Returns:
        Tuple of (midi_segments, number of corrections made).
    """
    tolerance = DIFFICULTY_TOLERANCE.get(difficulty, 2.0)
    effective_threshold = pitch_threshold_ht + tolerance
    corrections = 0

    for seg in midi_segments:
        detected_note = _get_note_pitch_from_window(
            pitched_data, seg.start, seg.end,
            vibrato_window=vibrato_window,
            vibrato_threshold_cents=vibrato_threshold_cents,
        )
        if detected_note is None:
            continue

        try:
            current_midi = librosa.note_to_midi(seg.note)
            detected_midi = librosa.note_to_midi(detected_note)
        except (ValueError, TypeError):
            continue

        deviation = abs(current_midi - detected_midi)
        if deviation > effective_threshold:
            seg.note = detected_note
            corrections += 1

    return midi_segments, corrections


def refine_timing(
    midi_segments: list[MidiSegment],
    onset_times: np.ndarray,
    pitched_data: PitchedData,
    timing_threshold_ms: float = 30.0,
    confidence_threshold: float = 0.4,
) -> tuple[list[MidiSegment], int]:
    """Refine note start/end times using detected audio onsets.

    For note starts: snaps to the nearest onset within the threshold.
    For note ends: looks for confidence drop-off near the note boundary.

    Args:
        midi_segments: Notes to refine (modified in-place).
        onset_times: Sorted array of onset times in seconds.
        pitched_data: For end-time refinement via confidence.
        timing_threshold_ms: Maximum snap distance in milliseconds.
        confidence_threshold: Minimum confidence for end-time detection.

    Returns:
        Tuple of (midi_segments, number of corrections made).
    """
    if not midi_segments:
        return midi_segments, 0

    threshold_s = timing_threshold_ms / 1000.0
    corrections = 0
    min_duration_s = 0.01  # 10ms minimum note duration

    prev_end = float("-inf")
    for i, seg in enumerate(midi_segments):
        # --- Start refinement: snap to nearest onset ---
        if len(onset_times) > 0:
            idx = np.searchsorted(onset_times, seg.start)

            candidates: list[float] = []
            if idx > 0:
                candidates.append(float(onset_times[idx - 1]))
            if idx < len(onset_times):
                candidates.append(float(onset_times[idx]))

            if candidates:
                nearest = min(candidates, key=lambda t: abs(t - seg.start))
                distance = abs(nearest - seg.start)

                if distance <= threshold_s:
                    # Don't overlap with previous note
                    candidate = max(nearest, prev_end)
                    # Re-check snap distance after prev_end enforcement
                    if (
                        abs(candidate - seg.start) <= threshold_s
                        and candidate < seg.end - min_duration_s
                    ):
                        if candidate != seg.start:
                            seg.start = candidate
                            corrections += 1

        # --- End refinement: detect confidence drop-off ---
        # Look ahead up to threshold_s for confidence drop
        search_end_idx = find_nearest_index(
            pitched_data.times, seg.end + threshold_s
        )
        search_start_idx = find_nearest_index(
            pitched_data.times, seg.end - threshold_s
        )

        if search_start_idx < search_end_idx:
            confs = pitched_data.confidence[search_start_idx:search_end_idx]
            # Find first frame where confidence drops below threshold
            for j, conf in enumerate(confs):
                if conf < confidence_threshold:
                    drop_time = pitched_data.times[search_start_idx + j]
                    if abs(drop_time - seg.end) <= threshold_s:
                        # Clamp to next segment's start to prevent overlap
                        next_start = (
                            midi_segments[i + 1].start
                            if i + 1 < len(midi_segments)
                            else math.inf
                        )
                        new_end = min(drop_time, next_start)
                        if new_end > seg.start + min_duration_s:
                            if new_end != seg.end:
                                seg.end = new_end
                                corrections += 1
                    break

        prev_end = seg.end

    return midi_segments, corrections


def refine_notes(
    midi_segments: list[MidiSegment],
    pitched_data: PitchedData,
    vocal_audio_path: str,
    *,
    refine_pitch_enabled: bool = True,
    refine_timing_enabled: bool = True,
    pitch_threshold_ht: float = 1.0,
    timing_threshold_ms: float = 30.0,
    vibrato_window: int = 5,
    vibrato_threshold_cents: float = 50.0,
    difficulty: str = "easy",
) -> list[MidiSegment]:
    """Orchestrate all refinement passes on the note list.

    Args:
        midi_segments: Notes to refine (modified in-place).
        pitched_data: SwiftF0 pitch contour.
        vocal_audio_path: Path to vocal-only audio (for onset detection).
        refine_pitch_enabled: Whether to run pitch refinement.
        refine_timing_enabled: Whether to run timing refinement.
        pitch_threshold_ht: Semitone threshold for pitch correction.
        timing_threshold_ms: Millisecond threshold for timing correction.
        vibrato_window: Smoothing window for vibrato damping.
        vibrato_threshold_cents: Vibrato detection threshold in cents.
        difficulty: Tolerance preset (``"easy"``, ``"medium"``, ``"hard"``).

    Returns:
        The refined midi_segments list.
    """
    if not midi_segments:
        return midi_segments

    print(
        f"{ULTRASINGER_HEAD} Refining notes from vocal audio "
        f"(difficulty={blue_highlighted(difficulty)}, "
        f"pitch={blue_highlighted(str(refine_pitch_enabled))}, "
        f"timing={blue_highlighted(str(refine_timing_enabled))})"
    )

    pitch_corrections = 0
    timing_corrections = 0

    # Phase 1: Pitch refinement
    if refine_pitch_enabled:
        midi_segments, pitch_corrections = refine_pitch(
            midi_segments,
            pitched_data,
            pitch_threshold_ht=pitch_threshold_ht,
            difficulty=difficulty,
            vibrato_window=vibrato_window,
            vibrato_threshold_cents=vibrato_threshold_cents,
        )

    # Phase 2: Timing refinement (requires onset detection)
    if refine_timing_enabled:
        from modules.Audio.onset_correction import detect_vocal_onsets

        onset_times = detect_vocal_onsets(vocal_audio_path)
        midi_segments, timing_corrections = refine_timing(
            midi_segments,
            onset_times,
            pitched_data,
            timing_threshold_ms=timing_threshold_ms,
        )

    total = pitch_corrections + timing_corrections
    print(
        f"{ULTRASINGER_HEAD} Refinement complete: "
        f"{blue_highlighted(str(pitch_corrections))} pitch, "
        f"{blue_highlighted(str(timing_corrections))} timing corrections "
        f"({blue_highlighted(str(total))} total)"
    )

    return midi_segments
