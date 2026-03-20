"""Reverse-scoring refinement — uses ultrastar-score's C++ ptAKF pitch
detection (the same algorithm as Vocaluxe/USDX) to identify notes that
would score poorly, then corrects them.

Pitch refinement:
    1. Write a temporary UltraStar TXT from the current midi_segments
    2. Score it against the vocal audio using ultrastar-score's ``score_song()``
    3. For notes with low ``hit_ratio``: replace the pitch with the median
       of ``detected_tones`` from the ptAKF detector
    4. Convert back to note names on the MidiSegments

Timing refinement remains librosa-based (onset detection), since ptAKF
only provides pitch, not onset timing.

This ensures the refinement uses the *exact same* pitch detection and
tolerance logic that the games use, eliminating systematic bias from
Python pitch libraries like librosa/SwiftF0.
"""

from __future__ import annotations

import math
import os
import tempfile

import librosa
import numpy as np

from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher.pitched_data import PitchedData
from modules.Midi.midi_creator import find_nearest_index
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted


# Difficulty presets: how many semitones off a note must be before correcting.
# These mirror ultrastar-score's Difficulty enum but as a simple mapping.
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


def _ptakf_tone_to_midi(tone: int) -> int:
    """Convert ptAKF tone index to MIDI note number.

    ptAKF: tone 0 = C2 = MIDI 36.
    """
    return tone + 36


def _write_temp_ultrastar_txt(
    midi_segments: list[MidiSegment],
    bpm: float,
    gap_ms: float = 0.0,
) -> str:
    """Write a minimal UltraStar TXT file from midi_segments for scoring.

    Returns the path to the temporary file.
    """
    from modules.Ultrastar.coverter.ultrastar_converter import (
        real_bpm_to_ultrastar_bpm,
        second_to_beat,
    )
    from modules.Ultrastar.coverter.ultrastar_midi_converter import (
        convert_midi_note_to_ultrastar_note,
    )
    from modules.Ultrastar.ultrastar_writer import get_multiplier

    multiplier = get_multiplier(bpm)
    ultrastar_bpm = real_bpm_to_ultrastar_bpm(bpm, multiplier)

    lines = []
    lines.append("#TITLE:_refine_temp")
    lines.append("#ARTIST:_refine_temp")
    lines.append(f"#BPM:{ultrastar_bpm}")
    lines.append(f"#GAP:{gap_ms}")
    lines.append("#VERSION:1.2.0")
    lines.append("#MP3:_refine_temp.mp3")

    for seg in midi_segments:
        start_beat = second_to_beat(seg.start, ultrastar_bpm, gap_ms)
        end_beat = second_to_beat(seg.end, ultrastar_bpm, gap_ms)
        duration = max(1, end_beat - start_beat)
        pitch = convert_midi_note_to_ultrastar_note(seg)
        word = seg.word if seg.word else "~"
        lines.append(f": {start_beat} {duration} {pitch} {word}")

    lines.append("E")

    fd, path = tempfile.mkstemp(suffix=".txt", prefix="usinger_refine_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path


def refine_pitch_with_uscore(
    midi_segments: list[MidiSegment],
    vocal_audio_path: str,
    bpm: float,
    difficulty: str = "easy",
    hit_ratio_threshold: float = 0.5,
) -> tuple[list[MidiSegment], int]:
    """Refine note pitches using ultrastar-score's C++ ptAKF detector.

    Scores the current notes against the vocal audio using the same
    algorithm as Vocaluxe/USDX.  Notes that score poorly (low hit_ratio)
    are corrected by taking the median of the ptAKF-detected tones.

    Args:
        midi_segments: Notes to refine (modified in-place).
        vocal_audio_path: Path to vocal-only audio file.
        bpm: Song BPM for beat/time conversion.
        difficulty: Tolerance preset (``"easy"``, ``"medium"``, ``"hard"``).
        hit_ratio_threshold: Notes below this hit ratio are corrected.

    Returns:
        Tuple of (midi_segments, number of corrections made).
    """
    from ultrastar_score import score_song, Difficulty
    from ultrastar_score.parser import parse_ultrastar

    # Map difficulty string to ultrastar-score enum
    diff_map = {
        "easy": Difficulty.EASY,
        "medium": Difficulty.MEDIUM,
        "hard": Difficulty.HARD,
    }
    uscore_difficulty = diff_map.get(difficulty, Difficulty.EASY)

    # Write temporary TXT for scoring
    tmp_txt = _write_temp_ultrastar_txt(midi_segments, bpm)
    try:
        song = parse_ultrastar(tmp_txt)
        result = score_song(song, vocal_audio_path, difficulty=uscore_difficulty)
    finally:
        try:
            os.unlink(tmp_txt)
        except OSError:
            pass

    # Flatten all NoteScores to match midi_segments order
    all_note_scores = [
        ns for ls in result.line_scores for ns in ls.note_scores
    ]

    if len(all_note_scores) != len(midi_segments):
        print(
            f"{ULTRASINGER_HEAD} Warning: note count mismatch "
            f"(segments={len(midi_segments)}, scores={len(all_note_scores)}), "
            f"skipping pitch refinement"
        )
        return midi_segments, 0

    corrections = 0
    for seg, ns in zip(midi_segments, all_note_scores, strict=True):
        if ns.beats_total == 0:
            continue

        # Note already scores well — skip
        if ns.hit_ratio >= hit_ratio_threshold:
            continue

        # Get the ptAKF-detected tones for this note (excluding unvoiced = -1)
        voiced_tones = [t for t in ns.detected_tones if t >= 0]
        if not voiced_tones:
            continue

        # Median of detected tones (ptAKF tone index)
        median_tone = int(np.median(voiced_tones))
        detected_midi = _ptakf_tone_to_midi(median_tone)

        try:
            current_midi = librosa.note_to_midi(seg.note)
        except (ValueError, TypeError):
            continue

        if detected_midi != current_midi:
            seg.note = librosa.midi_to_note(detected_midi)
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
    bpm: float,
    *,
    refine_pitch_enabled: bool = True,
    refine_timing_enabled: bool = True,
    timing_threshold_ms: float = 30.0,
    difficulty: str = "easy",
    hit_ratio_threshold: float = 0.5,
    vibrato_window: int = 5,
    vibrato_threshold_cents: float = 50.0,
) -> list[MidiSegment]:
    """Orchestrate all refinement passes on the note list.

    Pitch refinement uses ultrastar-score's C++ ptAKF detector (the same
    algorithm as Vocaluxe/USDX) to identify poorly-scoring notes, then
    corrects them based on what the game would actually detect.

    Timing refinement uses librosa onset detection (since ptAKF only
    provides pitch, not onset timing).

    Args:
        midi_segments: Notes to refine (modified in-place).
        pitched_data: SwiftF0 pitch contour (for timing refinement).
        vocal_audio_path: Path to vocal-only audio.
        bpm: Song BPM for beat/time conversion.
        refine_pitch_enabled: Whether to run pitch refinement.
        refine_timing_enabled: Whether to run timing refinement.
        timing_threshold_ms: Millisecond threshold for timing correction.
        difficulty: Tolerance preset (``"easy"``, ``"medium"``, ``"hard"``).
        hit_ratio_threshold: Notes below this hit ratio are pitch-corrected.
        vibrato_window: Smoothing window for vibrato damping (reserved
            for future timing confidence analysis).
        vibrato_threshold_cents: Vibrato detection threshold (reserved
            for future timing confidence analysis).

    Returns:
        The refined midi_segments list.
    """
    if not midi_segments:
        return midi_segments

    print(
        f"{ULTRASINGER_HEAD} Refining notes using game scoring engine "
        f"(difficulty={blue_highlighted(difficulty)}, "
        f"pitch={blue_highlighted(str(refine_pitch_enabled))}, "
        f"timing={blue_highlighted(str(refine_timing_enabled))})"
    )

    pitch_corrections = 0
    timing_corrections = 0

    # Phase 1: Pitch refinement via ultrastar-score (C++ ptAKF)
    if refine_pitch_enabled:
        try:
            midi_segments, pitch_corrections = refine_pitch_with_uscore(
                midi_segments,
                vocal_audio_path,
                bpm=bpm,
                difficulty=difficulty,
                hit_ratio_threshold=hit_ratio_threshold,
            )
        except (ImportError, OSError, ValueError, RuntimeError) as e:
            print(
                f"{ULTRASINGER_HEAD} Warning: uscore pitch refinement failed: {e}. "
                f"Skipping pitch refinement."
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
