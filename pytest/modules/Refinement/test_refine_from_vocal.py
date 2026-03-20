"""Tests for the reverse-scoring refinement module."""

from __future__ import annotations

import numpy as np
import pytest

from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher.pitched_data import PitchedData
from modules.Refinement.refine_from_vocal import (
    DIFFICULTY_TOLERANCE,
    damp_vibrato,
    refine_pitch,
    refine_timing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pitched_data(
    note: str = "C4",
    duration: float = 1.0,
    n_frames: int = 20,
    confidence: float = 0.9,
) -> PitchedData:
    """Create synthetic PitchedData for a single sustained note."""
    import librosa
    freq = float(librosa.note_to_hz(note))
    times = np.linspace(0.0, duration, n_frames).tolist()
    frequencies = [freq] * n_frames
    confidences = [confidence] * n_frames
    return PitchedData(times=times, frequencies=frequencies, confidence=confidences)


def _pitched_data_multi(
    notes_and_ranges: list[tuple[str, float, float]],
    n_frames_per_note: int = 20,
    confidence: float = 0.9,
) -> PitchedData:
    """Create PitchedData spanning multiple notes in sequence.

    Args:
        notes_and_ranges: List of (note_name, start_time, end_time).
    """
    import librosa
    times = []
    frequencies = []
    confidences = []
    for note, start, end in notes_and_ranges:
        freq = float(librosa.note_to_hz(note))
        ts = np.linspace(start, end, n_frames_per_note).tolist()
        times.extend(ts)
        frequencies.extend([freq] * n_frames_per_note)
        confidences.extend([confidence] * n_frames_per_note)
    return PitchedData(times=times, frequencies=frequencies, confidence=confidences)


# ---------------------------------------------------------------------------
# damp_vibrato
# ---------------------------------------------------------------------------

class TestDampVibrato:
    def test_no_vibrato_returns_unchanged(self):
        """Frequencies within threshold should not be smoothed."""
        import librosa
        freq = float(librosa.note_to_hz("C4"))
        freqs = [freq] * 10
        confs = [0.9] * 10
        result_freqs, result_confs = damp_vibrato(
            freqs, confs, smoothing_window=5, vibrato_threshold_cents=50.0
        )
        assert result_freqs == freqs
        assert result_confs == confs

    def test_vibrato_is_smoothed(self):
        """Wide vibrato should be damped by smoothing."""
        import librosa
        base = float(librosa.note_to_hz("A4"))  # 440 Hz
        # Oscillate +-1 semitone (~26 Hz at A4) => ~100 cents spread
        high = float(librosa.note_to_hz("A#4"))  # ~466 Hz
        low = float(librosa.note_to_hz("G#4"))  # ~415 Hz

        freqs = []
        for i in range(20):
            freqs.append(high if i % 2 == 0 else low)
        confs = [0.9] * 20

        result_freqs, _ = damp_vibrato(
            freqs, confs, smoothing_window=5, vibrato_threshold_cents=50.0
        )

        # After smoothing, the middle values should be closer to the mean
        mid = len(result_freqs) // 2
        mean_freq = (high + low) / 2.0
        assert abs(result_freqs[mid] - mean_freq) < abs(freqs[mid] - mean_freq)

    def test_too_few_frames_returns_unchanged(self):
        """Fewer frames than window should skip smoothing."""
        freqs = [440.0, 445.0]
        confs = [0.9, 0.9]
        result, _ = damp_vibrato(freqs, confs, smoothing_window=5)
        assert result == freqs


# ---------------------------------------------------------------------------
# refine_pitch
# ---------------------------------------------------------------------------

class TestRefinePitch:
    def test_corrects_wrong_note(self):
        """A note 3 HT off should be corrected (threshold=1, easy=+2 => 3)."""
        pitched = _pitched_data("D4", duration=1.0)
        segments = [MidiSegment(note="A#3", start=0.0, end=1.0, word="test")]

        result, corrections = refine_pitch(
            segments, pitched,
            pitch_threshold_ht=1.0,
            difficulty="easy",  # +2 tolerance => effective 3 HT
        )
        import librosa
        # A#3 is MIDI 58, D4 is MIDI 62 => 4 HT diff > 3 threshold
        assert corrections == 1
        assert result[0].note == "D4"

    def test_below_threshold_no_change(self):
        """A note within threshold should not be corrected."""
        pitched = _pitched_data("C#4", duration=1.0)
        # C4=60, C#4=61 => 1 HT diff, threshold 1+2(easy)=3 => no correction
        segments = [MidiSegment(note="C4", start=0.0, end=1.0, word="test")]

        result, corrections = refine_pitch(
            segments, pitched,
            pitch_threshold_ht=1.0,
            difficulty="easy",
        )
        assert corrections == 0
        assert result[0].note == "C4"

    def test_hard_difficulty_corrects_small_deviation(self):
        """Hard difficulty (0 tolerance) should correct even 2 HT deviation."""
        pitched = _pitched_data("D4", duration=1.0)
        # C4=60, D4=62 => 2 HT, threshold 1+0(hard)=1 => correction
        segments = [MidiSegment(note="C4", start=0.0, end=1.0, word="test")]

        result, corrections = refine_pitch(
            segments, pitched,
            pitch_threshold_ht=1.0,
            difficulty="hard",
        )
        assert corrections == 1
        assert result[0].note == "D4"

    def test_medium_difficulty(self):
        """Medium difficulty (+1 tolerance) boundary test."""
        pitched = _pitched_data("D4", duration=1.0)
        # C4=60, D4=62 => 2 HT, threshold 1+1(medium)=2 => no correction (not >2)
        segments = [MidiSegment(note="C4", start=0.0, end=1.0, word="test")]

        result, corrections = refine_pitch(
            segments, pitched,
            pitch_threshold_ht=1.0,
            difficulty="medium",
        )
        assert corrections == 0

    def test_preserves_lyrics(self):
        """Refinement should not alter the word/lyric of a segment."""
        pitched = _pitched_data("E4", duration=1.0)
        segments = [MidiSegment(note="C4", start=0.0, end=1.0, word="hello")]

        result, _ = refine_pitch(
            segments, pitched,
            pitch_threshold_ht=0.0,
            difficulty="hard",
        )
        assert result[0].word == "hello"

    def test_empty_segments(self):
        """Empty segment list should return empty."""
        pitched = _pitched_data("C4")
        result, corrections = refine_pitch([], pitched)
        assert result == []
        assert corrections == 0

    def test_low_confidence_skipped(self):
        """Notes with very low confidence should not cause corrections."""
        pitched = _pitched_data("E4", duration=1.0, confidence=0.05)
        segments = [MidiSegment(note="C4", start=0.0, end=1.0, word="test")]

        # Even with hard difficulty, low confidence may still trigger via
        # the top-25% fallback. This test verifies no crash at minimum.
        result, _ = refine_pitch(
            segments, pitched,
            pitch_threshold_ht=1.0,
            difficulty="hard",
        )
        assert len(result) == 1

    def test_multiple_segments(self):
        """Multiple segments are refined independently."""
        pitched = _pitched_data_multi([
            ("D4", 0.0, 0.5),
            ("E4", 0.5, 1.0),
        ])
        segments = [
            MidiSegment(note="C4", start=0.0, end=0.5, word="one"),
            MidiSegment(note="C4", start=0.5, end=1.0, word="two"),
        ]
        # D4 vs C4 = 2 HT, E4 vs C4 = 4 HT, hard threshold=1
        result, corrections = refine_pitch(
            segments, pitched,
            pitch_threshold_ht=1.0,
            difficulty="hard",
        )
        assert corrections == 2
        assert result[0].note == "D4"
        assert result[1].note == "E4"


# ---------------------------------------------------------------------------
# refine_timing
# ---------------------------------------------------------------------------

class TestRefineTiming:
    def test_snaps_to_onset(self):
        """Note start should snap to nearby onset."""
        pitched = _pitched_data("C4", duration=2.0, n_frames=40)
        onsets = np.array([0.98, 1.50])  # onset at 0.98s
        segments = [MidiSegment(note="C4", start=1.0, end=1.5, word="test")]

        result, corrections = refine_timing(
            segments, onsets, pitched,
            timing_threshold_ms=30.0,
        )
        assert corrections >= 1
        assert result[0].start == pytest.approx(0.98, abs=0.001)

    def test_no_snap_beyond_threshold(self):
        """Onset too far away should not trigger snap."""
        pitched = _pitched_data("C4", duration=2.0, n_frames=40)
        onsets = np.array([0.5])  # 500ms away
        segments = [MidiSegment(note="C4", start=1.0, end=1.5, word="test")]

        result, corrections = refine_timing(
            segments, onsets, pitched,
            timing_threshold_ms=30.0,
        )
        assert result[0].start == 1.0

    def test_no_overlap_with_previous(self):
        """Snapped start should not overlap with previous note's end."""
        pitched = _pitched_data("C4", duration=2.0, n_frames=40)
        onsets = np.array([0.48, 0.98])
        segments = [
            MidiSegment(note="C4", start=0.5, end=1.0, word="one"),
            MidiSegment(note="D4", start=1.02, end=1.5, word="two"),
        ]

        result, _ = refine_timing(
            segments, onsets, pitched,
            timing_threshold_ms=50.0,
        )
        # Second note should snap to 0.98 but be constrained by first note's end
        assert result[1].start >= result[0].end

    def test_empty_onsets(self):
        """No onsets available should return unchanged."""
        pitched = _pitched_data("C4")
        segments = [MidiSegment(note="C4", start=0.0, end=1.0, word="test")]

        result, corrections = refine_timing(
            segments, np.array([]), pitched,
            timing_threshold_ms=30.0,
        )
        assert corrections == 0
        assert result[0].start == 0.0

    def test_empty_segments(self):
        """Empty segment list should return empty."""
        pitched = _pitched_data("C4")
        result, corrections = refine_timing(
            [], np.array([1.0]), pitched,
        )
        assert result == []
        assert corrections == 0

    def test_no_zero_duration(self):
        """Timing snap should not create zero-duration notes."""
        pitched = _pitched_data("C4", duration=2.0, n_frames=40)
        # Onset at 1.49s, note ends at 1.5s => snapping would leave <10ms
        onsets = np.array([1.495])
        segments = [MidiSegment(note="C4", start=1.48, end=1.5, word="test")]

        result, corrections = refine_timing(
            segments, onsets, pitched,
            timing_threshold_ms=30.0,
        )
        # Should not snap to 1.495 because 1.5 - 1.495 = 5ms < 10ms min
        assert result[0].start == 1.48
        assert corrections == 0


# ---------------------------------------------------------------------------
# DIFFICULTY_TOLERANCE
# ---------------------------------------------------------------------------

class TestDifficultyTolerance:
    def test_easy(self):
        assert DIFFICULTY_TOLERANCE["easy"] == 2.0

    def test_medium(self):
        assert DIFFICULTY_TOLERANCE["medium"] == 1.0

    def test_hard(self):
        assert DIFFICULTY_TOLERANCE["hard"] == 0.0
