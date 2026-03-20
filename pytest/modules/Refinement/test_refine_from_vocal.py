"""Tests for the reverse-scoring refinement module."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher.pitched_data import PitchedData
from modules.Refinement.refine_from_vocal import (
    DIFFICULTY_TOLERANCE,
    _ptakf_tone_to_midi,
    damp_vibrato,
    refine_pitch_with_uscore,
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


# Minimal stand-ins for ultrastar-score types used in mocking

@dataclass
class FakeNoteScore:
    beats_hit: int = 0
    beats_total: int = 10
    detected_tones: list[int] = field(default_factory=list)

    @property
    def hit_ratio(self) -> float:
        return self.beats_hit / self.beats_total if self.beats_total else 0.0


@dataclass
class FakeLineScore:
    note_scores: list[FakeNoteScore] = field(default_factory=list)


@dataclass
class FakeSongScore:
    line_scores: list[FakeLineScore] = field(default_factory=list)


def _mock_score_song(note_scores: list[FakeNoteScore]):
    """Return a mock score_song that returns the given note scores."""
    result = FakeSongScore(line_scores=[FakeLineScore(note_scores=note_scores)])

    def _score_song(song, audio_path, difficulty=None):
        return result

    return _score_song


# ---------------------------------------------------------------------------
# _ptakf_tone_to_midi
# ---------------------------------------------------------------------------

class TestPtakfToneToMidi:
    def test_c2_is_midi_36(self):
        assert _ptakf_tone_to_midi(0) == 36

    def test_a4_is_midi_69(self):
        # A4 = MIDI 69, ptAKF tone = 69 - 36 = 33
        assert _ptakf_tone_to_midi(33) == 69

    def test_round_trip(self):
        for midi in range(36, 93):  # C2 to G#6
            tone = midi - 36
            assert _ptakf_tone_to_midi(tone) == midi


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
        high = float(librosa.note_to_hz("A#4"))  # ~466 Hz
        low = float(librosa.note_to_hz("G#4"))  # ~415 Hz

        freqs = [high if i % 2 == 0 else low for i in range(20)]
        confs = [0.9] * 20

        result_freqs, _ = damp_vibrato(
            freqs, confs, smoothing_window=5, vibrato_threshold_cents=50.0
        )

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
# refine_pitch_with_uscore (mocked ultrastar-score)
# ---------------------------------------------------------------------------

class TestRefinePitchWithUscore:
    """Test pitch refinement logic by mocking ultrastar-score internals."""

    def _run_refinement(self, segments, note_scores, **kwargs):
        """Helper: run refine_pitch_with_uscore with mocked scoring."""
        mock_score_fn = _mock_score_song(note_scores)

        # Mock the ultrastar_score module that gets imported inside the function
        mock_uscore = MagicMock()
        mock_uscore.score_song = mock_score_fn
        mock_uscore.Difficulty.EASY = "easy"
        mock_uscore.Difficulty.MEDIUM = "medium"
        mock_uscore.Difficulty.HARD = "hard"

        mock_parser = MagicMock()
        mock_parser.parse_ultrastar = MagicMock(return_value=MagicMock())

        import sys
        with patch.dict(sys.modules, {
            "ultrastar_score": mock_uscore,
            "ultrastar_score.parser": mock_parser,
        }), \
             patch("modules.Refinement.refine_from_vocal._write_temp_ultrastar_txt", return_value="fake.txt"), \
             patch("os.unlink"):
            return refine_pitch_with_uscore(
                segments,
                vocal_audio_path="fake_vocal.wav",
                bpm=120.0,
                **kwargs,
            )

    def test_corrects_low_hit_ratio_note(self):
        """A note with low hit ratio should be corrected to the detected pitch."""
        # C4 = MIDI 60, ptAKF tone = 60 - 36 = 24
        # Detected: D4 = MIDI 62, ptAKF tone = 26
        segments = [MidiSegment(note="C4", start=0.0, end=1.0, word="test")]
        note_scores = [
            FakeNoteScore(beats_hit=2, beats_total=10, detected_tones=[26] * 8 + [-1, -1]),
        ]

        result, corrections = self._run_refinement(segments, note_scores, hit_ratio_threshold=0.5)
        assert corrections == 1
        assert result[0].note == "D4"

    def test_high_hit_ratio_no_change(self):
        """A note scoring well should not be corrected."""
        segments = [MidiSegment(note="C4", start=0.0, end=1.0, word="test")]
        note_scores = [
            FakeNoteScore(beats_hit=8, beats_total=10, detected_tones=[24] * 10),
        ]

        result, corrections = self._run_refinement(segments, note_scores, hit_ratio_threshold=0.5)
        assert corrections == 0
        assert result[0].note == "C4"

    def test_preserves_lyrics(self):
        """Refinement should not alter the word/lyric of a segment."""
        segments = [MidiSegment(note="C4", start=0.0, end=1.0, word="hello")]
        note_scores = [
            FakeNoteScore(beats_hit=1, beats_total=10, detected_tones=[28] * 10),
        ]

        result, _ = self._run_refinement(segments, note_scores, hit_ratio_threshold=0.5)
        assert result[0].word == "hello"

    def test_all_unvoiced_no_correction(self):
        """If all detected tones are unvoiced (-1), skip correction."""
        segments = [MidiSegment(note="C4", start=0.0, end=1.0, word="test")]
        note_scores = [
            FakeNoteScore(beats_hit=0, beats_total=10, detected_tones=[-1] * 10),
        ]

        result, corrections = self._run_refinement(segments, note_scores, hit_ratio_threshold=0.5)
        assert corrections == 0
        assert result[0].note == "C4"

    def test_zero_beats_total_skipped(self):
        """Notes with zero total beats should be skipped."""
        segments = [MidiSegment(note="C4", start=0.0, end=1.0, word="test")]
        note_scores = [
            FakeNoteScore(beats_hit=0, beats_total=0, detected_tones=[]),
        ]

        _result, corrections = self._run_refinement(segments, note_scores, hit_ratio_threshold=0.5)
        assert corrections == 0

    def test_multiple_segments(self):
        """Multiple segments are refined independently."""
        segments = [
            MidiSegment(note="C4", start=0.0, end=0.5, word="one"),
            MidiSegment(note="C4", start=0.5, end=1.0, word="two"),
        ]
        # First note: low hit ratio, detected D4 (tone 26)
        # Second note: high hit ratio, stays C4
        note_scores = [
            FakeNoteScore(beats_hit=1, beats_total=10, detected_tones=[26] * 10),
            FakeNoteScore(beats_hit=9, beats_total=10, detected_tones=[24] * 10),
        ]

        result, corrections = self._run_refinement(segments, note_scores, hit_ratio_threshold=0.5)
        assert corrections == 1
        assert result[0].note == "D4"
        assert result[1].note == "C4"

    def test_note_count_mismatch_skips(self):
        """If score returns different note count, skip refinement gracefully."""
        segments = [
            MidiSegment(note="C4", start=0.0, end=0.5, word="one"),
            MidiSegment(note="C4", start=0.5, end=1.0, word="two"),
        ]
        # Only one NoteScore for two segments
        note_scores = [
            FakeNoteScore(beats_hit=1, beats_total=10, detected_tones=[26] * 10),
        ]

        result, corrections = self._run_refinement(segments, note_scores, hit_ratio_threshold=0.5)
        assert corrections == 0
        assert result[0].note == "C4"
        assert result[1].note == "C4"

    def test_threshold_boundary(self):
        """Note exactly at the threshold should NOT be corrected (>= means good)."""
        segments = [MidiSegment(note="C4", start=0.0, end=1.0, word="test")]
        # hit_ratio = 5/10 = 0.5, threshold = 0.5 => >= => no correction
        note_scores = [
            FakeNoteScore(beats_hit=5, beats_total=10, detected_tones=[26] * 10),
        ]

        _result, corrections = self._run_refinement(segments, note_scores, hit_ratio_threshold=0.5)
        assert corrections == 0

    def test_median_of_mixed_tones(self):
        """Median should be taken from voiced tones, ignoring unvoiced (-1)."""
        segments = [MidiSegment(note="C4", start=0.0, end=1.0, word="test")]
        # Detected tones: mix of 26 (D4) and 28 (E4), plus some unvoiced
        # Median of [26, 26, 26, 28, 28] = 26
        note_scores = [
            FakeNoteScore(
                beats_hit=2, beats_total=10,
                detected_tones=[-1, -1, 26, 26, 26, 28, 28, -1, -1, -1],
            ),
        ]

        result, corrections = self._run_refinement(segments, note_scores, hit_ratio_threshold=0.5)
        assert corrections == 1
        assert result[0].note == "D4"

    def test_same_pitch_detected_no_correction(self):
        """If detected pitch equals current pitch, no correction counted."""
        # C4 = MIDI 60, ptAKF tone 24
        segments = [MidiSegment(note="C4", start=0.0, end=1.0, word="test")]
        note_scores = [
            FakeNoteScore(beats_hit=2, beats_total=10, detected_tones=[24] * 10),
        ]

        result, corrections = self._run_refinement(segments, note_scores, hit_ratio_threshold=0.5)
        assert corrections == 0
        assert result[0].note == "C4"


# ---------------------------------------------------------------------------
# refine_timing
# ---------------------------------------------------------------------------

class TestRefineTiming:
    def test_snaps_to_onset(self):
        """Note start should snap to nearby onset."""
        pitched = _pitched_data("C4", duration=2.0, n_frames=40)
        onsets = np.array([0.98, 1.50])
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

        result, _corrections = refine_timing(
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
        onsets = np.array([1.495])
        segments = [MidiSegment(note="C4", start=1.48, end=1.5, word="test")]

        result, corrections = refine_timing(
            segments, onsets, pitched,
            timing_threshold_ms=30.0,
        )
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
