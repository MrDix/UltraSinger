"""Tests for growl/scream detection module."""

import unittest

import numpy as np

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Pitcher.growl_detector import (
    detect_growl_segments,
    _analyze_segment,
    _SegmentAnalysis,
)


def _make_pitched_data(
    start: float = 0.0,
    end: float = 5.0,
    step: float = 0.016,
    base_freq: float = 440.0,
    confidence: float = 0.9,
    freq_noise_semitones: float = 0.0,
) -> PitchedData:
    """Create synthetic PitchedData for testing."""
    n_frames = int((end - start) / step)
    times = [start + i * step for i in range(n_frames)]
    # Add pitch noise in semitones
    if freq_noise_semitones > 0:
        rng = np.random.RandomState(42)
        semitone_offsets = rng.normal(0, freq_noise_semitones, n_frames)
        frequencies = [base_freq * (2.0 ** (s / 12.0)) for s in semitone_offsets]
    else:
        frequencies = [base_freq] * n_frames
    confidences = [confidence] * n_frames
    return PitchedData(times=times, frequencies=frequencies, confidence=confidences)


def _make_segment(start: float, end: float, word: str = "test ", note: str = "C4") -> MidiSegment:
    return MidiSegment(note=note, start=start, end=end, word=word)


class TestGrowlDetection(unittest.TestCase):
    """Test the main detect_growl_segments function."""

    def test_clean_singing_not_marked(self):
        """Clean singing with high confidence and stable pitch → not freestyle."""
        pd = _make_pitched_data(confidence=0.9, freq_noise_semitones=0.5)
        segments = [_make_segment(0.5, 1.5), _make_segment(2.0, 3.0)]
        result = detect_growl_segments(segments, pd, use_spectral=False)
        for seg in result:
            self.assertEqual(seg.note_type, ":")

    def test_growl_detected_low_confidence_high_stdev(self):
        """Low confidence + erratic pitch → marked as freestyle."""
        pd = _make_pitched_data(confidence=0.2, freq_noise_semitones=8.0)
        segments = [_make_segment(0.5, 1.5)]
        result = detect_growl_segments(segments, pd, use_spectral=False)
        self.assertEqual(result[0].note_type, "F")

    def test_low_confidence_stable_pitch_not_growl(self):
        """Low confidence but stable pitch → not growl (might be quiet singing)."""
        pd = _make_pitched_data(confidence=0.2, freq_noise_semitones=0.5)
        segments = [_make_segment(0.5, 1.5)]
        result = detect_growl_segments(segments, pd, use_spectral=False)
        self.assertEqual(result[0].note_type, ":")

    def test_high_confidence_erratic_pitch_not_growl(self):
        """High confidence + erratic pitch → not growl (ornamentation)."""
        pd = _make_pitched_data(confidence=0.8, freq_noise_semitones=8.0)
        segments = [_make_segment(0.5, 1.5)]
        result = detect_growl_segments(segments, pd, use_spectral=False)
        self.assertEqual(result[0].note_type, ":")

    def test_existing_freestyle_preserved(self):
        """Segments already marked as freestyle should not be re-analyzed."""
        pd = _make_pitched_data(confidence=0.9, freq_noise_semitones=0.5)
        seg = _make_segment(0.5, 1.5)
        seg.note_type = "F"
        result = detect_growl_segments([seg], pd, use_spectral=False)
        self.assertEqual(result[0].note_type, "F")

    def test_empty_segments(self):
        """Empty segments list → return unchanged."""
        pd = _make_pitched_data()
        result = detect_growl_segments([], pd)
        self.assertEqual(result, [])

    def test_none_pitched_data(self):
        """None pitched_data → return unchanged."""
        segments = [_make_segment(0.5, 1.5)]
        result = detect_growl_segments(segments, None)
        self.assertEqual(result[0].note_type, ":")

    def test_mixed_segments(self):
        """Mix of clean and growl segments."""
        # 0-2.5s: clean singing
        # 2.5-5s: growl
        n_clean = 156  # 2.5s / 0.016
        n_growl = 156
        times = [i * 0.016 for i in range(n_clean + n_growl)]
        rng = np.random.RandomState(42)
        growl_semitones = rng.normal(0, 8, n_growl)
        frequencies = (
            [440.0] * n_clean  # Stable A4
            + [440.0 * (2.0 ** (s / 12.0)) for s in growl_semitones]  # Erratic
        )
        confidences = [0.9] * n_clean + [0.15] * n_growl
        pd = PitchedData(times=times, frequencies=frequencies, confidence=confidences)

        segments = [
            _make_segment(0.5, 2.0, "clean "),
            _make_segment(2.5, 4.5, "growl "),
        ]
        result = detect_growl_segments(segments, pd, use_spectral=False)
        self.assertEqual(result[0].note_type, ":")  # Clean
        self.assertEqual(result[1].note_type, "F")  # Growl

    def test_threshold_customization(self):
        """Custom thresholds should affect detection."""
        pd = _make_pitched_data(confidence=0.3, freq_noise_semitones=5.0)
        segments = [_make_segment(0.5, 1.5)]

        # With default thresholds (conf=0.35, stdev=4.0) → detected
        result = detect_growl_segments(segments, pd, use_spectral=False)
        self.assertEqual(result[0].note_type, "F")

        # Reset and use very strict threshold → not detected
        segments[0].note_type = ":"
        result = detect_growl_segments(
            segments, pd,
            confidence_threshold=0.1,  # Very strict
            use_spectral=False,
        )
        self.assertEqual(result[0].note_type, ":")

    def test_too_few_voiced_frames_not_growl(self):
        """Silence/breath (very low confidence, no voiced frames) → not marked."""
        pd = _make_pitched_data(confidence=0.05, base_freq=20.0)  # Below 50Hz threshold
        segments = [_make_segment(0.5, 1.5)]
        result = detect_growl_segments(segments, pd, use_spectral=False)
        self.assertEqual(result[0].note_type, ":")


class TestSegmentAnalysis(unittest.TestCase):
    """Test the internal _analyze_segment function."""

    def test_analysis_values_clean(self):
        """Clean singing should have high confidence, low stdev."""
        pd = _make_pitched_data(confidence=0.85, freq_noise_semitones=0.3)
        times = np.array(pd.times)
        freqs = np.array(pd.frequencies)
        confs = np.array(pd.confidence)
        seg = _make_segment(0.5, 1.5)

        result = _analyze_segment(
            seg, times, freqs, confs, None, None,
            0.35, 4.0, 0.25, 0.15,
        )
        self.assertFalse(result.is_growl)
        self.assertGreater(result.median_conf, 0.8)
        self.assertLess(result.pitch_stdev, 1.0)

    def test_analysis_values_growl(self):
        """Growl should have low confidence, high stdev."""
        pd = _make_pitched_data(confidence=0.15, freq_noise_semitones=10.0)
        times = np.array(pd.times)
        freqs = np.array(pd.frequencies)
        confs = np.array(pd.confidence)
        seg = _make_segment(0.5, 1.5)

        result = _analyze_segment(
            seg, times, freqs, confs, None, None,
            0.35, 4.0, 0.25, 0.15,
        )
        self.assertTrue(result.is_growl)
        self.assertLess(result.median_conf, 0.35)
        self.assertGreater(result.pitch_stdev, 4.0)


if __name__ == "__main__":
    unittest.main()
