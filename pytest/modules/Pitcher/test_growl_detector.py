"""Tests for growl/scream detection module."""

import unittest

import numpy as np

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Pitcher.growl_detector import (
    detect_growl_segments,
    _analyze_segment_pitch,
    _detect_by_harmonicity,
    _HpssData,
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


def _make_hpss_data(
    n_frames: int = 100,
    n_freq_bins: int = 1025,
    harmonicity: float = 0.8,
    duration: float = 5.0,
    sr: int = 22050,
    hop_length: int = 512,
) -> _HpssData:
    """Create synthetic HPSS data with controlled harmonicity ratio.

    harmonicity=0.8 means 80% of energy is harmonic (clean singing).
    harmonicity=0.1 means 10% of energy is harmonic (growl).
    """
    rng = np.random.RandomState(42)
    # Total magnitude = 1.0 per bin for simplicity
    base = rng.uniform(0.5, 1.5, (n_freq_bins, n_frames))
    # harm^2 / (harm^2 + perc^2) = harmonicity
    # So harm = sqrt(harmonicity), perc = sqrt(1 - harmonicity)
    harm_scale = np.sqrt(harmonicity)
    perc_scale = np.sqrt(1.0 - harmonicity)
    H = base * harm_scale
    P = base * perc_scale
    frame_times = np.linspace(0, duration, n_frames)
    return _HpssData(
        sr=sr,
        hop_length=hop_length,
        harmonic_mag=H,
        percussive_mag=P,
        frame_times=frame_times,
        duration=duration,
    )


class TestGrowlDetection(unittest.TestCase):
    """Test the main detect_growl_segments function."""

    def test_clean_singing_not_marked(self):
        """Clean singing with high confidence and stable pitch -> not freestyle."""
        pd = _make_pitched_data(confidence=0.9, freq_noise_semitones=0.5)
        segments = [_make_segment(0.5, 1.5), _make_segment(2.0, 3.0)]
        result = detect_growl_segments(segments, pd, use_spectral=False)
        for seg in result:
            self.assertEqual(seg.note_type, ":")

    def test_growl_detected_low_confidence_high_stdev(self):
        """Low confidence + erratic pitch -> marked as freestyle (fallback)."""
        pd = _make_pitched_data(confidence=0.2, freq_noise_semitones=8.0)
        segments = [_make_segment(0.5, 1.5)]
        result = detect_growl_segments(segments, pd, use_spectral=False)
        self.assertEqual(result[0].note_type, "F")

    def test_low_confidence_stable_pitch_not_growl(self):
        """Low confidence but stable pitch -> not growl (might be quiet singing)."""
        pd = _make_pitched_data(confidence=0.2, freq_noise_semitones=0.5)
        segments = [_make_segment(0.5, 1.5)]
        result = detect_growl_segments(segments, pd, use_spectral=False)
        self.assertEqual(result[0].note_type, ":")

    def test_high_confidence_erratic_pitch_not_growl(self):
        """High confidence + erratic pitch -> not growl (ornamentation)."""
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
        """Empty segments list -> return unchanged."""
        pd = _make_pitched_data()
        result = detect_growl_segments([], pd)
        self.assertEqual(result, [])

    def test_none_pitched_data(self):
        """None pitched_data -> return unchanged."""
        segments = [_make_segment(0.5, 1.5)]
        result = detect_growl_segments(segments, None)
        self.assertEqual(segments[0].note_type, ":")

    def test_mixed_segments(self):
        """Mix of clean and growl segments (fallback path)."""
        n_clean = 156  # 2.5s / 0.016
        n_growl = 156
        times = [i * 0.016 for i in range(n_clean + n_growl)]
        rng = np.random.RandomState(42)
        growl_semitones = rng.normal(0, 8, n_growl)
        frequencies = (
            [440.0] * n_clean
            + [440.0 * (2.0 ** (s / 12.0)) for s in growl_semitones]
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

        # With default thresholds -> detected
        result = detect_growl_segments(segments, pd, use_spectral=False)
        self.assertEqual(result[0].note_type, "F")

        # Reset and use very strict threshold -> not detected
        segments[0].note_type = ":"
        result = detect_growl_segments(
            segments, pd,
            confidence_threshold=0.1,
            use_spectral=False,
        )
        self.assertEqual(result[0].note_type, ":")

    def test_too_few_voiced_frames_not_growl(self):
        """Silence/breath (very low confidence, no voiced frames) -> not marked."""
        pd = _make_pitched_data(confidence=0.05, base_freq=20.0)
        segments = [_make_segment(0.5, 1.5)]
        result = detect_growl_segments(segments, pd, use_spectral=False)
        self.assertEqual(result[0].note_type, ":")


class TestSegmentAnalysis(unittest.TestCase):
    """Test the internal _analyze_segment_pitch function."""

    def test_analysis_values_clean(self):
        """Clean singing should have high confidence, low stdev."""
        pd = _make_pitched_data(confidence=0.85, freq_noise_semitones=0.3)
        times = np.array(pd.times)
        freqs = np.array(pd.frequencies)
        confs = np.array(pd.confidence)
        seg = _make_segment(0.5, 1.5)

        result = _analyze_segment_pitch(
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

        result = _analyze_segment_pitch(
            seg, times, freqs, confs, None, None,
            0.35, 4.0, 0.25, 0.15,
        )
        self.assertTrue(result.is_growl)
        self.assertLess(result.median_conf, 0.35)
        self.assertGreater(result.pitch_stdev, 4.0)


class TestHpssDetection(unittest.TestCase):
    """Test HPSS-based harmonicity detection."""

    def test_clean_singing_high_harmonicity(self):
        """Clean singing (harmonicity=0.8) should NOT be marked as growl."""
        hpss = _make_hpss_data(harmonicity=0.8)
        seg = _make_segment(1.0, 3.0)
        result = _detect_by_harmonicity(seg, hpss, 0.40, 0.01)
        self.assertFalse(result)

    def test_growl_low_harmonicity(self):
        """Growl (harmonicity=0.1) should be marked as growl."""
        hpss = _make_hpss_data(harmonicity=0.1)
        seg = _make_segment(1.0, 3.0)
        result = _detect_by_harmonicity(seg, hpss, 0.40, 0.01)
        self.assertTrue(result)

    def test_borderline_harmonicity(self):
        """Harmonicity just below threshold -> growl."""
        hpss = _make_hpss_data(harmonicity=0.35)
        seg = _make_segment(1.0, 3.0)
        result = _detect_by_harmonicity(seg, hpss, 0.40, 0.01)
        self.assertTrue(result)

    def test_borderline_above_threshold(self):
        """Harmonicity just above threshold -> not growl."""
        hpss = _make_hpss_data(harmonicity=0.45)
        seg = _make_segment(1.0, 3.0)
        result = _detect_by_harmonicity(seg, hpss, 0.40, 0.01)
        self.assertFalse(result)

    def test_short_segment_context_expansion(self):
        """Short segment (< 1.0s) should expand context window."""
        hpss = _make_hpss_data(harmonicity=0.1, n_frames=200, duration=10.0)
        seg = _make_segment(5.0, 5.3)  # 0.3s segment
        result = _detect_by_harmonicity(seg, hpss, 0.40, 0.01)
        self.assertTrue(result)

    def test_silence_not_marked_as_growl(self):
        """Near-zero energy should not be marked as growl."""
        hpss = _make_hpss_data(harmonicity=0.1)
        # Scale everything to near-zero
        hpss.harmonic_mag = hpss.harmonic_mag * 1e-6
        hpss.percussive_mag = hpss.percussive_mag * 1e-6
        seg = _make_segment(1.0, 3.0)
        result = _detect_by_harmonicity(seg, hpss, 0.40, 0.01)
        self.assertFalse(result)

    def test_custom_thresholds(self):
        """Custom thresholds affect HPSS detection."""
        hpss = _make_hpss_data(harmonicity=0.35)
        seg = _make_segment(1.0, 3.0)
        # Default threshold 0.40 -> growl (0.35 < 0.40)
        self.assertTrue(_detect_by_harmonicity(seg, hpss, 0.40, 0.01))
        # Strict threshold 0.20 -> not growl (0.35 > 0.20)
        self.assertFalse(_detect_by_harmonicity(seg, hpss, 0.20, 0.01))


if __name__ == "__main__":
    unittest.main()
