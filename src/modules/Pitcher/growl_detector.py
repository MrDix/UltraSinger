"""Detect unpitchable vocal passages (growl, scream, rap, spoken word).

Marks segments as freestyle (note_type = "F") so they are displayed
but not scored in UltraStar karaoke.

Tier 1: SwiftF0 confidence + pitch standard deviation (zero extra deps)
Tier 2: Spectral flatness via librosa (already a transitive dependency)
"""

import logging
import math
from typing import Optional

import librosa
import numpy as np

from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher.pitched_data import PitchedData
from modules.console_colors import ULTRASINGER_HEAD

logger = logging.getLogger(__name__)


def detect_growl_segments(
    midi_segments: list[MidiSegment],
    pitched_data: PitchedData,
    vocal_audio_path: Optional[str] = None,
    confidence_threshold: float = 0.35,
    pitch_stdev_threshold: float = 4.0,
    spectral_flatness_threshold: float = 0.25,
    min_voiced_ratio: float = 0.15,
    use_spectral: bool = True,
) -> list[MidiSegment]:
    """Analyze each MidiSegment and mark unpitchable ones as freestyle.

    A segment is considered unpitchable when SwiftF0 reports low confidence
    AND the pitch values are erratic (high standard deviation in semitones).
    Optionally, spectral flatness (Tier 2) provides a noise-like signal
    indicator as additional evidence.

    Args:
        midi_segments: List of MidiSegments to analyze.
        pitched_data: SwiftF0 output with times, frequencies, confidence.
        vocal_audio_path: Path to separated vocals audio (for Tier 2).
        confidence_threshold: Median confidence below this → suspect.
        pitch_stdev_threshold: Pitch stdev (semitones) above this → suspect.
        spectral_flatness_threshold: Spectral flatness above this → noisy.
        min_voiced_ratio: Minimum fraction of voiced frames required.
        use_spectral: Enable Tier 2 spectral flatness analysis.

    Returns:
        The same list with unpitchable segments marked as note_type = "F".
    """
    if not midi_segments or not pitched_data or not pitched_data.times:
        return midi_segments

    times = np.array(pitched_data.times)
    freqs = np.array(pitched_data.frequencies)
    confs = np.array(pitched_data.confidence)

    # Pre-compute spectral flatness if Tier 2 is enabled and audio available
    sf_times = None
    sf_values = None
    if use_spectral and vocal_audio_path:
        sf_times, sf_values = _compute_spectral_flatness(vocal_audio_path)

    growl_count = 0
    for seg in midi_segments:
        # Skip segments already marked as freestyle (e.g. from LRCLIB)
        if seg.note_type == "F":
            continue

        result = _analyze_segment(
            seg, times, freqs, confs,
            sf_times, sf_values,
            confidence_threshold,
            pitch_stdev_threshold,
            spectral_flatness_threshold,
            min_voiced_ratio,
        )
        if result.is_growl:
            seg.note_type = "F"
            growl_count += 1

    if growl_count > 0:
        total = len([s for s in midi_segments if s.note_type != "F" or True])
        print(
            f"{ULTRASINGER_HEAD} Growl detection: marked {growl_count}/{len(midi_segments)} "
            f"segments as freestyle (unpitchable)"
        )

    return midi_segments


class _SegmentAnalysis:
    """Result of analyzing a single segment."""
    __slots__ = ("is_growl", "median_conf", "pitch_stdev", "spectral_flat", "voiced_ratio")

    def __init__(self):
        self.is_growl = False
        self.median_conf = 1.0
        self.pitch_stdev = 0.0
        self.spectral_flat = 0.0
        self.voiced_ratio = 1.0


def _analyze_segment(
    seg: MidiSegment,
    times: np.ndarray,
    freqs: np.ndarray,
    confs: np.ndarray,
    sf_times: Optional[np.ndarray],
    sf_values: Optional[np.ndarray],
    confidence_threshold: float,
    pitch_stdev_threshold: float,
    spectral_flatness_threshold: float,
    min_voiced_ratio: float,
) -> _SegmentAnalysis:
    """Analyze a single segment for growl/scream characteristics."""
    result = _SegmentAnalysis()

    # Extract frames within segment's time range
    mask = (times >= seg.start) & (times <= seg.end)
    seg_confs = confs[mask]
    seg_freqs = freqs[mask]

    if len(seg_confs) == 0:
        return result

    # Tier 1a: Median confidence
    result.median_conf = float(np.median(seg_confs))

    # Voiced frames (confidence > 0.1 AND frequency > 50 Hz)
    voiced_mask = (seg_confs > 0.1) & (seg_freqs > 50.0)
    result.voiced_ratio = float(np.sum(voiced_mask)) / len(seg_confs)

    # Too few voiced frames → likely silence/breath, not growl
    if result.voiced_ratio < min_voiced_ratio:
        return result

    # Tier 1b: Pitch standard deviation in semitones
    voiced_freqs = seg_freqs[voiced_mask]
    if len(voiced_freqs) >= 2:
        # Convert frequencies to MIDI note numbers for semitone-based stdev
        midi_notes = 12.0 * np.log2(voiced_freqs / 440.0) + 69.0
        result.pitch_stdev = float(np.std(midi_notes))
    else:
        result.pitch_stdev = 0.0

    # Tier 1 decision: low confidence AND high pitch variance
    tier1_growl = (
        result.median_conf < confidence_threshold
        and result.pitch_stdev > pitch_stdev_threshold
    )

    # Tier 2: Spectral flatness (if available)
    tier2_growl = False
    if sf_times is not None and sf_values is not None:
        sf_mask = (sf_times >= seg.start) & (sf_times <= seg.end)
        sf_seg = sf_values[sf_mask]
        if len(sf_seg) > 0:
            result.spectral_flat = float(np.median(sf_seg))
            tier2_growl = result.spectral_flat > spectral_flatness_threshold

    # Combined decision:
    # - Tier 1 alone (low confidence + erratic pitch) is sufficient
    # - Tier 2 alone is NOT sufficient (spectral flatness can fire on consonants)
    # - Tier 2 relaxes the confidence threshold slightly when both indicate noise
    if tier1_growl:
        result.is_growl = True
    elif tier2_growl and result.median_conf < (confidence_threshold * 1.3):
        # Spectral evidence + slightly below-threshold confidence
        result.is_growl = True

    return result


def _compute_spectral_flatness(
    audio_path: str,
    hop_length: int = 512,
    sr: int = 22050,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-frame spectral flatness for the vocal audio.

    Returns:
        Tuple of (frame_times, flatness_values).
    """
    try:
        y, sr_actual = librosa.load(audio_path, sr=sr, mono=True)
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
        frame_times = librosa.frames_to_time(
            np.arange(len(flatness)), sr=sr_actual, hop_length=hop_length
        )
        return frame_times, flatness
    except Exception as e:
        print(f"{ULTRASINGER_HEAD} Spectral flatness computation failed: {e}")
        return None, None
