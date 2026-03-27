"""Detect unpitchable vocal passages (growl, scream, rap, spoken word).

Marks segments as freestyle (note_type = "F") so they are displayed
but not scored in UltraStar karaoke.

Primary: HPSS harmonicity analysis — genre/gender-independent, measures
         harmonic vs. percussive energy via librosa.decompose.hpss().
Fallback: SwiftF0 confidence + pitch stability (when no vocal audio).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np

from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher.pitched_data import PitchedData
from modules.console_colors import ULTRASINGER_HEAD

logger = logging.getLogger(__name__)

# Short segments get expanded to this duration (seconds) for reliable HPSS.
_MIN_CONTEXT_SECONDS = 1.0


@dataclass
class _HpssData:
    """Pre-computed HPSS decomposition for the entire vocal track."""
    sr: int
    hop_length: int
    harmonic_mag: np.ndarray   # |H| from HPSS
    percussive_mag: np.ndarray # |P| from HPSS
    frame_times: np.ndarray    # time of each STFT frame
    duration: float            # total audio duration in seconds


class _SegmentAnalysis:
    """Result of analyzing a single segment."""
    __slots__ = ("harmonicity_ratio", "is_growl", "median_conf", "pitch_stdev",
                 "spectral_flat", "voiced_ratio")

    def __init__(self):
        self.is_growl = False
        self.median_conf = 1.0
        self.pitch_stdev = 0.0
        self.spectral_flat = 0.0
        self.voiced_ratio = 1.0
        self.harmonicity_ratio = 1.0


def detect_growl_segments(
    midi_segments: list[MidiSegment],
    pitched_data: PitchedData,
    vocal_audio_path: Optional[str] = None,
    confidence_threshold: float = 0.35,
    pitch_stdev_threshold: float = 4.0,
    spectral_flatness_threshold: float = 0.25,
    min_voiced_ratio: float = 0.15,
    use_spectral: bool = True,
    harmonicity_threshold: float = 0.40,
    energy_threshold: float = 0.01,
) -> list[MidiSegment]:
    """Analyze each MidiSegment and mark unpitchable ones as freestyle.

    Primary detection uses HPSS (Harmonic-Percussive Source Separation):
    clean singing has a high harmonic-to-total energy ratio (0.7+),
    while growls/screams have a low ratio (< 0.40).

    When no vocal audio is available, falls back to SwiftF0 pitch
    confidence + pitch stability analysis.

    Args:
        midi_segments: List of MidiSegments to analyze.
        pitched_data: SwiftF0 output with times, frequencies, confidence.
        vocal_audio_path: Path to separated vocals audio (for HPSS).
        confidence_threshold: Median confidence below this -> suspect (fallback).
        pitch_stdev_threshold: Pitch stdev (semitones) above this -> suspect (fallback).
        spectral_flatness_threshold: Spectral flatness above this -> noisy (fallback).
        min_voiced_ratio: Minimum fraction of voiced frames required (fallback).
        use_spectral: Enable spectral flatness analysis (fallback).
        harmonicity_threshold: HPSS harmonic ratio below this -> unpitchable.
        energy_threshold: RMS energy below this -> silence, not growl.

    Returns:
        The same list with unpitchable segments marked as note_type = "F".
    """
    if not midi_segments or not pitched_data or not pitched_data.times:
        return midi_segments

    # Try HPSS primary detection
    hpss = None
    if vocal_audio_path:
        hpss = _precompute_hpss(vocal_audio_path)

    times = np.array(pitched_data.times)
    freqs = np.array(pitched_data.frequencies)
    confs = np.array(pitched_data.confidence)

    # Pre-compute spectral flatness for fallback Tier 2
    sf_times = None
    sf_values = None
    if hpss is None and use_spectral and vocal_audio_path:
        sf_times, sf_values = _compute_spectral_flatness(vocal_audio_path)

    growl_count = 0
    for seg in midi_segments:
        # Skip segments already marked as freestyle (e.g. from LRCLIB)
        if seg.note_type == "F":
            continue

        if hpss is not None:
            # Primary: HPSS harmonicity
            is_growl = _detect_by_harmonicity(
                seg, hpss, harmonicity_threshold, energy_threshold
            )
            if is_growl:
                seg.note_type = "F"
                growl_count += 1
        else:
            # Fallback: pitch confidence analysis
            result = _analyze_segment_pitch(
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
        print(
            f"{ULTRASINGER_HEAD} Growl detection: marked {growl_count}/{len(midi_segments)} "
            f"segments as freestyle (unpitchable)"
        )

    return midi_segments


def _precompute_hpss(
    audio_path: str,
    sr: int = 22050,
    hop_length: int = 512,
) -> Optional[_HpssData]:
    """Load vocal audio and compute HPSS decomposition once."""
    try:
        y, sr_actual = librosa.load(audio_path, sr=sr, mono=True)
        S = librosa.stft(y, hop_length=hop_length)
        S_mag = np.abs(S)
        H, P = librosa.decompose.hpss(S_mag)
        frame_times = librosa.frames_to_time(
            np.arange(H.shape[1]), sr=sr_actual, hop_length=hop_length
        )
        duration = len(y) / sr_actual
        return _HpssData(
            sr=sr_actual,
            hop_length=hop_length,
            harmonic_mag=H,
            percussive_mag=P,
            frame_times=frame_times,
            duration=duration,
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as e:
        logger.warning("HPSS pre-computation failed: %s", e)
        return None


def _detect_by_harmonicity(
    seg: MidiSegment,
    hpss: _HpssData,
    harmonicity_threshold: float,
    energy_threshold: float,
) -> bool:
    """Check if a segment is unpitchable using HPSS harmonicity ratio.

    For short segments (< 1.0s), the analysis window is expanded to
    _MIN_CONTEXT_SECONDS centered on the segment midpoint, so that
    HPSS has enough spectral context for reliable decomposition.

    Returns True if the segment should be marked as freestyle (growl).
    """
    seg_dur = seg.end - seg.start

    # Expand short segments for reliable HPSS analysis
    if seg_dur < _MIN_CONTEXT_SECONDS:
        mid = (seg.start + seg.end) / 2.0
        half = _MIN_CONTEXT_SECONDS / 2.0
        win_start = max(0.0, mid - half)
        win_end = mid + half
    else:
        win_start = seg.start
        win_end = seg.end

    # Select HPSS frames within the window
    mask = (hpss.frame_times >= win_start) & (hpss.frame_times <= win_end)
    if not np.any(mask):
        return False

    H_seg = hpss.harmonic_mag[:, mask]
    P_seg = hpss.percussive_mag[:, mask]

    harm_energy = np.sum(H_seg ** 2)
    perc_energy = np.sum(P_seg ** 2)
    total_energy = harm_energy + perc_energy

    # Below energy threshold → silence/breath, not growl
    if total_energy < energy_threshold:
        return False

    harmonicity_ratio = harm_energy / total_energy
    return harmonicity_ratio < harmonicity_threshold


def _analyze_segment_pitch(
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
    """Analyze a single segment using pitch confidence (fallback method)."""
    result = _SegmentAnalysis()

    # Extract frames within segment's time range
    mask = (times >= seg.start) & (times <= seg.end)
    seg_confs = confs[mask]
    seg_freqs = freqs[mask]

    if len(seg_confs) == 0:
        return result

    # Median confidence
    result.median_conf = float(np.median(seg_confs))

    # Voiced frames (confidence > 0.1 AND frequency > 50 Hz)
    voiced_mask = (seg_confs > 0.1) & (seg_freqs > 50.0)
    result.voiced_ratio = float(np.sum(voiced_mask)) / len(seg_confs)

    # Too few voiced frames → likely silence/breath, not growl
    if result.voiced_ratio < min_voiced_ratio:
        return result

    # Pitch standard deviation in semitones
    voiced_freqs = seg_freqs[voiced_mask]
    if len(voiced_freqs) >= 2:
        midi_notes = 12.0 * np.log2(voiced_freqs / 440.0) + 69.0
        result.pitch_stdev = float(np.std(midi_notes))
    else:
        result.pitch_stdev = 0.0

    # Decision: low confidence AND high pitch variance
    tier1_growl = (
        result.median_conf < confidence_threshold
        and result.pitch_stdev > pitch_stdev_threshold
    )

    # Spectral flatness (if available)
    tier2_growl = False
    if sf_times is not None and sf_values is not None:
        sf_mask = (sf_times >= seg.start) & (sf_times <= seg.end)
        sf_seg = sf_values[sf_mask]
        if len(sf_seg) > 0:
            result.spectral_flat = float(np.median(sf_seg))
            tier2_growl = result.spectral_flat > spectral_flatness_threshold

    # Combined decision
    if tier1_growl:
        result.is_growl = True
    elif tier2_growl and result.median_conf < (confidence_threshold * 1.3):
        result.is_growl = True

    return result


def _compute_spectral_flatness(
    audio_path: str,
    hop_length: int = 512,
    sr: int = 22050,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute per-frame spectral flatness for the vocal audio."""
    try:
        y, sr_actual = librosa.load(audio_path, sr=sr, mono=True)
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
        frame_times = librosa.frames_to_time(
            np.arange(len(flatness)), sr=sr_actual, hop_length=hop_length
        )
        return frame_times, flatness
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as e:
        logger.warning("Spectral flatness computation failed: %s", e)
        return None, None
