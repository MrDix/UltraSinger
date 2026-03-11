"""BPM detection module"""

import numpy as np
import librosa
import soundfile as sf

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted


def _pick_best_tempo(tempos: np.ndarray) -> float:
    """Pick the best tempo from multiple candidates.

    Applies half/double-tempo correction to prefer a tempo in the
    typical song range (60-200 BPM).  If the primary candidate is
    outside this range but a half or double variant is inside, the
    variant is preferred.  When multiple candidates fall inside the
    range the one closest to 120 BPM (a common pop/rock tempo) wins.

    Args:
        tempos: Array of tempo candidates from librosa (sorted by
            strength, strongest first).

    Returns:
        The best tempo estimate in BPM.
    """
    if len(tempos) == 0:
        return 120.0  # Fallback

    primary = float(tempos[0])

    # Build candidate list: primary + half/double variants
    candidates = {primary}
    for t in tempos[:3]:  # Consider top-3 from librosa
        t = float(t)
        candidates.add(t)
        candidates.add(t / 2)
        candidates.add(t * 2)

    # Prefer candidates in the typical song range (60-200 BPM)
    in_range = [c for c in candidates if 60 <= c <= 200]

    if in_range:
        # Pick the one closest to 120 BPM (common pop/rock tempo)
        best = min(in_range, key=lambda x: abs(x - 120))
    else:
        # Nothing in range — use the primary as-is
        best = primary

    return best


def get_bpm_from_data(data, sampling_rate):
    """Get real BPM from audio data.

    Uses librosa's tempo estimation with multiple candidates and
    applies half/double-tempo correction to avoid common detection
    errors where the tempo is reported as 2x or 0.5x the actual value.
    """
    onset_env = librosa.onset.onset_strength(y=data, sr=sampling_rate)

    # Get multiple tempo candidates instead of just the top one
    tempos = librosa.feature.tempo(
        onset_envelope=onset_env, sr=sampling_rate, aggregate=None
    )

    # Pick the best candidate with half/double correction
    best_bpm = _pick_best_tempo(tempos)

    print(
        f"{ULTRASINGER_HEAD} BPM is {blue_highlighted(str(round(best_bpm, 2)))}"
    )
    return best_bpm


def get_bpm_from_file(wav_file: str) -> float:
    """Get real BPM from audio file."""
    data, sampling_rate = sf.read(wav_file, dtype='float32')
    # Transpose if stereo to match librosa's expected format
    if len(data.shape) > 1:
        data = data.T
    # Convert to mono if stereo
    if data.ndim > 1:
        data = librosa.to_mono(data)
    return get_bpm_from_data(data, sampling_rate)
