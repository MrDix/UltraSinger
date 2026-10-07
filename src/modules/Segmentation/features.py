"""Frame features of a separated vocal for the segmentation model.

Training (tools/train_segmentation.py) and inference share these functions, so a
model always sees exactly the features it was trained on. One frame is 256
samples at 16 kHz (16 ms), the hop size of the SwiftF0 pitch detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

SAMPLE_RATE = 16000
HOP = 256
N_FFT = 1024
N_MELS = 80
FRAME_S = HOP / SAMPLE_RATE
VOICED_CONFIDENCE = 0.7
N_EXTRA = 6


@dataclass
class VocalAnalysis:
    """Raw per-frame analysis of a vocal track (what gets cached for training)."""
    f0_t: np.ndarray       # seconds
    f0_hz: np.ndarray
    f0_conf: np.ndarray
    logmel: np.ndarray     # (frames, N_MELS) float16
    rms: np.ndarray        # (frames,)
    duration: float


def load_vocal(path: str) -> np.ndarray:
    """Mono 16 kHz float32 signal of a vocal file."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y.astype(np.float32)


def analyse_vocal(y: np.ndarray) -> VocalAnalysis:
    """SwiftF0 pitch, log-mel spectrogram and RMS on the 16 ms frame grid."""
    from swift_f0 import SwiftF0

    detector = SwiftF0(fmin=46.875, fmax=2093.75, confidence_threshold=VOICED_CONFIDENCE)
    r = detector.detect_from_array(y, SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP,
                                         n_mels=N_MELS, fmin=50, fmax=8000)
    logmel = librosa.power_to_db(mel, ref=1.0, top_db=None).T.astype(np.float16)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0].astype(np.float32)
    return VocalAnalysis(
        f0_t=np.asarray(r.timestamps, dtype=np.float32),
        f0_hz=np.asarray(r.pitch_hz, dtype=np.float32),
        f0_conf=np.asarray(r.confidence, dtype=np.float32),
        logmel=logmel,
        rms=rms,
        duration=len(y) / SAMPLE_RATE,
    )


def frame_pitch(a: VocalAnalysis, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame MIDI pitch (NaN when unvoiced) and voiced mask, length ``n_frames``."""
    f = np.zeros(n_frames, np.float32)
    c = np.zeros(n_frames, np.float32)
    m = min(n_frames, len(a.f0_hz))
    f[:m] = a.f0_hz[:m]
    c[:m] = a.f0_conf[:m]
    voiced = (c >= VOICED_CONFIDENCE) & (f > 40)
    midi = np.full(n_frames, np.nan, np.float32)
    midi[f > 40] = 69 + 12 * np.log2(f[f > 40] / 440.0)
    return midi, voiced


def model_input(a: VocalAnalysis) -> np.ndarray:
    """Network input of shape (frames, N_MELS + N_EXTRA), float32."""
    mel = a.logmel.astype(np.float32)
    n = mel.shape[0]
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    midi, voiced = frame_pitch(a, n)
    c = np.zeros(n, np.float32)
    m = min(n, len(a.f0_conf))
    c[:m] = a.f0_conf[:m]
    center = float(np.median(midi[voiced])) if voiced.any() else 60.0
    rel = np.where(np.isfinite(midi), (midi - center) / 12.0, 0).astype(np.float32)
    raw = np.where(np.isfinite(midi), midi, 0).astype(np.float32)
    d1 = np.zeros(n, np.float32)
    d1[1:] = np.clip(np.diff(raw), -12, 12) / 12.0
    d1[~voiced] = 0
    rms = np.zeros(n, np.float32)
    rms[:min(n, len(a.rms))] = a.rms[:n]
    rmsdb = (20 * np.log10(rms + 1e-5) + 40) / 40.0
    extra = np.stack([rel, d1, c, voiced.astype(np.float32), rmsdb, np.ones(n, np.float32)], 1)
    return np.concatenate([mel, extra], 1).astype(np.float32)
