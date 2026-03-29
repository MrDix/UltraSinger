"""FCPE (Fast Context-based Pitch Estimation) pitch detection backend."""

import io
import sys
import threading
import numpy as np

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.Pitcher.pitched_data import PitchedData

_fcpe_model = None
_fcpe_lock = threading.Lock()


def _get_model():
    """Lazy initialize FCPE model on GPU if available."""
    global _fcpe_model
    if _fcpe_model is not None:
        return _fcpe_model
    with _fcpe_lock:
        if _fcpe_model is not None:
            return _fcpe_model
        import torch
        from torchfcpe import spawn_bundled_infer_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # torchfcpe prints noisy INFO/WARN to stdout (device info,
        # harmonic_emb defaults) — suppress to avoid confusing users.
        _prev_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            _fcpe_model = (spawn_bundled_infer_model(device=device), device)
        finally:
            sys.stdout = _prev_stdout
    return _fcpe_model


def _compute_frame_confidence(
    audio: np.ndarray,
    frequencies: list[float], hop_size: int,
) -> list[float]:
    """Derive per-frame confidence from local RMS energy and pitch stability.

    FCPE does not output native per-frame confidence. Instead we combine:
    - RMS energy: frames with louder audio are more likely correctly pitched
    - Pitch stability: frames whose pitch agrees with neighbors are more reliable

    The two signals are combined via geometric mean and scaled to [0.35, 0.95]
    for voiced frames. Unvoiced frames (frequency == 0) get confidence 0.0.
    """
    n_frames = len(frequencies)
    if n_frames == 0:
        return []

    # --- Per-frame RMS energy ---
    frame_len = hop_size
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * frame_len
        end = min(start + frame_len, len(audio))
        if start < len(audio):
            chunk = audio[start:end]
            energy[i] = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0.0

    # Normalize energy to 0-1
    max_energy = energy.max()
    if max_energy > 0:
        energy_norm = energy / max_energy
    else:
        energy_norm = energy

    # --- Pitch stability (low variance in neighborhood = high confidence) ---
    freqs_arr = np.array(frequencies, dtype=np.float64)
    stability = np.zeros(n_frames)
    neighborhood = 3  # frames on each side
    for i in range(n_frames):
        lo = max(0, i - neighborhood)
        hi = min(n_frames, i + neighborhood + 1)
        local = freqs_arr[lo:hi]
        voiced_local = local[local > 0]
        if len(voiced_local) >= 2 and freqs_arr[i] > 0:
            # Coefficient of variation (lower = more stable)
            cv = np.std(voiced_local) / np.mean(voiced_local)
            stability[i] = max(0.0, 1.0 - cv * 5.0)  # scale: cv=0.2 -> 0.0
        elif freqs_arr[i] > 0:
            stability[i] = 0.5  # isolated voiced frame, moderate confidence

    # --- Combine: geometric mean of energy and stability ---
    confidence = []
    for i in range(n_frames):
        if frequencies[i] <= 0:
            confidence.append(0.0)
        else:
            combined = np.sqrt(energy_norm[i] * stability[i])
            # Scale to [0.35, 0.95] for voiced frames. Two downstream gates:
            # - _find_voiced_regions uses > 0.3 → 0.35 floor passes this
            # - get_frequencies_with_high_confidence uses > 0.4 → only
            #   frames with sufficient energy+stability contribute to notes
            scaled = 0.35 + combined * 0.60
            confidence.append(float(min(0.95, scaled)))

    return confidence


def get_pitch_with_fcpe(
    audio: np.ndarray, sample_rate: int
) -> PitchedData:
    """Pitch detection using FCPE.

    FCPE processes audio at 16kHz with 160-sample hop size internally.
    Returns frames at approximately 10 ms intervals.
    GPU-accelerated when CUDA is available, falls back to CPU.
    """
    import torch
    import librosa

    print(
        f"{ULTRASINGER_HEAD} Pitching with {blue_highlighted('FCPE')} (torchfcpe)"
    )

    model, device = _get_model()

    # Resample to 16kHz if needed
    target_sr = 16000
    if sample_rate != target_sr:
        audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
    else:
        audio_16k = audio

    # FCPE expects [batch, samples] tensor
    audio_tensor = torch.from_numpy(audio_16k).float().unsqueeze(0).to(device)

    # Run inference
    hop_size = 160  # 10ms at 16kHz
    with torch.inference_mode():
        f0 = model.infer(audio_tensor, sr=target_sr, decoder_mode="local_argmax",
                         threshold=0.006)

    # f0 shape: [batch, frames, 1]
    f0_np = np.atleast_1d(f0.squeeze().cpu().numpy())
    n_frames = len(f0_np)

    # Generate timestamps
    times = [float(i * hop_size / target_sr) for i in range(n_frames)]
    frequencies = [max(float(f), 0.0) for f in f0_np]

    # Derive confidence from energy and pitch stability
    confidence = _compute_frame_confidence(
        audio_16k, frequencies, hop_size
    )

    return PitchedData(times, frequencies, confidence)
