"""FCPE (Fast Context-based Pitch Estimation) pitch detection backend."""

import numpy as np

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.Pitcher.pitched_data import PitchedData

_fcpe_model = None


def _get_model():
    """Lazy initialize FCPE model on GPU if available."""
    global _fcpe_model
    if _fcpe_model is None:
        import torch
        from torchfcpe import spawn_bundled_infer_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _fcpe_model = (spawn_bundled_infer_model(device=device), device)
    return _fcpe_model


def get_pitch_with_fcpe(
    audio: np.ndarray, sample_rate: int
) -> PitchedData:
    """Pitch detection using FCPE.

    FCPE processes audio at 16kHz with 160-sample hop size internally.
    Returns frames at approximately 10 ms intervals.
    GPU-accelerated when CUDA is available.
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
    f0 = model.infer(audio_tensor, sr=target_sr, decoder_mode="local_argmax",
                     threshold=0.006)

    # f0 shape: [batch, frames, 1]
    f0_np = f0.squeeze().cpu().numpy()
    n_frames = len(f0_np)

    # Generate timestamps
    times = [float(i * hop_size / target_sr) for i in range(n_frames)]
    frequencies = [max(float(f), 0.0) for f in f0_np]

    # FCPE doesn't output per-frame confidence directly.
    # Use voicing as binary confidence: voiced (f0 > 0) = 0.9, unvoiced = 0.0
    confidence = [0.9 if f > 0 else 0.0 for f in frequencies]

    return PitchedData(times, frequencies, confidence)
