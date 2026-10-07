"""Segmentation network and self-describing model files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from modules.Segmentation.features import N_EXTRA, N_MELS

MODEL_FORMAT = 1
DEFAULT_DECODE = {"onset_thr": 0.4, "act_thr": 0.4, "min_note_frames": 6, "min_gap_frames": 0}

# Frame classes predicted by the network
CLASS_NONE, CLASS_PITCHED, CLASS_FREESTYLE = 0, 1, 2


class SegNet(nn.Module):
    """CNN front end over the mel spectrogram + BiGRU over time.

    Outputs per frame: class logits (none / pitched note / freestyle) and an
    onset logit (a note starts here).
    """

    def __init__(self, n_mels: int = N_MELS, n_extra: int = N_EXTRA, hidden: int = 160):
        super().__init__()
        self.n_mels = n_mels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.BatchNorm2d(24), nn.GELU(), nn.MaxPool2d((1, 2)),
            nn.Conv2d(24, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d((1, 2)),
            nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.GELU(), nn.MaxPool2d((1, 4)),
        )
        self.proj = nn.Linear(48 * (n_mels // 16) + n_extra, hidden)
        self.rnn = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.cls = nn.Linear(2 * hidden, 3)
        self.onset = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor):  # x: (B, T, F)
        mel, extra = x[..., :self.n_mels], x[..., self.n_mels:]
        h = self.conv(mel.unsqueeze(1))
        b, c, t, m = h.shape
        h = h.permute(0, 2, 1, 3).reshape(b, t, c * m)
        h = torch.relu(self.proj(torch.cat([h, extra], -1)))
        h, _ = self.rnn(h)
        return self.cls(h), self.onset(h).squeeze(-1)


def _plain(value):
    """numpy scalars -> Python numbers, so the file loads with ``weights_only=True``."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_model(path: str | Path, model: SegNet, decode: dict | None = None, info: dict | None = None) -> None:
    """Write weights + decoding thresholds into one file."""
    torch.save({
        "format": MODEL_FORMAT,
        "state_dict": model.state_dict(),
        "config": {"n_mels": model.n_mels, "hidden": model.proj.out_features},
        "decode": _plain(dict(DEFAULT_DECODE, **(decode or {}))),
        "info": _plain(info or {}),
    }, str(path))


def load_model(path: str | Path, device: str = "cpu") -> tuple[SegNet, dict]:
    """Load a model file; returns the network (eval mode) and its decoding thresholds."""
    # weights_only: a model file can only carry tensors and plain values, never code
    data = torch.load(str(path), map_location=device, weights_only=True)
    if not isinstance(data, dict) or data.get("format") != MODEL_FORMAT:
        raise ValueError(f"{path} is not a segmentation model file (format {MODEL_FORMAT})")
    cfg = data.get("config", {})
    model = SegNet(n_mels=cfg.get("n_mels", N_MELS), hidden=cfg.get("hidden", 160))
    model.load_state_dict(data["state_dict"])
    model.to(device).eval()
    return model, dict(DEFAULT_DECODE, **data.get("decode", {}))


@torch.no_grad()
def predict(model: SegNet, x: np.ndarray, device: str = "cpu", chunk: int = 2000, overlap: int = 200):
    """Frame class probabilities (T, 3) and onset probabilities (T,), chunked for long songs."""
    model.eval()
    n = x.shape[0]
    probs = np.zeros((n, 3), np.float32)
    onset = np.zeros(n, np.float32)
    weight = np.zeros(n, np.float32)
    step = chunk - overlap
    for a in range(0, max(n - overlap, 1), step):
        b = min(a + chunk, n)
        lc, lo = model(torch.from_numpy(x[a:b]).unsqueeze(0).to(device))
        w = np.ones(b - a, np.float32)
        if a > 0:
            w[:overlap // 2] = 0.0
        probs[a:b] += torch.softmax(lc, -1)[0].cpu().numpy() * w[:, None]
        onset[a:b] += torch.sigmoid(lo)[0].cpu().numpy() * w
        weight[a:b] += w
        if b == n:
            break
    weight = np.maximum(weight, 1e-6)
    return probs / weight[:, None], onset / weight
