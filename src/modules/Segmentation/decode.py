"""Turn frame predictions into notes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modules.Segmentation.features import FRAME_S, VocalAnalysis, frame_pitch
from modules.Segmentation.model import CLASS_FREESTYLE, CLASS_PITCHED


@dataclass
class PredictedNote:
    start: float   # seconds
    end: float     # seconds
    midi: int
    freestyle: bool = False


def _onset_peaks(onset: np.ndarray, threshold: float) -> np.ndarray:
    peaks = np.zeros(len(onset), bool)
    if len(onset) > 2:
        mid = onset[1:-1]
        peaks[1:-1] = (mid >= threshold) & (mid >= onset[:-2]) & (mid >= onset[2:])
    return peaks


def _regions(mask: np.ndarray):
    """[start, end) frame ranges where mask is True."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def decode_notes(probs: np.ndarray, onset: np.ndarray, analysis: VocalAnalysis,
                 onset_thr: float = 0.4, act_thr: float = 0.4,
                 min_note_frames: int = 6, min_gap_frames: int = 0) -> list[PredictedNote]:
    """Split active regions at onset peaks; pitch = median of confident pitch frames.

    Pitched regions are split at onset peaks into notes; freestyle regions become
    single freestyle notes. Notes shorter than ``min_note_frames`` are dropped and
    ``min_gap_frames`` are left between notes split inside one region.
    """
    n = len(onset)
    midi, voiced = frame_pitch(analysis, n)
    pitched = probs[:, CLASS_PITCHED] >= act_thr
    free = (probs[:, CLASS_FREESTYLE] > probs[:, CLASS_PITCHED]) & (probs[:, CLASS_FREESTYLE] >= act_thr)
    peaks = _onset_peaks(onset, onset_thr)
    notes: list[PredictedNote] = []

    def emit(a: int, b: int, is_free: bool) -> None:
        if b - a < min_note_frames:
            return
        seg_midi = midi[a:b]
        sel = voiced[a:b]
        if is_free:
            # pitch is not scored for freestyle notes; keep it near the singer
            pitch = int(np.round(np.nanmedian(seg_midi))) if np.isfinite(seg_midi).any() else 60
        elif sel.sum() >= 2:
            pitch = int(np.round(np.median(seg_midi[sel])))
        elif notes:
            pitch = notes[-1].midi  # too few confident frames: continue the melody
        else:
            return
        notes.append(PredictedNote(a * FRAME_S, b * FRAME_S, pitch, is_free))

    for a, b in _regions(pitched):
        cuts = [a]
        for k in range(a + min_note_frames, b - min_note_frames + 1):
            if peaks[k] and k - cuts[-1] >= min_note_frames:
                cuts.append(k)
        cuts.append(b)
        for s, e in zip(cuts[:-1], cuts[1:]):
            if e < b and e - s > min_note_frames + min_gap_frames:
                e -= min_gap_frames
            emit(s, e, False)
    for a, b in _regions(free):
        emit(a, b, True)
    notes.sort(key=lambda x: x.start)
    return notes
