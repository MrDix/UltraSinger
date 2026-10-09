"""Turn frame predictions into notes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modules.Segmentation.features import FRAME_S, VocalAnalysis, frame_pitch
from modules.Segmentation.model import CLASS_FREESTYLE, CLASS_PITCHED

# Sung notes sag below their written pitch (scoops into the note, a falling end,
# vibrato), so the upper middle of a note's confident pitch frames matches charted
# notes better than their median does.
NOTE_PITCH_PERCENTILE = 60


@dataclass
class PredictedNote:
    start: float   # seconds
    end: float     # seconds
    midi: int
    freestyle: bool = False
    voiced: float = 0.0  # share of the note's frames with confident pitch on the track its pitch came from
    check_midi: int | None = None  # pitch on a second track (None: too few confident frames there)


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


def _track_pitch(midi: np.ndarray, voiced: np.ndarray, a: int, b: int, percentile: float = 50) -> int | None:
    """Pitch of frames [a, b) of one track, or ``None`` with fewer than 2 confident frames."""
    sel = voiced[a:b]
    if sel.sum() < 2:
        return None
    return int(np.round(np.percentile(midi[a:b][sel], percentile)))


def decode_notes(probs: np.ndarray, onset: np.ndarray, analysis: VocalAnalysis,
                 onset_thr: float = 0.4, act_thr: float = 0.4,
                 min_note_frames: int = 6, min_gap_frames: int = 0,
                 pitch_analysis: VocalAnalysis | None = None,
                 check_analysis: VocalAnalysis | None = None) -> list[PredictedNote]:
    """Split active regions at onset peaks; pitch = upper middle of the confident pitch frames.

    Pitched regions are split at onset peaks into notes; freestyle regions become
    single freestyle notes. Notes shorter than ``min_note_frames`` are dropped and
    ``min_gap_frames`` are left between notes split inside one region.

    ``pitch_analysis`` (e.g. of a lead-vocal stem) is the preferred source for
    note pitches; where it has too few confident frames inside a note, the pitch
    of ``analysis`` is used. Each note also records how much of it the track its
    pitch came from covers (``voiced``) and the pitch a second track measures
    there (``check_midi``): ``analysis`` for pitches from ``pitch_analysis``, or,
    without a ``pitch_analysis``, ``check_analysis`` (e.g. a lead stem not reliable
    enough to take the pitches from).
    """
    n = len(onset)
    midi, voiced = frame_pitch(analysis, n)
    if pitch_analysis is not None:
        p_midi, p_voiced = frame_pitch(pitch_analysis, n)
        c_midi, c_voiced = midi, voiced
    else:
        p_midi, p_voiced = midi, voiced
        c_midi, c_voiced = frame_pitch(check_analysis, n) if check_analysis is not None else (None, None)
    # Mutually exclusive: a frame is pitched only if that class is at least as
    # likely as freestyle, so the same interval never yields both kinds of note.
    pitched = (probs[:, CLASS_PITCHED] >= act_thr) & (probs[:, CLASS_PITCHED] >= probs[:, CLASS_FREESTYLE])
    free = (probs[:, CLASS_FREESTYLE] > probs[:, CLASS_PITCHED]) & (probs[:, CLASS_FREESTYLE] >= act_thr)
    peaks = _onset_peaks(onset, onset_thr)
    notes: list[PredictedNote] = []

    def emit(a: int, b: int, is_free: bool) -> None:
        if b - a < min_note_frames:
            return
        if is_free:
            # pitch is not scored for freestyle notes; keep it near the singer
            seg_midi = midi[a:b]
            pitch = int(np.round(np.nanmedian(seg_midi))) if np.isfinite(seg_midi).any() else 60
            notes.append(PredictedNote(a * FRAME_S, b * FRAME_S, pitch, True))
            return
        share, check = 0.0, None
        pitch = _track_pitch(p_midi, p_voiced, a, b, NOTE_PITCH_PERCENTILE)
        if pitch is not None:
            share = float(p_voiced[a:b].mean())
            if c_midi is not None:
                check = _track_pitch(c_midi, c_voiced, a, b)
        else:
            # the preferred track is silent here: take the vocal's pitch, no second opinion
            pitch = _track_pitch(midi, voiced, a, b, NOTE_PITCH_PERCENTILE)
            if pitch is not None:
                share = float(voiced[a:b].mean())
            elif notes:
                pitch = notes[-1].midi  # too few confident frames: continue the melody
            else:
                return
        notes.append(PredictedNote(a * FRAME_S, b * FRAME_S, pitch, False, share, check))

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
