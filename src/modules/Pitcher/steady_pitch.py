"""Notes that contradict a pitch the singing clearly stays on.

A note made from a word has its pitch from the pitch data of the denoised
vocal, moved to the key, and the refinement may move it once more to the
tone the games' pitch detection hears in it. Each of these steps can put a
note beside the pitch that is sung: the noise reduction smears quick
changes, the key has no place for a sung pitch outside of it, and the
games' pitch detection hears a fifth or another overtone in a short note.

This pass compares each note with the pitch track of the separated vocal
stem. Where the singing is steady over the note - most of its frames are
voiced and they lie close together - and the note is charted on another
pitch, the note takes the sung pitch: the nearest one of that pitch class,
so a note charted in another octave than it is sung in stays there. A note
sung with a wide vibrato, over a run, or without a sure pitch is not steady,
and stays as it is.

Not for the notes of the segmentation model: they have their pitch from the
vocal stem already.
"""

import librosa
import numpy as np

from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher.pitched_data import PitchedData
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted

VOICED_CONFIDENCE = 0.7  # pitch frames below this confidence are not voiced
MIN_FRAMES = 5  # pitch frames in a note from which its singing is judged
MIN_VOICED_SHARE = 0.6  # share of the note's frames that are voiced
MAX_SPREAD = 1.0  # semitones (standard deviation) the voiced frames lie apart at most
MIN_DEVIATION = 0.75  # semitones between the note and the sung pitch from which the note moves
PITCHED_TYPES = (":", "*")  # freestyle and rap notes are charted without a sung pitch


def _folded(semitones):
    """A difference in semitones as the nearest one between pitch classes (-6 ... +6)."""
    return (np.asarray(semitones, dtype=float) + 6.0) % 12.0 - 6.0


def steady_sung_pitch(midi: np.ndarray, voiced: np.ndarray) -> float | None:
    """The pitch (fractional MIDI) the singing stays on over the frames of one
    note; None when it is not steady. *midi* are the pitches of all frames of
    the note, *voiced* marks those with a sure pitch."""
    if len(midi) < MIN_FRAMES or float(np.mean(voiced)) < MIN_VOICED_SHARE:
        return None
    sung = midi[voiced]
    # Octave errors of the pitch detection neither make the singing unsteady
    # nor move its pitch: the frames are taken around the pitch class they
    # gather at (their mean on the circle of the twelve classes). The median
    # of the pitches themselves lies between the octaves when half of the
    # frames are an octave off.
    angles = sung * (2.0 * np.pi / 12.0)
    centre = float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))) * 12.0 / (2.0 * np.pi)
    around = _folded(sung - centre)
    if float(np.std(around)) > MAX_SPREAD:
        return None
    middle = centre + float(np.median(around))
    # in the octave of most of the frames
    return middle + 12.0 * round((float(np.median(sung)) - middle) / 12.0)


def follow_steady_sung_pitch(midi_segments: list[MidiSegment], pitched_data: PitchedData | None) -> list[MidiSegment]:
    """Move notes that are charted beside a pitch the singing clearly stays on.

    Args:
        midi_segments: The notes; a note that moves is changed in place.
        pitched_data: Pitch track of the separated vocal stem.

    Returns:
        The same list.
    """
    if not midi_segments or pitched_data is None or not pitched_data.times:
        return midi_segments
    times = np.asarray(pitched_data.times, dtype=float)
    freqs = np.asarray(pitched_data.frequencies, dtype=float)
    conf = np.asarray(pitched_data.confidence, dtype=float)
    pitched = (freqs > 0) & np.isfinite(freqs)
    midi = np.full(len(freqs), np.nan)
    midi[pitched] = librosa.hz_to_midi(freqs[pitched])
    voiced = pitched & (conf >= VOICED_CONFIDENCE)

    moved = 0
    for segment in midi_segments:
        if segment.note_type not in PITCHED_TYPES:
            continue
        first, end = np.searchsorted(times, [segment.start, segment.end])
        sung = steady_sung_pitch(midi[first:end], voiced[first:end])
        if sung is None:
            continue
        note_midi = int(librosa.note_to_midi(segment.note))
        deviation = float(_folded(sung - note_midi))
        if abs(deviation) <= MIN_DEVIATION:
            continue
        segment.note = librosa.midi_to_note(note_midi + int(round(deviation)))
        moved += 1

    if moved:
        print(f"{ULTRASINGER_HEAD} {blue_highlighted(str(moved))} note{'s' if moved != 1 else ''} "
              "moved to the pitch the singing stays on")
    return midi_segments
