"""Vocal runs, riffs and acrobatics as notes.

A singer often leaves a held note for a moment and comes back to it: a
quick flip a tone up or down within one syllable (mordents, turns, grace
notes, neighbour notes, trills, vocal runs and riffs). UltraSinger charts
such a syllable as one held note. This optional pass splits the held note
there, so the chart follows the flip: note, short note on the flipped
pitch, back to the note (all ``~`` continuations of the syllable).

The chart becomes noticeably harder to sing, and professional charts
rarely notate singing this finely - at most of these places they keep
the one held note. That is why the pass is off by default.

How a flip is found, in each normal or golden note held for at least
``MIN_NOTE_S``:

1. The confident pitch frames of the note (median-smoothed over 3 frames)
   are compared with their median, the note's held pitch.
2. A flip is a run of frames at least ``START_DEVIATION`` semitones to one
   side of it, lasting ``MIN_FLIP_MS`` to ``MAX_FLIP_MS``, with the voice
   steady on the held pitch for ``STEADY_FRAMES`` frames before and after.
3. A flip of more than ``MAX_INTERVAL`` semitones is a pitch detection
   error (most often an octave error), not singing, and is left out.
4. The flipped pitch is the rounded deviation, at least one semitone. A
   quick flip is sung short of its pitch (a tone often measures as one
   semitone or less), so with a known key the flip goes to the nearest
   pitch of the key's scale within one semitone of that, on the same side.
5. The flip is placed on the written beat grid and lasts at least one
   beat; a flip that leaves less than one beat of the held note before or
   after it is left out.
"""

from dataclasses import replace

import librosa
import numpy as np

from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher.pitched_data import PitchedData
from modules.Ultrastar.coverter.ultrastar_converter import real_bpm_to_ultrastar_bpm
from modules.Ultrastar.ultrastar_writer import get_multiplier
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted

MIN_NOTE_S = 0.35  # shorter notes are not held long enough to flip in
CONFIDENCE = 0.9  # pitch frames below this confidence are not used
START_DEVIATION = 1.0  # semitones from the held pitch that start a flip
MIN_FLIP_MS = 50.0
MAX_FLIP_MS = 400.0
MAX_INTERVAL = 4  # semitones; a larger jump is a pitch detection error
STEADY_FRAMES = 3  # frames on the held pitch before and after a flip
STEADY_SEMITONES = 1.0  # how close those frames stay to the held pitch
MAX_FRAME_GAP_S = 0.05  # a longer gap between confident frames ends a flip
FLIPPABLE_TYPES = (":", "*")  # freestyle and rap notes have no pitch to flip

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_SCALES = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}


def scale_pitch_classes(music_key: str | None) -> set[int] | None:
    """Pitch classes (0 = C) of the scale of *music_key* ("D minor", "F# major").

    None when there is no key or it cannot be read.
    """
    if not music_key:
        return None
    parts = music_key.split()
    if len(parts) != 2 or parts[0] not in _NOTE_NAMES or parts[1].lower() not in _SCALES:
        return None
    root = _NOTE_NAMES.index(parts[0])
    return {(root + step) % 12 for step in _SCALES[parts[1].lower()]}


def find_flips(times: np.ndarray, midi: np.ndarray) -> list[tuple[float, float, float]]:
    """Flips in the confident pitch frames of one held note.

    *times* (s) and *midi* (fractional MIDI pitch) are the note's confident
    frames in order. Returns (start s, end s, deviation in semitones from
    the held pitch) for each flip; the deviation is not limited here.
    """
    if len(times) < 2 * STEADY_FRAMES + 2:
        return []
    smooth = _median3(midi)
    held = float(np.median(smooth))
    deviation = smooth - held
    hop = float(np.median(np.diff(times)))
    flips = []
    i, n = 0, len(smooth)
    while i < n:
        if abs(deviation[i]) < START_DEVIATION:
            i += 1
            continue
        side = np.sign(deviation[i])
        j = i
        while (j + 1 < n and np.sign(deviation[j + 1]) == side
               and abs(deviation[j + 1]) >= START_DEVIATION
               and times[j + 1] - times[j] < MAX_FRAME_GAP_S):
            j += 1
        start, end = float(times[i]), float(times[j] + hop)
        steady_before = i >= STEADY_FRAMES and np.all(
            np.abs(deviation[i - STEADY_FRAMES:i]) < STEADY_SEMITONES)
        steady_after = j + STEADY_FRAMES < n and np.all(
            np.abs(deviation[j + 1:j + 1 + STEADY_FRAMES]) < STEADY_SEMITONES)
        if (MIN_FLIP_MS <= (end - start) * 1000 <= MAX_FLIP_MS
                and steady_before and steady_after):
            flips.append((start, end, float(np.median(smooth[i:j + 1]) - held)))
        i = j + 1
    return flips


def _median3(values: np.ndarray) -> np.ndarray:
    """Median over each frame and its two neighbours (the ends stay as they are)."""
    out = values.astype(float).copy()
    if len(values) >= 3:
        out[1:-1] = np.median(np.stack([values[:-2], values[1:-1], values[2:]]), axis=0)
    return out


def flip_interval(note_midi: int, deviation: float, scale: set[int] | None) -> int | None:
    """Semitones from *note_midi* to the pitch of a flip *deviation* semitones off.

    None for a flip larger than ``MAX_INTERVAL`` (a pitch detection error).
    The rounded deviation, at least one semitone; with a *scale*, the
    nearest pitch of the scale within one semitone of that, on the same
    side (the rounded deviation when there is none).
    """
    if abs(deviation) > MAX_INTERVAL + 0.5:
        return None
    side = 1 if deviation > 0 else -1
    rounded = side * min(MAX_INTERVAL, max(1, round(abs(deviation))))
    if scale is None or (note_midi + rounded) % 12 in scale:
        return rounded
    candidates = [side * size for size in range(1, MAX_INTERVAL + 1)
                  if abs(side * size - rounded) <= 1 and (note_midi + side * size) % 12 in scale]
    if not candidates:
        return rounded
    return min(candidates, key=lambda c: (abs(c - deviation), abs(c)))


def written_beat_seconds(real_bpm: float) -> float:
    """Length in seconds of one beat of the written chart (see ``ultrastar_writer.note_beats``)."""
    return 60.0 / (real_bpm * get_multiplier(real_bpm_to_ultrastar_bpm(real_bpm)))


def split_notes_at_vocal_runs(
    midi_segments: list[MidiSegment],
    pitched_data: PitchedData,
    bpm: float,
    music_key: str | None = None,
) -> list[MidiSegment]:
    """Split held notes where the singing flips to another pitch and back.

    Args:
        midi_segments: The notes, in order. Not changed; split notes are
            replaced by their pieces in the returned list.
        pitched_data: Pitch track of the vocal.
        bpm: Real BPM, for the written beat grid.
        music_key: Key of the song ("D minor"); flips go to its scale.

    Returns:
        The notes with each flip as a note of its own between two pieces of
        the held note.
    """
    if not midi_segments or pitched_data is None or not pitched_data.times or bpm <= 0:
        return midi_segments
    times = np.asarray(pitched_data.times, dtype=float)
    freqs = np.asarray(pitched_data.frequencies, dtype=float)
    conf = np.asarray(pitched_data.confidence, dtype=float)
    keep = (conf >= CONFIDENCE) & (freqs > 0) & np.isfinite(freqs)
    times, midi = times[keep], librosa.hz_to_midi(freqs[keep])
    scale = scale_pitch_classes(music_key)
    beat = written_beat_seconds(bpm)
    gap = midi_segments[0].start  # the writer counts beats from the first note

    def beat_of(t: float) -> int:
        return round((t - gap) / beat)

    result: list[MidiSegment] = []
    flips_total = 0
    for segment in midi_segments:
        if segment.note_type not in FLIPPABLE_TYPES or segment.end - segment.start < MIN_NOTE_S:
            result.append(segment)
            continue
        inside = (times >= segment.start) & (times <= segment.end)
        note_midi = int(librosa.note_to_midi(segment.note))
        cuts: list[tuple[int, int, int]] = []  # (start beat, end beat, interval)
        last_end, note_end = beat_of(segment.start), beat_of(segment.end)
        for start, end, deviation in find_flips(times[inside], midi[inside]):
            interval = flip_interval(note_midi, deviation, scale)
            if interval is None:
                continue
            flip_start = beat_of(start)
            flip_end = max(beat_of(end), flip_start + 1)
            if flip_start - last_end < 1 or note_end - flip_end < 1:
                continue
            cuts.append((flip_start, flip_end, interval))
            last_end = flip_end
        if not cuts:
            result.append(segment)
            continue
        result.extend(_pieces(segment, cuts, gap, beat, note_midi))
        flips_total += len(cuts)

    if flips_total:
        print(
            f"{ULTRASINGER_HEAD} Vocal runs as notes: "
            f"{blue_highlighted(str(flips_total))} pitch flip{'s' if flips_total != 1 else ''} "
            "charted as notes of their own"
        )
    return result


def _pieces(segment: MidiSegment, cuts: list[tuple[int, int, int]], gap: float,
            beat: float, note_midi: int) -> list[MidiSegment]:
    """*segment* cut at *cuts* (start beat, end beat, interval) into held pieces and flips."""
    word = segment.word
    trailing = word[len(word.rstrip()):]  # the word end goes to the last piece
    pieces: list[MidiSegment] = []
    position = segment.start
    for flip_start, flip_end, interval in cuts:
        t0, t1 = gap + flip_start * beat, gap + flip_end * beat
        pieces.append(replace(segment, start=position, end=t0, word="~", line_break_after=False))
        pieces.append(replace(segment, note=librosa.midi_to_note(note_midi + interval),
                              start=t0, end=t1, word="~", line_break_after=False))
        position = t1
    pieces.append(replace(segment, start=position, word="~" + trailing))
    pieces[0].word = word.rstrip() or "~"
    return pieces
