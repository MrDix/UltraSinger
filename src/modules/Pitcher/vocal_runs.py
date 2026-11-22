"""Vocal runs, riffs and acrobatics as notes.

A singer often sings more than one pitch on a syllable: a step up or down
that is held, a run over several pitches, or a quick flip a tone away and
back (mordents, turns, grace notes, neighbour notes, trills, vocal runs and
riffs). UltraSinger charts such a syllable as one held note, or as fewer
notes than are sung. This optional pass splits the note there, so the chart
follows the singing (all pieces after the first are ``~`` continuations of
the syllable).

The chart becomes noticeably harder to sing, and professional charts
rarely notate singing this finely - at most of these places they keep
the one held note. That is why the pass is off by default.

Two things are found in each normal or golden note, in this order:

**Steps and runs** (``find_plateaus``), in a note of at least
``MIN_STEP_NOTE_S``: the voiced pitch frames of the note are cut into
plateaus, stretches the singing stays on one pitch. The cut is the one that
fits the frames best when every change of pitch has a price (the lower the
threshold, the lower the price), with plateaus of at least
``MIN_PLATEAU_S``. Pitches count by their pitch class, as the
games score them: an octave error of the pitch detection never splits a
note. A wide vibrato is averaged out over its period first. Then:

1. A plateau the singing only passes through (a glide: fewer than
   ``PLATEAU_SHARE`` of its frames are within ``PLATEAU_SEMITONES`` of its
   pitch, or its halves are ``GLIDE_DRIFT`` semitones apart) goes to its
   neighbours, and so does a first or last plateau shorter than
   ``EDGE_PLATEAU_S``: the scoop into a note, or the voice falling off at
   its end.
2. A plateau more than ``MAX_INTERVAL`` semitones from its neighbour is a
   pitch detection error, not singing, and joins it too.
3. The plateau the note has its pitch from keeps that pitch; the others
   get theirs by their distance to it. Without a plateau on the note's
   pitch the note is left alone.

**Flips** (``find_flips``), in each note or piece held for at least
``MIN_NOTE_S``: quick departures that are too short for a plateau.

1. The confident pitch frames (median-smoothed over 3 frames) are compared
   with their median, the held pitch.
2. A flip is a run of frames at least ``threshold`` semitones (by default
   ``START_DEVIATION``; lower finds more) to one side of it, lasting
   ``MIN_FLIP_MS`` to ``MAX_FLIP_MS``, with the voice steady on the held
   pitch for ``STEADY_FRAMES`` frames before and after. In a note sung with
   vibrato, a flip leaves the held pitch by at least ``VIBRATO_MARGIN``
   times the extent of the vibrato: its swings are no flips.
3. A flip of more than ``MAX_INTERVAL`` semitones is a pitch detection
   error (most often an octave error) and is left out.
4. The flipped pitch is the rounded deviation, at least one semitone. A
   quick flip is sung short of its pitch (a tone often measures as one
   semitone or less), so with a known key the flip goes to the nearest
   pitch of the key's scale within one semitone of that, on the same side.

Steps and flips are placed on the written beat grid; a piece that would
come out shorter than its minimum there (``MIN_PLATEAU_BEATS``, one beat
for a flip and the held note around it) is not cut off.

A held note may be charted in several notes already, all on one pitch (a
segmentation model cuts it where the voice sets in anew - often right where
it flips). Such notes are also looked at as the one held note they are, and
charted from that when more steps and flips are found in it than in the
pieces: a flip at a cut has no steady voice before it in its own piece.

**Without the option**, the clearest of the steps are charted in notes made
from words (``split_notes_at_clear_steps``): those found at the highest
price of a change, a pitch the singing stays on for long or far away from
the note's. Professional charts have these as notes of their own - a word
note that covers two sung pitches is on the wrong pitch for one of them.
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
CONFIDENCE = 0.9  # pitch frames below this confidence are not used for flips
START_DEVIATION = 1.0  # semitones from the held pitch that start a flip (default threshold)
MIN_FLIP_MS = 50.0
MAX_FLIP_MS = 400.0
MAX_INTERVAL = 4  # semitones; a larger jump is a pitch detection error
STEADY_FRAMES = 3  # frames on the held pitch before and after a flip
STEADY_SEMITONES = 1.0  # how close those frames stay to the held pitch
MAX_FRAME_GAP_S = 0.05  # a longer gap between confident frames ends a flip
FLIPPABLE_TYPES = (":", "*")  # freestyle and rap notes have no pitch to flip

MIN_STEP_NOTE_S = 0.2  # shorter notes have no room for two plateaus
VOICED_CONFIDENCE = 0.7  # pitch frames below this confidence are not used for plateaus
MIN_PLATEAU_S = 0.08  # a plateau is sung for at least this long
CHANGE_PRICE = 0.08  # price of a change of pitch at the default threshold, in semitones x seconds
MIN_CHANGE_PRICE, MAX_CHANGE_PRICE = 0.032, 0.24
CLEAR_STEP_PRICE = MAX_CHANGE_PRICE  # price for the steps charted without the option (see split_notes_at_clear_steps)
FRAME_COST_CAP = 3.0  # semitones; a frame further off (an outlier) costs no more
PLATEAU_SEMITONES = 0.6  # how close the frames of a plateau are to its pitch
PLATEAU_SHARE = 0.6  # share of its frames that are; fewer is a glide
EDGE_PLATEAU_S = 0.15  # a shorter first or last plateau is a scoop or a fall-off
GLIDE_DRIFT = 0.8  # semitones between the halves of a plateau from which it is a glide
MIN_PLATEAU_BEATS = 2  # written beats a piece of a step lasts at least

VIBRATO_HZ = (4.0, 8.5)  # rates of a vibrato
VIBRATO_CORRELATION = 0.4  # the pitch one period later is at least this alike
MIN_VIBRATO_EXTENT = 0.15  # semitones; less is a steady voice
MIN_VIBRATO_PERIODS = 1.5  # a vibrato is heard over at least this many periods
VIBRATO_MARGIN = 1.5  # a flip leaves the held pitch by this many times the vibrato's extent
SMOOTHED_VIBRATO = 0.4  # semitones; from this extent on, plateaus are searched in the averaged pitch

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


def _folded(semitones):
    """A difference in semitones as the nearest one between pitch classes (-6 ... +6)."""
    return (np.asarray(semitones, dtype=float) + 6.0) % 12.0 - 6.0


def change_price(threshold: float) -> float:
    """Price of one change of pitch for the plateaus, from the flip *threshold*."""
    return float(min(MAX_CHANGE_PRICE, max(MIN_CHANGE_PRICE, CHANGE_PRICE * threshold / START_DEVIATION)))


def vibrato(times: np.ndarray, midi: np.ndarray) -> tuple[float, int]:
    """Extent in semitones and period in frames of a regular vibrato in the
    pitch frames of one note; ``(0.0, 0)`` without one.

    The extent is that of the swing to one side, taken from the median
    distance to the middle, so a flip or a step in the note hardly counts.
    """
    n = len(times)
    if n < 4:
        return 0.0, 0
    hop = float(np.median(np.diff(times)))
    shortest, longest = max(2, round(1 / VIBRATO_HZ[1] / hop)), round(1 / VIBRATO_HZ[0] / hop)
    swing = _folded(np.asarray(midi, dtype=float) - float(np.median(midi)))
    swing = swing - np.median(swing)
    extent = float(np.sqrt(2.0) * np.median(np.abs(swing)))  # of a sine, the median distance is 0.71 of it
    if extent < MIN_VIBRATO_EXTENT or n < MIN_VIBRATO_PERIODS * shortest:
        return 0.0, 0
    swing = np.clip(swing, -2.0 * extent, 2.0 * extent)
    best, period = 0.0, 0
    for lag in range(shortest, longest + 1):
        if n - lag < max(lag // 2, 2) or n < MIN_VIBRATO_PERIODS * lag:
            break
        a, b = swing[:-lag], swing[lag:]
        scale = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
        correlation = float(np.dot(a, b)) / scale if scale > 0 else 0.0
        if correlation > best:
            best, period = correlation, lag
    return (extent, period) if best >= VIBRATO_CORRELATION else (0.0, 0)


def _best_cut(pitch_classes: np.ndarray, price: float, min_frames: int) -> list[tuple[int, int, int]]:
    """The frames cut into stretches of one pitch class each, of at least
    *min_frames* frames: (first frame, end frame, pitch class), with the
    lowest sum of the frames' distances to their stretch's pitch plus *price*
    (in semitones x frames) for every change."""
    n = len(pitch_classes)
    cost = np.minimum(np.abs(_folded(pitch_classes[None, :] - np.arange(12)[:, None])), FRAME_COST_CAP)
    summed = np.concatenate([np.zeros((12, 1)), np.cumsum(cost, axis=1)], axis=1)
    best = np.full(n + 1, np.inf)
    best[0] = -price  # the first stretch is no change
    came_from = np.zeros((n + 1, 2), dtype=int)  # (first frame, pitch class) of the stretch ending here
    for end in range(min_frames, n + 1):
        starts = np.arange(0, end - min_frames + 1)
        starts = starts[(starts == 0) | (starts >= min_frames)]
        stretch = summed[:, end:end + 1] - summed[:, starts]
        pitch = stretch.argmin(axis=0)
        total = best[starts] + stretch[pitch, np.arange(len(starts))] + price
        choice = int(total.argmin())
        best[end] = total[choice]
        came_from[end] = (starts[choice], pitch[choice])
    stretches = []
    end = n
    while end > 0:
        start, pitch = came_from[end]
        stretches.append((int(start), int(end), int(pitch)))
        end = int(start)
    return stretches[::-1]


def find_plateaus(times: np.ndarray, midi: np.ndarray, price: float = CHANGE_PRICE) -> list[tuple[float, float, int]]:
    """Plateaus in the voiced pitch frames of one note.

    *times* (s) and *midi* (fractional MIDI pitch) are the note's voiced
    frames in order. Returns (start s, end s, pitch class) for each plateau,
    from the first frame of a plateau to the first of the next; an empty
    list when the note is sung on one pitch.
    """
    if len(times) < 4:
        return []
    hop = float(np.median(np.diff(times)))
    min_frames = max(2, round(MIN_PLATEAU_S / hop))
    if len(times) < 2 * min_frames:
        return []
    pitch = np.asarray(midi, dtype=float)
    sung_classes = pitch % 12.0
    extent, period = vibrato(times, pitch)
    if extent >= SMOOTHED_VIBRATO:
        # The mean over one period is the pitch the vibrato swings around;
        # nothing shorter than a period is told apart in it
        middle = float(np.median(pitch))
        kernel = np.ones(period) / period
        padded = np.pad(_folded(pitch - middle), (period // 2, period - 1 - period // 2), mode="edge")
        pitch = middle + np.convolve(padded, kernel, mode="valid")
        min_frames = max(min_frames, period)
        if len(times) < 2 * min_frames:
            return []
    pitch_classes = pitch % 12.0
    plateaus = _best_cut(pitch_classes, price / hop, min_frames)

    def is_glide(start: int, end: int, pitch_class: int) -> bool:
        off = _folded(pitch_classes[start:end] - pitch_class)
        if float((np.abs(off) <= PLATEAU_SEMITONES).mean()) < PLATEAU_SHARE:
            return True
        half = (end - start) // 2
        return half >= 2 and abs(float(np.median(off[-half:]) - np.median(off[:half]))) >= GLIDE_DRIFT

    def seconds(start: int, end: int) -> float:
        return float(times[end - 1] - times[start]) + hop

    def interval(a: int, b: int) -> float:
        return abs(float(_folded(plateaus[a][2] - plateaus[b][2])))

    def tidy() -> None:
        k = 1
        while k < len(plateaus):  # neighbours of one pitch are one plateau
            if plateaus[k][2] == plateaus[k - 1][2]:
                plateaus[k - 1:k + 1] = [(plateaus[k - 1][0], plateaus[k][1], plateaus[k][2])]
            else:
                k += 1

    def join(index: int, into: int) -> None:
        low, high = min(index, into), max(index, into)
        plateaus[low:high + 1] = [(plateaus[low][0], plateaus[high][1], plateaus[into][2])]
        tidy()

    def share(index: int) -> None:
        """A glide between two plateaus: each takes the half next to it."""
        start, end, _ = plateaus[index]
        middle = (start + end) // 2
        before, after = plateaus[index - 1], plateaus[index + 1]
        plateaus[index - 1:index + 2] = [(before[0], middle, before[2]), (middle, after[1], after[2])]
        tidy()

    joined = True
    while joined and len(plateaus) > 1:
        joined = False
        longest = max(range(len(plateaus)), key=lambda k: plateaus[k][1] - plateaus[k][0])
        for index, (start, end, pitch_class) in enumerate(plateaus):
            if index == longest:
                continue
            first, last = index == 0, index == len(plateaus) - 1
            neighbour = index + 1 if first else index - 1
            if is_glide(start, end, pitch_class):
                if first or last:
                    join(index, neighbour)
                else:
                    share(index)
            elif (((first or last) and seconds(start, end) < EDGE_PLATEAU_S)
                    or interval(index, neighbour) > MAX_INTERVAL
                    or (not first and not last and interval(index, index + 1) > MAX_INTERVAL)):
                join(index, neighbour)
            else:
                continue
            joined = True
            break
    if extent >= SMOOTHED_VIBRATO:
        # The pitch of a plateau is the one sung in it, not the averaged one
        for index, (start, end, pitch_class) in enumerate(plateaus):
            off = float(np.median(_folded(sung_classes[start:end] - pitch_class)))
            plateaus[index] = (start, end, int(round(pitch_class + off)) % 12)
        tidy()
    if len(plateaus) < 2:
        return []
    starts = [float(times[start]) for start, _, _ in plateaus]
    ends = starts[1:] + [float(times[-1]) + hop]
    return [(start, end, pitch) for start, end, (_, _, pitch) in zip(starts, ends, plateaus)]


def find_flips(times: np.ndarray, midi: np.ndarray,
               threshold: float = START_DEVIATION) -> list[tuple[float, float, float]]:
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
    # The swings of a vibrato are no flips
    threshold = max(threshold, VIBRATO_MARGIN * vibrato(times, midi)[0])
    # Frames on the held pitch (vibrato included) stay within STEADY_SEMITONES
    # of it, whatever the threshold: a lower one must not lose the flips of a
    # note sung with vibrato
    steady = STEADY_SEMITONES
    flips = []
    i, n = 0, len(smooth)
    while i < n:
        if abs(deviation[i]) < threshold:
            i += 1
            continue
        side = np.sign(deviation[i])
        j = i
        while (j + 1 < n and np.sign(deviation[j + 1]) == side
               and abs(deviation[j + 1]) >= threshold
               and times[j + 1] - times[j] < MAX_FRAME_GAP_S):
            j += 1
        start, end = float(times[i]), float(times[j] + hop)
        steady_before = i >= STEADY_FRAMES and np.all(
            np.abs(deviation[i - STEADY_FRAMES:i]) < steady)
        steady_after = j + STEADY_FRAMES < n and np.all(
            np.abs(deviation[j + 1:j + 1 + STEADY_FRAMES]) < steady)
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
    threshold: float = START_DEVIATION,
) -> list[MidiSegment]:
    """Split notes where the singing goes to another pitch: held steps, runs and quick flips.

    Args:
        midi_segments: The notes, in order. Not changed; split notes are
            replaced by their pieces in the returned list.
        pitched_data: Pitch track of the vocal.
        bpm: Real BPM, for the written beat grid.
        music_key: Key of the song ("D minor"); flips go to its scale.
        threshold: Semitones the singing has to leave the held pitch by to
            count as a flip, and with it how clearly it has to settle on
            another pitch to count as a step; lower finds more.

    Returns:
        The notes with each sung pitch as a note of its own.
    """
    result, steps, flips = _split_notes(midi_segments, pitched_data, bpm, change_price(threshold),
                                        flip_threshold=threshold, music_key=music_key)
    if steps or flips:
        found = []
        if steps:
            found.append(f"{blue_highlighted(str(steps))} pitch step{'s' if steps != 1 else ''}")
        if flips:
            found.append(f"{blue_highlighted(str(flips))} pitch flip{'s' if flips != 1 else ''}")
        print(f"{ULTRASINGER_HEAD} Vocal runs as notes: {' and '.join(found)} charted as notes of their own")
    return result


def split_notes_at_clear_steps(midi_segments: list[MidiSegment], pitched_data: PitchedData,
                               bpm: float) -> list[MidiSegment]:
    """Split notes where the singing clearly settles on another pitch.

    The steps of ``split_notes_at_vocal_runs`` at the highest price of a
    change (``CLEAR_STEP_PRICE``) and without the flips: what is left are
    the steps that professional charts have as notes of their own.

    Args:
        midi_segments: The notes, in order. Not changed; split notes are
            replaced by their pieces in the returned list.
        pitched_data: Pitch track of the vocal.
        bpm: Real BPM, for the written beat grid.

    Returns:
        The notes, split at the clear steps.
    """
    result, steps, _ = _split_notes(midi_segments, pitched_data, bpm, CLEAR_STEP_PRICE)
    if steps:
        print(f"{ULTRASINGER_HEAD} {blue_highlighted(str(steps))} note{'s' if steps != 1 else ''} split "
              "where the singing settles on another pitch")
    return result


def _split_notes(midi_segments: list[MidiSegment], pitched_data: PitchedData, bpm: float, price: float,
                 flip_threshold: float | None = None,
                 music_key: str | None = None) -> tuple[list[MidiSegment], int, int]:
    """The notes split at their steps (found at *price* a change) and, with a
    *flip_threshold*, at their flips; with the number of steps and of flips."""
    if not midi_segments or pitched_data is None or not pitched_data.times or bpm <= 0:
        return midi_segments, 0, 0
    times = np.asarray(pitched_data.times, dtype=float)
    freqs = np.asarray(pitched_data.frequencies, dtype=float)
    conf = np.asarray(pitched_data.confidence, dtype=float)
    pitched = (freqs > 0) & np.isfinite(freqs)
    midi = np.full(len(freqs), np.nan)
    midi[pitched] = librosa.hz_to_midi(freqs[pitched])
    voiced = pitched & (conf >= VOICED_CONFIDENCE)
    confident = pitched & (conf >= CONFIDENCE)
    scale = scale_pitch_classes(music_key)
    beat = written_beat_seconds(bpm)
    gap = midi_segments[0].start  # the writer counts beats from the first note

    def beat_of(t: float) -> int:
        return round((t - gap) / beat)

    def frames(segment: MidiSegment, usable: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inside = usable & (times >= segment.start) & (times <= segment.end)
        return times[inside], midi[inside]

    def at_steps(segment: MidiSegment) -> list[MidiSegment]:
        if segment.end - segment.start < MIN_STEP_NOTE_S:
            return [segment]
        note_midi = int(librosa.note_to_midi(segment.note))
        plateaus = find_plateaus(*frames(segment, voiced), price)
        return _step_pieces(segment, plateaus, note_midi, beat_of, gap, beat) if plateaus else [segment]

    def at_flips(segment: MidiSegment) -> list[MidiSegment]:
        if flip_threshold is None or segment.end - segment.start < MIN_NOTE_S:
            return [segment]
        note_midi = int(librosa.note_to_midi(segment.note))
        cuts: list[tuple[int, int, int]] = []  # (start beat, end beat, interval)
        last_end, note_end = beat_of(segment.start), beat_of(segment.end)
        for start, end, deviation in find_flips(*frames(segment, confident), flip_threshold):
            interval = flip_interval(note_midi, deviation, scale)
            if interval is None:
                continue
            flip_start = beat_of(start)
            flip_end = max(beat_of(end), flip_start + 1)
            if flip_start - last_end < 1 or note_end - flip_end < 1:
                continue
            cuts.append((flip_start, flip_end, interval))
            last_end = flip_end
        return _pieces(segment, cuts, gap, beat, note_midi) if cuts else [segment]

    def split(segment: MidiSegment) -> tuple[list[MidiSegment], int, int]:
        """*segment* split at its steps and flips, with their numbers."""
        pieces: list[MidiSegment] = []
        stepped = at_steps(segment)
        flips = 0
        for piece in stepped:
            flipped = at_flips(piece)
            flips += (len(flipped) - 1) // 2
            pieces.extend(flipped)
        return pieces, len(stepped) - 1, flips

    def held_until(first: int) -> int:
        """Index of the last note of the held note that starts at *first*: the
        notes after it that continue its syllable on its pitch without a pause."""
        last = first
        while (last + 1 < len(midi_segments)
               and not midi_segments[last].line_break_after
               and midi_segments[last + 1].word.strip() == "~"
               and midi_segments[last + 1].note == midi_segments[first].note
               and midi_segments[last + 1].note_type == midi_segments[first].note_type
               and midi_segments[last + 1].start - midi_segments[last].end <= beat):
            last += 1
        return last

    result: list[MidiSegment] = []
    steps_total = flips_total = 0
    index = 0
    while index < len(midi_segments):
        segment = midi_segments[index]
        if segment.note_type not in FLIPPABLE_TYPES:
            result.append(segment)
            index += 1
            continue
        # With the flips (the option), a held note that is cut into several
        # notes on one pitch is also looked at as a whole: a flip right at
        # such a cut has no steady voice before it in its own piece. The
        # whole is taken when more is found in it than in its pieces.
        last = held_until(index) if flip_threshold is not None else index
        found = [split(piece) for piece in midi_segments[index:last + 1]]
        pieces = [piece for split_pieces, _, _ in found for piece in split_pieces]
        steps, flips = sum(f[1] for f in found), sum(f[2] for f in found)
        if last > index:
            final = midi_segments[last]
            whole = replace(segment, end=final.end, line_break_after=final.line_break_after,
                            word=segment.word.rstrip() + final.word[len(final.word.rstrip()):])
            whole_pieces, whole_steps, whole_flips = split(whole)
            if whole_steps + whole_flips > steps + flips:
                pieces, steps, flips = whole_pieces, whole_steps, whole_flips
                # The whole has the mark of the pitch refinement of its first note
                # only: each piece gets the mark of the note its middle lies in
                parts = midi_segments[index:last + 1]
                for piece in pieces:
                    middle = (piece.start + piece.end) / 2
                    part = min(parts, key=lambda p: max(p.start - middle, middle - p.end, 0.0))
                    piece.pitch_refined = part.pitch_refined
        result.extend(pieces)
        steps_total += steps
        flips_total += flips
        index = last + 1

    return result, steps_total, flips_total


def _step_pieces(segment: MidiSegment, plateaus: list[tuple[float, float, int]], note_midi: int,
                 beat_of, gap: float, beat: float) -> list[MidiSegment]:
    """*segment* cut at its *plateaus* (start s, end s, pitch class); ``[segment]``
    when no plateau is on the note's pitch or no cut is left on the beat grid."""
    off_pitch = [abs(float(_folded(note_midi - pitch))) for _, _, pitch in plateaus]
    if min(off_pitch) > 1:
        return [segment]
    # The plateau the note has its pitch from: the longest one on its pitch,
    # else the longest one a semitone beside it
    on_pitch = [k for k, off in enumerate(off_pitch) if off == min(off_pitch)]
    anchor = max(on_pitch, key=lambda k: plateaus[k][1] - plateaus[k][0])
    pitches = [note_midi] * len(plateaus)
    for k in range(anchor + 1, len(plateaus)):
        pitches[k] = pitches[k - 1] + int(round(float(_folded(plateaus[k][2] - plateaus[k - 1][2]))))
    for k in range(anchor - 1, -1, -1):
        pitches[k] = pitches[k + 1] + int(round(float(_folded(plateaus[k][2] - plateaus[k + 1][2]))))
    # On the beat grid; a piece too short there joins the piece before it (the first one the next)
    bounds = [beat_of(segment.start)] + [beat_of(start) for start, _, _ in plateaus[1:]] + [beat_of(segment.end)]
    pieces = [[bounds[k], bounds[k + 1], pitches[k], k == anchor] for k in range(len(plateaus))]
    k = 0
    while len(pieces) > 1 and k < len(pieces):
        start, end, _, is_anchor = pieces[k]
        if end - start >= MIN_PLATEAU_BEATS:
            k += 1
            continue
        into = k - 1 if k > 0 else 1
        pieces[into][0], pieces[into][1] = min(start, pieces[into][0]), max(end, pieces[into][1])
        pieces[into][3] = pieces[into][3] or is_anchor
        if is_anchor:
            pieces[into][2] = note_midi  # the note's own pitch stays in the chart
        del pieces[k]
        k = 0
    merged = [pieces[0]]
    for piece in pieces[1:]:
        if piece[2] == merged[-1][2]:
            merged[-1][1] = piece[1]
        else:
            merged.append(piece)
    if len(merged) < 2:
        return [segment]
    word = segment.word
    trailing = word[len(word.rstrip()):]  # the word end goes to the last piece
    result = []
    for index, (start, end, pitch, _) in enumerate(merged):
        result.append(replace(
            segment,
            note=librosa.midi_to_note(pitch),
            start=segment.start if index == 0 else gap + start * beat,
            end=segment.end if index == len(merged) - 1 else gap + end * beat,
            word=(word.rstrip() or "~") if index == 0 else "~",
            line_break_after=False,
        ))
    result[-1].word += trailing
    result[-1].line_break_after = segment.line_break_after
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
