"""Notes without a pitch to sing: harsh vocals, shouting, whispering.

A scream or a growl, a shouted or a whispered line has no pitch a singer
could hit and a game could score: the pitch detection finds none in it, and
a note charted there gets an arbitrary pitch. This pass finds the notes that
are sung without a pitch and makes them freestyle notes (``F``): their
words are shown, they are not scored. Harsh vocals that carry a pitch - a
scream on a note - keep their notes.

How the notes are judged, in the separated vocal stem:

1. A frame *has a pitch* when the pitch detection is sure of one
   (``VOICED_CONFIDENCE``), or when partials stand clearly out of its
   spectrum (``partial_prominence`` of at least ``MIN_PROMINENCE_DB``). The
   second is for several voices at once: harmonies and choirs leave the
   pitch detection unsure which voice to follow, but every voice keeps its
   partials. A scream has neither: its spectrum is noise, shaped by the
   mouth.
2. Of the loud frames of a note (``LOUD_BELOW_PEAK_DB``), those with a
   pitch are counted. Few of them speak for a note without a pitch, many
   against it; a note with hardly a loud frame says nothing.
3. The notes are judged together in their order: a note is most likely of
   the kind of the note before it (``STAY``), and without evidence it is a
   sung note (``PRIOR_UNPITCHED``). So a short note follows its neighbours,
   and one unsure note within singing stays a sung note.
4. The notes of one syllable are of one kind, the one most of their loud
   frames are in.
5. A passage without a pitch has some length (``MIN_PASSAGE_S``, with the
   freestyle notes beside it): a note or two among sung ones are far more
   often a quick, breathy or half-spoken syllable of a sung line than a
   scream, and stay sung notes.
6. A note without a pitch becomes a freestyle note. The pitch it was
   charted on was arbitrary: a run of such notes is charted on the pitch of
   the last sung note before it (of the first one after it at the start of
   the song), which reads calmer than a zigzag.

Notes that are freestyle already (backing vocals of the lyrics, the
segmentation model's freestyle notes) count as neighbours and stay
freestyle; next to notes without a pitch they go to the pitch of that run,
so a passage is charted on one height. Rap notes stay as they are.
"""

import math
from dataclasses import replace

import librosa
import numpy as np

from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher.pitched_data import PitchedData
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, gold_highlighted

SAMPLE_RATE = 16000  # the audio is analysed at this rate
VOICED_CONFIDENCE = 0.7  # the pitch detection is sure of a pitch from here on
LOUD_BELOW_PEAK_DB = 30.0  # frames less than this below the loud singing are loud

# Frames that are analysed at once: the windows of a whole song at once take
# gigabytes (10 minutes are 37,500 frames)
BLOCK_FRAMES = 2000
LOUDNESS_WINDOW = 1024  # samples (64 ms) a frame's loudness is taken over
PROMINENCE_WINDOW = 2048  # samples (128 ms): long enough to tell the partials of several voices apart
PROMINENCE_BAND_HZ = (200.0, 4000.0)  # where the partials of singing are
ENVELOPE_HZ = 300.0  # the spectrum is compared with its mean over this width
TOP_SHARE = 0.1  # the partials are the highest tenth of the spectrum
# Partials this far above the spectrum's middle are sung ones. Screams have a
# median of 10 dB, harmonies of several voices of 16 dB; at 12.5 dB, four in
# five screamed stretches of an annotated set of metal vocals were judged to
# be without a pitch, and one in sixteen of its cleanly sung ones.
MIN_PROMINENCE_DB = 12.5

# Share of the loud frames with a pitch in a sung note and in one without a
# pitch. Measured: 0.94 and 0.11; a sung note with fewer of them is common
# enough (a tenth of the sung notes has less than 0.73), so the evidence of
# a frame without a pitch counts for less than the mean would make it.
P_PITCH_IN_SUNG = 0.80
P_PITCH_IN_UNPITCHED = 0.08
FRAME_WEIGHT = 0.5  # neighbouring frames are not independent evidence
MAX_FRAMES = 40  # one note counts as this many frames at most
PRIOR_UNPITCHED = 0.03  # share of notes without a pitch, before any evidence
STAY = 0.9  # probability that a note without a pitch follows one
# Probability that a note without a pitch follows a sung one: the one that
# leaves PRIOR_UNPITCHED of the notes without a pitch in the long run
START_UNPITCHED = (1.0 - STAY) * PRIOR_UNPITCHED / (1.0 - PRIOR_UNPITCHED)
NEW_PASSAGE_S = 4.0  # after a pause this long, the notes are judged anew
# Notes without a pitch in a row (with the freestyle notes beside them) are a
# passage from this length. Shorter ones stay sung notes: on 100 songs with
# hand-made charts this halves the notes charted with a pitch there that
# become freestyle (0.9 % instead of 1.8 %), and in songs with harsh vocals
# 6 % fewer notes do.
MIN_PASSAGE_S = 1.0
JUDGED_TYPES = (":", "*")  # freestyle and rap notes stay what they are


def frame_samples(audio: np.ndarray, sample_rate: int, times: np.ndarray, window: int) -> np.ndarray:
    """*window* samples of *audio* around each of *times* (rows), zero-padded at the ends."""
    centres = np.round(np.asarray(times, dtype=float) * sample_rate).astype(int)
    padded = np.pad(audio, (window, window))
    index = centres[:, None] + window - window // 2 + np.arange(window)[None, :]
    return padded[np.clip(index, 0, len(padded) - 1)]


def loudness_db(audio: np.ndarray, sample_rate: int, times: np.ndarray) -> np.ndarray:
    """Level in dB (0 = full scale) of *audio* around each of *times*."""
    result = np.zeros(len(times))
    for first in range(0, len(times), BLOCK_FRAMES):
        frames = frame_samples(audio, sample_rate, times[first:first + BLOCK_FRAMES], LOUDNESS_WINDOW)
        result[first:first + len(frames)] = np.mean(frames.astype(np.float64) ** 2, axis=1)
    return 10.0 * np.log10(result + 1e-12)


def loud_frames(level_db: np.ndarray) -> np.ndarray:
    """Which frames are loud: less than ``LOUD_BELOW_PEAK_DB`` below the loud
    singing (the level only one frame in twenty of the sounding ones exceeds)."""
    sounding = level_db[level_db > -60.0]
    if not len(sounding):
        return np.zeros(len(level_db), dtype=bool)
    return level_db > float(np.percentile(sounding, 95)) - LOUD_BELOW_PEAK_DB


def partial_prominence(audio: np.ndarray, sample_rate: int, times: np.ndarray) -> np.ndarray:
    """How far partials stand out of the spectrum around each of *times*, in dB.

    The spectrum (in dB) is taken relative to its mean over ``ENVELOPE_HZ``,
    which leaves its fine structure without the shape of the vowel; the value
    is the mean of the highest tenth of it above its median. The partials of
    sung voices - of several at once too - are narrow peaks far above the
    rest; a noise-like sound has none, whatever its vowel.
    """
    n_fft = 2 * PROMINENCE_WINDOW
    low, high = (int(round(hz * n_fft / sample_rate)) for hz in PROMINENCE_BAND_HZ)
    width = max(3, int(round(ENVELOPE_HZ * n_fft / sample_rate)) | 1)
    window = np.hanning(PROMINENCE_WINDOW).astype(np.float32)
    result = np.zeros(len(times))
    for first in range(0, len(times), BLOCK_FRAMES):
        frames = frame_samples(audio, sample_rate, times[first:first + BLOCK_FRAMES], PROMINENCE_WINDOW) * window
        spectrum = 20.0 * np.log10(np.abs(np.fft.rfft(frames, n_fft, axis=1)) + 1e-7)
        half = width // 2
        padded = np.pad(spectrum, ((0, 0), (half, half)), mode="edge")
        summed = np.cumsum(np.pad(padded, ((0, 0), (1, 0))), axis=1)
        envelope = (summed[:, width:] - summed[:, :-width]) / width
        fine = (spectrum - envelope)[:, low:high + 1]
        ordered = np.sort(fine, axis=1)
        top = ordered[:, int((1.0 - TOP_SHARE) * ordered.shape[1]):].mean(axis=1)
        result[first:first + len(frames)] = top - np.median(fine, axis=1)
    return result


def note_evidence(loud: int, with_pitch: int) -> float:
    """How much the frames of one note speak for a note without a pitch
    (above 0) or for a sung note (below 0): the logarithm of how much more
    likely *with_pitch* of *loud* frames are in the one than in the other."""
    if loud <= 0:
        return 0.0
    counted = min(loud, MAX_FRAMES)
    pitched = with_pitch / loud * counted
    unpitched = (pitched * math.log(P_PITCH_IN_UNPITCHED)
                 + (counted - pitched) * math.log(1.0 - P_PITCH_IN_UNPITCHED))
    sung = pitched * math.log(P_PITCH_IN_SUNG) + (counted - pitched) * math.log(1.0 - P_PITCH_IN_SUNG)
    return FRAME_WEIGHT * (unpitched - sung)


def judge_notes(spans: list[tuple[float, float]], evidence: list[float]) -> list[bool]:
    """Which notes are without a pitch: the most likely sequence of kinds
    for the notes (start s, end s) with their *evidence*, in order."""
    result = [False] * len(spans)
    # step[kind before][kind after], 0 = sung, 1 = without a pitch
    step = ((math.log(1.0 - START_UNPITCHED), math.log(START_UNPITCHED)),
            (math.log(1.0 - STAY), math.log(STAY)))
    first = 0
    while first < len(spans):
        last = first
        while last + 1 < len(spans) and spans[last + 1][0] - spans[last][1] < NEW_PASSAGE_S:
            last += 1
        count = last - first + 1
        score = np.zeros((count, 2))
        came_from = np.zeros((count, 2), dtype=int)
        score[0] = (math.log(1.0 - PRIOR_UNPITCHED), math.log(PRIOR_UNPITCHED) + evidence[first])
        for k in range(1, count):
            for kind in (0, 1):
                options = (score[k - 1, 0] + step[0][kind], score[k - 1, 1] + step[1][kind])
                came_from[k, kind] = int(options[1] > options[0])
                score[k, kind] = max(options) + (evidence[first + k] if kind else 0.0)
        kind = int(score[count - 1, 1] > score[count - 1, 0])
        for k in range(count - 1, -1, -1):
            result[first + k] = bool(kind)
            kind = came_from[k, kind]
        first = last + 1
    return result


def find_unpitched_notes(midi_segments: list[MidiSegment], times: np.ndarray,
                         has_pitch: np.ndarray, loud: np.ndarray) -> list[bool]:
    """Which of the notes are sung without a pitch.

    Args:
        midi_segments: The notes, in order.
        times: Times (s) of the frames of the vocal stem.
        has_pitch: Which frames have a pitch.
        loud: Which frames are loud.

    Returns:
        For each note whether it is without a pitch. Notes of other types
        than normal and golden are never (they count as neighbours with
        what is sung in them).
    """
    counts = []
    for segment in midi_segments:
        first, end = np.searchsorted(times, [segment.start, segment.end])
        end = max(end, first + 1)
        note_loud = loud[first:end]
        counts.append((int(note_loud.sum()), int((has_pitch[first:end] & note_loud).sum())))
    unpitched = judge_notes([(segment.start, segment.end) for segment in midi_segments],
                            [note_evidence(*count) for count in counts])
    # One kind for the notes of a syllable: the one most of its loud frames are in
    first = 0
    while first < len(midi_segments):
        last = first
        while last + 1 < len(midi_segments) and midi_segments[last + 1].word.strip() == "~":
            last += 1
        if last > first:
            without = sum(counts[k][0] for k in range(first, last + 1) if unpitched[k])
            sung = sum(counts[k][0] for k in range(first, last + 1) if not unpitched[k])
            for k in range(first, last + 1):
                unpitched[k] = without > sung
        first = last + 1
    unpitched = [flag and segment.note_type in JUDGED_TYPES for flag, segment in zip(unpitched, midi_segments)]
    # A passage without a pitch has some length: a note or two among sung ones
    # are far more often a quick, breathy or spoken syllable of a sung line than
    # a scream. Freestyle notes that are there already belong to the passage.
    first = 0
    while first < len(midi_segments):
        if not (unpitched[first] or midi_segments[first].note_type == "F"):
            first += 1
            continue
        last = first
        # (a long pause ends a passage, as it does where the notes are judged)
        while (last + 1 < len(midi_segments)
               and (unpitched[last + 1] or midi_segments[last + 1].note_type == "F")
               and midi_segments[last + 1].start - midi_segments[last].end < NEW_PASSAGE_S):
            last += 1
        # (a hair under the length is the length: the times are sums of beats)
        if sum(segment.end - segment.start for segment in midi_segments[first:last + 1]) < MIN_PASSAGE_S - 1e-6:
            unpitched[first:last + 1] = [False] * (last - first + 1)
        first = last + 1
    return unpitched


def as_freestyle(midi_segments: list[MidiSegment], unpitched: list[bool]) -> list[MidiSegment]:
    """The notes with the *unpitched* ones as freestyle notes, each run of
    freestyle notes with one of them on one pitch: that of the last sung
    note before it (of the first sung note of the song for a run at its
    start; in a song without a sung note, the middle one of their pitches).

    Freestyle notes that were there before (of the lyrics, of a segmentation
    model) and stand in such a run go to its pitch with it, so a passage
    never jumps between two heights; a run of them alone stays as it is.
    """
    sung = [segment.note for segment, flag in zip(midi_segments, unpitched)
            if not flag and segment.note_type in JUDGED_TYPES]
    marked = sorted((segment.note for segment, flag in zip(midi_segments, unpitched) if flag),
                    key=librosa.note_to_midi)
    result = list(midi_segments)
    if not marked:
        return result
    pitch = sung[0] if sung else marked[len(marked) // 2]
    index = 0
    while index < len(result):
        if not unpitched[index] and result[index].note_type != "F":
            if result[index].note_type in JUDGED_TYPES:
                pitch = result[index].note
            index += 1
            continue
        end = index
        while end < len(result) and (unpitched[end] or result[end].note_type == "F"):
            end += 1
        if any(unpitched[index:end]):
            result[index:end] = [replace(segment, note_type="F", note=pitch) for segment in result[index:end]]
        index = end
    return result


def mark_unpitched_notes(midi_segments: list[MidiSegment], pitched_data: PitchedData | None,
                         vocal_audio_path: str) -> list[MidiSegment]:
    """Make the notes that are sung without a pitch freestyle notes.

    Args:
        midi_segments: The notes, in order. Not changed.
        pitched_data: Pitch track of the vocal stem.
        vocal_audio_path: The vocal stem the pitch track is of.

    Returns:
        The notes, those without a pitch as freestyle notes; the given list
        when there is nothing to judge, the stem cannot be read or its
        analysis fails.
    """
    if not midi_segments or pitched_data is None or not pitched_data.times:
        return midi_segments
    try:
        audio, _ = librosa.load(vocal_audio_path, sr=SAMPLE_RATE, mono=True)
        times = np.asarray(pitched_data.times, dtype=float)
        confidence = np.asarray(pitched_data.confidence, dtype=float)
        loud = loud_frames(loudness_db(audio, SAMPLE_RATE, times))
        has_pitch = ((confidence >= VOICED_CONFIDENCE)
                     | (partial_prominence(audio, SAMPLE_RATE, times) >= MIN_PROMINENCE_DB))
        unpitched = find_unpitched_notes(midi_segments, times, has_pitch, loud)
    except Exception as error:  # noqa: BLE001 - fail open, never lose the chart (also when memory runs out)
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} the freestyle detection failed ({error!r}) "
              "- every note is charted with a pitch")
        return midi_segments
    count = sum(unpitched)
    if not count:
        return midi_segments
    print(f"{ULTRASINGER_HEAD} {blue_highlighted(str(count))} of {len(midi_segments)} notes are sung without "
          "a pitch (harsh vocals, shouting, whispering): charted as freestyle notes")
    return as_freestyle(midi_segments, unpitched)
