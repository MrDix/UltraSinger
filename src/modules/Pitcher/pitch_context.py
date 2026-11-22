"""Note pitches from their musical context.

A note of the segmentation model has the pitch that was sung: the upper
middle of the confident pitch frames inside the note. A hand-made chart
writes the pitch the melody means, and that is often the semitone next to
it: a note sung flat, scooped into or left early still is the note of the
scale it belongs to. Of the generated notes that sit one to one on a note
of a hand-made chart, about seven in ten have exactly its pitch and one in
six is one semitone off.

A small learned model decides per note whether the charted pitch is one or
two semitones beside the sung one. It sees how the note's pitch frames lie
around its pitch, the two notes on each side of it, and the song as a whole:
how often each pitch class is sung, the scale that fits best and the tuning.
For each shift of -2 to +2 semitones it predicts on what share of the note's
time the shifted pitch would be within one semitone of a hand-made chart,
and on what share it would be exactly on it. A note moves when a shift
promises more than staying, by a margin that the chosen strength sets.

Only a note that is at the pitch of its frames is judged. A note that
another step moved away from them stays where that step put it: the pitch
refinement moves a note to the tone the games' pitch detection hears, and
pulling it back to the sung pitch costs game score and agreement with
hand-made charts alike.

Training (``tools/train_pitch_context.py``) and the conversion share the
feature code of this module, so a model always sees the features it was
trained on. A model file holds gradient-boosted regression trees as plain
arrays of numbers (a numpy ``.npz`` archive, read without pickle): loading
it never runs code, and predicting needs nothing but numpy. No model is
shipped; everyone trains their own from their UltraStar song library.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, gold_highlighted

# Semitones a note can move by
SHIFTS = (-2, -1, 0, 1, 2)
_KEEP = SHIFTS.index(0)

# The strengths of the setting: how much being exactly on the charted pitch
# counts next to being within one semitone of it (the tolerance most games
# score with), and by how much a shift has to beat staying.
# "balanced" moves about one note in twelve and keeps the agreement within one
# semitone; "strong" moves about one in seven, gains the most exact pitches
# and gives a little of the agreement within one semitone for them.
STRENGTH_OFF = "off"
STRENGTHS = {
    "balanced": (0.3, 0.1),
    "strong": (1.0, 0.0),
}
STRENGTH_CHOICES = (STRENGTH_OFF, *STRENGTHS)

VOICED_CONFIDENCE = 0.7  # pitch frames below this confidence are not voiced
MIN_HZ = 40.0
# The pitch a note gets from its confident pitch frames (as modules.Segmentation.decode)
NOTE_PITCH_PERCENTILE = 60
MIN_FRAMES = 2  # confident pitch frames a note needs to be judged
# A note that lies further than this from the pitch of its frames (semitones,
# octaves aside) was put there by another step: the pitch refinement moves a
# note to the tone the games' pitch detection hears in it. The model has seen
# no such note - in training every note has the pitch of its frames - and
# would mostly pull it back to the frames. Such a note stays as it is.
MAX_FRAME_DISTANCE = 1.5
DEFAULT_FRAME_MS = 16.0
PITCHED_TYPES = (":", "*")  # freestyle and rap notes are charted without a sung pitch

_MAJOR_SCALE = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], bool)
_NEIGHBOURS = (-2, -1, 1, 2)

# The features of a note, in the order of the columns of ``note_features``.
# A model file names the features it was trained on; a model for other
# features is refused.
FEATURE_NAMES = (
    # the note's confident pitch frames, in semitones from the note's pitch
    "frames_p10", "frames_p25", "frames_p50", "frames_p60", "frames_p75", "frames_p90",
    "frames_mean", "frames_spread", "frames_count", "frames_share",
    # share of the frames within half a semitone of each candidate pitch
    *(f"frames_near_{d:+d}" for d in SHIFTS),
    "frames_slope",
    "duration_ms", "duration_log",
    # the two pitched notes before and after (of the notes that have pitch frames
    # themselves): semitones from the note, pause in between
    *(f"note_{k:+d}_{what}" for k in _NEIGHBOURS for what in ("pitch", "gap_ms")),
    # the song: share of the note time on the pitch class of each candidate pitch,
    # and whether it lies in the major scale that holds most of the note time
    *(f"song_share_{d:+d}" for d in SHIFTS),
    *(f"in_scale_{d:+d}" for d in SHIFTS),
    "scale_fit", "scale_lead", "scale_step", "tuning",
)
N_FEATURES = len(FEATURE_NAMES)

MODEL_FORMAT = 1
MODEL_KIND = "ultrasinger-pitch-context"
# The outputs of a model: for each shift the share of the note's time within
# one semitone of the charted pitch, then for each shift the share exactly on it
N_OUTPUTS = 2 * len(SHIFTS)
_MAX_TREE_DEPTH = 64
_MAX_MODEL_BYTES = 64 * 1024 * 1024  # unpacked; a real model unpacks to about 5 MB
_MAX_NODES = 2_000_000  # of all trees of a model together; a real model has about 200 000
_MAX_TREES = 20_000  # a real model has 3000
_MAX_TEXT_BYTES = 256 * 1024  # of the names and of the info of a model (text takes 4 bytes a character)
_PREDICT_CHUNK = 256  # notes predicted at once (bounds the memory of the tree walk)
_PREDICT_CELLS = 4_000_000  # notes x trees walked at once at most


# ---------------------------------------------------------------------------
# Pitch track and features
# ---------------------------------------------------------------------------

class PitchTrack:
    """The confident frames of a pitch track: their times (ms) and pitches (fractional MIDI)."""

    def __init__(self, times_s, frequencies_hz, confidence):
        times = np.asarray(times_s, dtype=float)
        hz = np.asarray(frequencies_hz, dtype=float)
        conf = np.asarray(confidence, dtype=float)
        n = min(len(times), len(hz), len(conf))
        times, hz, conf = times[:n], hz[:n], conf[:n]
        keep = np.isfinite(hz) & (hz > MIN_HZ) & (conf >= VOICED_CONFIDENCE)
        self.times_ms = times[keep] * 1000.0
        self.midi = 69.0 + 12.0 * np.log2(hz[keep] / 440.0)
        step = float(np.median(np.diff(times))) * 1000.0 if n > 1 else 0.0
        self.frame_ms = step if step > 0 else DEFAULT_FRAME_MS

    @classmethod
    def from_pitched_data(cls, pitched_data) -> "PitchTrack":
        return cls(pitched_data.times, pitched_data.frequencies, pitched_data.confidence)

    def frames(self, start_ms: float, end_ms: float) -> np.ndarray:
        """The pitches of the confident frames from ``start_ms`` up to ``end_ms``."""
        first, end = np.searchsorted(self.times_ms, [start_ms, end_ms])
        return self.midi[first:end]


def track_pitch(frames: np.ndarray) -> int | None:
    """The pitch a note gets from its confident pitch frames; None with too few of them."""
    if len(frames) < MIN_FRAMES:
        return None
    return int(np.round(np.percentile(frames, NOTE_PITCH_PERCENTILE)))


def _best_scale(pitch_class_time: np.ndarray) -> tuple[np.ndarray, int, float, float]:
    """The major scale that holds most of the note time: its pitch classes, its
    tonic, the share of the note time in it and its lead over the next best scale."""
    in_scale = np.array([pitch_class_time[np.roll(_MAJOR_SCALE, k)].sum() for k in range(12)])
    tonic = int(np.argmax(in_scale))
    total = max(float(pitch_class_time.sum()), 1.0)
    runner_up = float(np.max(np.delete(in_scale, tonic)))
    return (np.roll(_MAJOR_SCALE, tonic), tonic, float(in_scale[tonic]) / total,
            (float(in_scale[tonic]) - runner_up) / total)


def _slope(frames: np.ndarray) -> float:
    """Semitones per frame the pitch rises by over the note (least squares)."""
    if len(frames) < 4:
        return 0.0
    position = np.arange(len(frames), dtype=float)
    position -= position.mean()
    return float(np.dot(position, frames - frames.mean()) / np.dot(position, position))


def note_features(starts_ms, ends_ms, pitches, track: PitchTrack) -> tuple[np.ndarray, np.ndarray]:
    """The features of the pitched notes of one song.

    ``starts_ms``, ``ends_ms`` and ``pitches`` (MIDI) describe the notes in
    the order they are sung; ``track`` is the pitch track their pitches come
    from. Returns the feature matrix (notes x ``N_FEATURES``, float32; NaN
    where a note has nothing to measure, e.g. no note before it) and, per
    note, whether it has enough confident pitch frames to be judged.

    Only the judged notes describe each other and the song. A note without
    enough pitch frames has its pitch from elsewhere (another stem, or the
    note before it): it is no neighbour of a note, adds nothing to the song's
    pitch classes, and its own row stays empty (NaN). Training leaves such
    notes out before it asks for the features and the conversion hands in
    every pitched note; this way both get the same features for a note.
    """
    starts = np.asarray(starts_ms, dtype=float)
    ends = np.asarray(ends_ms, dtype=float)
    pitches = np.asarray(pitches, dtype=np.int64)
    n = len(pitches)
    features = np.full((n, N_FEATURES), np.nan, dtype=np.float64)
    durations = ends - starts
    note_frames = [track.frames(starts[i], ends[i]) for i in range(n)]
    judged = np.array([len(frames) >= MIN_FRAMES for frames in note_frames], dtype=bool)
    known = np.flatnonzero(judged)  # the notes with a pitch of their own in the track, in order
    if not len(known):
        return features.astype(np.float32), judged

    # The song: time on each pitch class, best-fitting scale, tuning
    pitch_class_time = np.bincount(pitches[known] % 12, weights=durations[known], minlength=12)
    song_share = pitch_class_time / max(float(pitch_class_time.sum()), 1.0)
    scale, tonic, scale_fit, scale_lead = _best_scale(pitch_class_time)
    sung = np.concatenate([note_frames[i] for i in known])
    # how far the singing lies from the semitones on average (-0.5 ... 0.5)
    tuning = float(np.angle(np.mean(np.exp(2j * np.pi * (sung - np.round(sung))))) / (2 * np.pi))

    shifts = np.array(SHIFTS)
    for position, i in enumerate(known):
        pitch = int(pitches[i])
        row = features[i]
        # in the octave of the note (the octave steps of the conversion may have moved it)
        frames = note_frames[i] + 12.0 * np.round((pitch - np.median(note_frames[i])) / 12.0)
        row[0:6] = np.percentile(frames, [10, 25, 50, NOTE_PITCH_PERCENTILE, 75, 90]) - pitch
        row[6] = frames.mean() - pitch
        row[7] = frames.std()
        row[8] = len(frames)
        row[9] = len(frames) * track.frame_ms / max(durations[i], 1.0)
        row[10:15] = [np.mean(np.abs(frames - (pitch + d)) <= 0.5) for d in SHIFTS]
        row[15] = _slope(frames)
        row[16] = durations[i]
        row[17] = np.log(max(durations[i], 10.0))
        for k, offset in enumerate(_NEIGHBOURS):
            if 0 <= position + offset < len(known):
                j = known[position + offset]
                gap = starts[i] - ends[j] if offset < 0 else starts[j] - ends[i]
                row[18 + 2 * k] = np.clip(pitches[j] - pitch, -12, 12)
                row[19 + 2 * k] = np.clip(gap, -100.0, 2000.0)
        candidates = (pitch + shifts) % 12
        row[26:31] = song_share[candidates]
        row[31:36] = scale[candidates]
        row[36] = scale_fit
        row[37] = scale_lead
        row[38] = (pitch - tonic) % 12
        row[39] = tuning
    return features.astype(np.float32), judged


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class PitchContextModel:
    """Gradient-boosted regression trees as arrays; one row per node or per tree.

    A node with ``feature`` -1 is a leaf and adds its ``value`` to the output
    of its tree. Any other node sends a note to its ``left`` child when the
    note's feature is at most ``threshold`` (a missing feature goes where
    ``missing_left`` says) and to its ``right`` child otherwise; a child
    always comes after its parent. An output is its ``baseline`` plus the
    leaves that the trees of that output (``tree_output``) end in.
    """
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    missing_left: np.ndarray
    value: np.ndarray
    tree_root: np.ndarray
    tree_output: np.ndarray
    baseline: np.ndarray
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        # (before the arrays are made as wide as the walk needs them, and before the check
        # of the trees, which both take memory for every node and tree)
        if np.size(self.feature) > _MAX_NODES:
            raise ValueError("the model has more nodes than any real one")
        if np.size(self.tree_root) > _MAX_TREES or np.size(self.tree_output) > _MAX_TREES:
            raise ValueError("the model has more trees than any real one")
        self.feature = np.asarray(self.feature, dtype=np.int32)
        self.threshold = np.asarray(self.threshold, dtype=np.float64)
        self.left = np.asarray(self.left, dtype=np.int32)
        self.right = np.asarray(self.right, dtype=np.int32)
        self.missing_left = np.asarray(self.missing_left, dtype=bool)
        self.value = np.asarray(self.value, dtype=np.float64)
        self.tree_root = np.asarray(self.tree_root, dtype=np.int32)
        self.tree_output = np.asarray(self.tree_output, dtype=np.int32)
        self.baseline = np.asarray(self.baseline, dtype=np.float64)
        self._check()

    def _check(self) -> None:
        """Refuse arrays that are no trees: walking them must end and stay inside them."""
        nodes = len(self.feature)
        per_node = (self.threshold, self.left, self.right, self.missing_left, self.value)
        if any(a.ndim != 1 for a in (self.feature, *per_node, self.tree_root, self.tree_output, self.baseline)):
            raise ValueError("the arrays of the model have a wrong shape")
        if nodes == 0 or any(len(a) != nodes for a in per_node):
            raise ValueError("the node arrays of the model differ in length")
        if len(self.tree_root) == 0 or len(self.tree_root) != len(self.tree_output):
            raise ValueError("the tree arrays of the model differ in length")
        if self.baseline.shape != (N_OUTPUTS,) or not np.isfinite(self.baseline).all():
            raise ValueError(f"the model needs {N_OUTPUTS} outputs")
        if self.tree_output.min() < 0 or self.tree_output.max() >= N_OUTPUTS:
            raise ValueError("a tree of the model belongs to no output")
        if self.tree_root.min() < 0 or self.tree_root.max() >= nodes:
            raise ValueError("a tree of the model starts outside of it")
        if self.feature.min() < -1 or self.feature.max() >= N_FEATURES:
            raise ValueError("a node of the model asks for a feature that does not exist")
        inner = self.feature >= 0
        index = np.arange(nodes)
        for child in (self.left, self.right):
            if (child[inner] <= index[inner]).any() or (child[inner] >= nodes).any():
                raise ValueError("a node of the model leads outside of its tree")
        if np.isnan(self.threshold[inner]).any() or not np.isfinite(self.value[~inner]).all():
            raise ValueError("the model holds a number that is none")
        # A child comes after its parent, so one pass in order gives the depths
        depth = [0] * nodes
        left, right = self.left.tolist(), self.right.tolist()
        for parent in np.flatnonzero(inner).tolist():
            below = depth[parent] + 1
            depth[left[parent]] = max(depth[left[parent]], below)
            depth[right[parent]] = max(depth[right[parent]], below)
        self._depth = max(depth)
        if self._depth > _MAX_TREE_DEPTH:
            raise ValueError("a tree of the model is too deep")

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per note and shift: the share of its time within one semitone of the
        charted pitch, and the share exactly on it (two arrays, notes x shifts)."""
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != N_FEATURES:
            raise ValueError(f"the features need {N_FEATURES} columns")
        out = np.empty((len(features), N_OUTPUTS), dtype=np.float64)
        of_output = [np.flatnonzero(self.tree_output == k) for k in range(N_OUTPUTS)]
        chunk = max(1, min(_PREDICT_CHUNK, _PREDICT_CELLS // len(self.tree_root)))
        for a in range(0, len(features), chunk):
            x = features[a:a + chunk]
            rows = np.arange(len(x))[:, None]
            node = np.broadcast_to(self.tree_root, (len(x), len(self.tree_root))).copy()
            for _ in range(self._depth):
                asked = self.feature[node]
                inner = asked >= 0
                if not inner.any():
                    break
                value = x[rows, np.maximum(asked, 0)]
                go_left = np.where(np.isnan(value), self.missing_left[node], value <= self.threshold[node])
                node = np.where(inner, np.where(go_left, self.left[node], self.right[node]), node)
            leaves = self.value[node]
            for k, trees in enumerate(of_output):
                out[a:a + len(x), k] = self.baseline[k] + leaves[:, trees].sum(axis=1)
        return out[:, :len(SHIFTS)], out[:, len(SHIFTS):]

    def save(self, path: str | Path) -> None:
        """Write the model file through a temporary file in its folder, put in its
        place in one step: there is never a half-written model. Each save has a
        temporary file of its own, so two trainings that write the same model
        file do not write into each other's (the one that ends last stays).
        Where files have permission bits, the model is readable by its owner
        only, as temporary files are."""
        path = Path(path)
        handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".part")
        try:
            with os.fdopen(handle, "wb") as f:  # a file object, so numpy does not append ".npz" to the name
                np.savez_compressed(
                    f, kind=np.array(MODEL_KIND), format=np.array(MODEL_FORMAT, dtype=np.int64),
                    features=np.array(FEATURE_NAMES), shifts=np.array(SHIFTS, dtype=np.int64),
                    feature=self.feature.astype(np.int16), threshold=self.threshold,
                    left=self.left, right=self.right, missing_left=self.missing_left, value=self.value,
                    tree_root=self.tree_root, tree_output=self.tree_output.astype(np.int16),
                    baseline=self.baseline, info=np.array(json.dumps(_plain(self.info))),
                )
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def _plain(value):
    """numpy scalars -> Python numbers, so the model's info is plain JSON."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# The arrays of a model file: the kind of their numbers and their number of dimensions
_MODEL_ARRAYS = {
    "feature": ("i", 1), "threshold": ("f", 1), "left": ("i", 1), "right": ("i", 1),
    "missing_left": ("b", 1), "value": ("f", 1), "tree_root": ("i", 1), "tree_output": ("i", 1),
    "baseline": ("f", 1),
}


def _read_array(archive: zipfile.ZipFile, name: str, kind: str, ndim: int,
                max_bytes: int = _MAX_MODEL_BYTES) -> np.ndarray:
    """The array ``name`` of a model archive, read only when it is what it says.

    numpy makes an array as large as the shape in its header names before it
    reads any data, so a header that names billions of values asks for that
    much memory, whatever the size of the file. The header is read first here:
    the array must be of numbers of the expected ``kind`` (never of Python
    objects, which would be unpickled) and its data must be exactly the rest
    of the archive member, at most ``max_bytes``.

    A member is stored or deflated, as numpy writes it. Other compressions are
    refused before a byte is read: Python unpacks a whole chunk of a bzip2 or
    lzma member at once, however much that is (a few bytes can stand for
    gigabytes of zeros), and only then cuts it down to the size the archive
    names.
    """
    entry = archive.getinfo(name + ".npy")
    if entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ValueError(f"the array '{name}' is packed in a way a model file never is")
    if entry.flag_bits & 0x1:
        raise ValueError(f"the array '{name}' is encrypted")
    with archive.open(entry) as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(member)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(member)
        else:
            raise ValueError(f"the array '{name}' is of an unknown format")
        if dtype.hasobject or dtype.kind != kind or dtype.itemsize == 0 or len(shape) != ndim:
            raise ValueError(f"the array '{name}' has a wrong type")
        size = math.prod(shape) * dtype.itemsize
        if size != entry.file_size - member.tell():
            raise ValueError(f"the array '{name}' is not as large as its header says")
        if size > max_bytes:
            raise ValueError(f"the array '{name}' is too large")
        data = member.read(size)
    if len(data) != size:
        raise ValueError(f"the array '{name}' is cut off")
    array = np.frombuffer(data, dtype=dtype).reshape(shape, order="F" if fortran_order else "C")
    # (in the byte order of this machine: a copy only when the file has the other one)
    return array if dtype.isnative else array.astype(dtype.newbyteorder("="))


def load_model(path: str | Path) -> PitchContextModel:
    """Read a model file. Raises ValueError for a file that is none.

    The file is a numpy ``.npz`` archive of number and text arrays. Each array
    is checked before it is read (see ``_read_array``): nothing in the file is
    ever executed, and a pickle or an archive that holds one is refused. A
    file may unpack to ``_MAX_MODEL_BYTES`` and hold ``_MAX_NODES`` nodes in
    ``_MAX_TREES`` trees at most: the most a crafted file can make the loader
    hold, before it is refused or loaded, is a few hundred megabytes, and a
    model that loads is walked in pieces of a bounded size.
    """
    try:
        with zipfile.ZipFile(str(path)) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > _MAX_MODEL_BYTES:
                raise ValueError("too large")
            file_kind = _read_array(archive, "kind", "U", 0, _MAX_TEXT_BYTES)
            file_format = _read_array(archive, "format", "i", 0, _MAX_TEXT_BYTES)
            if str(file_kind[()]) != MODEL_KIND or int(file_format[()]) != MODEL_FORMAT:
                raise ValueError(f"not format {MODEL_FORMAT}")
            features = _read_array(archive, "features", "U", 1, _MAX_TEXT_BYTES)
            shifts = _read_array(archive, "shifts", "i", 1, _MAX_TEXT_BYTES)
            if (tuple(str(name) for name in features) != FEATURE_NAMES
                    or tuple(int(d) for d in shifts) != SHIFTS):
                raise ValueError("made for other note features than this version uses; train a new one")
            arrays = {name: _read_array(archive, name, kind, ndim)
                      for name, (kind, ndim) in _MODEL_ARRAYS.items()}
            info = {}
            if "info.npy" in archive.namelist():
                info = json.loads(str(_read_array(archive, "info", "U", 0, _MAX_TEXT_BYTES)[()]))
        return PitchContextModel(**arrays, info=info if isinstance(info, dict) else {})
    except Exception as e:  # noqa: BLE001 - whatever a file makes the readers raise, it is no model
        # (e.g. zipfile.BadZipFile for a pickle or any other file that is no archive,
        # RecursionError for an info nested too deeply, MemoryError, zlib.error)
        raise ValueError(f"{path} is not a pitch context model file ({e!r})") from e


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def choose_shifts(within_one: np.ndarray, exact: np.ndarray, strength: str) -> np.ndarray:
    """The shift of each note in semitones (0: it stays).

    ``within_one`` and ``exact`` are the predictions of a model (notes x
    shifts). A note takes the shift that promises most, when that is more
    than staying promises by more than the margin of ``strength``.
    """
    weight, margin = STRENGTHS[strength]
    promise = np.asarray(within_one, dtype=float) + weight * np.asarray(exact, dtype=float)
    best = np.argmax(promise, axis=1)
    rows = np.arange(len(promise))
    moves = promise[rows, best] - promise[:, _KEEP] > margin
    return np.where(moves, np.array(SHIFTS)[best], 0)


def predict_shifts(model: PitchContextModel, starts_ms, ends_ms, pitches, track: PitchTrack,
                   strength: str) -> np.ndarray:
    """The shift (semitones) of each pitched note of a song; see ``note_features``
    for the notes. A note without enough confident pitch frames stays (and is
    no neighbour of the notes around it), and so does a note that is not at
    the pitch of its frames (``MAX_FRAME_DISTANCE``)."""
    features, judged = note_features(starts_ms, ends_ms, pitches, track)
    if not len(features):
        return np.zeros(0, dtype=np.int64)
    with np.errstate(invalid="ignore"):  # (no frames: no distance, not judged anyway)
        judged &= np.abs(features[:, FEATURE_NAMES.index("frames_p60")]) <= MAX_FRAME_DISTANCE
    within_one, exact = model.predict(features)
    return np.where(judged, choose_shifts(within_one, exact, strength), 0)


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict = {}


def _get_model(path: str) -> PitchContextModel:
    key = (os.path.abspath(path), os.path.getmtime(path))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = load_model(path)
    return _MODEL_CACHE[key]


@dataclass
class PitchContextResult:
    """What the step did: for the console and the settings info file."""
    applied: bool
    notes: int = 0
    moved: int = 0
    reason: str = ""  # why it did not run

    def __str__(self) -> str:
        if not self.applied:
            return f"not applied ({self.reason})"
        return f"{self.moved} of {self.notes} note pitches moved"


def correct_note_pitches(midi_segments: list, pitched_data, model_path: str | None,
                         strength: str) -> PitchContextResult:
    """Move the pitched notes whose musical context says so by up to two semitones.

    Args:
        midi_segments: The notes; a note that moves is changed in place, in its octave.
            Freestyle and rap notes are neither moved nor looked at.
        pitched_data: Pitch track of the stem the note pitches come from.
        model_path: The model file (see ``tools/train_pitch_context.py``).
        strength: "balanced" or "strong" (see ``STRENGTHS``).

    Fails open: without a usable model file or pitch track, and on any error,
    the notes stay as they are and the reason is printed.
    """
    def skipped(reason: str) -> PitchContextResult:
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} pitch context not applied: {reason} "
              f"- the note pitches stay as they are")
        return PitchContextResult(False, reason=reason)

    if strength not in STRENGTHS:
        return skipped(f"unknown strength '{strength}'")
    if not model_path:
        return skipped("no model file is set (--pitch_context_model)")
    if not os.path.isfile(model_path):
        return skipped(f"model file not found: {model_path}")
    if pitched_data is None or len(pitched_data.times) == 0:
        return skipped("no pitch track of the vocal")
    try:
        import librosa

        model = _get_model(model_path)
        notes = [segment for segment in midi_segments if segment.note_type in PITCHED_TYPES]
        notes.sort(key=lambda segment: segment.start)
        pitches = [int(librosa.note_to_midi(segment.note)) for segment in notes]
        shifts = predict_shifts(
            model, [segment.start * 1000.0 for segment in notes], [segment.end * 1000.0 for segment in notes],
            pitches, PitchTrack.from_pitched_data(pitched_data), strength)
        # (the new names first: no note is changed when one of them fails)
        moved = [(segment, librosa.midi_to_note(pitch + int(shift)))
                 for segment, pitch, shift in zip(notes, pitches, shifts) if shift]
    except Exception as e:  # noqa: BLE001 - an improvement, never a requirement
        return skipped(f"{e!r}")
    for segment, note in moved:
        segment.note = note
    up = sum(1 for shift in shifts if shift > 0)
    print(f"{ULTRASINGER_HEAD} Pitch context ({strength}, model {blue_highlighted(os.path.basename(model_path))}): "
          f"{blue_highlighted(str(len(moved)))} of {len(notes)} note pitches moved "
          f"({up} up, {len(moved) - up} down)")
    return PitchContextResult(True, notes=len(notes), moved=len(moved))
