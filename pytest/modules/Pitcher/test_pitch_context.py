"""Tests for the pitch context step: note features, the model file, the decision and the pipeline step."""

import contextlib
import io
import os
import pickle
import struct
import tracemalloc
import zipfile

import librosa
import numpy as np
import pytest

from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher import pitch_context
from modules.Pitcher.pitch_context import (
    FEATURE_NAMES,
    N_FEATURES,
    N_OUTPUTS,
    SHIFTS,
    STRENGTHS,
    PitchContextModel,
    PitchTrack,
    choose_shifts,
    correct_note_pitches,
    load_model,
    note_features,
    predict_shifts,
    track_pitch,
)
from modules.Pitcher.pitched_data import PitchedData

FRAME_S = 0.016


def _column(name: str) -> int:
    return FEATURE_NAMES.index(name)


def _pitched_data(spans, length=4.0) -> PitchedData:
    """A pitch track with a frame every 16 ms: the (start s, end s, MIDI pitch) spans are sung."""
    times, frequencies, confidence = [], [], []
    for i in range(int(length / FRAME_S) + 1):
        t = i * FRAME_S
        hz, conf = 0.0, 0.0
        for start, end, midi in spans:
            if start <= t < end:
                hz, conf = float(librosa.midi_to_hz(midi)), 0.9
        times.append(t)
        frequencies.append(hz)
        confidence.append(conf)
    return PitchedData(times, frequencies, confidence)


def _track(spans, length=4.0) -> PitchTrack:
    return PitchTrack.from_pitched_data(_pitched_data(spans, length))


def _toy_model(up: float = 1.0, feature: str = "frames_p50", threshold: float = 0.3) -> PitchContextModel:
    """One tree per output. Staying promises 0.5 (within one semitone); one semitone up
    promises ``up`` for a note whose frames lie more than ``threshold`` above its pitch."""
    nodes, roots = [], []
    for output in range(N_OUTPUTS):
        roots.append(len(nodes))
        if output == SHIFTS.index(1):
            first = len(nodes)
            nodes += [(_column(feature), threshold, first + 1, first + 2, 0.0),
                      (-1, 0.0, 0, 0, 0.0), (-1, 0.0, 0, 0, up)]
        else:
            nodes.append((-1, 0.0, 0, 0, 0.5 if output == SHIFTS.index(0) else 0.0))
    feature_index, thresholds, left, right, value = zip(*nodes)
    return PitchContextModel(feature=feature_index, threshold=thresholds, left=left, right=right,
                             missing_left=[False] * len(nodes), value=value, tree_root=roots,
                             tree_output=list(range(N_OUTPUTS)), baseline=np.zeros(N_OUTPUTS))


def _model_file(tmp_path, **kwargs) -> str:
    path = tmp_path / "pitch_context.npz"
    _toy_model(**kwargs).save(path)
    return str(path)


def _midi(segment: MidiSegment) -> int:
    return int(librosa.note_to_midi(segment.note))


def _quiet(function, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return function(*args, **kwargs), out.getvalue()


# ── Pitch track ───────────────────────────────────────────────────────────


class TestPitchTrack:
    def test_only_confident_frames_with_a_pitch_count(self):
        track = PitchTrack([0.0, 0.016, 0.032, 0.048, 0.064], [440.0, 440.0, 0.0, float("nan"), 880.0],
                           [0.9, 0.5, 0.9, 0.9, 0.7])
        assert track.times_ms.tolist() == [0.0, 64.0]
        assert track.midi.tolist() == [69.0, 81.0]
        assert track.frame_ms == pytest.approx(16.0)

    def test_frames_of_a_span(self):
        track = _track([(0.0, 1.0, 60.0)])
        assert len(track.frames(0.0, 160.0)) == 10
        assert len(track.frames(2000.0, 3000.0)) == 0

    def test_an_empty_track(self):
        track = PitchTrack([], [], [])
        assert len(track.frames(0.0, 1000.0)) == 0
        assert track.frame_ms == pitch_context.DEFAULT_FRAME_MS

    def test_the_note_pitch_is_the_upper_middle_of_the_frames(self):
        frames = np.array([59.0] * 4 + [60.0] * 6)
        assert track_pitch(frames) == 60
        assert track_pitch(np.array([60.4])) is None
        assert track_pitch(np.array([])) is None

    def test_the_conversion_pitches_its_notes_the_same_way(self):
        from modules.Segmentation.decode import NOTE_PITCH_PERCENTILE
        from modules.Segmentation.features import VOICED_CONFIDENCE

        assert pitch_context.NOTE_PITCH_PERCENTILE == NOTE_PITCH_PERCENTILE
        assert pitch_context.VOICED_CONFIDENCE == VOICED_CONFIDENCE


# ── Features ──────────────────────────────────────────────────────────────


class TestNoteFeatures:
    def test_one_column_per_feature_name(self):
        assert N_FEATURES == len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 40
        features, judged = note_features([0.0], [500.0], [60], _track([(0.0, 0.5, 60.0)]))
        assert features.shape == (1, N_FEATURES) and features.dtype == np.float32
        assert judged.tolist() == [True]

    def test_the_frames_are_measured_from_the_note_pitch(self):
        features, _ = note_features([0.0], [1000.0], [60], _track([(0.0, 1.0, 60.4)]))
        row = features[0]
        for name in ("frames_p10", "frames_p50", "frames_p60", "frames_p90", "frames_mean"):
            assert row[_column(name)] == pytest.approx(0.4, abs=1e-3), name
        assert row[_column("frames_spread")] == pytest.approx(0.0, abs=1e-3)
        assert row[_column("frames_count")] == 63
        assert row[_column("frames_share")] == pytest.approx(1.0, abs=0.02)
        assert [row[_column(f"frames_near_{d:+d}")] for d in SHIFTS] == [0, 0, 1, 0, 0]
        assert row[_column("frames_slope")] == pytest.approx(0.0, abs=1e-4)
        assert row[_column("duration_ms")] == 1000.0
        assert row[_column("duration_log")] == pytest.approx(np.log(1000.0))

    def test_frames_near_the_neighbouring_semitone(self):
        features, _ = note_features([0.0], [1000.0], [60], _track([(0.0, 0.5, 60.1), (0.5, 1.0, 60.9)]))
        row = features[0]
        assert row[_column("frames_near_+0")] == pytest.approx(0.5, abs=0.02)
        assert row[_column("frames_near_+1")] == pytest.approx(0.5, abs=0.02)
        assert row[_column("frames_slope")] > 0

    def test_a_note_in_another_octave_than_its_frames(self):
        """The octave steps of the conversion may have moved the note: the frames are
        compared in the note's octave."""
        low, _ = note_features([0.0], [1000.0], [60], _track([(0.0, 1.0, 60.4)]))
        high, _ = note_features([0.0], [1000.0], [72], _track([(0.0, 1.0, 60.4)]))
        frame_columns = slice(0, _column("duration_ms"))
        assert np.allclose(low[0, frame_columns], high[0, frame_columns], atol=1e-5)

    def test_the_notes_around_a_note(self):
        starts, ends, pitches = [0.0, 500.0, 1000.0, 5000.0], [400.0, 1000.0, 1300.0, 5200.0], [60, 62, 79, 60]
        features, _ = note_features(starts, ends, pitches, _track([(0.0, 5.2, 60.0)], length=6.0))
        first, second, third = features[0], features[1], features[2]
        # nothing before the first note
        assert np.isnan(first[_column("note_-1_pitch")]) and np.isnan(first[_column("note_-2_gap_ms")])
        assert first[_column("note_+1_pitch")] == 2 and first[_column("note_+1_gap_ms")] == 100
        assert first[_column("note_+2_pitch")] == 12  # 19 semitones, limited to an octave
        assert second[_column("note_-1_pitch")] == -2 and second[_column("note_-1_gap_ms")] == 100
        assert second[_column("note_+1_gap_ms")] == 0
        assert third[_column("note_+1_gap_ms")] == 2000  # 3.7 s, limited
        assert np.isnan(features[3][_column("note_+1_pitch")])

    def test_the_song_as_a_whole(self):
        # C major, twice as long on C as on each other note, sung a quarter tone sharp
        scale = [60, 62, 64, 65, 67, 69, 71]
        starts = [i * 1000.0 for i in range(8)]
        ends = [start + 500.0 for start in starts]
        pitches = [60] + scale
        spans = [(start / 1000, end / 1000, pitch + 0.25) for start, end, pitch in zip(starts, ends, pitches)]
        features, _ = note_features(starts, ends, pitches, _track(spans, length=9.0))
        row = features[0]
        assert row[_column("song_share_+0")] == pytest.approx(2 / 8)
        assert row[_column("song_share_+2")] == pytest.approx(1 / 8)
        assert row[_column("song_share_+1")] == 0
        assert [row[_column(f"in_scale_{d:+d}")] for d in SHIFTS] == [0, 1, 1, 0, 1]
        assert row[_column("scale_fit")] == pytest.approx(1.0)
        assert row[_column("scale_lead")] > 0
        assert row[_column("scale_step")] == 0
        assert features[2][_column("scale_step")] == 2  # D in C major
        assert row[_column("tuning")] == pytest.approx(0.25, abs=1e-3)

    def test_a_note_without_pitch_frames_is_not_judged(self):
        features, judged = note_features([0.0, 2000.0], [500.0, 2500.0], [60, 62], _track([(0.0, 0.5, 60.0)]))
        assert judged.tolist() == [True, False]
        assert np.isnan(features[1]).all()
        # one frame is not enough either
        _, judged = note_features([0.0, 2000.0], [500.0, 2500.0], [60, 62],
                                  _track([(0.0, 0.5, 60.0), (2.0, 2.01, 62.0)]))
        assert judged.tolist() == [True, False]

    def test_a_note_without_pitch_frames_does_not_describe_the_others(self):
        """Its pitch is from elsewhere (another stem, or the note before it): it is no
        neighbour and no part of the song. Training never sees such a note."""
        starts = [0.0, 600.0, 1200.0, 1800.0, 2400.0]
        ends = [500.0, 1100.0, 1700.0, 2300.0, 2900.0]
        pitches = [60, 66, 62, 61, 64]  # the second and the fourth are not sung in the track
        track = _track([(0.0, 0.5, 60.2), (1.2, 1.7, 62.0), (2.4, 2.9, 63.8)])
        features, judged = note_features(starts, ends, pitches, track)
        assert judged.tolist() == [True, False, True, False, True]
        first, third, fifth = features[0], features[2], features[4]
        assert first[_column("note_+1_pitch")] == 2 and first[_column("note_+1_gap_ms")] == 700
        assert first[_column("note_+2_pitch")] == 4
        assert third[_column("note_-1_pitch")] == -2 and third[_column("note_+1_pitch")] == 2
        assert np.isnan(third[_column("note_-2_pitch")]) and np.isnan(third[_column("note_+2_pitch")])
        assert fifth[_column("note_-1_gap_ms")] == 700 and np.isnan(fifth[_column("note_+1_pitch")])
        # the song is C, D and E, a third of its time each: no F sharp, no C sharp
        assert first[_column("song_share_+0")] == pytest.approx(1 / 3)
        assert first[_column("song_share_+1")] == 0
        assert third[_column("song_share_-1")] == 0 and third[_column("song_share_-2")] == pytest.approx(1 / 3)

    def test_the_notes_with_pitch_frames_get_the_same_features_alone(self):
        """What the training does (it leaves the other notes out before it asks for the
        features) and what the conversion does (it hands in every pitched note) is the same."""
        rng = np.random.default_rng(3)
        starts = np.arange(40) * 500.0
        ends = starts + rng.integers(150, 450, 40)
        pitches = rng.integers(55, 70, 40)
        sung = rng.random(40) > 0.3
        track = _track([(a / 1000, b / 1000, p + rng.normal(0, 0.3)) for a, b, p, s in
                        zip(starts, ends, pitches, sung) if s], length=21.0)
        features, judged = note_features(starts, ends, pitches, track)
        assert judged.tolist() == sung.tolist() and 5 < (~judged).sum() < 20
        alone, all_judged = note_features(starts[judged], ends[judged], pitches[judged], track)
        assert all_judged.all()
        assert np.array_equal(features[judged], alone, equal_nan=True)
        # and whatever pitch the other notes have
        other = np.where(judged, pitches, pitches + 5)
        assert np.array_equal(note_features(starts, ends, other, track)[0], features, equal_nan=True)

    def test_no_notes(self):
        features, judged = note_features([], [], [], _track([]))
        assert features.shape == (0, N_FEATURES) and len(judged) == 0

    def test_the_same_notes_give_the_same_features(self):
        args = ([0.0, 600.0], [500.0, 900.0], [60, 64], _track([(0.0, 0.5, 60.3), (0.6, 0.9, 63.8)]))
        first, _ = note_features(*args)
        again, _ = note_features(*args)
        assert np.array_equal(first, again, equal_nan=True)


# ── Model ─────────────────────────────────────────────────────────────────


class TestModel:
    def test_the_trees_are_walked(self):
        model = _toy_model(up=0.8)
        features = np.zeros((3, N_FEATURES), dtype=np.float32)
        features[:, _column("frames_p50")] = [0.1, 0.5, np.nan]
        within_one, exact = model.predict(features)
        assert within_one.shape == exact.shape == (3, len(SHIFTS))
        assert within_one[:, SHIFTS.index(0)].tolist() == [0.5, 0.5, 0.5]
        # a missing feature goes where the node says: to the right here
        assert within_one[:, SHIFTS.index(1)].tolist() == [0.0, 0.8, 0.8]
        assert not exact.any()

    def test_a_missing_feature_can_go_left(self):
        model = _toy_model()
        model.missing_left[:] = True
        features = np.full((1, N_FEATURES), np.nan, dtype=np.float32)
        assert model.predict(features)[0][0, SHIFTS.index(1)] == 0.0

    def test_the_baseline_and_several_trees_add_up(self):
        model = _toy_model()
        twice = PitchContextModel(
            feature=np.tile(model.feature, 2), threshold=np.tile(model.threshold, 2),
            left=np.concatenate([model.left, np.where(model.feature >= 0, model.left + len(model.feature), 0)]),
            right=np.concatenate([model.right, np.where(model.feature >= 0, model.right + len(model.feature), 0)]),
            missing_left=np.tile(model.missing_left, 2), value=np.tile(model.value, 2),
            tree_root=np.concatenate([model.tree_root, model.tree_root + len(model.feature)]),
            tree_output=np.tile(model.tree_output, 2), baseline=np.full(N_OUTPUTS, 0.25))
        features = np.zeros((1, N_FEATURES), dtype=np.float32)
        features[0, _column("frames_p50")] = 0.5
        within_one, exact = twice.predict(features)
        assert within_one[0, SHIFTS.index(1)] == 2.25 and within_one[0, SHIFTS.index(0)] == 1.25
        assert exact[0].tolist() == [0.25] * len(SHIFTS)

    def test_many_notes_are_predicted_like_few(self):
        model = _toy_model()
        features = np.zeros((1000, N_FEATURES), dtype=np.float32)
        features[::2, _column("frames_p50")] = 0.5
        within_one, _ = model.predict(features)
        assert within_one[:, SHIFTS.index(1)].tolist() == [1.0, 0.0] * 500

    def test_features_of_another_shape_are_refused(self):
        with pytest.raises(ValueError):
            _toy_model().predict(np.zeros((2, N_FEATURES - 1)))

    def test_save_and_load(self, tmp_path):
        model = _toy_model(up=0.7)
        model.info = {"songs": 3, "notes": np.int64(120), "held_out": {"strong": {"exact_pct": np.float64(80.5)}}}
        path = tmp_path / "model.npz"
        model.save(path)
        assert os.listdir(tmp_path) == ["model.npz"]  # no temporary file is left
        loaded = load_model(path)
        for name in ("feature", "threshold", "left", "right", "missing_left", "value", "tree_root",
                     "tree_output", "baseline"):
            assert np.array_equal(getattr(loaded, name), getattr(model, name)), name
        assert loaded.info == {"songs": 3, "notes": 120, "held_out": {"strong": {"exact_pct": 80.5}}}
        features = np.random.default_rng(0).normal(size=(50, N_FEATURES)).astype(np.float32)
        assert np.array_equal(np.hstack(loaded.predict(features)), np.hstack(model.predict(features)))

    def test_the_file_holds_no_pickle(self, tmp_path):
        path = tmp_path / "model.npz"
        _toy_model().save(path)
        with np.load(path, allow_pickle=False) as data:
            for name in data.files:
                assert data[name].dtype.kind in "iufbU", name


class _Marker:
    """Unpickling this creates a file: proof that a pickle was executed."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (open, (self.path, "w"))


class TestFilesThatAreNoModel:
    def _refused(self, path):
        with pytest.raises(ValueError, match="not a pitch context model file"):
            load_model(path)

    def test_a_pickle_is_refused_and_not_executed(self, tmp_path):
        marker = tmp_path / "executed"
        path = tmp_path / "model.npz"
        path.write_bytes(pickle.dumps({"model": _Marker(str(marker))}))
        self._refused(path)
        assert not marker.exists()

    def test_an_array_file_with_a_pickle_is_refused_and_not_executed(self, tmp_path):
        marker = tmp_path / "executed"
        path = tmp_path / "model.npz"
        with open(path, "wb") as f:
            np.save(f, np.array([_Marker(str(marker))], dtype=object), allow_pickle=True)
        self._refused(path)
        assert not marker.exists()

    def test_an_archive_with_a_pickled_array_is_refused_and_not_executed(self, tmp_path):
        marker = tmp_path / "executed"
        good = tmp_path / "good.npz"
        _toy_model().save(good)
        with np.load(good, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["value"] = np.array([_Marker(str(marker))] * len(arrays["value"]), dtype=object)
        path = tmp_path / "model.npz"
        with open(path, "wb") as f:
            np.savez(f, **arrays)
        self._refused(path)
        assert not marker.exists()

    def test_other_files_are_refused(self, tmp_path):
        text = tmp_path / "text.npz"
        text.write_text("not a model", encoding="utf-8")
        empty = tmp_path / "empty.npz"
        empty.write_bytes(b"")
        array = tmp_path / "array.npz"
        with open(array, "wb") as f:
            np.save(f, np.zeros(3))
        archive = tmp_path / "archive.npz"  # e.g. a segmentation model: a zip archive too
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("archive/data.pkl", b"\x80\x02}q\x00.")
        for path in (text, empty, array, archive, tmp_path / "missing.npz", tmp_path):
            self._refused(path)

    def _changed(self, tmp_path, **changes):
        good = tmp_path / "good.npz"
        _toy_model().save(good)
        with np.load(good, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        for name, value in changes.items():
            if value is None:
                del arrays[name]
            else:
                arrays[name] = value(arrays[name]) if callable(value) else value
        path = tmp_path / "changed.npz"
        with open(path, "wb") as f:
            np.savez(f, **arrays)
        return path

    def test_a_model_of_another_kind_or_format_is_refused(self, tmp_path):
        self._refused(self._changed(tmp_path, kind=np.array("something-else")))
        self._refused(self._changed(tmp_path, format=np.array(99)))
        self._refused(self._changed(tmp_path, threshold=None))

    def test_a_model_for_other_features_is_refused(self, tmp_path):
        other = self._changed(tmp_path, features=lambda names: names[::-1])
        with pytest.raises(ValueError, match="other note features"):
            load_model(other)
        self._refused(self._changed(tmp_path, shifts=np.array([-1, 0, 1])))

    def test_arrays_that_are_no_trees_are_refused(self, tmp_path):
        def root_points_to_itself(left):
            left = left.copy()
            root = np.flatnonzero(left)[0]  # the one node of the toy model that has children
            left[root] = root
            return left

        for changes in (
                {"left": root_points_to_itself},                               # walking it would never end
                {"right": lambda right: np.where(right > 0, 10_000, 0)},       # leads outside
                {"feature": lambda feature: np.where(feature >= 0, 999, -1)},  # no such feature
                {"tree_output": lambda outputs: outputs + 1},                  # no such output
                {"tree_root": lambda roots: roots + 10_000},
                {"value": lambda value: value[:-1]},                           # arrays of different length
                {"threshold": lambda threshold: np.full(len(threshold), np.nan)},
                {"value": lambda value: np.full(len(value), np.inf)},
                {"baseline": np.zeros(3)},
                {"feature": lambda feature: feature.astype(np.float64)},       # wrong type
                {"threshold": lambda threshold: threshold.reshape(-1, 1)},
        ):
            self._refused(self._changed(tmp_path, **changes))


def _array_file(descr: str, shape: tuple, data: bytes, version: int = 1) -> bytes:
    """The bytes of an array file whose header names ``shape``, whatever its data is."""
    out = io.BytesIO()
    write_header = np.lib.format.write_array_header_1_0 if version == 1 else np.lib.format.write_array_header_2_0
    write_header(out, {"descr": descr, "fortran_order": False, "shape": shape})
    return out.getvalue() + data


class TestCraftedModelFiles:
    """A model file is read from a path a user was given: whatever it holds, the
    loader says "no model" and asks for no more memory than the file is large."""

    @staticmethod
    def _crafted(tmp_path, **members) -> str:
        """A model archive with the given members (name -> bytes) put in place of its own."""
        good = tmp_path / "good.npz"
        _toy_model().save(good)
        with zipfile.ZipFile(good) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        files.update({f"{name}.npy": data for name, data in members.items()})
        path = tmp_path / "crafted.npz"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        return str(path)

    @staticmethod
    def _refused(path) -> float:
        """Loading ``path`` raises ValueError; returns the most memory it held meanwhile (MB)."""
        tracemalloc.start()
        try:
            with pytest.raises(ValueError, match="not a pitch context model file"):
                load_model(path)
            return tracemalloc.get_traced_memory()[1] / 1e6
        finally:
            tracemalloc.stop()

    def test_the_helper_makes_a_model_that_loads(self, tmp_path):
        nodes = len(_toy_model().value)
        path = self._crafted(tmp_path, value=_array_file("<f8", (nodes,), _toy_model().value.tobytes()))
        assert np.array_equal(load_model(path).value, _toy_model().value)

    @pytest.mark.parametrize("method", ["ZIP_BZIP2", "ZIP_LZMA"])
    def test_a_member_packed_in_another_way_is_not_unpacked(self, tmp_path, method):
        """Python unpacks a chunk of such a member at once, whatever its size: a few
        hundred bytes of bzip2 stand for hundreds of megabytes of zeros."""
        good = tmp_path / "good.npz"
        _toy_model().save(good)
        with zipfile.ZipFile(good) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        header = _array_file("<U1", (), b"")[:128]
        files["kind.npy"] = header + bytes(200 * 1024 * 1024)
        path = tmp_path / "packed.npz"
        try:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, data in files.items():
                    archive.writestr(name, data, getattr(zipfile, method) if name == "kind.npy" else None)
        except (RuntimeError, NotImplementedError):
            pytest.skip(f"{method} is not available here")
        assert path.stat().st_size < 1024 * 1024
        assert self._refused(str(path)) < 20

    def test_a_header_that_names_more_values_than_the_file_holds(self, tmp_path):
        """numpy makes the array a header names before it reads the data: 2.4 GB here,
        and for the larger one more memory than there is (a MemoryError)."""
        for shape in ((3 * 10 ** 8,), (10 ** 11,), (10 ** 6, 10 ** 6)):
            path = self._crafted(tmp_path, threshold=_array_file("<f8", shape, b"\0" * 64))
            assert self._refused(path) < 50, shape

    def test_no_array_is_trusted_by_its_header(self, tmp_path):
        huge = {
            "kind": _array_file("<U1000000000", (), b"\0" * 64),
            "format": _array_file("<i8", (10 ** 10,), b"\0" * 64),
            "features": _array_file("<U14", (10 ** 9,), b"\0" * 56),
            "shifts": _array_file("<i8", (10 ** 10,), b"\0" * 40),
            "info": _array_file("<U2", (10 ** 10,), b"\0" * 8),
            "missing_left": _array_file("|b1", (10 ** 12,), b"\0" * 12),
            "tree_root": _array_file("<i4", (10 ** 11,), b"\0" * 40),
        }
        for name, data in huge.items():
            assert self._refused(self._crafted(tmp_path, **{name: data})) < 50, name

    def test_an_array_with_other_data_than_its_header_names(self, tmp_path):
        value = _toy_model().value
        for data in (value.tobytes() + b"\0" * 8,          # more data than values
                     value.tobytes()[:-8],                 # less
                     b""):
            self._refused(self._crafted(tmp_path, value=_array_file("<f8", (len(value),), data)))
        # nothing of a size of zero, and no size below zero
        self._refused(self._crafted(tmp_path, value=_array_file("<V0", (len(value),), b"")))
        self._refused(self._crafted(tmp_path, value=_array_file("<f8", (-1,), b"")))

    def test_an_array_of_another_kind_or_shape(self, tmp_path):
        value = _toy_model().value
        for name, data in (
                ("value", _array_file("<i8", (len(value),), value.astype("<i8").tobytes())),
                ("kind", _array_file("<i8", (), b"\0" * 8)),
                ("format", _array_file("<f8", (), b"\0" * 8)),
                ("format", _array_file("<i8", (1,), np.array([1], "<i8").tobytes())),
                ("info", _array_file("<f8", (), b"\0" * 8)),
                ("info", _array_file("|O", (), b"\0" * 8)),
                ("value", b"not an array file at all"),
                ("value", b"\x93NUMPY\x03\x00" + _array_file("<f8", (len(value),), value.tobytes())[8:])):
            self._refused(self._crafted(tmp_path, **{name: data}))

    def test_a_long_or_deeply_nested_info(self, tmp_path, monkeypatch):
        def info(text):
            return _array_file(f"<U{len(text)}", (), np.array(text).tobytes())

        # far longer than what a training writes
        self._refused(self._crafted(tmp_path, info=info("[" * 100_000 + "]" * 100_000)))
        self._refused(self._crafted(tmp_path, info=info(" " * 70_000 + "{}")))
        # short enough to be read: it is refused, or (where the JSON reader gets through
        # the nesting) it is no description of a model and left out; no other error
        nested = self._crafted(tmp_path, info=info("[" * 30_000 + "]" * 30_000))
        try:
            assert load_model(nested).info == {}
        except ValueError as error:
            assert "not a pitch context model file" in str(error)
        self._refused(self._crafted(tmp_path, info=info("{no json")))
        # whatever the JSON reader raises
        for error in (RecursionError("maximum recursion depth exceeded"), MemoryError()):
            def fail(text, error=error):
                raise error

            monkeypatch.setattr(pitch_context.json, "loads", fail)
            self._refused(self._crafted(tmp_path, info=info("{}")))

    def test_a_damaged_archive(self, tmp_path):
        good = tmp_path / "good.npz"
        _toy_model().save(good)
        data = good.read_bytes()
        cut = tmp_path / "cut.npz"
        cut.write_bytes(data[:len(data) // 2])
        self._refused(cut)
        # a member whose packed data is damaged (the archive's list of members is whole)
        with zipfile.ZipFile(good) as archive:
            entry = archive.getinfo("value.npy")
        name_length, extra_length = struct.unpack("<HH", data[entry.header_offset + 26:entry.header_offset + 30])
        packed = entry.header_offset + 30 + name_length + extra_length
        damaged = bytearray(data)
        damaged[packed:packed + entry.compress_size] = b"\xff" * entry.compress_size
        broken = tmp_path / "broken.npz"
        broken.write_bytes(bytes(damaged))
        self._refused(broken)

    def test_a_model_of_another_byte_order_or_header_version(self, tmp_path):
        """Not crafted: a model written on another machine or by another numpy."""
        model = _toy_model(up=0.7)
        path = self._crafted(
            tmp_path,
            value=_array_file(">f8", (len(model.value),), model.value.astype(">f8").tobytes()),
            left=_array_file(">i4", (len(model.left),), model.left.astype(">i4").tobytes()),
            threshold=_array_file("<f8", (len(model.threshold),), model.threshold.tobytes(), version=2))
        loaded = load_model(path)
        features = np.random.default_rng(1).normal(size=(20, N_FEATURES)).astype(np.float32)
        assert np.array_equal(np.hstack(loaded.predict(features)), np.hstack(model.predict(features)))
        assert loaded.value.dtype == np.float64 and loaded.value.dtype.isnative


# ── Decision ──────────────────────────────────────────────────────────────


def _promises(**shifts):
    row = np.zeros((1, len(SHIFTS)))
    for name, value in shifts.items():
        row[0, SHIFTS.index({"down2": -2, "down": -1, "stay": 0, "up": 1, "up2": 2}[name])] = value
    return row


class TestChooseShifts:
    def test_the_strengths(self):
        assert STRENGTHS == {"balanced": (0.3, 0.1), "strong": (1.0, 0.0)}
        assert pitch_context.STRENGTH_CHOICES == ("off", "balanced", "strong")

    def test_a_note_takes_the_shift_that_promises_most(self):
        within_one = _promises(stay=0.5, up=0.9, down2=0.7)
        nothing = np.zeros_like(within_one)
        assert choose_shifts(within_one, nothing, "strong").tolist() == [1]
        assert choose_shifts(within_one, nothing, "balanced").tolist() == [1]

    def test_staying_wins_when_nothing_promises_more(self):
        within_one = _promises(stay=0.9, up=0.8, down=0.2)
        assert choose_shifts(within_one, np.zeros_like(within_one), "strong").tolist() == [0]

    def test_a_shift_that_promises_the_same_does_not_move_the_note(self):
        within_one = _promises(stay=0.8, down2=0.8, up=0.8)
        assert choose_shifts(within_one, np.zeros_like(within_one), "strong").tolist() == [0]

    def test_balanced_needs_a_margin(self):
        nothing = np.zeros((1, len(SHIFTS)))
        little = _promises(stay=0.80, down=0.85)
        assert choose_shifts(little, nothing, "strong").tolist() == [-1]
        assert choose_shifts(little, nothing, "balanced").tolist() == [0]
        assert choose_shifts(_promises(stay=0.80, down=0.95), nothing, "balanced").tolist() == [-1]

    def test_the_exact_pitch_counts_more_for_strong(self):
        # two semitones up is a little less often within one semitone, but much more often exact
        within_one = _promises(stay=0.90, up2=0.85)
        exact = _promises(stay=0.30, up2=0.55)
        assert choose_shifts(within_one, exact, "strong").tolist() == [2]      # -0.05 + 1.0 * 0.25
        assert choose_shifts(within_one, exact, "balanced").tolist() == [0]    # -0.05 + 0.3 * 0.25 < 0.1

    def test_each_note_is_decided_on_its_own(self):
        within_one = np.vstack([_promises(stay=0.5, up=0.9), _promises(stay=0.9), _promises(stay=0.1, down2=0.6)])
        assert choose_shifts(within_one, np.zeros_like(within_one), "strong").tolist() == [1, 0, -2]

    def test_an_unknown_strength(self):
        with pytest.raises(KeyError):
            choose_shifts(_promises(stay=1.0), _promises(stay=1.0), "off")

    def test_a_note_without_pitch_frames_stays(self):
        # the toy model sends a missing feature to "one semitone up"
        track = _track([(0.0, 0.5, 60.5)])
        shifts = predict_shifts(_toy_model(), [0.0, 2000.0], [500.0, 2500.0], [60, 60], track, "strong")
        assert shifts.tolist() == [1, 0]

    def test_a_note_that_is_not_at_the_pitch_of_its_frames_stays(self):
        """Another step put it there (the pitch refinement moves a note to the tone the
        games' pitch detection hears): the model has seen no such note."""
        assert pitch_context.MAX_FRAME_DISTANCE == 1.5
        # sung at 61.4, 61.6 and 63: 1.4, 1.6 and 3 semitones above the note; the toy
        # model promises most for one semitone up in all three
        track = _track([(0.0, 0.5, 61.4), (1.0, 1.5, 61.6), (2.0, 2.5, 63.0)])
        starts, ends = [0.0, 1000.0, 2000.0], [500.0, 1500.0, 2500.0]
        assert predict_shifts(_toy_model(), starts, ends, [60, 60, 60], track, "strong").tolist() == [1, 0, 0]
        # octaves aside: a note charted an octave above its frames is at their pitch
        assert predict_shifts(_toy_model(), starts, ends, [72, 72, 72], track, "strong").tolist() == [1, 0, 0]

    def test_the_same_notes_get_the_same_shifts(self):
        track = _track([(0.0, 0.5, 60.5), (0.6, 0.9, 62.0)])
        args = (_toy_model(), [0.0, 600.0], [500.0, 900.0], [60, 62], track, "strong")
        assert predict_shifts(*args).tolist() == predict_shifts(*args).tolist() == [1, 0]


# ── Pipeline step ─────────────────────────────────────────────────────────


def _notes():
    """Three notes sung half a semitone sharp (they move up), one sung on its pitch (it stays)."""
    segments = [MidiSegment("C4", 0.0, 0.5, "one "), MidiSegment("D4", 0.6, 1.0, "two "),
                MidiSegment("E4", 1.2, 1.6, "three "), MidiSegment("B3", 2.0, 2.5, "four ")]
    sung = _pitched_data([(0.0, 0.5, 60.5), (0.6, 1.0, 62.0), (1.2, 1.6, 64.5), (2.0, 2.5, 59.5)])
    return segments, sung


class TestCorrectNotePitches:
    def test_notes_move_in_place(self, tmp_path):
        segments, sung = _notes()
        result, out = _quiet(correct_note_pitches, segments, sung, _model_file(tmp_path), "strong")
        assert [_midi(s) for s in segments] == [61, 62, 65, 60]
        assert [s.word for s in segments] == ["one ", "two ", "three ", "four "]
        assert (result.applied, result.notes, result.moved) == (True, 4, 3)
        assert str(result) == "3 of 4 note pitches moved"
        assert "of 4 note pitches moved (3 up, 0 down)" in out and "Pitch context (strong" in out

    def test_the_strengths_differ(self, tmp_path):
        model = _model_file(tmp_path, up=0.55)  # promises 0.05 more than staying
        segments, sung = _notes()
        result, _ = _quiet(correct_note_pitches, segments, sung, model, "balanced")
        assert (result.applied, result.moved) == (True, 0)
        assert [_midi(s) for s in segments] == [60, 62, 64, 59]
        result, _ = _quiet(correct_note_pitches, segments, sung, model, "strong")
        assert result.moved == 3

    def test_freestyle_and_rap_notes_are_never_touched(self, tmp_path):
        segments, sung = _notes()
        segments[0].note_type = "F"
        segments[2].note_type = "R"
        result, _ = _quiet(correct_note_pitches, segments, sung, _model_file(tmp_path), "strong")
        assert [_midi(s) for s in segments] == [60, 62, 64, 60]
        assert [s.note_type for s in segments] == ["F", ":", "R", ":"]
        assert (result.notes, result.moved) == (2, 1)

    def test_freestyle_notes_are_no_neighbours(self, tmp_path):
        # the note moves when the note before it is two semitones below; a freestyle note in between is skipped
        model = _model_file(tmp_path, feature="note_-1_pitch", threshold=-1.5)
        segments = [MidiSegment("C4", 0.0, 0.5, "one "), MidiSegment("A4", 0.6, 1.0, "two ", note_type="F"),
                    MidiSegment("D4", 1.2, 1.6, "three ")]
        sung = _pitched_data([(0.0, 0.5, 60.0), (0.6, 1.0, 69.0), (1.2, 1.6, 62.0)])
        _quiet(correct_note_pitches, segments, sung, model, "strong")
        # C4 has no note before it (missing: moves); D4 has C4 two below (-2 is not above -1.5: stays)
        assert [_midi(s) for s in segments] == [61, 69, 62]

    def test_notes_without_pitch_frames_are_no_neighbours_and_stay(self, tmp_path):
        # as above, with a pitched note in between that the pitch track has nothing for
        # (its pitch is from another stem, or continues the note before it)
        model = _model_file(tmp_path, feature="note_-1_pitch", threshold=-1.5)
        segments = [MidiSegment("C4", 0.0, 0.5, "one "), MidiSegment("A4", 0.6, 1.0, "two "),
                    MidiSegment("D4", 1.2, 1.6, "three ")]
        sung = _pitched_data([(0.0, 0.5, 60.0), (1.2, 1.6, 62.0)])
        result, _ = _quiet(correct_note_pitches, segments, sung, model, "strong")
        # with A4 as the note before it, D4 would move
        assert [_midi(s) for s in segments] == [61, 69, 62]
        assert (result.notes, result.moved) == (3, 1)

    def test_golden_notes_move_like_other_notes(self, tmp_path):
        segments, sung = _notes()
        segments[0].note_type = "*"
        _quiet(correct_note_pitches, segments, sung, _model_file(tmp_path), "strong")
        assert _midi(segments[0]) == 61 and segments[0].note_type == "*"

    def test_a_note_keeps_its_octave(self, tmp_path):
        # charted an octave above the singing (an octave step of the conversion moved it there)
        segments = [MidiSegment("C5", 0.0, 0.5, "one "), MidiSegment("C5", 0.6, 1.0, "two ")]
        sung = _pitched_data([(0.0, 0.5, 60.5), (0.6, 1.0, 60.0)])
        _quiet(correct_note_pitches, segments, sung, _model_file(tmp_path), "strong")
        assert [_midi(s) for s in segments] == [73, 72]

    def test_a_note_the_refinement_moved_away_from_its_frames_stays(self, tmp_path):
        segments, sung = _notes()
        segments[0].note = "G4"  # sung around C4: seven semitones from its frames
        result, _ = _quiet(correct_note_pitches, segments, sung, _model_file(tmp_path), "strong")
        assert [_midi(s) for s in segments] == [67, 62, 65, 60]
        assert (result.notes, result.moved) == (4, 2)

    def test_the_order_of_the_notes_does_not_matter(self, tmp_path):
        segments, sung = _notes()
        shuffled = [segments[2], segments[0], segments[3], segments[1]]
        _quiet(correct_note_pitches, shuffled, sung, _model_file(tmp_path), "strong")
        assert [_midi(s) for s in segments] == [61, 62, 65, 60]
        assert shuffled[0] is segments[2]

    def test_no_notes(self, tmp_path):
        result, _ = _quiet(correct_note_pitches, [], _notes()[1], _model_file(tmp_path), "strong")
        assert (result.applied, result.notes, result.moved) == (True, 0, 0)


class TestFailsOpen:
    """Whatever is missing or goes wrong: the notes stay as they are, and the reason is printed."""

    def _unchanged(self, sung, model_path, strength="strong"):
        segments, _ = _notes()
        result, out = _quiet(correct_note_pitches, segments, sung, model_path, strength)
        assert [s.note for s in segments] == ["C4", "D4", "E4", "B3"]
        assert not result.applied and result.moved == 0
        assert "pitch context not applied" in out and "the note pitches stay as they are" in out
        return result, out

    def test_no_model_file_is_set(self):
        for model_path in (None, ""):
            result, out = self._unchanged(_notes()[1], model_path)
            assert "no model file is set (--pitch_context_model)" in out
            assert str(result) == "not applied (no model file is set (--pitch_context_model))"

    def test_the_model_file_is_missing(self, tmp_path):
        _, out = self._unchanged(_notes()[1], str(tmp_path / "missing.npz"))
        assert "model file not found" in out

    def test_the_model_file_is_no_model(self, tmp_path):
        path = tmp_path / "model.npz"
        path.write_bytes(pickle.dumps({"weights": [1, 2, 3]}))
        _, out = self._unchanged(_notes()[1], str(path))
        assert "is not a pitch context model file" in out

    def test_no_pitch_track(self, tmp_path):
        model = _model_file(tmp_path)
        for sung in (None, PitchedData([], [], [])):
            _, out = self._unchanged(sung, model)
            assert "no pitch track of the vocal" in out

    def test_an_unknown_strength(self, tmp_path):
        _, out = self._unchanged(_notes()[1], _model_file(tmp_path), strength="loud")
        assert "unknown strength 'loud'" in out

    def test_an_error_while_predicting(self, tmp_path, monkeypatch):
        def fail(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(pitch_context, "predict_shifts", fail)
        _, out = self._unchanged(_notes()[1], _model_file(tmp_path))
        assert "boom" in out

    def test_a_note_that_cannot_be_read_changes_none(self, tmp_path):
        segments, sung = _notes()
        segments[3].note = "no note"
        result, _ = _quiet(correct_note_pitches, segments, sung, _model_file(tmp_path), "strong")
        assert not result.applied
        assert [s.note for s in segments] == ["C4", "D4", "E4", "no note"]

    def test_a_changed_model_file_is_read_again(self, tmp_path):
        path = _model_file(tmp_path, up=0.0)  # never promises more than staying
        segments, sung = _notes()
        result, _ = _quiet(correct_note_pitches, segments, sung, path, "strong")
        assert result.moved == 0
        _toy_model(up=1.0).save(path)
        os.utime(path, (os.path.getmtime(path) + 10, os.path.getmtime(path) + 10))
        result, _ = _quiet(correct_note_pitches, segments, sung, path, "strong")
        assert result.moved == 3
