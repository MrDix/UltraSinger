"""Tests for tools/train_pitch_context.py on tiny synthetic examples (no audio, no GPU)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import train_pitch_context as tp  # noqa: E402
from modules.Pitcher.pitch_context import SHIFTS, load_model  # noqa: E402

FRAME_S = 0.016


def _example(seed: int, notes: int = 60, fit: float = 0.9, kind_of=lambda i: 0) -> dict:
    """A song of ``notes`` notes of 0.3 s. Every other note is sung half a semitone sharp and
    charted one semitone above the pitch its frames give it; the others are sung and charted
    on the same pitch."""
    rng = np.random.default_rng(seed)
    duration = notes * 0.5 + 1.0
    times = np.arange(int(duration / FRAME_S)) * FRAME_S
    midi = np.zeros(len(times))
    conf = np.zeros(len(times))
    rows = []
    for i in range(notes):
        start, end = 0.5 * i + 0.1, 0.5 * i + 0.4
        base = int(rng.integers(55, 70))
        sharp = i % 2 == 1
        inside = (times >= start) & (times < end)
        midi[inside] = base + (0.45 if sharp else 0.0) + rng.normal(0, 0.03, inside.sum())
        conf[inside] = 0.9
        rows.append([start * 1000, end * 1000, base + (1 if sharp else 0), kind_of(i)])
    hz = np.where(conf > 0, 440.0 * 2 ** ((midi - 69) / 12), 0.0)
    return {"f0_t": times.astype(np.float32), "f0_hz": hz.astype(np.float32), "f0_conf": conf.astype(np.float32),
            "notes": np.array(rows, dtype=np.float32), "ref_fit": fit, "duration": duration}


def _write(folder: Path, name: str, example: dict) -> Path:
    """An example file as tools/train_segmentation.py writes it (with the chart's words as objects)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.npz"
    words = np.array(["la"] * len(example["notes"]), dtype=object)
    with open(path, "wb") as f:
        np.savez_compressed(f, logmel=np.zeros((4, 80), np.float16), rms=np.zeros(4, np.float32),
                            words=words, offset_ms=0.0, **example)
    return path


class TestTargets:
    def test_the_note_kinds_are_those_of_the_segmentation_tool(self):
        import train_segmentation as ts

        assert tp.PITCHED_KIND_CODES == (ts.KIND_CODES[":"], ts.KIND_CODES["*"])
        assert tp.FRAME_MS == ts.cb.FRAME_MS

    def test_reference_grid(self):
        grid = tp.reference_grid([100.0, 300.0], [200.0, 5000.0], [60, 62], n_frames=40)
        assert np.isnan(grid[:10]).all() and (grid[10:20] == 60).all()
        assert np.isnan(grid[20:30]).all() and (grid[30:] == 62).all()

    def test_what_a_shift_is_worth(self):
        grid = tp.reference_grid([0.0], [1000.0], [61], n_frames=200)
        within_one, exact, charted = tp.shift_targets([0.0], [1000.0], [60], [grid])
        assert SHIFTS == (-2, -1, 0, 1, 2)
        assert within_one.tolist() == [[0, 0, 1, 1, 1]]
        assert exact.tolist() == [[0, 0, 0, 1, 0]]
        assert charted.tolist() == [True]

    def test_octaves_do_not_count(self):
        grid = tp.reference_grid([0.0], [1000.0], [73], n_frames=200)
        within_one, exact, _ = tp.shift_targets([0.0], [1000.0], [60], [grid])
        assert within_one.tolist() == [[0, 0, 1, 1, 1]] and exact.tolist() == [[0, 0, 0, 1, 0]]

    def test_the_share_of_the_note_time(self):
        # the chart has a note under the first half of the note only
        grid = tp.reference_grid([0.0], [500.0], [60], n_frames=200)
        within_one, exact, charted = tp.shift_targets([0.0, 1500.0], [1000.0, 1800.0], [60, 60], [grid])
        assert within_one[0].tolist() == [0, 0.5, 0.5, 0.5, 0]
        assert exact[0].tolist() == [0, 0, 0.5, 0, 0]
        assert charted.tolist() == [True, False]
        assert not within_one[1].any()

    def test_a_shift_agrees_with_any_voice(self):
        first = tp.reference_grid([0.0], [1000.0], [60], n_frames=200)
        second = tp.reference_grid([0.0], [1000.0], [64], n_frames=200)
        _, exact, _ = tp.shift_targets([0.0], [1000.0], [62], [first, second])
        assert exact.tolist() == [[1, 0, 0, 0, 1]]


class TestExamples:
    def test_rows_of_an_example(self):
        rows = tp.example_rows(_example(1))
        assert rows.features.shape == (60, 40) and rows.weights.shape == (60,)
        assert np.allclose(rows.weights, 300.0, atol=1.0)
        keep = SHIFTS.index(0)
        # the notes sung on their pitch are exact as they are, the sharp ones one semitone up
        assert rows.exact[0::2, keep].tolist() == [1.0] * 30
        assert rows.exact[1::2, keep + 1].tolist() == [1.0] * 30
        assert not rows.exact[1::2, keep].any()
        assert (rows.within_one[:, keep] == 1.0).all()

    def test_only_pitched_notes_inside_the_audio(self):
        # every third note is freestyle; they are neither learned from nor neighbours
        rows = tp.example_rows(_example(2, notes=90, kind_of=lambda i: 2 if i % 3 == 0 else 0))
        assert len(rows.weights) == 60
        late = _example(3)
        late["notes"][-5:, :2] += 1e6  # a part appended after the end of the audio
        assert len(tp.example_rows(late).weights) == 55

    def test_too_few_notes(self):
        assert tp.example_rows(_example(4, notes=20)) is None
        silent = _example(5)
        silent["f0_conf"][:] = 0.0
        assert tp.example_rows(silent) is None

    def test_the_words_of_an_example_are_not_unpickled(self, tmp_path):
        path = _write(tmp_path, "tr_0001", _example(6))
        example = tp.load_example(path)
        assert set(example) == {"f0_t", "f0_hz", "f0_conf", "notes", "ref_fit", "duration"}

    def test_examples_of_the_work_folder_or_its_data_folder(self, tmp_path):
        path = _write(tmp_path / "data", "tr_0001", _example(7))
        assert tp.find_examples(tmp_path) == [path]
        assert tp.find_examples(tmp_path / "data") == [path]
        assert tp.find_examples(tmp_path / "data" / "..") == [tmp_path / "data" / ".." / "data" / "tr_0001.npz"]

    def test_excluded_names(self, tmp_path):
        listed = tmp_path / "bench.json"
        listed.write_text(json.dumps(["tr_0003", "tr_0004.npz"]), encoding="utf-8")
        assert tp.excluded_ids(["tr_0001", "tr_0002.npz", str(listed)]) == {
            "tr_0001", "tr_0002", "tr_0003", "tr_0004"}
        assert tp.excluded_ids(None) == set()
        listed.write_text(json.dumps({"songs": []}), encoding="utf-8")
        with pytest.raises(SystemExit):
            tp.excluded_ids([str(listed)])


class TestTraining:
    @pytest.fixture(scope="class")
    def trained(self, tmp_path_factory):
        folder = tmp_path_factory.mktemp("pitch_context")
        for i in range(6):
            _write(folder / "work" / "data", f"tr_{i:04d}", _example(10 + i))
        _write(folder / "work" / "data", "tr_0100", _example(30, fit=0.3))   # the chart does not fit the vocal
        _write(folder / "work" / "data", "tr_0101", _example(31, notes=10))  # too short
        _write(folder / "work" / "data", "tr_0102", _example(32))            # excluded
        (folder / "work" / "data" / "tr_0103.npz").write_bytes(b"broken")
        out = folder / "models" / "pitch_context.npz"
        return folder, out

    def test_trains_and_writes_a_model(self, trained, capsys):
        folder, out = trained
        assert tp.main(["--data", str(folder / "work"), "--out", str(out), "--trees", "20",
                        "--val-fraction", "0.34", "--exclude", "tr_0102"]) == 0
        printed = capsys.readouterr().out
        assert "1 of 10 examples left out (--exclude)" in printed
        assert "6 usable songs of 9 (left out: 1 whose chart fits the vocal worse than 60%, " \
               "1 with fewer than 50 usable notes, 1 unreadable)" in printed
        assert "360 notes; before any correction 50.0 % of their time is exactly on the charted pitch, " \
               "100.0 % within one semitone" in printed
        assert "measuring: training on 4 songs, 2 held out" in printed
        assert "held-out songs (2)" in printed and "training on all 6 songs" in printed
        assert out.is_file() and [p.name for p in out.parent.iterdir()] == ["pitch_context.npz"]

        model = load_model(out)
        assert len(model.tree_root) == 20 * 2 * len(SHIFTS)
        assert model.info["songs"] == 6 and model.info["notes"] == 360
        assert model.info["held_out"]["songs"] == 2
        # what it learned: the notes sung sharp move up, the others stay
        held_out = model.info["held_out"]
        assert held_out["off"]["exact_pct"] == pytest.approx(50.0)
        assert held_out["strong"]["exact_pct"] == pytest.approx(100.0)
        assert held_out["strong"]["moved_pct"] == pytest.approx(50.0)
        assert held_out["strong"]["within_one_pct"] == pytest.approx(100.0)
        rows = tp.example_rows(_example(99))
        result = tp.evaluate(model, rows)
        assert result["off"]["moved_pct"] == 0.0
        assert result["strong"]["exact_pct"] == pytest.approx(100.0)

    def test_the_same_data_gives_the_same_model(self, trained, tmp_path):
        folder, _ = trained
        outs = [tmp_path / "a.npz", tmp_path / "b.npz"]
        for out in outs:
            tp.main(["--data", str(folder / "work" / "data"), "--out", str(out), "--trees", "3",
                     "--val-fraction", "0", "--limit", "4"])
        first, second = (load_model(out) for out in outs)
        assert np.array_equal(first.value, second.value) and np.array_equal(first.threshold, second.threshold)
        assert "held_out" not in first.info and first.info["songs"] == 4

    def test_the_exported_trees_predict_like_the_trained_ones(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        rows = tp.stacked([tp.example_rows(_example(seed)) for seed in (40, 41)])
        rows.features[::7, 3] = np.nan  # a missing feature takes the side its node names
        targets = np.hstack([rows.within_one, rows.exact]) + np.nan_to_num(rows.features[:, 3:4])
        regressors = [HistGradientBoostingRegressor(max_iter=8, max_depth=3, random_state=0)
                      .fit(rows.features, targets[:, k]) for k in range(targets.shape[1])]
        model = tp.export_trees(regressors, rows.features)
        trained = np.column_stack([regressor.predict(rows.features) for regressor in regressors])
        assert np.allclose(np.hstack(model.predict(rows.features)), trained, atol=1e-9)

    def test_an_export_that_predicts_something_else_fails(self):
        class Other:
            """A regressor whose trees say one thing and whose prediction another."""
            _baseline_prediction = 0.0
            _predictors = [[type("Tree", (), {"nodes": np.zeros(1, dtype=[
                ("value", "f8"), ("feature_idx", "i8"), ("num_threshold", "f8"), ("missing_go_to_left", "u1"),
                ("left", "u4"), ("right", "u4"), ("is_leaf", "u1"), ("is_categorical", "u1")])})()]]

            def predict(self, features):
                return np.ones(len(features))

        Other._predictors[0][0].nodes["is_leaf"] = 1
        with pytest.raises(RuntimeError, match="do not predict what scikit-learn predicts"):
            tp.export_trees([Other() for _ in range(2 * len(SHIFTS))], np.zeros((5, 40), np.float32))

    def test_the_model_stays_outside_the_repository(self, trained):
        folder, _ = trained
        with pytest.raises(SystemExit, match="outside the repository"):
            tp.main(["--data", str(folder / "work"), "--out", str(tp.REPO / "pitch_context.npz")])
        assert not (tp.REPO / "pitch_context.npz").exists()

    def test_nothing_to_train_on(self, tmp_path):
        with pytest.raises(SystemExit, match="data folder not found"):
            tp.main(["--data", str(tmp_path / "missing"), "--out", str(tmp_path / "m.npz")])
        with pytest.raises(SystemExit, match="no training examples"):
            tp.main(["--data", str(tmp_path), "--out", str(tmp_path / "m.npz")])
        _write(tmp_path, "tr_0001", _example(50))
        with pytest.raises(SystemExit, match="only 1 usable songs"):
            tp.main(["--data", str(tmp_path), "--out", str(tmp_path / "m.npz")])
        for arguments in (["--trees", "0"], ["--val-fraction", "1"]):
            with pytest.raises(SystemExit):
                tp.main(["--data", str(tmp_path), "--out", str(tmp_path / "m.npz"), *arguments])
        assert not (tmp_path / "m.npz").exists()
