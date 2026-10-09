"""Tests for tools/train_segmentation.py — training a note segmentation model.

No vocals are separated here: training examples are written directly with
synthetic analyses, and training runs a couple of steps on the CPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import train_segmentation as ts  # noqa: E402
from chart_benchmark import ChartNote  # noqa: E402
from modules.Segmentation.features import FRAME_S, N_MELS, VocalAnalysis  # noqa: E402
from modules.Segmentation.model import load_model  # noqa: E402


def _analysis(n=1000, midi=60.0):
    hz = 440.0 * 2 ** ((midi - 69) / 12)
    rng = np.random.default_rng(int(midi))
    return VocalAnalysis(
        f0_t=(np.arange(n) * FRAME_S).astype(np.float32),
        f0_hz=np.full(n, hz, np.float32),
        f0_conf=np.full(n, 0.95, np.float32),
        logmel=rng.normal(size=(n, N_MELS)).astype(np.float16),
        rms=np.full(n, 0.1, np.float32),
        duration=n * FRAME_S,
    )


def _notes():
    return [ChartNote(1000, 1500, 60, ":", "la "), ChartNote(1500, 2000, 62, ":", "la "),
            ChartNote(4000, 4800, 60, "F", "hey "), ChartNote(6000, 6400, 64, "*", "gold ")]


class TestLabels:
    def test_classes_and_onsets(self):
        d = {"notes": np.array([[n.start_ms, n.end_ms, n.midi, ts.KIND_CODES[n.kind]] for n in _notes()])}
        cls, onset = ts.frame_labels(d, 600)
        f = lambda ms: int(round(ms / 1000 / FRAME_S))  # noqa: E731
        assert (cls[f(1000):f(2000)] == 1).all()
        assert (cls[f(4000):f(4800)] == 2).all()   # freestyle
        assert (cls[f(6000):f(6400)] == 1).all()   # golden counts as pitched
        assert cls[f(3000)] == 0
        assert onset[f(1000)] == 1.0 and onset[f(1500)] == 1.0 and onset[f(1000) + 1] == 0.5
        assert onset[f(4000)] == 0.0               # no onsets inside freestyle

    def test_notes_outside_range_ignored(self):
        d = {"notes": np.array([[-500, -100, 60, 0], [99000, 99500, 60, 0]], np.float32)}
        cls, onset = ts.frame_labels(d, 100)
        assert not cls.any() and not onset.any()


class TestExamples:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "x.npz"
        ts.save_example(path, _analysis(300), _notes(), offset_ms=40.0, ref_fit=0.8)
        d = ts.load_example(path)
        ref = ts.example_reference(d)
        assert [n.kind for n in ref] == [":", ":", "F", "*"]
        assert ref[0].start_ms == pytest.approx(1040) and ref[0].word == "la "
        assert float(d["ref_fit"]) == pytest.approx(0.8)
        a = ts.example_analysis(d)
        assert a.logmel.shape == (300, N_MELS)
        assert [p.name for p in tmp_path.iterdir()] == ["x.npz"]

    def test_interrupted_write_leaves_no_example(self, tmp_path, monkeypatch):
        def broken(f, **arrays):
            f.write(b"PK\x03\x04 truncated")
            raise KeyboardInterrupt
        monkeypatch.setattr(ts.np, "savez_compressed", broken)
        path = tmp_path / "x.npz"
        with pytest.raises(KeyboardInterrupt):
            ts.save_example(path, _analysis(300), _notes(), offset_ms=0.0, ref_fit=0.9)
        assert not list(tmp_path.iterdir())


class TestGuards:
    def test_workdir_inside_repo_refused(self):
        with pytest.raises(SystemExit, match="outside the repository"):
            ts.main(["extract", str(ts.REPO), str(ts.REPO / "segdata")])

    def test_model_inside_repo_refused(self, tmp_path):
        with pytest.raises(SystemExit, match="model file must be outside"):
            ts.main(["train", str(tmp_path), "--out", str(ts.REPO / "x.pt")])

    def test_zero_epochs_refused(self, tmp_path):
        with pytest.raises(SystemExit, match="at least 1"):
            ts.main(["train", str(tmp_path), "--out", str(tmp_path / "m.pt"), "--epochs", "0"])

    def test_missing_library(self, tmp_path):
        with pytest.raises(SystemExit, match="library folder not found"):
            ts.main(["extract", str(tmp_path / "nope"), str(tmp_path / "w")])

    def test_too_few_songs(self, tmp_path):
        with pytest.raises(SystemExit, match="usable songs"):
            ts.main(["train", str(tmp_path), "--out", str(tmp_path / "m.pt"), "--cpu"])


class TestTraining:
    def _workdir(self, tmp_path, n=6):
        data = tmp_path / "data"
        data.mkdir()
        for i in range(n):
            ts.save_example(data / f"tr_{i:05d}.npz", _analysis(900, midi=55 + i), _notes(),
                            offset_ms=0.0, ref_fit=0.9 if i else 0.2)  # first song below min fit
        return tmp_path

    def test_dataset_skips_poor_fit(self, tmp_path):
        items = ts.load_dataset(self._workdir(tmp_path), min_fit=0.5)
        assert len(items) == 5 and "tr_00000" not in [i[0] for i in items]

    def test_dataset_skips_chart_with_appended_voice(self, tmp_path, capsys):
        work = self._workdir(tmp_path)
        # 900 frames = 14.4 s of audio; the chart goes on for another song length
        song = [ChartNote(1000 + 400 * k, 1300 + 400 * k, 60, ":", "la ") for k in range(30)]
        appended = [ChartNote(n.start_ms + 15000, n.end_ms + 15000, 62, ":", "la ") for n in song]
        ts.save_example(work / "data" / "tr_00099.npz", _analysis(900), song + appended,
                        offset_ms=0.0, ref_fit=0.9)
        ids = [i[0] for i in ts.load_dataset(work, min_fit=0.5)]
        assert "tr_00099" not in ids and len(ids) == 5
        assert "skipped 1 songs" in capsys.readouterr().out

    def test_dataset_skips_chart_with_a_few_notes_after_the_audio(self, tmp_path):
        work = self._workdir(tmp_path)
        song = [ChartNote(1000 + 400 * k, 1300 + 400 * k, 60, ":", "la ") for k in range(30)]
        # too few to count as an appended voice, but still not in the 14.4 s of audio
        late = [ChartNote(16000 + 400 * k, 16300 + 400 * k, 62, ":", "la ") for k in range(3)]
        ts.save_example(work / "data" / "tr_00099.npz", _analysis(900), song + late,
                        offset_ms=0.0, ref_fit=0.9)
        assert "tr_00099" not in [i[0] for i in ts.load_dataset(work, min_fit=0.5)]

    def test_dataset_keeps_chart_ending_just_after_the_audio(self, tmp_path):
        work = self._workdir(tmp_path)
        # the last note starts 0.2 s after the end of the 14.4 s of audio
        song = [ChartNote(1000 + 400 * k, 1300 + 400 * k, 60, ":", "la ") for k in range(35)]
        ts.save_example(work / "data" / "tr_00099.npz", _analysis(900), song,
                        offset_ms=0.0, ref_fit=0.9)
        assert "tr_00099" in [i[0] for i in ts.load_dataset(work, min_fit=0.5)]

    def test_train_writes_loadable_model(self, tmp_path):
        work = self._workdir(tmp_path)
        out = tmp_path / "models" / "m.pt"
        assert ts.main(["train", str(work), "--out", str(out), "--epochs", "1", "--steps", "2",
                        "--batch-size", "2", "--cpu", "--val-fraction", "0.3"]) == 0
        model, decode = load_model(out)
        assert set(decode) >= {"onset_thr", "act_thr", "min_note_frames", "min_gap_frames"}
        assert decode["onset_thr"] in ts.DECODE_GRID["onset_thr"]


class TestExtractSongList:
    def test_exclude_and_resume(self, tmp_path, monkeypatch):
        lib = tmp_path / "lib"
        for name in ("A", "B", "C"):
            d = lib / f"Artist - {name}"
            d.mkdir(parents=True)
            (d / "song.mp3").write_bytes(b"x")
            lines = ["#TITLE:T", "#ARTIST:A", "#MP3:song.mp3", "#BPM:300", "#GAP:0"]
            lines += [f": {i * 4} 2 60 la" for i in range(150)]
            (d / "song.txt").write_text("\n".join(lines + ["E"]), encoding="utf-8")
        exclude = tmp_path / "bench.json"
        exclude.write_text(json.dumps([{"folder": str(lib / "Artist - B")}]), encoding="utf-8")

        class _Stop(Exception):
            pass

        # stop right after the song list is written (no separation in tests)
        import audio_separator.separator as sepmod
        monkeypatch.setattr(sepmod, "Separator", lambda *a, **k: (_ for _ in ()).throw(_Stop()))
        work = tmp_path / "work"
        with pytest.raises(_Stop):
            ts.main(["extract", str(lib), str(work), "--exclude", str(exclude)])
        songs = json.loads((work / "songs.json").read_text(encoding="utf-8"))
        assert sorted(Path(s["folder"]).name for s in songs) == ["Artist - A", "Artist - C"]

    def test_exclude_applied_when_resuming(self, tmp_path, monkeypatch):
        lib = tmp_path / "lib"
        lib.mkdir()
        work = tmp_path / "work"
        (work / "data").mkdir(parents=True)
        songs = [{"id": "tr_00000", "folder": "F/A", "txt": "", "media": ""},
                 {"id": "tr_00001", "folder": "F/B", "txt": "", "media": ""}]
        (work / "songs.json").write_text(json.dumps(songs), encoding="utf-8")
        for s in songs:  # both already extracted in an earlier run without --exclude
            (work / "data" / f"{s['id']}.npz").write_bytes(b"x")
        exclude = tmp_path / "bench.json"
        exclude.write_text(json.dumps([{"folder": "F/B"}]), encoding="utf-8")

        class _Stop(Exception):
            pass

        import audio_separator.separator as sepmod
        monkeypatch.setattr(sepmod, "Separator", lambda *a, **k: (_ for _ in ()).throw(_Stop()))
        with pytest.raises(_Stop):
            ts.main(["extract", str(lib), str(work), "--exclude", str(exclude)])
        kept = json.loads((work / "songs.json").read_text(encoding="utf-8"))
        assert [s["id"] for s in kept] == ["tr_00000"]
        assert (work / "data" / "tr_00000.npz").exists()
        assert not (work / "data" / "tr_00001.npz").exists()

    def test_train_uses_only_listed_songs(self, tmp_path):
        data = tmp_path / "data"
        data.mkdir()
        for i in range(3):
            ts.save_example(data / f"tr_{i:05d}.npz", _analysis(500), _notes(), offset_ms=0.0, ref_fit=0.9)
        (tmp_path / "songs.json").write_text(json.dumps([{"id": "tr_00000"}, {"id": "tr_00002"}]),
                                             encoding="utf-8")
        assert sorted(i[0] for i in ts.load_dataset(tmp_path, 0.5)) == ["tr_00000", "tr_00002"]


class TestExtractProgressOutput:
    def test_progress_lines_for_the_gui(self, tmp_path, monkeypatch, capsys):
        lib = tmp_path / "lib"
        work = tmp_path / "work"
        (work / "data").mkdir(parents=True)
        songs = []
        for i in range(2):
            d = lib / f"Artist - Title {i}"
            d.mkdir(parents=True)
            (d / "song.mp3").write_bytes(b"x")
            txt = d / "song.txt"
            lines = ["#TITLE:T", "#ARTIST:A", "#MP3:song.mp3", "#BPM:300", "#GAP:0"]
            txt.write_text("\n".join(lines + [f": {k * 4} 2 60 la" for k in range(150)] + ["E"]), encoding="utf-8")
            songs.append({"id": f"tr_{i:05d}", "folder": str(d), "txt": str(txt), "media": str(d / "song.mp3"),
                          "audio": str(d / "song.mp3")})
        (work / "songs.json").write_text(json.dumps(songs), encoding="utf-8")
        ts.save_example(work / "data" / "tr_00000.npz", _analysis(300), _notes(), 0.0, 0.9)  # already done

        class FakeSeparator:
            def __init__(self, output_dir, **kwargs):
                self.out = Path(output_dir)

            def load_model(self, model_filename):
                pass

            def separate(self, path, custom_output_names=None):
                (self.out / "vocals.wav").write_bytes(b"wav")

        import audio_separator.separator as sepmod
        import modules.Segmentation.features as feats
        monkeypatch.setattr(sepmod, "Separator", FakeSeparator)
        monkeypatch.setattr(feats, "load_vocal", lambda p: np.zeros(16000, np.float32))
        monkeypatch.setattr(ts, "analyse_vocal", lambda y: _analysis(300))
        assert ts.main(["extract", str(lib), str(work)]) == 0
        out = capsys.readouterr().out
        assert "2 songs, 1 already extracted" in out
        assert "[2/2] tr_00001: ok" in out
        assert "tr_00000: ok" not in out  # existing examples are skipped
