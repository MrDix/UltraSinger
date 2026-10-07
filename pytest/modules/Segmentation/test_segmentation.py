"""Tests for model-based note segmentation (modules.Segmentation).

Everything runs on small synthetic data: no audio is separated and no trained
model is needed (an untrained network is enough for shape/IO checks).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from modules.Midi.MidiSegment import MidiSegment
from modules.Segmentation import lyrics as seg_lyrics
from modules.Segmentation import segmenter
from modules.Segmentation.decode import PredictedNote, decode_notes
from modules.Segmentation.features import (
    FRAME_S, N_EXTRA, N_MELS, VocalAnalysis, frame_pitch, model_input,
)
from modules.Segmentation.model import (
    DEFAULT_DECODE, SegNet, load_model, predict, save_model,
)


def _analysis(n=500, midi=60.0, conf=0.95):
    hz = 440.0 * 2 ** ((midi - 69) / 12)
    return VocalAnalysis(
        f0_t=(np.arange(n) * FRAME_S).astype(np.float32),
        f0_hz=np.full(n, hz, np.float32),
        f0_conf=np.full(n, conf, np.float32),
        logmel=np.random.default_rng(0).normal(size=(n, N_MELS)).astype(np.float16),
        rms=np.full(n, 0.1, np.float32),
        duration=n * FRAME_S,
    )


# ── features ────────────────────────────────────────────────────────────────

class TestFeatures:
    def test_model_input_shape_and_dtype(self):
        x = model_input(_analysis(300))
        assert x.shape == (300, N_MELS + N_EXTRA)
        assert x.dtype == np.float32
        assert np.isfinite(x).all()

    def test_relative_pitch_is_zero_on_constant_pitch(self):
        x = model_input(_analysis(100, midi=64))
        assert np.allclose(x[:, N_MELS + 0], 0.0)  # pitch relative to the song median
        assert np.allclose(x[:, N_MELS + 3], 1.0)  # voiced flag

    def test_frame_pitch_marks_unvoiced(self):
        a = _analysis(10)
        a.f0_conf[:5] = 0.1
        midi, voiced = frame_pitch(a, 12)  # longer than the analysis
        assert not voiced[:5].any() and voiced[5:10].all() and not voiced[10:].any()
        assert np.isnan(midi[10:]).all()
        assert midi[7] == pytest.approx(60.0, abs=1e-3)


# ── decoding ────────────────────────────────────────────────────────────────

def _probs(n, pitched=(), free=()):
    p = np.zeros((n, 3), np.float32)
    p[:, 0] = 1.0
    for a, b in pitched:
        p[a:b] = [0.0, 1.0, 0.0]
    for a, b in free:
        p[a:b] = [0.0, 0.0, 1.0]
    return p


class TestDecode:
    def test_region_split_at_onset_peak(self):
        n = 100
        onset = np.zeros(n, np.float32)
        onset[30] = 0.9
        notes = decode_notes(_probs(n, pitched=[(10, 50)]), onset, _analysis(n, midi=62), **DEFAULT_DECODE)
        assert [(round(x.start / FRAME_S), round(x.end / FRAME_S)) for x in notes] == [(10, 30), (30, 50)]
        assert all(x.midi == 62 and not x.freestyle for x in notes)

    def test_short_notes_dropped_and_close_peaks_ignored(self):
        n = 100
        onset = np.zeros(n, np.float32)
        onset[[22, 24]] = 0.9  # second peak too close to the first cut
        notes = decode_notes(_probs(n, pitched=[(10, 40), (60, 63)]), onset, _analysis(n),
                             onset_thr=0.4, act_thr=0.4, min_note_frames=6, min_gap_frames=0)
        assert [(round(x.start / FRAME_S), round(x.end / FRAME_S)) for x in notes] == [(10, 22), (22, 40)]

    def test_gap_between_split_notes(self):
        n = 100
        onset = np.zeros(n, np.float32)
        onset[30] = 0.9
        notes = decode_notes(_probs(n, pitched=[(10, 50)]), onset, _analysis(n), min_gap_frames=2,
                             onset_thr=0.4, act_thr=0.4, min_note_frames=6)
        assert round(notes[0].end / FRAME_S) == 28 and round(notes[1].start / FRAME_S) == 30

    def test_freestyle_region(self):
        n = 80
        notes = decode_notes(_probs(n, free=[(20, 60)]), np.zeros(n, np.float32), _analysis(n), **DEFAULT_DECODE)
        assert len(notes) == 1 and notes[0].freestyle

    def test_unvoiced_note_continues_previous_pitch_or_is_dropped(self):
        n = 100
        a = _analysis(n, midi=65)
        a.f0_conf[:] = 0.1
        a.f0_conf[10:20] = 0.95
        onset = np.zeros(n, np.float32)
        onset[20] = 0.9
        notes = decode_notes(_probs(n, pitched=[(10, 40)]), onset, a, **DEFAULT_DECODE)
        assert [x.midi for x in notes] == [65, 65]
        a.f0_conf[:] = 0.1
        assert decode_notes(_probs(n, pitched=[(10, 40)]), np.zeros(n, np.float32), a, **DEFAULT_DECODE) == []


# ── model files ─────────────────────────────────────────────────────────────

class TestModel:
    def test_save_load_roundtrip_and_predict(self, tmp_path):
        torch.manual_seed(0)
        net = SegNet()
        path = tmp_path / "m.pt"
        save_model(path, net, {"onset_thr": 0.3}, {"train_songs": 1})
        loaded, decode = load_model(path)
        assert decode["onset_thr"] == 0.3 and decode["min_note_frames"] == DEFAULT_DECODE["min_note_frames"]
        x = model_input(_analysis(2300))  # longer than one chunk
        p1, o1 = predict(net.eval(), x)
        p2, o2 = predict(loaded, x)
        assert p1.shape == (2300, 3) and o1.shape == (2300,)
        assert np.allclose(p1, p2, atol=1e-5) and np.allclose(o1, o2, atol=1e-5)
        assert np.allclose(p1.sum(1), 1.0, atol=1e-4)

    def test_rejects_foreign_file(self, tmp_path):
        path = tmp_path / "x.pt"
        torch.save({"something": 1}, path)
        with pytest.raises(ValueError, match="not a segmentation model"):
            load_model(path)


# ── lyrics placement ────────────────────────────────────────────────────────

class _FakeHyphenator:
    def syllables(self, word):
        cut = {"hello": 3, "dancing": 3}.get(word.lower())
        return [word[:cut], word[cut:]] if cut else [word]


@pytest.fixture
def fake_hyphen(monkeypatch):
    monkeypatch.setattr(seg_lyrics, "_hyphenator", lambda language: _FakeHyphenator())


def _seg(word, start, end, line_break=False):
    s = MidiSegment("C4", start, end, word)
    s.line_break_after = line_break
    return s


class TestSyllables:
    def test_tokens_continuations_and_line_starts(self, fake_hyphen):
        segs = [_seg("Hello ", 0.0, 0.4), _seg("~ ", 0.4, 0.8, line_break=True),
                _seg("world ", 1.0, 1.5)]
        syl = seg_lyrics.syllables_from_segments(segs, "en")
        assert [s.text for s in syl] == ["Hel", "lo ", "world "]
        assert syl[0].line_start and not syl[1].line_start and syl[2].line_start
        assert syl[1].end_ms == pytest.approx(800)  # continuation extends the word
        assert syl[0].start_ms == 0 and syl[0].end_ms == pytest.approx(syl[1].start_ms)

    def test_without_hyphenator(self, monkeypatch):
        monkeypatch.setattr(seg_lyrics, "_hyphenator", lambda language: None)
        syl = seg_lyrics.syllables_from_segments([_seg("dancing ", 0, 1)], "en")
        assert [s.text for s in syl] == ["dancing "]


def _notes(*spans):
    return [PredictedNote(a, b, 60) for a, b in spans]


class TestPlaceLyrics:
    def test_one_syllable_per_note(self):
        syl = [seg_lyrics.Syllable("one ", 0, 400), seg_lyrics.Syllable("two ", 500, 900)]
        segs = seg_lyrics.place_lyrics(_notes((0.0, 0.4), (0.5, 0.9)), syl)
        assert [s.word for s in segs] == ["one ", "two "]

    def test_extra_notes_become_continuations_with_space_on_last(self):
        syl = [seg_lyrics.Syllable("mind ", 0, 1500), seg_lyrics.Syllable("now ", 2000, 2400)]
        segs = seg_lyrics.place_lyrics(_notes((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (2.0, 2.4)), syl)
        assert [s.word for s in segs] == ["mind", "~", "~ ", "now "]

    def test_missing_notes_merge_syllables(self):
        syl = [seg_lyrics.Syllable("a ", 0, 200), seg_lyrics.Syllable("b ", 250, 400),
               seg_lyrics.Syllable("c ", 2000, 2300)]
        segs = seg_lyrics.place_lyrics(_notes((0.0, 0.4), (2.0, 2.3)), syl)
        assert [s.word for s in segs] == ["a b ", "c "]

    def test_trailing_syllables_appended(self):
        syl = [seg_lyrics.Syllable("a ", 0, 200), seg_lyrics.Syllable("end ", 5000, 5200)]
        segs = seg_lyrics.place_lyrics(_notes((0.0, 0.2)), syl)
        assert segs[0].word.split() == ["a", "end"]

    def test_line_break_before_line_start(self):
        syl = [seg_lyrics.Syllable("a ", 0, 200, line_start=True), seg_lyrics.Syllable("b ", 1000, 1200, line_start=True)]
        segs = seg_lyrics.place_lyrics(_notes((0.0, 0.2), (1.0, 1.2)), syl)
        assert segs[0].line_break_after and not segs[1].line_break_after

    def test_freestyle_and_note_names(self):
        notes = [PredictedNote(0.0, 0.5, 69, freestyle=True)]
        segs = seg_lyrics.place_lyrics(notes, [seg_lyrics.Syllable("hey ", 0, 500)])
        assert segs[0].note_type == "F" and segs[0].note == "A4"

    def test_no_syllables_gives_no_segments(self):
        assert seg_lyrics.place_lyrics(_notes((0.0, 0.5)), []) == []

    def test_note_far_from_all_syllables_keeps_path(self):
        # syllables only at the start, then a note 30 s later (outside the band)
        syl = [seg_lyrics.Syllable("a ", 0, 300), seg_lyrics.Syllable("b ", 400, 700)]
        segs = seg_lyrics.place_lyrics(_notes((0.0, 0.3), (0.4, 0.7), (30.0, 30.5)), syl)
        assert len(segs) == 3
        assert segs[0].word.startswith("a")
        assert "b" in "".join(s.word for s in segs)  # no syllable is lost

    def test_widening_when_band_starts_beyond_reachable_syllables(self):
        # 30 syllables in the first 15 s, 3 more at 60 s; only two early notes and
        # one at 60 s. The band of the last note starts at syllable 30, far beyond
        # what the two early notes can reach, so the row must be widened.
        syl = [seg_lyrics.Syllable(f"e{i} ", i * 500, i * 500 + 400) for i in range(30)]
        syl += [seg_lyrics.Syllable(f"l{i} ", 60000 + i * 500, 60000 + i * 500 + 400) for i in range(3)]
        path = seg_lyrics.align_syllables(np.array([0.0, 500.0, 60000.0]), syl)
        assert len(path) == 3
        assert path[0] == (0, True)                     # a real path, not a fabricated one
        assert [j for j, _ in path] == sorted(j for j, _ in path)
        segs = seg_lyrics.place_lyrics(_notes((0.0, 0.4), (0.5, 0.9), (60.0, 60.4)), syl)
        text = "".join(s.word for s in segs)
        assert all(f"e{i} " in text for i in range(30)) and all(f"l{i} " in text for i in range(3))

    def test_unreachable_rows_are_not_finite(self):
        syl = [seg_lyrics.Syllable("a ", 0, 300)]
        path = seg_lyrics.align_syllables(np.array([0.0, 400.0]), syl)
        assert path == [(0, True), (0, False)]

    def test_alignment_is_monotonic(self):
        rng = np.random.default_rng(1)
        starts = np.sort(rng.uniform(0, 60000, 120))
        syl = [seg_lyrics.Syllable(f"s{i} ", t, t + 200) for i, t in enumerate(np.sort(rng.uniform(0, 60000, 90)))]
        path = seg_lyrics.align_syllables(starts, syl)
        idx = [j for j, _ in path]
        assert idx == sorted(idx)


# ── pipeline step ───────────────────────────────────────────────────────────

class TestSegmenter:
    def test_missing_model_keeps_segments(self, tmp_path):
        assert segmenter.segment_with_model([_seg("a ", 0, 1)], str(tmp_path / "v.wav"),
                                            str(tmp_path / "none.pt"), "en") is None

    def test_missing_vocal_keeps_segments(self, tmp_path):
        model = tmp_path / "m.pt"
        save_model(model, SegNet())
        assert segmenter.segment_with_model([_seg("a ", 0, 1)], str(tmp_path / "missing.wav"),
                                            str(model), "en") is None

    def test_errors_fail_open(self, tmp_path, monkeypatch):
        model = tmp_path / "m.pt"
        save_model(model, SegNet())
        vocal = tmp_path / "v.wav"
        vocal.write_bytes(b"not audio")
        import modules.Segmentation.features as feats
        monkeypatch.setattr(feats, "load_vocal", lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
        assert segmenter.segment_with_model([_seg("a ", 0, 1)], str(vocal), str(model), "en") is None

    def test_unplaceable_lyrics_keep_segments(self, tmp_path, monkeypatch):
        model = tmp_path / "m.pt"
        save_model(model, SegNet())
        vocal = tmp_path / "v.wav"
        vocal.write_bytes(b"x")
        import modules.Segmentation.decode as dec
        import modules.Segmentation.features as feats
        monkeypatch.setattr(feats, "load_vocal", lambda p: np.zeros(16000, np.float32))
        monkeypatch.setattr(feats, "analyse_vocal", lambda y: _analysis(200))
        monkeypatch.setattr(dec, "decode_notes", lambda *a, **k: _notes((0.0, 0.5)))
        monkeypatch.setattr(seg_lyrics, "place_lyrics", lambda notes, syl: [])
        assert segmenter.segment_with_model([_seg("hello ", 0, 1)], str(vocal), str(model), "en") is None

    def test_success_path(self, tmp_path, monkeypatch, fake_hyphen):
        model = tmp_path / "m.pt"
        save_model(model, SegNet())
        vocal = tmp_path / "v.wav"
        vocal.write_bytes(b"x")
        import modules.Segmentation.decode as dec
        import modules.Segmentation.features as feats
        monkeypatch.setattr(feats, "load_vocal", lambda p: np.zeros(16000, np.float32))
        monkeypatch.setattr(feats, "analyse_vocal", lambda y: _analysis(200))
        monkeypatch.setattr(dec, "decode_notes", lambda *a, **k: _notes((0.0, 0.5), (0.6, 1.0)))
        segs = segmenter.segment_with_model([_seg("hello ", 0, 1)], str(vocal), str(model), "en")
        assert [s.word for s in segs] == ["hel", "lo "]
