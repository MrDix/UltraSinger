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

    def test_pitched_and_freestyle_never_overlap(self):
        n = 60
        probs = np.zeros((n, 3), np.float32)
        probs[:] = [1.0, 0.0, 0.0]
        probs[10:50] = [0.17, 0.41, 0.42]  # both classes above act_thr, freestyle slightly ahead
        notes = decode_notes(probs, np.zeros(n, np.float32), _analysis(n), **DEFAULT_DECODE)
        assert len(notes) == 1 and notes[0].freestyle

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

    def test_pitch_is_upper_middle_of_the_frames(self):
        # half of the note at 60, half at 62: the median would round to 61
        n = 60
        a = _analysis(n, midi=60)
        a.f0_hz[25:40] = 440.0 * 2 ** ((62 - 69) / 12)
        notes = decode_notes(_probs(n, pitched=[(10, 40)]), np.zeros(n, np.float32), a, **DEFAULT_DECODE)
        assert [x.midi for x in notes] == [62]

    def test_coverage_and_second_opinion_of_lead_pitches(self):
        n = 60
        vocals, lead = _analysis(n, midi=60), _analysis(n, midi=64)
        lead.f0_conf[10:25] = 0.1  # the lead covers half of the note
        notes = decode_notes(_probs(n, pitched=[(10, 40)]), np.zeros(n, np.float32), vocals,
                             pitch_analysis=lead, **DEFAULT_DECODE)
        assert [(x.midi, x.voiced, x.check_midi) for x in notes] == [(64, 0.5, 60)]

    def test_vocal_pitch_where_the_lead_is_silent_has_no_second_opinion(self):
        n = 60
        vocals, lead = _analysis(n, midi=60), _analysis(n, midi=64)
        lead.f0_conf[:] = 0.1
        notes = decode_notes(_probs(n, pitched=[(10, 40)]), np.zeros(n, np.float32), vocals,
                             pitch_analysis=lead, **DEFAULT_DECODE)
        assert [(x.midi, x.voiced, x.check_midi) for x in notes] == [(60, 1.0, None)]

    def test_check_track_gives_the_second_opinion_on_vocal_pitches(self):
        n = 60
        vocals, check = _analysis(n, midi=60), _analysis(n, midi=67)
        notes = decode_notes(_probs(n, pitched=[(10, 40)]), np.zeros(n, np.float32), vocals,
                             check_analysis=check, **DEFAULT_DECODE)
        assert [(x.midi, x.check_midi) for x in notes] == [(60, 67)]
        check.f0_conf[:] = 0.1
        notes = decode_notes(_probs(n, pitched=[(10, 40)]), np.zeros(n, np.float32), vocals,
                             check_analysis=check, **DEFAULT_DECODE)
        assert notes[0].check_midi is None
        # without any check track there is no second opinion
        assert decode_notes(_probs(n, pitched=[(10, 40)]), np.zeros(n, np.float32), vocals,
                            **DEFAULT_DECODE)[0].check_midi is None

    def test_freestyle_and_continued_notes_are_not_tracked(self):
        n = 100
        a = _analysis(n, midi=65)
        a.f0_conf[:] = 0.1
        a.f0_conf[10:20] = 0.95
        onset = np.zeros(n, np.float32)
        onset[20] = 0.9
        notes = decode_notes(_probs(n, pitched=[(10, 40)], free=[(60, 80)]), onset, a, **DEFAULT_DECODE)
        assert [(x.voiced, x.freestyle) for x in notes] == [(1.0, False), (0.0, False), (0.0, True)]


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

    def test_hyphen_chain_is_sung_piece_by_piece(self, fake_hyphen):
        syl = seg_lyrics.syllables_from_segments([_seg("Ooh-ooh-oh, ", 0.0, 1.1)], "en")
        assert [s.text for s in syl] == ["Ooh-", "ooh-", "oh, "]
        assert syl[0].start_ms == 0 and syl[-1].end_ms == pytest.approx(1100)

    def test_no_syllable_of_hyphens_only(self, fake_hyphen):
        syl = seg_lyrics.syllables_from_segments([_seg("la--la ", 0.0, 1.0), _seg("-oh ", 1.0, 1.5)], "en")
        assert [s.text for s in syl] == ["la--", "la ", "-oh "]

    def test_pieces_of_a_hyphenated_word_are_hyphenated_further(self, fake_hyphen):
        syl = seg_lyrics.syllables_from_segments([_seg("hello-dancing ", 0.0, 1.2)], "en")
        assert [s.text for s in syl] == ["hel", "lo-", "dan", "cing "]


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

    def test_syllables_without_notes_keep_their_order(self):
        """b is nearer the next note and c nearer the previous one, but c is sung after b."""
        syl = [seg_lyrics.Syllable("a ", 0, 250), seg_lyrics.Syllable("b ", 300, 500),
               seg_lyrics.Syllable("c ", 950, 1000), seg_lyrics.Syllable("d ", 1100, 1300)]
        segs = seg_lyrics.place_lyrics(_notes((0.0, 1.0), (1.1, 1.3)), syl)
        assert [s.word for s in segs] == ["a ", "b c d "]

    def test_no_syllable_moves_into_the_next_line(self):
        """The last word of a line stays there, although the next line's note is nearer."""
        syl = [seg_lyrics.Syllable("the ", 0, 500, line_start=True), seg_lyrics.Syllable("end ", 1500, 1800),
               seg_lyrics.Syllable("You ", 2000, 2200, line_start=True)]
        segs = seg_lyrics.place_lyrics(_notes((0.0, 0.5), (1.95, 2.2)), syl)
        assert [s.word for s in segs] == ["the end ", "You "]
        assert segs[0].line_break_after

    def test_no_syllable_moves_into_the_previous_line(self):
        syl = [seg_lyrics.Syllable("a ", 0, 200, line_start=True), seg_lyrics.Syllable("b ", 300, 500),
               seg_lyrics.Syllable("c ", 600, 800, line_start=True), seg_lyrics.Syllable("d ", 2800, 2900),
               seg_lyrics.Syllable("e ", 3000, 3200)]
        segs = seg_lyrics.place_lyrics(_notes((0.0, 0.5), (3.0, 3.2)), syl)
        assert [s.word for s in segs] == ["a b ", "c d e "]  # c is nearer the first note
        assert segs[0].line_break_after

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
        result = segmenter.segment_with_model([_seg("hello ", 0, 1)], str(vocal), str(model), "en")
        assert [s.word for s in result.segments] == ["hel", "lo "]
        assert result.pitch_audio_path == str(vocal)

    def test_well_tracked_notes_lock_their_pitch(self):
        notes = [PredictedNote(0.0, 0.5, 60, voiced=0.5, check_midi=62),
                 PredictedNote(0.5, 1.0, 62, voiced=0.49, check_midi=62),
                 PredictedNote(1.0, 1.5, 64, freestyle=True, voiced=1.0)]
        segments = [_seg("a ", 0.0, 0.5), _seg("b ", 0.5, 1.0), _seg("c ", 1.0, 1.5)]
        assert segmenter.lock_pitches(segments, notes) == 1
        assert [(s.pitch_locked, s.check_midi) for s in segments] == [(True, 62), (False, 62), (False, None)]

    def test_new_segments_are_not_locked(self):
        seg = MidiSegment("C4", 0.0, 1.0, "a ")
        assert seg.pitch_locked is False and seg.check_midi is None


# ── lead-vocal pitch ────────────────────────────────────────────────────────

from modules.Segmentation import lead_vocal  # noqa: E402


class TestLeadVocalPitch:
    def test_decode_prefers_pitch_analysis(self):
        n = 60
        vocals, lead = _analysis(n, midi=60), _analysis(n, midi=64)
        notes = decode_notes(_probs(n, pitched=[(10, 40)]), np.zeros(n, np.float32), vocals,
                             pitch_analysis=lead, **DEFAULT_DECODE)
        assert [x.midi for x in notes] == [64]

    def test_decode_falls_back_where_lead_is_silent(self):
        n = 80
        vocals, lead = _analysis(n, midi=60), _analysis(n, midi=64)
        lead.f0_conf[40:] = 0.1  # lead silent in the second note
        onset = np.zeros(n, np.float32)
        onset[40] = 0.9
        notes = decode_notes(_probs(n, pitched=[(10, 70)]), onset, vocals, pitch_analysis=lead, **DEFAULT_DECODE)
        assert [x.midi for x in notes] == [64, 60]

    def test_rule(self):
        vocals = _analysis(100)
        good, bad = _analysis(100), _analysis(100)
        bad.f0_conf[:30] = 0.1  # lost 30 % of the singing
        assert lead_vocal.voiced_ratio(good, vocals) == pytest.approx(1.0)
        assert lead_vocal.choose_pitch_source(vocals, good)[0] is good
        chosen, ratio = lead_vocal.choose_pitch_source(vocals, bad)
        assert chosen is None and ratio == pytest.approx(0.7)
        assert lead_vocal.choose_pitch_source(vocals, None) == (None, 0.0)

    def test_separation_is_cached(self, tmp_path, monkeypatch):
        calls = []

        class FakeSeparator:
            def __init__(self, output_dir, **kw):
                self.out = output_dir

            def load_model(self, model_filename):
                calls.append(model_filename)

            def separate(self, path, custom_output_names=None):
                open(os.path.join(self.out, "lead.wav"), "wb").close()

        import os
        import audio_separator.separator as sepmod
        monkeypatch.setattr(sepmod, "Separator", FakeSeparator)
        vocals = tmp_path / "vocals.wav"
        vocals.write_bytes(b"one")
        p1 = lead_vocal.separate_lead_vocal(str(vocals), str(tmp_path))
        p2 = lead_vocal.separate_lead_vocal(str(vocals), str(tmp_path))
        assert p1 == p2 and p1.endswith("lead.wav") and calls == [lead_vocal.KARAOKE_MODEL]
        # another song's vocal file in the same cache folder gets its own lead stem
        other = tmp_path / "other" / "vocals.wav"
        other.parent.mkdir()
        other.write_bytes(b"two")
        p3 = lead_vocal.separate_lead_vocal(str(other), str(tmp_path))
        assert p3 != p1 and len(calls) == 2

    def test_analysis_comes_with_the_stem_path(self, monkeypatch):
        import modules.Segmentation.features as feats
        monkeypatch.setattr(lead_vocal, "separate_lead_vocal", lambda v, c: "cache/lead.wav")
        monkeypatch.setattr(feats, "load_vocal", lambda p: np.zeros(160, np.float32))
        monkeypatch.setattr(feats, "analyse_vocal", lambda y: "analysis")
        assert lead_vocal.lead_vocal_analysis("v.wav", "cache") == ("cache/lead.wav", "analysis")


class TestSegmenterLeadPitch:
    def _run(self, tmp_path, monkeypatch, lead_result):
        model = tmp_path / "m.pt"
        save_model(model, SegNet())
        vocal = tmp_path / "v.wav"
        vocal.write_bytes(b"x")
        import modules.Segmentation.decode as dec
        import modules.Segmentation.features as feats
        monkeypatch.setattr(feats, "load_vocal", lambda p: np.zeros(16000, np.float32))
        monkeypatch.setattr(feats, "analyse_vocal", lambda y: _analysis(200))
        seen = {}

        def fake_decode(probs, onset, analysis, pitch_analysis=None, check_analysis=None, **kw):
            seen["pitch_analysis"] = pitch_analysis
            seen["check_analysis"] = check_analysis
            return _notes((0.0, 0.5))
        monkeypatch.setattr(dec, "decode_notes", fake_decode)
        monkeypatch.setattr(lead_vocal, "lead_vocal_analysis", lead_result)
        result = segmenter.segment_with_model([_seg("hello ", 0, 1)], str(vocal), str(model), "en",
                                              lead_vocal_pitch=True, cache_folder=str(tmp_path))
        return result, seen

    def test_reliable_lead_is_used(self, tmp_path, monkeypatch):
        lead = _analysis(200, midi=64)
        result, seen = self._run(tmp_path, monkeypatch, lambda p, c: ("lead.wav", lead))
        assert result.segments and seen["pitch_analysis"] is lead and seen["check_analysis"] is None
        assert result.pitch_audio_path == "lead.wav"  # later steps compare the notes with the lead stem

    def test_unreliable_lead_only_checks_the_pitches(self, tmp_path, monkeypatch):
        lead = _analysis(200, midi=64)
        lead.f0_conf[:150] = 0.1
        result, seen = self._run(tmp_path, monkeypatch, lambda p, c: ("lead.wav", lead))
        assert result.segments and seen["pitch_analysis"] is None and seen["check_analysis"] is lead
        assert result.pitch_audio_path == str(tmp_path / "v.wav")

    def test_separation_error_falls_back(self, tmp_path, monkeypatch):
        def boom(p, c):
            raise RuntimeError("no model")
        result, seen = self._run(tmp_path, monkeypatch, boom)
        assert result.segments and seen["pitch_analysis"] is None and seen["check_analysis"] is None
        assert result.pitch_audio_path == str(tmp_path / "v.wav")
