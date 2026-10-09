"""Tests for tools/chart_benchmark.py — generated chart vs reference chart.

Everything runs on small synthetic charts and synthetic pitch data; no audio
is processed and UltraSinger itself is never started.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import chart_benchmark as cb  # noqa: E402
from chart_benchmark import ChartNote, SungPitch  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

def _note(start, end, midi, kind=":", word="la"):
    return ChartNote(float(start), float(end), midi, kind, word)


def _melody(n=40, step_ms=500, dur_ms=400, base=60):
    """Simple alternating melody: n notes, one every step_ms."""
    return [_note(i * step_ms, i * step_ms + dur_ms, base + (i % 5)) for i in range(n)]


def _sung_from(notes, shift_ms=0.0, hop_s=0.016):
    """Confident sung pitch exactly on the given notes, shifted in time."""
    times, midi = [], []
    for n in notes:
        if not n.pitched:
            continue
        t = (n.start_ms + shift_ms) / 1000.0
        while t < (n.end_ms + shift_ms) / 1000.0:
            times.append(t)
            midi.append(float(n.midi))
            t += hop_s
    return SungPitch(np.array(times), np.array(midi))


def _write_chart(path: Path, notes_lines: list[str], bpm="300", gap="1000",
                 extra_headers: list[str] | None = None) -> Path:
    head = ["#TITLE:Title", "#ARTIST:Artist", "#MP3:song.mp3", f"#BPM:{bpm}", f"#GAP:{gap}"]
    path.write_text("\n".join(head + (extra_headers or []) + notes_lines + ["E"]), encoding="utf-8")
    return path


def _note_lines(count, pitch=60, length=2, step=4):
    return [f": {i * step} {length} {pitch + (i % 3)} la" for i in range(count)]


# ── basic helpers ────────────────────────────────────────────────────────────

class TestFold:
    def test_octaves_fold_to_zero(self):
        assert cb.fold(12) == 0
        assert cb.fold(-24) == 0

    def test_small_differences_unchanged(self):
        assert cb.fold(1) == 1
        assert cb.fold(-1) == -1

    def test_range(self):
        vals = cb.fold(np.arange(-30, 30))
        assert vals.min() >= -6 and vals.max() < 6


class TestFrameGrid:
    def test_marks_note_frames(self):
        g = cb.frame_grid([_note(100, 200, 64)], 50)
        assert np.isnan(g[9]) and g[10] == 64 and g[19] == 64 and np.isnan(g[20])

    def test_freestyle_excluded_from_pitched_grid(self):
        g = cb.frame_grid([_note(0, 100, 64, kind="F")], 20)
        assert np.isnan(g).all()

    def test_freestyle_grid(self):
        g = cb.frame_grid([_note(0, 100, 64, kind="R")], 20, cb.FREESTYLE_TYPES)
        assert not np.isnan(g[:10]).any()

    def test_negative_times_clipped(self):
        g = cb.frame_grid([_note(-100, 50, 60)], 20)
        assert g[0] == 60 and g[4] == 60 and np.isnan(g[5])


class TestLoadChart:
    def test_legacy_pitch_and_timing(self, tmp_path):
        p = _write_chart(tmp_path / "a.txt", [": 0 4 60 Hel", ": 4 4 62 lo"], bpm="300", gap="1000")
        notes = cb.load_chart(p)
        assert [n.midi for n in notes] == [60, 62]
        # 300 header BPM -> 1200 beats/min -> 50 ms per beat
        assert notes[0].start_ms == pytest.approx(1000)
        assert notes[0].end_ms == pytest.approx(1200)
        assert notes[1].word == "lo"

    def test_relative_format_pitch(self, tmp_path):
        p = _write_chart(tmp_path / "a.txt", [": 0 4 0 la"], extra_headers=["#VERSION:1.2.0"])
        assert cb.load_chart(p)[0].midi == 48


# ── metrics ──────────────────────────────────────────────────────────────────

class TestChartMetrics:
    def test_identical_charts(self):
        ref = _melody()
        m = cb.chart_metrics(ref, list(ref))
        assert m["chart_agreement_pct"] == 100.0
        assert m["onset_hit_50_pct"] == 100.0
        assert m["onset_precision_100_pct"] == 100.0
        assert m["note_count_ratio"] == 1.0
        assert m["extra_time_pct"] == 0.0

    def test_octave_shift_is_ignored(self):
        ref = _melody()
        gen = [_note(n.start_ms, n.end_ms, n.midi - 12) for n in ref]
        assert cb.chart_metrics(ref, gen)["chart_agreement_pct"] == 100.0

    def test_wrong_pitch_lowers_agreement(self):
        ref = _melody()
        gen = [_note(n.start_ms, n.end_ms, n.midi + 3) for n in ref]
        m = cb.chart_metrics(ref, gen)
        assert m["chart_agreement_pct"] == 0.0
        assert m["ref_coverage_pct"] == 100.0

    def test_late_onsets(self):
        ref = _melody()
        gen = [_note(n.start_ms + 150, n.end_ms + 150, n.midi) for n in ref]
        m = cb.chart_metrics(ref, gen)
        assert m["onset_hit_100_pct"] == 0.0
        assert m["chart_agreement_pct"] < 100.0

    def test_extra_notes_count_as_extra_time(self):
        ref = _melody(n=10)
        gen = list(ref) + [_note(10_000, 12_000, 60)]
        m = cb.chart_metrics(ref, gen)
        assert m["extra_time_pct"] > 0
        assert m["onset_precision_100_pct"] < 100.0

    def test_freestyle_charted_as_melody(self):
        ref = _melody(n=4) + [_note(3000, 4000, 60, kind="F")]
        gen = _melody(n=4) + [_note(3000, 4000, 60)]
        assert cb.chart_metrics(ref, gen)["freestyle_charted_pct"] == 100.0

    def test_generated_freestyle_is_not_pitched(self):
        ref = _melody(n=4)
        gen = [_note(n.start_ms, n.end_ms, n.midi, kind="F") for n in ref]
        m = cb.chart_metrics(ref, gen)
        assert m["gen_notes"] == 0
        assert m["chart_agreement_pct"] == 0.0
        assert m["median_note_ms"] is None

    def test_empty_generated_chart(self):
        m = cb.chart_metrics(_melody(n=4), [])
        assert m["onset_hit_100_pct"] == 0.0
        assert m["note_count_ratio"] == 0.0

    def test_vocal_metrics(self):
        ref = _melody()
        sung = _sung_from(ref)
        m = cb.chart_metrics(ref, list(ref), sung)
        assert m["oracle_pitch_pct"] == 100.0
        assert m["vocal_hits_ref_pct"] == 100.0
        assert m["vocal_hits_gen_pct"] == 100.0


class TestOffsetFit:
    def test_finds_shift(self):
        ref = _melody()
        sung = _sung_from(ref, shift_ms=200)
        offset, fit = cb.fit_reference_offset(ref, sung)
        assert abs(offset - 200) <= 20
        assert fit > 0.9

    def test_zero_shift(self):
        ref = _melody()
        offset, _ = cb.fit_reference_offset(ref, _sung_from(ref))
        assert abs(offset) <= 10

    def test_no_sung_frames(self):
        offset, fit = cb.fit_reference_offset(_melody(), SungPitch(np.array([]), np.array([])))
        assert offset == 0.0 and fit == 0.0


class TestSungPitch:
    def test_confidence_and_silence_filtered(self, tmp_path):
        p = tmp_path / "pitch.json"
        p.write_text(json.dumps({"times": [0.0, 0.1, 0.2], "frequencies": [440.0, 440.0, 10.0],
                                 "confidence": [0.9, 0.2, 0.9]}), encoding="utf-8")
        s = SungPitch.from_pitch_json(p)
        assert list(s.times) == [0.0]
        assert s.midi[0] == pytest.approx(69.0)


# ── lyrics ───────────────────────────────────────────────────────────────────

def _words(path, shift_ms=0.0):
    return [(w.text, round(w.start_ms)) for w in cb.load_words(path, shift_ms)]


def _word(text, *spans):
    return cb.ChartWord(text, [(float(a), float(b)) for a, b in spans])


class TestNormalizeWord:
    @pytest.mark.parametrize("raw, key", [("Don't", "dont"), ("Café", "cafe"), ("ÜBER!", "uber"),
                                          ("Straße", "strasse"), ("rock-n-roll", "rocknroll"), ("...", "")])
    def test_keys(self, raw, key):
        assert cb.normalize_word(raw) == key


class TestLoadWords:
    def test_leading_space_starts_a_word(self, tmp_path):
        p = _write_chart(tmp_path / "a.txt", [": 0 2 60 Hel", ": 2 2 60 lo", ": 4 2 60  world"])
        assert _words(p) == [("hello", 1000), ("world", 1200)]

    def test_trailing_space_ends_a_word(self, tmp_path):
        p = _write_chart(tmp_path / "a.txt", [": 0 2 60 Hel", ": 2 2 60 lo ", ": 4 2 60 world"])
        assert _words(p) == [("hello", 1000), ("world", 1200)]

    def test_line_break_ends_a_word(self, tmp_path):
        p = _write_chart(tmp_path / "a.txt", [": 0 2 60 one", "- 3", ": 4 2 60 two"])
        assert _words(p) == [("one", 1000), ("two", 1200)]

    def test_continuation_notes_extend_the_word(self, tmp_path):
        p = _write_chart(tmp_path / "a.txt", [": 0 2 60 mind", ": 2 2 62 ~", ": 4 2 64 ~ ", ": 6 2 60 again"])
        words = cb.load_words(p)
        assert [w.text for w in words] == ["mind", "again"]
        assert words[0].spans == [(1000, 1100), (1100, 1200), (1200, 1300)]

    def test_continuation_after_a_word_end_keeps_the_next_word_separate(self, tmp_path):
        p = _write_chart(tmp_path / "a.txt", [": 0 2 60 mind ", ": 2 2 62 ~", ": 4 2 60 again"])
        assert [w.text for w in cb.load_words(p)] == ["mind", "again"]

    def test_several_words_on_one_note_share_its_time(self, tmp_path):
        p = _write_chart(tmp_path / "a.txt", [": 0 4 60 ab cd "])
        words = cb.load_words(p)
        assert [w.text for w in words] == ["ab", "cd"]
        assert words[0].spans == [(1000, 1100)] and words[1].spans == [(1100, 1200)]

    def test_punctuation_only_and_shift(self, tmp_path):
        p = _write_chart(tmp_path / "a.txt", [": 0 2 60 Hi ", ": 2 2 60 !", "F 4 2 60  yo"])
        assert _words(p, shift_ms=-100) == [("hi", 900), ("yo", 1100)]

    def test_cp1252_text(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_bytes("\n".join(["#BPM:300", "#GAP:0", ": 0 2 60 sch\xf6n", "E"]).encode("cp1252"))
        assert _words(p) == [("schon", 0)]


class TestAlignWords:
    def test_order_preserving(self):
        ref = [_word("a", (0, 100)), _word("b", (200, 300))]
        assert cb.align_words(ref, [_word("b", (0, 100)), _word("a", (200, 300))]) in ([(0, 1)], [(1, 0)])

    def test_missing_word_skipped(self):
        ref = [_word("a", (0, 100)), _word("b", (200, 300)), _word("c", (400, 500))]
        assert cb.align_words(ref, [_word("a", (0, 100)), _word("c", (400, 500))]) == [(0, 0), (2, 1)]

    def test_repeated_word_matched_to_the_closer_occurrence(self):
        ref = [_word("oh", (10_000, 10_200))]
        gen = [_word("oh", (0, 200)), _word("oh", (10_100, 10_300))]
        assert cb.align_words(ref, gen) == [(0, 1)]

    def test_empty(self):
        assert cb.align_words([], [_word("a", (0, 1))]) == []


class TestLyricsMetrics:
    def _ref(self):
        return [_word(t, (i * 500, i * 500 + 400)) for i, t in enumerate(["one", "two", "three", "four"])]

    def test_identical(self):
        m = cb.lyrics_metrics(self._ref(), self._ref())
        assert m["lyrics_agreement_pct"] == 100.0 and m["lyrics_agree_pct"] == 100.0
        assert m["lyrics_words_found_pct"] == 100.0 and m["word_start_100_pct"] == 100.0

    def test_words_one_note_late(self):
        ref = self._ref()
        gen = [_word("one", (0, 900))] + [_word(w.text, (w.spans[0][0] + 500, w.spans[0][1] + 500)) for w in ref[1:]]
        m = cb.lyrics_metrics(ref, gen)
        assert m["lyrics_agreement_pct"] == 25.0  # only "one" is under its notes
        assert m["lyrics_words_found_pct"] == 100.0
        assert m["word_start_250_pct"] == 25.0

    def test_missing_word_and_missing_notes(self):
        ref = self._ref()
        gen = [ref[0], ref[1], ref[3]]  # "three" neither sung nor charted
        m = cb.lyrics_metrics(ref, gen)
        assert m["lyrics_words_found_pct"] == 75.0
        assert m["lyrics_agreement_pct"] == 75.0
        assert m["lyrics_agree_pct"] == 100.0  # where both charts have a note, the text matches

    def test_wrong_word(self):
        ref = self._ref()
        gen = [ref[0], _word("too", ref[1].spans[0]), ref[2], ref[3]]
        m = cb.lyrics_metrics(ref, gen)
        assert m["lyrics_agreement_pct"] == 75.0 and m["lyrics_agree_pct"] == 75.0

    def test_no_reference_words(self):
        m = cb.lyrics_metrics([], self._ref())
        assert m["lyrics_agreement_pct"] is None and m["gen_words"] == 4


# ── library discovery ────────────────────────────────────────────────────────

class TestInspectSongFolder:
    def _song(self, tmp_path, name="Artist - Title", lines=None, headers=None, media=("song.mp3",)):
        d = tmp_path / name
        d.mkdir()
        for m in media:
            (d / m).write_bytes(b"x")
        _write_chart(d / f"{name}.txt", lines if lines is not None else _note_lines(150),
                     extra_headers=headers)
        return d

    def test_solo_song_qualifies(self, tmp_path):
        e = cb.inspect_song_folder(self._song(tmp_path))
        assert e is not None
        assert e["media"].endswith("song.mp3")
        assert e["pitched_notes"] == 150

    def test_duet_rejected(self, tmp_path):
        d = self._song(tmp_path, lines=["P1"] + _note_lines(150))
        assert cb.inspect_song_folder(d) is None

    def test_too_few_notes_rejected(self, tmp_path):
        assert cb.inspect_song_folder(self._song(tmp_path, lines=_note_lines(10))) is None

    def test_missing_media_rejected(self, tmp_path):
        assert cb.inspect_song_folder(self._song(tmp_path, media=())) is None

    def test_two_txt_rejected(self, tmp_path):
        d = self._song(tmp_path)
        (d / "other.txt").write_text("x", encoding="utf-8")
        assert cb.inspect_song_folder(d) is None

    def test_prefer_video(self, tmp_path):
        d = self._song(tmp_path, headers=["#VIDEO:clip.mp4"], media=("song.mp3", "clip.mp4"))
        assert cb.inspect_song_folder(d)["media"].endswith("song.mp3")
        assert cb.inspect_song_folder(d, prefer_video=True)["media"].endswith("clip.mp4")

    def test_cp1252_chart(self, tmp_path):
        d = tmp_path / "Artist - Title"
        d.mkdir()
        (d / "song.mp3").write_bytes(b"x")
        text = "\n".join(["#TITLE:Gr\xfc\xdfe", "#MP3:song.mp3", "#BPM:300", "#GAP:0"] + _note_lines(150))
        (d / "song.txt").write_bytes(text.encode("cp1252"))
        assert cb.inspect_song_folder(d) is not None


class TestSampling:
    def test_reproducible_and_anonymous(self, tmp_path):
        cands = [{"folder": f"f{i}", "txt": "", "media": "", "pitched_notes": 200} for i in range(20)]
        a = cb.sample_songs(cands, 5, seed=3)
        b = cb.sample_songs(cands, 5, seed=3)
        assert a == b
        assert [s["id"] for s in a] == [f"song_{i:03d}" for i in range(1, 6)]

    def test_count_capped(self):
        cands = [{"folder": "f", "txt": "", "media": "", "pitched_notes": 200}]
        assert len(cb.sample_songs(cands, 5, seed=1)) == 1


class TestCleanInputName:
    def test_strips_bracket_tags(self):
        assert cb._clean_input_name(Path("Artist - Title [VD#0].AVI")) == "Artist - Title.avi"

    def test_fallback(self):
        assert cb._clean_input_name(Path("[x].mp3")) == "song.mp3"


class TestChooseInput:
    def _song(self, media, audio="song.mp3"):
        return {"media": media, "audio": audio}

    def test_silent_video_falls_back_to_audio(self, monkeypatch):
        monkeypatch.setattr(cb, "has_audio_stream", lambda p: False)
        assert cb.choose_input(self._song("clip.mp4")).name == "song.mp3"

    def test_video_with_sound_kept(self, monkeypatch):
        monkeypatch.setattr(cb, "has_audio_stream", lambda p: True)
        assert cb.choose_input(self._song("clip.mp4")).name == "clip.mp4"

    def test_silent_video_without_audio_file_kept(self, monkeypatch):
        monkeypatch.setattr(cb, "has_audio_stream", lambda p: False)
        assert cb.choose_input(self._song("clip.mp4", audio=None)).name == "clip.mp4"

    def test_audio_media_not_probed(self, monkeypatch):
        def boom(p):
            raise AssertionError("must not probe audio files")
        monkeypatch.setattr(cb, "has_audio_stream", boom)
        assert cb.choose_input(self._song("song.mp3")).name == "song.mp3"

    def test_missing_ffprobe_assumes_sound(self, monkeypatch):
        def no_exe(*a, **k):
            raise OSError("not found")
        monkeypatch.setattr(cb.subprocess, "run", no_exe)
        assert cb.has_audio_stream(Path("clip.mp4")) is True


class TestConvertSong:
    def _song(self, tmp_path):
        media = tmp_path / "lib" / "Artist - Title.mp3"
        media.parent.mkdir(exist_ok=True)
        media.write_bytes(b"x")
        return {"id": "song_001", "media": str(media), "audio": str(media)}

    def test_timeout_recorded_as_failure(self, tmp_path, monkeypatch):
        def hang(cmd, **kwargs):
            assert kwargs["timeout"] == 5
            raise cb.subprocess.TimeoutExpired(cmd, 5, output=b"partial out", stderr=None)
        monkeypatch.setattr(cb.subprocess, "run", hang)
        run_dir = tmp_path / "run"
        res = cb.convert_song(self._song(tmp_path), run_dir, [], "python", timeout_s=5)
        assert res["timed_out"] is True
        assert res["returncode"] == -1 and res["txt"] is None
        log = (run_dir / "log.txt").read_text(encoding="utf-8")
        assert "partial out" in log and "TIMEOUT" in log
        assert (run_dir / "result.json").exists()
        assert not (run_dir / "input").exists()

    def test_success_and_resume(self, tmp_path, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            out = Path(cmd[cmd.index("-o") + 1]) / "Artist - Title"
            out.mkdir(parents=True)
            (out / "Artist - Title.txt").write_text("x", encoding="utf-8")
            (out / "Artist - Title.mp3").write_bytes(b"0")
            return cb.subprocess.CompletedProcess(cmd, 0, "ok", "")
        monkeypatch.setattr(cb.subprocess, "run", fake_run)
        run_dir = tmp_path / "run"
        res = cb.convert_song(self._song(tmp_path), run_dir, ["--x"], "python")
        assert res["returncode"] == 0 and res["txt"].endswith("Artist - Title.txt")
        assert "--keep_cache" in calls[0] and "--x" in calls[0]
        assert not list(run_dir.rglob("*.mp3"))  # pruned
        cb.convert_song(self._song(tmp_path), run_dir, [], "python")
        assert len(calls) == 1  # second call resumed from result.json


class TestInputName:
    def _song(self, tmp_path, headers):
        txt = tmp_path / "ref.txt"
        txt.write_text("\n".join(headers + [": 0 1 60 la", "E"]), encoding="utf-8")
        return {"txt": str(txt)}

    def test_name_from_chart_headers(self, tmp_path):
        song = self._song(tmp_path, ["#ARTIST:Some Artist", "#TITLE:Some Title"])
        assert cb.input_name(song, Path("clip [VD#0].AVI")) == "Some Artist - Some Title.avi"

    def test_invalid_characters_removed(self, tmp_path):
        song = self._song(tmp_path, ["#ARTIST:A/B", "#TITLE:Why? "])
        assert cb.input_name(song, Path("x.mp3")) == "AB - Why.mp3"

    def test_long_names_truncated(self, tmp_path):
        song = self._song(tmp_path, ["#ARTIST:" + "A" * 200, "#TITLE:" + "T" * 200])
        name = cb.input_name(song, Path("x.MP4"))
        assert name.endswith(".mp4")
        stem = Path(name).stem
        assert len(stem) <= cb.MAX_INPUT_STEM
        artist, title = stem.split(" - ")
        assert artist.startswith("A") and title.startswith("T")

    @pytest.mark.parametrize("artist_len, title_len", [(200, 5), (5, 200), (60, 60), (97, 1)])
    def test_truncation_keeps_both_fields(self, tmp_path, artist_len, title_len):
        song = self._song(tmp_path, ["#ARTIST:" + "A" * artist_len, "#TITLE:" + "T" * title_len])
        stem = Path(cb.input_name(song, Path("x.mp3"))).stem
        assert len(stem) <= cb.MAX_INPUT_STEM
        artist, title = stem.split(" - ")
        assert artist and title
        if artist_len + title_len + 3 <= cb.MAX_INPUT_STEM:
            assert (len(artist), len(title)) == (artist_len, title_len)

    def test_leading_separators_do_not_empty_title(self, tmp_path):
        song = self._song(tmp_path, ["#ARTIST:" + "A" * 200, "#TITLE:" + "-" * 80 + "Real Title"])
        stem = Path(cb.input_name(song, Path("x.mp3"))).stem
        artist, title = stem.split(" - ")
        assert artist and title.startswith("Real")

    def test_separator_only_field_falls_back(self, tmp_path):
        song = self._song(tmp_path, ["#ARTIST:Artist", "#TITLE: - . -"])
        assert cb.input_name(song, Path("Media Name.mp3")) == "Media Name.mp3"

    def test_short_field_kept_whole(self, tmp_path):
        song = self._song(tmp_path, ["#ARTIST:" + "A" * 200, "#TITLE:Short Title"])
        assert Path(cb.input_name(song, Path("x.mp3"))).stem.endswith(" - Short Title")

    def test_fallback_to_media_name(self, tmp_path):
        song = self._song(tmp_path, ["#TITLE:Only Title"])
        assert cb.input_name(song, Path("Artist - Title [CO].mp3")) == "Artist - Title.mp3"

    def test_missing_txt_falls_back(self, tmp_path):
        assert cb.input_name({"txt": str(tmp_path / "nope.txt")}, Path("a.mp3")) == "a.mp3"


class TestJoinArgsValue:
    def test_flag_value_joined(self):
        assert cb._join_args_value(["convert", "w", "--args", "--syllable_split"]) == \
            ["convert", "w", "--args=--syllable_split"]

    def test_equals_form_untouched(self):
        argv = ["convert", "w", "--args=--x 1"]
        assert cb._join_args_value(argv) == argv

    @pytest.mark.parametrize("own", ["--keep-audio", "--label", "--timeout=5", "--help"])
    def test_benchmark_options_not_consumed(self, own):
        argv = ["convert", "w", "--args", own]
        assert cb._join_args_value(argv) == argv

    def test_keep_audio_stays_a_benchmark_option(self):
        args = cb.build_parser().parse_args(
            cb._join_args_value(["convert", "w", "--args=--x", "--keep-audio"]))
        assert args.keep_audio is True and args.args == "--x"

    def test_parser_accepts_flag_value(self):
        args = cb.build_parser().parse_args(cb._join_args_value(["convert", "w", "--args", "--syllable_split"]))
        assert args.args == "--syllable_split"


class TestPrune:
    def test_keeps_only_txt_and_json(self, tmp_path):
        song = tmp_path / "out" / "Artist - Title"
        (song / "cache" / "separated" / "m").mkdir(parents=True)
        (song / "Artist - Title.txt").write_text("x", encoding="utf-8")
        (song / "Artist - Title.mp3").write_bytes(b"0" * 10)
        (song / "cache" / "pitch.json").write_text("{}", encoding="utf-8")
        (song / "cache" / "separated" / "m" / "vocals.wav").write_bytes(b"0" * 20)
        freed = cb.prune_run_output(tmp_path / "out")
        assert freed == 30
        assert (song / "Artist - Title.txt").exists()
        assert (song / "cache" / "pitch.json").exists()
        assert not (song / "cache" / "separated").exists()


# ── evaluation and reports ───────────────────────────────────────────────────

class TestEvaluateSong:
    def _setup(self, tmp_path, shift_beats=0, ref_fit_ok=True):
        lines = _note_lines(120)
        ref_txt = _write_chart(tmp_path / "ref.txt", lines)
        run = tmp_path / "run"
        song_out = run / "out" / "Artist - Title"
        (song_out / "cache").mkdir(parents=True)
        gen_lines = [f": {int(l.split()[1]) + shift_beats} {' '.join(l.split()[2:])}" for l in lines]
        gen_txt = _write_chart(song_out / "Artist - Title.txt", gen_lines)
        ref_notes = cb.load_chart(ref_txt)
        sung = _sung_from(ref_notes if ref_fit_ok else
                          [_note(n.start_ms, n.end_ms, n.midi + 4) for n in ref_notes])
        (song_out / "cache" / "swiftf0_False.json").write_text(json.dumps(
            {"times": sung.times.tolist(), "frequencies": (440 * 2 ** ((sung.midi - 69) / 12)).tolist(),
             "confidence": [0.95] * len(sung.times)}), encoding="utf-8")
        (run / "result.json").write_text(json.dumps({"returncode": 0, "seconds": 1, "txt": str(gen_txt)}),
                                         encoding="utf-8")
        return {"id": "song_001", "txt": str(ref_txt)}, run

    def test_perfect_generated_chart(self, tmp_path):
        song, run = self._setup(tmp_path)
        row = cb.evaluate_song(song, run, min_ref_fit=50, plots=False)
        assert row["status"] == "ok"
        assert row["chart_agreement_pct"] == 100.0
        assert abs(row["ref_offset_ms"]) <= 10
        assert row["lyrics_agreement_pct"] == 100.0 and row["lyrics_words_found_pct"] == 100.0

    def test_unreliable_reference_flagged(self, tmp_path):
        song, run = self._setup(tmp_path, ref_fit_ok=False)
        row = cb.evaluate_song(song, run, min_ref_fit=50, plots=False)
        assert row["status"] == "unreliable_reference"

    def test_no_output(self, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        (run / "result.json").write_text(json.dumps({"returncode": 1, "seconds": 1, "txt": None}),
                                         encoding="utf-8")
        assert cb.evaluate_song({"id": "song_001", "txt": ""}, run, 50, False)["status"] == "no_output"

    def test_plots_written(self, tmp_path):
        song, run = self._setup(tmp_path)
        cb.evaluate_song(song, run, min_ref_fit=50, plots=True)
        assert list((run / "plots").glob("page_*.png"))


class TestSummaries:
    def test_summary_uses_reliable_songs_only(self):
        rows = [{"id": "a", "status": "ok", "chart_agreement_pct": 80.0},
                {"id": "b", "status": "ok", "chart_agreement_pct": 60.0},
                {"id": "c", "status": "unreliable_reference", "chart_agreement_pct": 0.0},
                {"id": "d", "status": "no_output"}]
        s = cb.summarize(rows)
        assert s["songs_reliable"] == 2
        assert s["songs_unreliable_reference"] == 1
        assert s["songs_failed"] == 1
        assert s["median"]["chart_agreement_pct"] == 70.0

    def test_compare_deltas(self):
        a = {"median": {k: 50.0 for k in cb.SUMMARY_METRICS}}
        b = {"median": {k: 55.0 for k in cb.SUMMARY_METRICS}}
        b["median"]["onset_hit_50_pct"] = None
        rows = {k: d for k, _, _, d in cb.compare_summaries(a, b)}
        assert rows["chart_agreement_pct"] == 5.0
        assert rows["onset_hit_50_pct"] is None

    def test_markdown_has_only_ids(self):
        rows = [{"id": "song_001", "status": "ok", "chart_agreement_pct": 70.0}]
        md = cb.format_summary_md("base", cb.summarize(rows), rows)
        assert "song_001" in md and "chart_agreement_pct" in md


# ── CLI ──────────────────────────────────────────────────────────────────────

class TestCli:
    def test_sample_refuses_workdir_inside_repo(self, tmp_path):
        with pytest.raises(SystemExit, match="outside the repository"):
            cb.main(["sample", str(tmp_path), str(cb.REPO / "bench_tmp")])

    def test_sample_writes_songs_json(self, tmp_path):
        lib = tmp_path / "lib"
        for i in range(3):
            d = lib / f"Artist - Title {i}"
            d.mkdir(parents=True)
            (d / "song.mp3").write_bytes(b"x")
            _write_chart(d / "song.txt", _note_lines(150))
        work = tmp_path / "work"
        assert cb.main(["sample", str(lib), str(work), "--count", "2"]) == 0
        songs = json.loads((work / "songs.json").read_text(encoding="utf-8"))
        assert len(songs) == 2
        with pytest.raises(SystemExit, match="exists"):
            cb.main(["sample", str(lib), str(work)])

    @pytest.mark.parametrize("argv", [
        ["convert", "{wd}"],
        ["evaluate", "{wd}"],
        ["compare", "{wd}", "a", "b"],
    ])
    def test_all_commands_refuse_workdir_inside_repo(self, argv):
        wd = str(cb.REPO / "bench_tmp")
        with pytest.raises(SystemExit, match="outside the repository"):
            cb.main([a.format(wd=wd) for a in argv])

    def test_repo_root_refused(self):
        with pytest.raises(SystemExit, match="outside the repository"):
            cb.main(["evaluate", str(cb.REPO)])

    def test_invalid_label(self, tmp_path):
        with pytest.raises(SystemExit, match="invalid label"):
            cb.main(["evaluate", str(tmp_path), "--label", "../x"])

    def test_evaluate_and_compare(self, tmp_path, capsys):
        work = tmp_path / "work"
        ev = TestEvaluateSong()
        song, run = ev._setup(tmp_path)
        target = work / "runs" / "base" / "song_001"
        target.parent.mkdir(parents=True)
        run.rename(target)
        # result.json points at the old location of the generated TXT -> rewrite
        res = json.loads((target / "result.json").read_text(encoding="utf-8"))
        res["txt"] = str(next((target / "out").rglob("Artist - Title.txt")))
        (target / "result.json").write_text(json.dumps(res), encoding="utf-8")
        (work / "songs.json").write_text(json.dumps([song]), encoding="utf-8")
        assert cb.main(["evaluate", str(work), "--label", "base"]) == 0
        assert (work / "reports" / "base.json").exists()
        assert cb.main(["compare", str(work), "base", "base"]) == 0
        out = capsys.readouterr().out
        assert "chart_agreement_pct" in out and "lyrics_agreement_pct" in out

        other = tmp_path / "other_reports"
        assert cb.main(["evaluate", str(work), "--label", "base", "--reports-dir", str(other)]) == 0
        assert (other / "base.json").exists()
        capsys.readouterr()
        assert cb.main(["compare", str(work), "base", "base", "--reports-dir", str(other),
                        "--metric", "lyrics_agreement_pct"]) == 0
        assert "lyrics_agreement_pct per song: 0 higher, 0 lower of 1" in capsys.readouterr().out
        with pytest.raises(SystemExit, match="unknown metric"):
            cb.main(["compare", str(work), "base", "base", "--metric", "nope"])
