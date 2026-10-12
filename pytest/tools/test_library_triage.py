"""Tests for tools/library_triage.py -- destructive library triage tool.

Uses generic placeholder song names only (no copyrighted titles/artists),
per repo policy. No real audio/GPU is ever touched: audio files are fake
byte blobs, ffprobe is disabled (``ffprobe_path=None``) unless a test is
specifically exercising the ffprobe-failure path (mocked), and stage 2
separation/scoring are always injected fakes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

# The tool is a standalone script in tools/ -- import its internals directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import library_triage as lt  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_ffprobe(monkeypatch):
    """Tests use fake (non-media) audio bytes, so real ffprobe would always
    (correctly) reject them. Disable auto-detection by default; tests that
    specifically exercise the ffprobe path call evaluate_stage1() directly
    with an explicit ffprobe_path and/or a mocked probe_audio."""
    monkeypatch.setattr(lt, "_resolve_ffprobe", lambda: None)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_chart(
    dir_: Path,
    name: str,
    *,
    bpm: str = "120",
    gap: str = "0",
    title: str = "Song",
    notes: bool = True,
    audio_tag: str | None = "__default__",
    extra_lines: list[str] | None = None,
) -> Path:
    """Write a minimal UltraStar .txt chart. ``audio_tag`` of "__default__"
    means "<name>.ogg"; pass None to omit the #MP3 tag entirely, or an
    explicit string for a custom/missing reference."""
    lines = [f"#TITLE:{title}", f"#BPM:{bpm}", f"#GAP:{gap}"]
    if audio_tag == "__default__":
        lines.append(f"#MP3:{name}.ogg")
    elif audio_tag is not None:
        lines.append(f"#MP3:{audio_tag}")
    if extra_lines:
        lines.extend(extra_lines)
    if notes:
        lines.append(": 0 4 60 test")
    lines.append("E")
    txt_path = dir_ / f"{name}.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return txt_path


def _make_song(
    base: Path,
    name: str,
    *,
    with_audio: bool = True,
    audio_ext: str = "ogg",
    notes: bool = True,
    bpm: str = "120",
    audio_tag: str | None = "__default__",
) -> Path:
    """Create a fake song folder under *base* (creates parents too, so
    callers can pass a nested name like "Lib A/Sub/Song X")."""
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    leaf = Path(name).name
    _write_chart(d, leaf, bpm=bpm, notes=notes, audio_tag=audio_tag)
    if with_audio:
        (d / f"{leaf}.{audio_ext}").write_bytes(b"fake audio bytes, not real media")
    return d


def _song_folder(base: Path, rel: Path, txt_name: str) -> lt.SongFolder:
    d = base / rel
    return lt.SongFolder(dir_path=d, rel_path=str(rel), txt_path=d / txt_name)


# ── Discovery ────────────────────────────────────────────────────────────────

class TestDiscovery:
    def test_song_folder_is_discovered(self, tmp_path):
        _make_song(tmp_path, "Song A")
        folders = lt.find_song_folders(tmp_path)
        assert len(folders) == 1
        assert folders[0].rel_path == "Song A"
        assert folders[0].txt_path.name == "Song A.txt"

    def test_nested_song_folders_are_discovered(self, tmp_path):
        _make_song(tmp_path, "Lib A/Song One")
        _make_song(tmp_path, "Lib B/Sub/Song Two")
        folders = lt.find_song_folders(tmp_path)
        rels = sorted(f.rel_path for f in folders)
        assert rels == sorted([str(Path("Lib A/Song One")), str(Path("Lib B/Sub/Song Two"))])

    def test_collection_folder_without_txt_is_ignored(self, tmp_path):
        d = tmp_path / "Just Some Folder"
        d.mkdir()
        (d / "cover.jpg").write_bytes(b"not a chart")
        folders = lt.find_song_folders(tmp_path)
        assert folders == []

    def test_folder_with_unrelated_txt_is_not_a_song(self, tmp_path):
        d = tmp_path / "Random Folder"
        d.mkdir()
        (d / "readme.txt").write_text("just some notes about this folder", encoding="utf-8")
        folders = lt.find_song_folders(tmp_path)
        assert folders == []

    def test_skip_dir_is_pruned(self, tmp_path):
        _make_song(tmp_path, "Song A")
        skip = tmp_path / "_corrupt"
        _make_song(skip, "Song B")
        folders = lt.find_song_folders(tmp_path, skip_dir=skip)
        rels = [f.rel_path for f in folders]
        assert rels == ["Song A"]


# ── Stage 1 verdicts ─────────────────────────────────────────────────────────

class TestStage1:
    def test_ok_song_passes(self, tmp_path):
        _make_song(tmp_path, "Good Song")
        song = _song_folder(tmp_path, Path("Good Song"), "Good Song.txt")
        result = lt.evaluate_stage1(song, ffprobe_path=None)
        assert result.verdict == "ok"
        assert result.audio_path is not None

    def test_unparsable_txt_is_corrupt(self, tmp_path):
        # A non-numeric #BPM value makes the real parser raise ValueError.
        _make_song(tmp_path, "Broken Header", bpm="not-a-number")
        song = _song_folder(tmp_path, Path("Broken Header"), "Broken Header.txt")
        result = lt.evaluate_stage1(song, ffprobe_path=None)
        assert result.verdict == "corrupt"
        assert "unparsable_txt" in result.reason

    def test_zero_notes_is_corrupt(self, tmp_path):
        _make_song(tmp_path, "No Notes", notes=False)
        song = _song_folder(tmp_path, Path("No Notes"), "No Notes.txt")
        result = lt.evaluate_stage1(song, ffprobe_path=None)
        assert result.verdict == "corrupt"
        assert result.reason == "no_notes"

    def test_missing_audio_is_corrupt(self, tmp_path):
        _make_song(tmp_path, "No Audio", with_audio=False)
        song = _song_folder(tmp_path, Path("No Audio"), "No Audio.txt")
        result = lt.evaluate_stage1(song, ffprobe_path=None)
        assert result.verdict == "corrupt"
        assert result.reason == "no_audio"

    def test_untagged_audio_still_found_by_extension(self, tmp_path):
        # #MP3 tag omitted entirely, but a .ogg file sits in the folder --
        # must still be found and treated as present.
        _make_song(tmp_path, "Loose Audio", audio_tag=None)
        song = _song_folder(tmp_path, Path("Loose Audio"), "Loose Audio.txt")
        result = lt.evaluate_stage1(song, ffprobe_path=None)
        assert result.verdict == "ok"

    def test_ffprobe_failure_is_corrupt(self, tmp_path, monkeypatch):
        _make_song(tmp_path, "Bad Audio")
        song = _song_folder(tmp_path, Path("Bad Audio"), "Bad Audio.txt")
        monkeypatch.setattr(lt, "probe_audio", lambda path, ffprobe_path: (False, "decode failed"))
        result = lt.evaluate_stage1(song, ffprobe_path="fake-ffprobe")
        assert result.verdict == "corrupt"
        assert "audio_undecodable" in result.reason

    def test_ffprobe_success_passes(self, tmp_path, monkeypatch):
        _make_song(tmp_path, "Good Audio")
        song = _song_folder(tmp_path, Path("Good Audio"), "Good Audio.txt")
        monkeypatch.setattr(lt, "probe_audio", lambda path, ffprobe_path: (True, ""))
        result = lt.evaluate_stage1(song, ffprobe_path="fake-ffprobe")
        assert result.verdict == "ok"


# ── Move logic ───────────────────────────────────────────────────────────────

class TestMove:
    def test_dry_run_moves_nothing(self, tmp_path):
        source = tmp_path / "source"
        corrupt = tmp_path / "corrupt"
        song_dir = _make_song(source, "No Notes", notes=False)

        summary = lt.run(source, corrupt, apply=False, log=lambda *a: None)

        assert song_dir.exists()
        assert summary.would_move == 1
        assert summary.moved == 0

    def test_apply_moves_and_preserves_relative_structure(self, tmp_path):
        source = tmp_path / "source"
        corrupt = tmp_path / "corrupt"
        song_dir = _make_song(source, "Lib A/Sub/Broken Song", notes=False)
        assert song_dir.exists()

        summary = lt.run(source, corrupt, apply=True, log=lambda *a: None)

        assert not song_dir.exists()
        # Moved songs mirror the source folder's own name too ("source" here).
        dest = corrupt / "source" / "Lib A" / "Sub" / "Broken Song"
        assert dest.is_dir()
        assert (dest / "Broken Song.txt").exists()
        assert summary.moved == 1

    def test_healthy_song_is_kept_in_place(self, tmp_path):
        source = tmp_path / "source"
        corrupt = tmp_path / "corrupt"
        song_dir = _make_song(source, "Good Song")

        summary = lt.run(source, corrupt, apply=True, log=lambda *a: None)

        assert song_dir.exists()
        assert summary.kept == 1
        assert summary.moved == 0

    def test_existing_destination_is_not_overwritten(self, tmp_path):
        source = tmp_path / "source"
        corrupt = tmp_path / "corrupt"
        song_dir = _make_song(source, "Broken Song", notes=False)

        dest = corrupt / "source" / "Broken Song"
        dest.mkdir(parents=True)
        (dest / "sentinel.txt").write_text("do not touch", encoding="utf-8")

        summary = lt.run(source, corrupt, apply=True, log=lambda *a: None)

        # Original stays put, pre-existing destination content is untouched.
        assert song_dir.exists()
        assert (dest / "sentinel.txt").read_text(encoding="utf-8") == "do not touch"
        assert not (dest / "Broken Song.txt").exists()
        assert summary.errors == 1
        assert summary.moved == 0

    def test_corrupt_dir_inside_source_dir_raises(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        corrupt = source / "nested_corrupt"

        with pytest.raises(ValueError):
            lt.run(source, corrupt, apply=True, log=lambda *a: None)

    def test_source_equals_corrupt_dir_raises(self, tmp_path):
        same = tmp_path / "same"
        same.mkdir()
        with pytest.raises(ValueError):
            lt.run(same, same, apply=True, log=lambda *a: None)


# ── Stage 2 fail-safe ────────────────────────────────────────────────────────

class TestStage2FailSafe:
    def test_scoring_exception_never_moves_the_song(self, tmp_path):
        source = tmp_path / "source"
        corrupt = tmp_path / "corrupt"
        song_dir = _make_song(source, "Flaky Song")

        def raising_separate(audio_path, temp_dir):
            raise RuntimeError("separation blew up")

        def unused_score(txt_path, vocals_path):
            raise AssertionError("should never be reached")

        summary = lt.run(
            source, corrupt, apply=True, stage2=True,
            separate_fn=raising_separate, score_fn=unused_score,
            log=lambda *a: None,
        )

        assert song_dir.exists()  # never moved
        assert summary.errors == 1
        assert summary.moved == 0
        row = summary.all_rows["Flaky Song"]
        assert row["action"] == "error"
        assert row["stage1_verdict"] == "ok"

    def test_low_score_moves_song_under_apply(self, tmp_path):
        source = tmp_path / "source"
        corrupt = tmp_path / "corrupt"
        song_dir = _make_song(source, "Desynced Song")

        def fake_separate(audio_path, temp_dir):
            (temp_dir / "vocals.wav").write_bytes(b"fake vocals")

        def low_score(txt_path, vocals_path):
            return 10.0

        summary = lt.run(
            source, corrupt, apply=True, stage2=True, stage2_threshold=40.0,
            separate_fn=fake_separate, score_fn=low_score,
            log=lambda *a: None,
        )

        assert not song_dir.exists()
        assert (corrupt / "source" / "Desynced Song").is_dir()
        assert summary.moved == 1
        assert summary.stage2_fail == 1
        assert summary.stage1_corrupt == 0

    def test_high_score_keeps_song(self, tmp_path):
        source = tmp_path / "source"
        corrupt = tmp_path / "corrupt"
        song_dir = _make_song(source, "Fine Song")

        def fake_separate(audio_path, temp_dir):
            (temp_dir / "vocals.wav").write_bytes(b"fake vocals")

        def high_score(txt_path, vocals_path):
            return 90.0

        summary = lt.run(
            source, corrupt, apply=True, stage2=True, stage2_threshold=40.0,
            separate_fn=fake_separate, score_fn=high_score,
            log=lambda *a: None,
        )

        assert song_dir.exists()
        assert summary.kept == 1
        assert summary.moved == 0

    def test_stage1_corrupt_song_never_reaches_stage2(self, tmp_path):
        source = tmp_path / "source"
        corrupt = tmp_path / "corrupt"
        _make_song(source, "No Notes At All", notes=False)

        calls = []

        def spy_separate(audio_path, temp_dir):
            calls.append(audio_path)
            (temp_dir / "vocals.wav").write_bytes(b"x")

        summary = lt.run(
            source, corrupt, apply=True, stage2=True,
            separate_fn=spy_separate, score_fn=lambda *a: 90.0,
            log=lambda *a: None,
        )

        assert calls == []  # stage 2 never invoked for a stage-1 failure
        assert summary.moved == 1
        assert summary.stage1_corrupt == 1
        assert summary.stage2_fail == 0

    def test_temp_vocals_removed_after_scoring(self, tmp_path):
        source = tmp_path / "source"
        corrupt = tmp_path / "corrupt"
        _make_song(source, "Song")

        seen_temp_dirs = []

        def fake_separate(audio_path, temp_dir):
            seen_temp_dirs.append(temp_dir)
            (temp_dir / "vocals.wav").write_bytes(b"x")
            (temp_dir / "no_vocals.wav").write_bytes(b"y")

        lt.run(
            source, corrupt, apply=True, stage2=True,
            separate_fn=fake_separate, score_fn=lambda *a: 90.0,
            log=lambda *a: None,
        )

        assert seen_temp_dirs
        # The temp dir itself is torn down at the end of the whole run.
        assert not seen_temp_dirs[0].exists()


# ── Resume ───────────────────────────────────────────────────────────────────

class TestResume:
    def test_kept_song_is_skipped_on_resume(self, tmp_path):
        source = tmp_path / "source"
        corrupt = tmp_path / "corrupt"
        _make_song(source, "Good Song")
        log_path = tmp_path / "log.csv"

        summary1 = lt.run(source, corrupt, apply=True, log_path=log_path, log=lambda *a: None)
        assert summary1.kept == 1
        assert summary1.skipped_resume == 0

        summary2 = lt.run(source, corrupt, apply=True, log_path=log_path, log=lambda *a: None)
        assert summary2.skipped_resume == 1
        assert summary2.kept == 0

    def test_moved_song_is_skipped_on_resume(self, tmp_path):
        source = tmp_path / "source"
        corrupt = tmp_path / "corrupt"
        _make_song(source, "Broken Song", notes=False)
        log_path = tmp_path / "log.csv"

        summary1 = lt.run(source, corrupt, apply=True, log_path=log_path, log=lambda *a: None)
        assert summary1.moved == 1

        # Song folder is gone now (really moved) so it can't be rediscovered
        # anyway -- but the resume bookkeeping must also treat it as done.
        summary2 = lt.run(source, corrupt, apply=True, log_path=log_path, log=lambda *a: None)
        assert summary2.total == 0
        assert summary2.moved == 0

    def test_error_rows_are_retried_not_skipped(self, tmp_path):
        source = tmp_path / "source"
        corrupt = tmp_path / "corrupt"
        _make_song(source, "Flaky Song")
        log_path = tmp_path / "log.csv"

        call_count = [0]

        def counting_separate(audio_path, temp_dir):
            call_count[0] += 1
            raise RuntimeError("boom")

        lt.run(
            source, corrupt, apply=True, stage2=True, log_path=log_path,
            separate_fn=counting_separate, score_fn=lambda *a: 0.0,
            log=lambda *a: None,
        )
        assert call_count[0] == 1

        lt.run(
            source, corrupt, apply=True, stage2=True, log_path=log_path,
            separate_fn=counting_separate, score_fn=lambda *a: 0.0,
            log=lambda *a: None,
        )
        # An "error" verdict is not terminal -- it must be retried.
        assert call_count[0] == 2

    def test_would_move_rows_are_retried_not_skipped(self, tmp_path):
        source = tmp_path / "source"
        corrupt = tmp_path / "corrupt"
        _make_song(source, "Broken Song", notes=False)
        log_path = tmp_path / "log.csv"

        summary1 = lt.run(source, corrupt, apply=False, log_path=log_path, log=lambda *a: None)
        assert summary1.would_move == 1

        summary2 = lt.run(source, corrupt, apply=False, log_path=log_path, log=lambda *a: None)
        assert summary2.skipped_resume == 0
        assert summary2.would_move == 1


# ── CSV round trip ───────────────────────────────────────────────────────────

class TestCsv:
    def test_load_existing_results_missing_file(self, tmp_path):
        assert lt.load_existing_results(tmp_path / "nope.csv") == {}

    def test_append_and_reload_round_trip(self, tmp_path):
        p = tmp_path / "log.csv"
        row = {"rel_path": "Song A", "stage1_verdict": "ok", "stage2_score": "",
               "action": "kept", "reason": ""}
        lt.append_result(p, row, write_header=True)
        loaded = lt.load_existing_results(p)
        assert "Song A" in loaded
        assert loaded["Song A"]["action"] == "kept"
        assert list(loaded["Song A"].keys()) == list(lt.CSV_FIELDS)


class TestStage1KeptRecheckedInStage2:
    """A song kept by a stage-1-only pass must be re-evaluated when a later
    pass enables stage 2 (both sharing the same log)."""

    def _make_song(self, root, name):
        d = root / name
        d.mkdir(parents=True)
        (d / "s.txt").write_text("#TITLE:x\n#BPM:200\n#MP3:s.mp3\n: 0 4 60 la\nE\n",
                                 encoding="utf-8")
        (d / "s.mp3").write_bytes(b"x")
        return d

    def test_kept_without_stage2_is_rechecked(self, tmp_path, monkeypatch):
        src = tmp_path / "src"; corrupt = tmp_path / "corrupt"
        self._make_song(src, "SongA")
        log = tmp_path / "log.csv"
        # Simulate a prior stage-1-only run: song kept, no stage2_score.
        log.write_text(
            "rel_path,stage1_verdict,stage2_score,action,reason\n"
            "SongA,ok,,kept,\n", encoding="utf-8")
        called = {"n": 0}
        def fake_stage2(*a, **k):
            called["n"] += 1
            return "ok", "", 95.0
        monkeypatch.setattr(lt, "evaluate_stage2", fake_stage2)
        # A fake separator, so no real separation model is loaded.
        lt.run(src, corrupt, apply=False, stage2=True, log_path=log,
               ffprobe_path=None, separate_fn=lambda audio_path, temp_dir: None,
               log=lambda *a: None)
        assert called["n"] == 1  # stage2 ran on the previously-kept song


# ── Nested song folders ──────────────────────────────────────────────────────

class TestNestedSongFolders:
    """A flagged folder that holds other song folders (e.g. a collection
    folder with a stray chart) must never be moved: the songs inside would
    go along."""

    def _library(self, tmp_path):
        source = tmp_path / "source"
        collection = source / "Collection"
        collection.mkdir(parents=True)
        _write_chart(collection, "Stray Chart")  # no audio anywhere -> no_audio
        healthy = _make_song(collection, "Song A")
        broken = _make_song(collection, "Sub/Song B", notes=False)
        return source, collection, healthy, broken

    def test_dry_run_reports_the_folder_instead_of_moving_it(self, tmp_path):
        source, _, _, _ = self._library(tmp_path)
        summary = lt.run(source, tmp_path / "corrupt", apply=False, log=lambda *a: None)
        rows = summary.all_rows
        assert rows["Collection"]["action"] == "error"
        assert "contains_song_folders_skipped: 2 song folder(s)" in rows["Collection"]["reason"]
        assert rows[str(Path("Collection/Song A"))]["action"] == "kept"
        assert rows[str(Path("Collection/Sub/Song B"))]["action"] == "would_move"
        assert summary.would_move == 1

    def test_apply_moves_only_the_broken_song(self, tmp_path):
        source, collection, healthy, broken = self._library(tmp_path)
        corrupt = tmp_path / "corrupt"
        logged = []

        summary = lt.run(source, corrupt, apply=True, log=logged.append)

        assert (collection / "Stray Chart.txt").exists()
        assert (healthy / "Song A.txt").exists() and (healthy / "Song A.ogg").exists()
        assert not broken.exists()
        assert (corrupt / "source" / "Collection" / "Sub" / "Song B" / "Song B.txt").exists()
        assert not (corrupt / "source" / "Collection" / "Song A").exists()
        assert summary.moved == 1 and summary.kept == 1 and summary.errors == 1
        assert summary.all_rows[str(Path("Collection/Song A"))]["action"] == "kept"
        assert any(line.startswith("ERROR Collection") for line in logged)

    def test_song_folder_with_other_subfolders_is_still_moved(self, tmp_path):
        source = tmp_path / "source"
        song_dir = _make_song(source, "Broken Song", notes=False)
        (song_dir / "covers").mkdir()
        (song_dir / "covers" / "front.jpg").write_bytes(b"x")
        (song_dir / "covers" / "readme.txt").write_text("not a chart", encoding="utf-8")

        summary = lt.run(source, tmp_path / "corrupt", apply=True, log=lambda *a: None)

        assert summary.moved == 1 and not song_dir.exists()

    def test_unreadable_nested_txt_counts_as_a_song(self, tmp_path, monkeypatch):
        source = tmp_path / "source"
        song_dir = _make_song(source, "Broken Song", notes=False)
        (song_dir / "inner").mkdir()
        locked = song_dir / "inner" / "locked.txt"
        locked.write_text("#TITLE:x", encoding="utf-8")
        real_read, real_parse = lt._read_text_lenient, lt.parse_score_song

        def read(path):
            if path == locked:
                raise PermissionError(13, "locked")
            return real_read(path)

        def parse(path):
            if Path(path) == locked:
                raise PermissionError(13, "locked")
            return real_parse(path)
        monkeypatch.setattr(lt, "_read_text_lenient", read)
        monkeypatch.setattr(lt, "parse_score_song", parse)

        summary = lt.run(source, tmp_path / "corrupt", apply=True, log=lambda *a: None)

        assert song_dir.exists() and locked.exists() and summary.moved == 0
        assert "contains_song_folders_skipped" in summary.all_rows["Broken Song"]["reason"]
        assert summary.all_rows[str(Path("Broken Song/inner"))]["action"] == "error"

    def test_utf16_song_inside_a_flagged_folder_is_not_taken_along(self, tmp_path):
        source = tmp_path / "source"
        collection = source / "Collection"
        collection.mkdir(parents=True)
        _write_chart(collection, "Stray Chart")  # no audio anywhere -> no_audio
        song_dir = collection / "Song U"
        song_dir.mkdir()
        chart = "#TITLE:Song U\r\n#ARTIST:Artist\r\n#MP3:Song U.ogg\r\n#BPM:300\r\n#GAP:0\r\n: 0 4 60 la\r\nE\r\n"
        (song_dir / "Song U.txt").write_bytes(chart.encode("utf-16"))
        (song_dir / "Song U.ogg").write_bytes(b"fake audio")

        summary = lt.run(source, tmp_path / "corrupt", apply=True, log=lambda *a: None)

        assert (song_dir / "Song U.txt").exists() and summary.moved == 0
        assert "contains_song_folders_skipped: 1" in summary.all_rows["Collection"]["reason"]
        assert summary.all_rows[str(Path("Collection/Song U"))]["action"] == "kept"


# ── Read errors and tool failures are not corruption ────────────────────────

class TestStage1ErrorsNeverMove:
    def test_missing_score_library_stops_the_run_before_anything_moves(self, tmp_path, monkeypatch):
        """Without the optional scoring package every chart would look
        unparsable: the run must refuse to start instead of moving the library."""
        source = tmp_path / "source"
        song_dir = _make_song(source, "Song A")
        monkeypatch.setitem(sys.modules, "ultrastar_score", None)
        monkeypatch.setitem(sys.modules, "ultrastar_score.parser", None)

        with pytest.raises(ValueError, match="scoring"):
            lt.run(source, tmp_path / "corrupt", apply=True, log=lambda *a: None)

        assert (song_dir / "Song A.txt").exists()
        assert not (tmp_path / "corrupt").exists()

    @pytest.mark.parametrize("error", [MemoryError(), ImportError("no module"), RuntimeError("boom")])
    def test_parser_failure_that_is_no_parse_error_never_moves(self, tmp_path, monkeypatch, error):
        source = tmp_path / "source"
        song_dir = _make_song(source, "Song A")

        def failing_parse(path):
            raise error
        monkeypatch.setattr(lt, "parse_score_song", failing_parse)

        assert lt.evaluate_stage1(lt.find_song_folders(source)[0], ffprobe_path=None).verdict == "error"
        summary = lt.run(source, tmp_path / "corrupt", apply=True, log=lambda *a: None)
        assert song_dir.exists() and summary.moved == 0
        row = summary.all_rows["Song A"]
        assert row["action"] == "error" and "parse_failed" in row["reason"]

    def test_unreadable_chart_is_not_judged_by_another_chart(self, tmp_path, monkeypatch):
        """The real chart can't be read right now, a stray chart next to it is
        broken: the folder must not be judged (and moved) by the stray one."""
        source = tmp_path / "source"
        song_dir = _make_song(source, "Song")  # Song.txt + Song.ogg
        _write_chart(song_dir, "Aaa Stray", bpm="fast")  # sorts first, unparsable
        real_parse = lt.parse_score_song

        def parse(path):
            if Path(path).name == "Song.txt":
                raise PermissionError(13, "file is locked")
            return real_parse(path)
        monkeypatch.setattr(lt, "parse_score_song", parse)

        summary = lt.run(source, tmp_path / "corrupt", apply=True, log=lambda *a: None)

        assert song_dir.exists() and summary.moved == 0
        assert "txt_unreadable" in summary.all_rows["Song"]["reason"]

    def test_folder_whose_only_txt_cannot_be_read_is_reported(self, tmp_path, monkeypatch):
        source = tmp_path / "source"
        song_dir = source / "Locked Song"
        song_dir.mkdir(parents=True)
        locked = song_dir / "Locked Song.txt"
        locked.write_text("#TITLE:x\n#BPM:120\n: 0 4 60 la\nE\n", encoding="utf-8")
        (song_dir / "Locked Song.ogg").write_bytes(b"fake audio")

        def unreadable(path):
            raise PermissionError(13, "file is locked")
        monkeypatch.setattr(lt, "_read_text_lenient", unreadable)
        monkeypatch.setattr(lt, "parse_score_song", unreadable)

        summary = lt.run(source, tmp_path / "corrupt", apply=True, log=lambda *a: None)

        assert locked.exists() and summary.moved == 0
        assert summary.all_rows["Locked Song"]["action"] == "error"

    def test_empty_gap_is_not_corrupt(self, tmp_path):
        d = tmp_path / "Empty Gap"
        d.mkdir()
        _write_chart(d, "Empty Gap", gap="")
        (d / "Empty Gap.ogg").write_bytes(b"fake audio")
        result = lt.evaluate_stage1(_song_folder(tmp_path, Path("Empty Gap"), "Empty Gap.txt"), ffprobe_path=None)
        assert result.verdict == "ok"

    def test_unreadable_chart_is_an_error_not_corrupt(self, tmp_path, monkeypatch):
        source = tmp_path / "source"
        song_dir = _make_song(source, "Locked Song")

        def locked(path):
            raise PermissionError(13, "file is locked")
        monkeypatch.setattr(lt, "parse_score_song", locked)

        summary = lt.run(source, tmp_path / "corrupt", apply=True, log=lambda *a: None)

        row = summary.all_rows["Locked Song"]
        assert song_dir.exists() and summary.moved == 0
        assert row["action"] == "error" and row["stage1_verdict"] == "error"
        assert "txt_unreadable" in row["reason"]
        assert summary.stage1_corrupt == 0

    def test_vanished_chart_is_an_error(self, tmp_path):
        _make_song(tmp_path, "Gone")
        song = _song_folder(tmp_path, Path("Gone"), "Gone.txt")
        (tmp_path / "Gone" / "Gone.txt").unlink()
        assert lt.evaluate_stage1(song, ffprobe_path=None).verdict == "error"

    def test_ffprobe_that_cannot_start_is_an_error(self, tmp_path, monkeypatch):
        _make_song(tmp_path, "Song")
        song = _song_folder(tmp_path, Path("Song"), "Song.txt")

        def cannot_start(*a, **k):
            raise OSError("cannot start")
        monkeypatch.setattr(lt.subprocess, "run", cannot_start)

        result = lt.evaluate_stage1(song, ffprobe_path="ffprobe")
        assert result.verdict == "error" and "ffprobe_failed" in result.reason

    @pytest.mark.parametrize("returncode", [-9, 3221225477])
    def test_ffprobe_crash_or_kill_is_an_error(self, tmp_path, monkeypatch, returncode):
        """Killed (e.g. out of memory) or crashed while probing one file: not
        evidence that the audio is broken."""
        _make_song(tmp_path, "Song")
        song = _song_folder(tmp_path, Path("Song"), "Song.txt")
        monkeypatch.setattr(lt.subprocess, "run",
                            lambda cmd, **k: subprocess.CompletedProcess(cmd, returncode, "", ""))
        result = lt.evaluate_stage1(song, ffprobe_path="ffprobe")
        assert result.verdict == "error" and "ffprobe_failed" in result.reason

    def test_ffprobe_rejecting_the_file_is_corrupt(self, tmp_path, monkeypatch):
        _make_song(tmp_path, "Song")
        song = _song_folder(tmp_path, Path("Song"), "Song.txt")
        monkeypatch.setattr(lt.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(
            cmd, 1, "", "Invalid data found when processing input"))
        result = lt.evaluate_stage1(song, ffprobe_path="ffprobe")
        assert result.verdict == "corrupt" and "audio_undecodable" in result.reason

    def test_broken_ffprobe_stops_the_run_before_anything_moves(self, tmp_path, monkeypatch):
        source = tmp_path / "source"
        song_dir = _make_song(source, "Song")
        monkeypatch.setattr(lt.subprocess, "run",
                            lambda cmd, **k: subprocess.CompletedProcess(cmd, 3221225781, b"", b""))
        with pytest.raises(ValueError, match="ffprobe"):
            lt.run(source, tmp_path / "corrupt", apply=True, ffprobe_path="ffprobe", log=lambda *a: None)
        assert song_dir.exists()


class TestProbeAudioOutput:
    def test_non_ascii_tags_do_not_lose_the_output(self, tmp_path, monkeypatch):
        """ffprobe writes UTF-8. Decoded with the Windows console codepage,
        tags like ” or Ł turned the output into None and crashed the run."""
        payload = json.dumps({"streams": [{"codec_type": "audio"}],
                              "format": {"duration": "12.5", "tags": {"title": "”A” Á Ł"}}},
                             ensure_ascii=False)
        child = f"import sys; sys.stdout.buffer.write({ascii(payload)}.encode('utf-8'))"
        real_run = subprocess.run

        def fake_ffprobe(cmd, **kwargs):
            return real_run([sys.executable, "-c", child], **kwargs)
        monkeypatch.setattr(lt.subprocess, "run", fake_ffprobe)

        assert lt.probe_audio(tmp_path / "x.ogg", "ffprobe") == (True, "")


class TestRunSurvivesSongErrors:
    def test_unexpected_error_in_one_song_does_not_stop_the_run(self, tmp_path, monkeypatch):
        source = tmp_path / "source"
        _make_song(source, "Song A")
        _make_song(source, "Song B", notes=False)
        _make_song(source, "Song C")
        real = lt._process_one

        def flaky(song, **kwargs):
            if song.rel_path == "Song A":
                raise TypeError("the JSON object must be str, bytes or bytearray, not NoneType")
            return real(song, **kwargs)
        monkeypatch.setattr(lt, "_process_one", flaky)

        summary = lt.run(source, tmp_path / "corrupt", apply=True, log=lambda *a: None)

        assert summary.all_rows["Song A"]["action"] == "error"
        assert (source / "Song A").exists()
        assert summary.moved == 1 and summary.kept == 1 and summary.errors == 1


# ── #RELATIVE charts in stage 2 ──────────────────────────────────────────────

class TestStage2RelativeChart:
    def test_relative_chart_is_scored_with_absolute_beats(self, tmp_path, monkeypatch):
        us = pytest.importorskip("ultrastar_score")
        d = tmp_path / "Song"
        d.mkdir()
        txt = _write_chart(d, "Song", extra_lines=["#RELATIVE:YES", ": 0 4 60 a", "- 8 10"])
        seen = {}

        def fake_score_song(song, vocals_path, difficulty=None, pitch_frames=None):
            seen["beats"] = [n.start_beat for n in song.all_notes]
            return types.SimpleNamespace(percentage=90.0)
        monkeypatch.setattr(us, "score_song", fake_score_song)
        monkeypatch.setattr(us, "detect_pitch_frames", lambda path: [], raising=False)

        assert lt._default_score_medium(txt, tmp_path / "vocals.wav") == 90.0
        assert seen["beats"] == [0, 10]
