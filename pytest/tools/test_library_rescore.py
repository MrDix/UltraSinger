"""Tests for tools/library_rescore.py — library-wide chart rescoring tool.

Uses generic placeholder song names only (no copyrighted titles/artists),
per repo policy. Real scoring (``ultrastar_score``) is mocked out — these
tests cover discovery, CSV resume, and error-path behaviour only.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

# The tool is a standalone script in tools/ — import its internals directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from library_rescore import (  # noqa: E402
    CSV_FIELDS,
    RunSummary,
    SongEntry,
    append_result,
    find_song_dirs,
    load_existing_results,
    run,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_song_dir(base: Path, name: str, *, with_vocals: bool = True,
                    vocal_ext: str = "ogg", with_txt: bool = True) -> Path:
    """Create a fake UltraSinger output folder under *base*."""
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    if with_txt:
        (d / f"{name}.txt").write_text("#TITLE:Test\nE\n", encoding="utf-8")
    # Always write a full-mix file, only sometimes the vocals file
    (d / f"{name}.ogg").write_bytes(b"fake full mix")
    if with_vocals:
        (d / f"{name} [Vocals].{vocal_ext}").write_bytes(b"fake vocals")
        (d / f"{name} [Instrumental].{vocal_ext}").write_bytes(b"fake instrumental")
    return d


# ── Discovery ────────────────────────────────────────────────────────────────

class TestDiscovery:
    """Feature: recursively find song folders with chart + vocals track."""

    def test_dir_with_vocals_is_discovered(self, tmp_path):
        _make_song_dir(tmp_path, "Song A", with_vocals=True)
        result = find_song_dirs(tmp_path)
        assert len(result.entries) == 1
        assert result.skipped_no_vocals == 0
        entry = result.entries[0]
        assert entry.txt_path.name == "Song A.txt"
        assert "[Vocals]" in entry.vocal_path.name

    def test_dir_without_vocals_is_skipped_and_counted(self, tmp_path):
        _make_song_dir(tmp_path, "Song B", with_vocals=False)
        result = find_song_dirs(tmp_path)
        assert result.entries == []
        assert result.skipped_no_vocals == 1

    def test_dir_without_txt_is_ignored_entirely(self, tmp_path):
        d = tmp_path / "Not A Song Folder"
        d.mkdir()
        (d / "random.ogg").write_bytes(b"noise")
        result = find_song_dirs(tmp_path)
        assert result.entries == []
        assert result.skipped_no_vocals == 0

    def test_mp3_and_wav_vocal_extensions_supported(self, tmp_path):
        _make_song_dir(tmp_path, "Track One", with_vocals=True, vocal_ext="mp3")
        _make_song_dir(tmp_path, "Track Two", with_vocals=True, vocal_ext="wav")
        result = find_song_dirs(tmp_path)
        assert len(result.entries) == 2
        exts = sorted(e.vocal_path.suffix for e in result.entries)
        assert exts == [".mp3", ".wav"]

    def test_recursive_nested_dirs(self, tmp_path):
        _make_song_dir(tmp_path / "Library A", "Demo Song", with_vocals=True)
        _make_song_dir(tmp_path / "Library B" / "Sub", "Other Song", with_vocals=True)
        result = find_song_dirs(tmp_path)
        assert len(result.entries) == 2

    def test_mixed_library_counts_correctly(self, tmp_path):
        _make_song_dir(tmp_path, "Good Song", with_vocals=True)
        _make_song_dir(tmp_path, "Rip Only", with_vocals=False)
        d = tmp_path / "Junk Folder"
        d.mkdir()
        (d / "notes.md").write_text("nothing", encoding="utf-8")
        result = find_song_dirs(tmp_path)
        assert len(result.entries) == 1
        assert result.skipped_no_vocals == 1

    def test_unicode_song_names(self, tmp_path):
        _make_song_dir(tmp_path, "Söng Nämé Ünïcödé", with_vocals=True)
        result = find_song_dirs(tmp_path)
        assert len(result.entries) == 1
        assert "Ünïcödé" in str(result.entries[0].dir_path)


# ── CSV resume logic ─────────────────────────────────────────────────────────

class TestCsvResume:
    """Feature: results are appended to CSV and re-runs skip known paths."""

    def test_load_existing_results_missing_file(self, tmp_path):
        assert load_existing_results(tmp_path / "nope.csv") == {}

    def test_load_existing_results_empty_file(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        assert load_existing_results(p) == {}

    def test_append_result_writes_header_once(self, tmp_path):
        p = tmp_path / "results.csv"
        append_result(p, {"pfad": "a", "medium_pct": "50.0", "notes": 10,
                           "status": "ok", "fehler": ""}, write_header=True)
        append_result(p, {"pfad": "b", "medium_pct": "60.0", "notes": 20,
                           "status": "ok", "fehler": ""}, write_header=False)
        with p.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["pfad"] == "a"
        assert rows[1]["pfad"] == "b"
        assert list(rows[0].keys()) == list(CSV_FIELDS)

    def test_load_existing_results_round_trip(self, tmp_path):
        p = tmp_path / "results.csv"
        append_result(p, {"pfad": "song/one", "medium_pct": "42.5", "notes": 100,
                           "status": "ok", "fehler": ""}, write_header=True)
        loaded = load_existing_results(p)
        assert "song/one" in loaded
        assert loaded["song/one"]["medium_pct"] == "42.5"

    def test_run_resumes_and_skips_already_scored(self, tmp_path):
        _make_song_dir(tmp_path, "Song One", with_vocals=True)
        _make_song_dir(tmp_path, "Song Two", with_vocals=True)
        output = tmp_path / "out.csv"

        calls = []

        def fake_score(txt_path, vocal_path):
            calls.append(txt_path)
            return 55.5, 42

        summary1 = run(tmp_path, output, score_fn=fake_score, log=lambda *a: None)
        assert summary1.scored_this_run == 2
        assert len(calls) == 2

        # Second run: everything should be skipped via resume
        calls.clear()
        summary2 = run(tmp_path, output, score_fn=fake_score, log=lambda *a: None)
        assert summary2.scored_this_run == 0
        assert summary2.skipped_resume == 2
        assert calls == []

    def test_run_fresh_ignores_existing_csv(self, tmp_path):
        _make_song_dir(tmp_path, "Song One", with_vocals=True)
        output = tmp_path / "out.csv"

        calls = []

        def fake_score(txt_path, vocal_path):
            calls.append(txt_path)
            return 55.5, 42

        run(tmp_path, output, score_fn=fake_score, log=lambda *a: None)
        assert len(calls) == 1

        calls.clear()
        summary = run(tmp_path, output, score_fn=fake_score, fresh=True, log=lambda *a: None)
        assert summary.scored_this_run == 1
        assert len(calls) == 1

    def test_run_respects_limit(self, tmp_path):
        for i in range(5):
            _make_song_dir(tmp_path, f"Song {i}", with_vocals=True)
        output = tmp_path / "out.csv"

        def fake_score(txt_path, vocal_path):
            return 10.0, 1

        summary = run(tmp_path, output, limit=2, score_fn=fake_score, log=lambda *a: None)
        assert summary.scored_this_run == 2


# ── Error path ───────────────────────────────────────────────────────────────

class TestErrorPath:
    """Feature: a single broken song is recorded as an error row, not fatal."""

    def test_scoring_exception_is_caught_and_recorded(self, tmp_path):
        _make_song_dir(tmp_path, "Broken Song", with_vocals=True)
        _make_song_dir(tmp_path, "Fine Song", with_vocals=True)
        output = tmp_path / "out.csv"

        def fake_score(txt_path, vocal_path):
            if "Broken" in str(txt_path):
                raise ValueError("corrupt chart")
            return 77.0, 5

        summary = run(tmp_path, output, score_fn=fake_score, log=lambda *a: None)
        assert summary.errors_this_run == 1
        assert summary.scored_this_run == 1

        loaded = load_existing_results(output)
        broken_row = next(r for k, r in loaded.items() if "Broken" in k)
        assert broken_row["status"] == "error"
        assert "corrupt chart" in broken_row["fehler"]

        fine_row = next(r for k, r in loaded.items() if "Fine" in k)
        assert fine_row["status"] == "ok"
        assert fine_row["medium_pct"] == "77.0"

    def test_error_does_not_abort_remaining_songs(self, tmp_path):
        for i in range(3):
            _make_song_dir(tmp_path, f"Song {i}", with_vocals=True)
        output = tmp_path / "out.csv"

        def fake_score(txt_path, vocal_path):
            if "Song 1" in str(txt_path):
                raise RuntimeError("boom")
            return 33.3, 3

        summary = run(tmp_path, output, score_fn=fake_score, log=lambda *a: None)
        assert summary.scored_this_run == 2
        assert summary.errors_this_run == 1
        assert len(summary.all_rows) == 3

    def test_resumed_error_row_is_not_retried(self, tmp_path):
        _make_song_dir(tmp_path, "Broken Song", with_vocals=True)
        output = tmp_path / "out.csv"

        call_count = [0]

        def failing_score(txt_path, vocal_path):
            call_count[0] += 1
            raise ValueError("still broken")

        run(tmp_path, output, score_fn=failing_score, log=lambda *a: None)
        assert call_count[0] == 1

        run(tmp_path, output, score_fn=failing_score, log=lambda *a: None)
        # Resume treats the previously-recorded error as done, not retried.
        assert call_count[0] == 1


# ── Report formatting sanity (worst-first ordering) ─────────────────────────

class TestSummaryOrdering:
    def test_worst_songs_sort_ascending_by_percentage(self, tmp_path):
        names = ["High Song", "Low Song", "Mid Song"]
        scores = {"High Song": 90.0, "Low Song": 10.0, "Mid Song": 50.0}
        for n in names:
            _make_song_dir(tmp_path, n, with_vocals=True)
        output = tmp_path / "out.csv"

        def fake_score(txt_path, vocal_path):
            for n, pct in scores.items():
                if n in str(txt_path):
                    return pct, 1
            raise AssertionError("unexpected path")

        summary = run(tmp_path, output, score_fn=fake_score, log=lambda *a: None)
        ok_rows = [r for r in summary.all_rows.values() if r["status"] == "ok"]
        ok_rows.sort(key=lambda r: float(r["medium_pct"]))
        ordered_names = [Path(r["pfad"]).name for r in ok_rows]
        assert ordered_names == ["Low Song", "Mid Song", "High Song"]
