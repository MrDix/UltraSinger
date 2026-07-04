"""Tests for tools/regression_benchmark.py — game-score regression tool.

No real audio scoring happens here: score_song_folder / score_all are
either exercised against a missing file (to hit the fail-open path) or
monkeypatched so the comparison/reporting/CLI logic can be tested with
canned scores.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The tool is a standalone script in tools/ — import its internals directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from regression_benchmark import (
    ComparisonReport,
    SongDelta,
    compare_scores,
    find_song_pairs,
    format_table,
    main,
    score_song_folder,
)


# ── Song discovery ───────────────────────────────────────────────────────────

class TestFindSongPairs:
    """Feature: pair up a .txt and a '[Vocals]' audio file per song folder."""

    def _make_song_dir(self, tmp_path, name, txt=True, vocals=True, extra_txt=False, extra_vocals=False):
        d = tmp_path / name
        d.mkdir()
        if txt:
            (d / f"{name}.txt").write_text("dummy", encoding="utf-8")
        if extra_txt:
            (d / f"{name}_2.txt").write_text("dummy", encoding="utf-8")
        if vocals:
            (d / f"{name} [Vocals].mp3").write_bytes(b"")
        if extra_vocals:
            (d / f"{name} [Vocals] (alt).mp3").write_bytes(b"")
        return d

    def test_valid_pair_found(self, tmp_path):
        """A folder with exactly one .txt and one [Vocals] file is paired."""
        self._make_song_dir(tmp_path, "song_a")
        pairs, skipped = find_song_pairs(tmp_path)
        assert list(pairs.keys()) == ["song_a"]
        txt_path, audio_path = pairs["song_a"]
        assert txt_path.suffix == ".txt"
        assert "[Vocals]" in audio_path.name
        assert skipped == []

    def test_missing_txt_skipped(self, tmp_path):
        """A folder with no .txt file is skipped with a reason."""
        self._make_song_dir(tmp_path, "song_a", txt=False)
        pairs, skipped = find_song_pairs(tmp_path)
        assert pairs == {}
        assert len(skipped) == 1
        assert "no .txt file" in skipped[0]

    def test_missing_vocals_skipped(self, tmp_path):
        """A folder with no '[Vocals]' audio file is skipped with a reason."""
        self._make_song_dir(tmp_path, "song_a", vocals=False)
        pairs, skipped = find_song_pairs(tmp_path)
        assert pairs == {}
        assert len(skipped) == 1
        assert "Vocals" in skipped[0]

    def test_multiple_txt_skipped(self, tmp_path):
        """A folder with more than one .txt file is skipped."""
        self._make_song_dir(tmp_path, "song_a", extra_txt=True)
        pairs, skipped = find_song_pairs(tmp_path)
        assert pairs == {}
        assert len(skipped) == 1
        assert "multiple .txt" in skipped[0]

    def test_multiple_vocals_skipped(self, tmp_path):
        """A folder with more than one '[Vocals]' file is skipped."""
        self._make_song_dir(tmp_path, "song_a", extra_vocals=True)
        pairs, skipped = find_song_pairs(tmp_path)
        assert pairs == {}
        assert len(skipped) == 1
        assert "multiple '[Vocals]'" in skipped[0]

    def test_multiple_song_folders(self, tmp_path):
        """Several valid folders all get paired independently."""
        self._make_song_dir(tmp_path, "song_a")
        self._make_song_dir(tmp_path, "song_b")
        pairs, skipped = find_song_pairs(tmp_path)
        assert set(pairs.keys()) == {"song_a", "song_b"}
        assert skipped == []

    def test_non_directory_entries_ignored(self, tmp_path):
        """Loose files directly under the root are not treated as song folders."""
        (tmp_path / "stray.txt").write_text("x", encoding="utf-8")
        self._make_song_dir(tmp_path, "song_a")
        pairs, skipped = find_song_pairs(tmp_path)
        assert list(pairs.keys()) == ["song_a"]


# ── Scoring fail-open path ───────────────────────────────────────────────────

class TestScoreSongFolder:
    """score_song_folder must fail open (warn + None) rather than crash."""

    def test_missing_files_returns_none(self, tmp_path, capsys):
        result = score_song_folder(tmp_path / "missing.txt", tmp_path / "missing.mp3")
        assert result is None
        assert "scoring skipped" in capsys.readouterr().out


# ── Comparison logic ─────────────────────────────────────────────────────────

def _scores(easy, medium, hard, notes=100):
    return {"easy": easy, "medium": medium, "hard": hard, "notes": notes}


class TestCompareScores:
    """Feature: delta computation, tolerance handling, new/missing folders."""

    def test_identical_scores_no_regression(self):
        baseline = {"song_a": _scores(90.0, 80.0, 70.0)}
        current = {"song_a": _scores(90.0, 80.0, 70.0)}
        report = compare_scores(current, baseline, tolerance=0.5)
        assert report.has_regression is False
        assert report.deltas[0].deltas == {"easy": 0.0, "medium": 0.0, "hard": 0.0}
        assert report.deltas[0].regressed is False

    def test_improvement_no_regression(self):
        """A score increase is never a regression."""
        baseline = {"song_a": _scores(90.0, 80.0, 70.0)}
        current = {"song_a": _scores(95.0, 85.0, 75.0)}
        report = compare_scores(current, baseline, tolerance=0.5)
        assert report.has_regression is False

    def test_drop_beyond_tolerance_is_regression(self):
        """A drop larger than the tolerance flags a regression."""
        baseline = {"song_a": _scores(90.0, 80.0, 70.0)}
        current = {"song_a": _scores(89.0, 80.0, 70.0)}  # easy dropped by 1.0
        report = compare_scores(current, baseline, tolerance=0.5)
        assert report.has_regression is True
        assert report.deltas[0].regressed is True
        assert report.deltas[0].deltas["easy"] == pytest.approx(-1.0)

    def test_drop_within_tolerance_is_not_regression(self):
        """A small drop within the tolerance band is not flagged."""
        baseline = {"song_a": _scores(90.0, 80.0, 70.0)}
        current = {"song_a": _scores(89.7, 80.0, 70.0)}  # dropped by 0.3, tol=0.5
        report = compare_scores(current, baseline, tolerance=0.5)
        assert report.has_regression is False
        assert report.deltas[0].regressed is False

    def test_drop_exactly_at_tolerance_boundary_is_not_regression(self):
        """A drop exactly equal to the tolerance is not a regression (strict '<')."""
        baseline = {"song_a": _scores(90.0, 80.0, 70.0)}
        current = {"song_a": _scores(89.5, 80.0, 70.0)}  # dropped by exactly 0.5
        report = compare_scores(current, baseline, tolerance=0.5)
        assert report.has_regression is False

    def test_one_regressed_song_among_several(self):
        """Only the regressed song is flagged; others stay ok."""
        baseline = {
            "song_a": _scores(90.0, 80.0, 70.0),
            "song_b": _scores(90.0, 80.0, 70.0),
        }
        current = {
            "song_a": _scores(90.0, 80.0, 70.0),
            "song_b": _scores(90.0, 80.0, 60.0),  # hard dropped by 10
        }
        report = compare_scores(current, baseline, tolerance=0.5)
        assert report.has_regression is True
        by_name = {sd.name: sd for sd in report.deltas}
        assert by_name["song_a"].regressed is False
        assert by_name["song_b"].regressed is True

    def test_new_song_reported_not_a_regression(self):
        """A song present now but absent from the baseline is reported separately."""
        baseline = {"song_a": _scores(90.0, 80.0, 70.0)}
        current = {
            "song_a": _scores(90.0, 80.0, 70.0),
            "song_b": _scores(50.0, 40.0, 30.0),
        }
        report = compare_scores(current, baseline, tolerance=0.5)
        assert report.has_regression is False
        assert report.new_songs == ["song_b"]
        assert report.missing_songs == []

    def test_missing_song_reported_not_a_regression(self):
        """A song present in the baseline but not found now is reported separately."""
        baseline = {
            "song_a": _scores(90.0, 80.0, 70.0),
            "song_b": _scores(50.0, 40.0, 30.0),
        }
        current = {"song_a": _scores(90.0, 80.0, 70.0)}
        report = compare_scores(current, baseline, tolerance=0.5)
        assert report.has_regression is False
        assert report.missing_songs == ["song_b"]
        assert report.new_songs == []

    def test_no_common_songs(self):
        """No overlap between baseline and current yields an empty delta list."""
        baseline = {"song_a": _scores(90.0, 80.0, 70.0)}
        current = {"song_b": _scores(50.0, 40.0, 30.0)}
        report = compare_scores(current, baseline, tolerance=0.5)
        assert report.deltas == []
        assert report.has_regression is False
        assert report.new_songs == ["song_b"]
        assert report.missing_songs == ["song_a"]


# ── Table formatting ─────────────────────────────────────────────────────────

class TestFormatTable:
    """Human-readable table rendering."""

    def test_ok_status_shown(self):
        baseline = {"song_a": _scores(90.0, 80.0, 70.0)}
        current = {"song_a": _scores(90.0, 80.0, 70.0)}
        report = compare_scores(current, baseline, tolerance=0.5)
        table = format_table(report, tolerance=0.5)
        assert "song_a" in table
        assert "ok" in table
        assert "no regression" in table

    def test_regression_status_shown(self):
        baseline = {"song_a": _scores(90.0, 80.0, 70.0)}
        current = {"song_a": _scores(80.0, 80.0, 70.0)}
        report = compare_scores(current, baseline, tolerance=0.5)
        table = format_table(report, tolerance=0.5)
        assert "REGRESSION" in table
        assert "regression(s) detected" in table

    def test_new_and_missing_sections(self):
        baseline = {"song_a": _scores(90.0, 80.0, 70.0), "song_c": _scores(1.0, 1.0, 1.0)}
        current = {"song_a": _scores(90.0, 80.0, 70.0), "song_b": _scores(50.0, 40.0, 30.0)}
        report = compare_scores(current, baseline, tolerance=0.5)
        table = format_table(report, tolerance=0.5)
        assert "New folders" in table
        assert "song_b" in table
        assert "Missing folders" in table
        assert "song_c" in table

    def test_empty_report_no_crash(self):
        report = ComparisonReport(deltas=[], new_songs=[], missing_songs=[], has_regression=False)
        table = format_table(report, tolerance=0.5)
        assert "No songs found" in table


# ── CLI / main() orchestration ───────────────────────────────────────────────

class TestMainUpdateBaseline:
    """--update-baseline mode writes a JSON baseline from mocked scores."""

    def test_writes_baseline_file(self, tmp_path, monkeypatch):
        song_dir = tmp_path / "song_a"
        song_dir.mkdir()
        (song_dir / "song_a.txt").write_text("dummy", encoding="utf-8")
        (song_dir / "song_a [Vocals].mp3").write_bytes(b"")

        monkeypatch.setattr(
            "regression_benchmark.score_song_folder",
            lambda txt, audio: _scores(90.0, 80.0, 70.0, notes=42),
        )

        baseline_path = tmp_path / "baseline.json"
        exit_code = main([
            str(tmp_path), "--baseline", str(baseline_path), "--update-baseline",
        ])

        assert exit_code == 0
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
        assert data == {"song_a": _scores(90.0, 80.0, 70.0, notes=42)}


class TestMainCompare:
    """Default (compare) mode against a pre-written baseline file."""

    def _make_song_dir(self, tmp_path, name):
        d = tmp_path / name
        d.mkdir()
        (d / f"{name}.txt").write_text("dummy", encoding="utf-8")
        (d / f"{name} [Vocals].mp3").write_bytes(b"")
        return d

    def test_no_regression_exits_zero(self, tmp_path, monkeypatch):
        self._make_song_dir(tmp_path, "song_a")
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(
            json.dumps({"song_a": _scores(90.0, 80.0, 70.0)}), encoding="utf-8"
        )

        monkeypatch.setattr(
            "regression_benchmark.score_song_folder",
            lambda txt, audio: _scores(90.0, 80.0, 70.0),
        )

        exit_code = main([str(tmp_path), "--baseline", str(baseline_path)])
        assert exit_code == 0

    def test_regression_exits_one(self, tmp_path, monkeypatch, capsys):
        self._make_song_dir(tmp_path, "song_a")
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(
            json.dumps({"song_a": _scores(90.0, 80.0, 70.0)}), encoding="utf-8"
        )

        monkeypatch.setattr(
            "regression_benchmark.score_song_folder",
            lambda txt, audio: _scores(50.0, 80.0, 70.0),
        )

        exit_code = main([str(tmp_path), "--baseline", str(baseline_path)])
        assert exit_code == 1
        assert "REGRESSION" in capsys.readouterr().out

    def test_missing_baseline_file_exits_two(self, tmp_path):
        self._make_song_dir(tmp_path, "song_a")
        exit_code = main([str(tmp_path), "--baseline", str(tmp_path / "does_not_exist.json")])
        assert exit_code == 2

    def test_no_song_folders_exits_two(self, tmp_path):
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text("{}", encoding="utf-8")
        exit_code = main([str(tmp_path), "--baseline", str(baseline_path)])
        assert exit_code == 2

    def test_non_directory_songs_dir_exits_two(self, tmp_path):
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text("{}", encoding="utf-8")
        not_a_dir = tmp_path / "file.txt"
        not_a_dir.write_text("x", encoding="utf-8")
        exit_code = main([str(not_a_dir), "--baseline", str(baseline_path)])
        assert exit_code == 2
