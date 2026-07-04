"""Tests for the game-score report module."""

from __future__ import annotations

from modules.uscore_report import calculate_uscore_report, format_uscore_report


def _diff(total_pct: float) -> dict:
    return {
        "total_pct": total_pct,
        "notes_points": 8000,
        "golden_points": 100,
        "line_bonus": 900,
        "beats_hit": 540,
        "beats_total": 600,
    }


class TestFormatUscoreReport:
    def test_all_difficulties(self):
        line = format_uscore_report(
            {"easy": _diff(96.7), "medium": _diff(87.1), "hard": _diff(66.6)}
        )
        assert line == "Easy 96.7% | Medium 87.1% | Hard 66.6%"

    def test_stable_order_regardless_of_dict_order(self):
        line = format_uscore_report(
            {"hard": _diff(1.0), "easy": _diff(3.0), "medium": _diff(2.0)}
        )
        assert line == "Easy 3.0% | Medium 2.0% | Hard 1.0%"

    def test_missing_difficulty_skipped(self):
        line = format_uscore_report({"easy": _diff(50.0)})
        assert line == "Easy 50.0%"


class TestCalculateUscoreReport:
    def test_fails_open_on_missing_files(self, capsys):
        result = calculate_uscore_report("does_not_exist.txt", "no_audio.wav")
        assert result is None
        assert "game-score report skipped" in capsys.readouterr().out
