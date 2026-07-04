"""Tests for the game-score report module."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

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


@dataclass
class FakeScoreResult:
    percentage: float = 90.0
    score_notes: float = 8000.0
    score_golden: float = 100.0
    score_line_bonus: float = 900.0
    notes_hit: int = 540
    notes_total: int = 600


def _mock_ultrastar_score_modules(score_song_fn):
    mock_uscore = MagicMock()
    mock_uscore.score_song = score_song_fn
    mock_uscore.Difficulty.EASY = "easy"
    mock_uscore.Difficulty.MEDIUM = "medium"
    mock_uscore.Difficulty.HARD = "hard"

    mock_parser = MagicMock()
    mock_parser.parse_ultrastar = MagicMock(return_value=MagicMock())
    return mock_uscore, mock_parser


class TestCalculateUscoreReportPitchFrames:
    """Verify pitch_frames is threaded into every score_song call."""

    def test_pitch_frames_passed_to_every_call(self):
        received: list[object] = []

        def _score_song(song, audio_path, difficulty=None, pitch_frames=None):
            received.append(pitch_frames)
            return FakeScoreResult()

        mock_uscore, mock_parser = _mock_ultrastar_score_modules(_score_song)
        frames = [{"tone": 24, "time": 0.0}]

        with patch.dict(sys.modules, {
            "ultrastar_score": mock_uscore,
            "ultrastar_score.parser": mock_parser,
        }):
            result = calculate_uscore_report(
                "fake.txt", "fake_vocal.wav", pitch_frames=frames
            )

        assert result is not None
        assert len(received) == 3  # easy, medium, hard
        assert all(r is frames for r in received)

    def test_no_pitch_frames_calls_without_kwarg(self):
        received: list[tuple] = []

        def _score_song(song, audio_path, difficulty=None, pitch_frames=None):
            received.append(pitch_frames)
            return FakeScoreResult()

        mock_uscore, mock_parser = _mock_ultrastar_score_modules(_score_song)

        with patch.dict(sys.modules, {
            "ultrastar_score": mock_uscore,
            "ultrastar_score.parser": mock_parser,
        }):
            result = calculate_uscore_report("fake.txt", "fake_vocal.wav")

        assert result is not None
        assert all(r is None for r in received)

    def test_falls_back_when_score_song_rejects_pitch_frames(self):
        """Older ultrastar-score without the pitch_frames kwarg: TypeError
        on the first attempt should be caught and retried without it."""
        calls: list[dict] = []

        def _score_song(song, audio_path, difficulty=None, **kwargs):
            calls.append(kwargs)
            if "pitch_frames" in kwargs:
                raise TypeError("score_song() got an unexpected keyword argument 'pitch_frames'")
            return FakeScoreResult()

        mock_uscore, mock_parser = _mock_ultrastar_score_modules(_score_song)
        frames = [{"tone": 24, "time": 0.0}]

        with patch.dict(sys.modules, {
            "ultrastar_score": mock_uscore,
            "ultrastar_score.parser": mock_parser,
        }):
            result = calculate_uscore_report(
                "fake.txt", "fake_vocal.wav", pitch_frames=frames
            )

        assert result is not None
        # Each difficulty: first attempt with pitch_frames (fails), retry without
        assert len(calls) == 6
        assert result["easy"]["total_pct"] == 90.0
