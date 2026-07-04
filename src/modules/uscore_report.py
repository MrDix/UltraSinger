"""Report the real game score for the final UltraStar TXT.

UltraSinger's internal score (``ultrastar_score_calculator``) measures the
chart against the SwiftF0 pitch data it was built from.  The games however
score with the Vocaluxe/USDX ptAKF detector, which disagrees with SwiftF0
on a noticeable share of beats — so the internal score can rank charts in
the wrong order (it systematically undervalues ptAKF-refit charts and
overvalues SwiftF0-split charts).

This module scores the *written* TXT against the vocal audio with
``ultrastar-score`` (the same C++ ptAKF algorithm the games use) at all
three difficulties.  That is the number a perfect singer — or the vocal
track itself — would actually achieve in game.

Fails open: without the optional ``ultrastar-score`` dependency or on any
error the report is simply skipped.
"""

from __future__ import annotations

from modules.console_colors import ULTRASINGER_HEAD

# Difficulty tolerance (octave-folded semitones): Easy +-2, Medium +-1, Hard 0
_DIFFICULTY_ORDER = ("easy", "medium", "hard")


def calculate_uscore_report(
    txt_path: str, vocal_audio_path: str, pitch_frames: list[dict] | None = None
) -> dict[str, dict] | None:
    """Score the final TXT against the vocal audio at Easy/Medium/Hard.

    Args:
        txt_path: Path to the written UltraStar TXT.
        vocal_audio_path: Path to the vocal-only audio.
        pitch_frames: Optional pre-computed ptAKF pitch frames (from
            ``ultrastar_score.detect_pitch_frames``). When given, all three
            ``score_song`` calls reuse them instead of each re-loading and
            re-analysing ``vocal_audio_path`` from scratch. Ignored on older
            ``ultrastar-score`` versions that don't accept the parameter
            (falls back to per-call analysis).

    Returns a dict per difficulty with the detailed USDX score breakdown::

        {
            "easy": {
                "total_pct": 96.7,       # percentage of the 10000-point scale
                "notes_points": 8532,    # rounded normal-note points
                "golden_points": 120,    # rounded golden-note points
                "line_bonus": 850,       # rounded line bonus points
                "beats_hit": 543,        # beats matched within tolerance
                "beats_total": 600,      # total scoreable beats
            },
            "medium": {...},
            "hard": {...},
        }

    or ``None`` when scoring is unavailable.
    """
    try:
        from ultrastar_score import Difficulty, score_song
        from ultrastar_score.parser import parse_ultrastar

        song = parse_ultrastar(txt_path)
        difficulties = {
            "easy": Difficulty.EASY,
            "medium": Difficulty.MEDIUM,
            "hard": Difficulty.HARD,
        }
        report = {}
        for name, diff in difficulties.items():
            if pitch_frames is not None:
                try:
                    result = score_song(
                        song, vocal_audio_path, difficulty=diff,
                        pitch_frames=pitch_frames,
                    )
                except TypeError:
                    # Installed ultrastar-score predates the pitch_frames
                    # parameter — fall back to per-call analysis.
                    result = score_song(song, vocal_audio_path, difficulty=diff)
            else:
                result = score_song(song, vocal_audio_path, difficulty=diff)
            report[name] = {
                "total_pct": round(result.percentage, 1),
                "notes_points": round(result.score_notes),
                "golden_points": round(result.score_golden),
                "line_bonus": round(result.score_line_bonus),
                "beats_hit": result.notes_hit,
                "beats_total": result.notes_total,
            }
        return report
    except (ImportError, OSError, ValueError, RuntimeError,
            AttributeError, KeyError, TypeError) as e:
        print(
            f"{ULTRASINGER_HEAD} Warning: game-score report skipped: {e}"
        )
        return None


def format_uscore_report(scores: dict[str, dict]) -> str:
    """Human-readable one-liner, stable for log parsing by the GUI."""
    return " | ".join(
        f"{name.capitalize()} {scores[name]['total_pct']:.1f}%"
        for name in _DIFFICULTY_ORDER
        if name in scores
    )
