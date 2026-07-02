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
    txt_path: str, vocal_audio_path: str
) -> dict[str, float] | None:
    """Score the final TXT against the vocal audio at Easy/Medium/Hard.

    Returns ``{"easy": pct, "medium": pct, "hard": pct}`` (percentages of
    the 10000-point USDX scale) or ``None`` when scoring is unavailable.
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
        return {
            name: round(score_song(song, vocal_audio_path, difficulty=diff).percentage, 1)
            for name, diff in difficulties.items()
        }
    except (ImportError, OSError, ValueError, RuntimeError,
            AttributeError, KeyError, TypeError) as e:
        print(
            f"{ULTRASINGER_HEAD} Warning: game-score report skipped: {e}"
        )
        return None


def format_uscore_report(scores: dict[str, float]) -> str:
    """Human-readable one-liner, stable for log parsing by the GUI."""
    return " | ".join(
        f"{name.capitalize()} {scores[name]:.1f}%"
        for name in _DIFFICULTY_ORDER
        if name in scores
    )
