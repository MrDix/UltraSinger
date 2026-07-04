"""Regression benchmark for the real game score of already-converted songs.

UltraSinger writes an UltraStar TXT next to a "[Vocals]"-suffixed vocal
audio file for every converted song. This tool scores those TXT/vocal
pairs with ``ultrastar-score`` (the same ptAKF algorithm the games use, see
``src/modules/uscore_report.py``) at Easy/Medium/Hard and lets you compare
the result across code changes: run it once with ``--update-baseline`` to
snapshot today's scores, then re-run it after a pipeline change to see
whether any song's score dropped.

Baseline files are plain JSON of the shape::

    {"<song folder name>": {"easy": 87.3, "medium": 71.2, "hard": 52.0, "notes": 412}, ...}

They are runtime/user data (song folder names, scores measured against a
user's private song library) and must live OUTSIDE this repository -- do
not check baseline JSON files into git.

Usage::

    python tools/regression_benchmark.py D:/path/to/output --baseline D:/path/baseline.json --update-baseline
    python tools/regression_benchmark.py D:/path/to/output --baseline D:/path/baseline.json [--tolerance 0.5]

The first form (re)writes the baseline. The second form re-scores every
song, compares against the baseline and exits 1 if any difficulty's score
dropped by more than ``--tolerance`` percentage points for any song
(default exit 0 otherwise). Folders that are new or missing relative to
the baseline are reported but do not affect the exit code.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_DIFFICULTIES = ("easy", "medium", "hard")


# ---------------------------------------------------------------------------
# Song discovery
# ---------------------------------------------------------------------------

def find_song_pairs(root: Path) -> tuple[dict[str, tuple[Path, Path]], list[str]]:
    """Find (txt, vocals-audio) pairs in each immediate subfolder of ``root``.

    Returns ``(pairs, skip_reasons)`` where ``pairs`` maps the folder name to
    ``(txt_path, vocal_audio_path)`` and ``skip_reasons`` is a list of
    human-readable messages for folders that could not be paired up.
    """
    pairs: dict[str, tuple[Path, Path]] = {}
    skip_reasons: list[str] = []

    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        txts = [f for f in entry.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
        vocals = [f for f in entry.iterdir() if f.is_file() and "[Vocals]" in f.name]

        if not txts:
            skip_reasons.append(f"{entry.name}: no .txt file found, skipping")
            continue
        if len(txts) > 1:
            skip_reasons.append(f"{entry.name}: multiple .txt files found, skipping")
            continue
        if not vocals:
            skip_reasons.append(f"{entry.name}: no '[Vocals]' audio file found, skipping")
            continue
        if len(vocals) > 1:
            skip_reasons.append(f"{entry.name}: multiple '[Vocals]' audio files found, skipping")
            continue

        pairs[entry.name] = (txts[0], vocals[0])

    return pairs, skip_reasons


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_song_folder(txt_path: Path, audio_path: Path) -> dict[str, float] | None:
    """Score a single song's TXT against its vocal audio at Easy/Medium/Hard.

    Returns ``{"easy": pct, "medium": pct, "hard": pct, "notes": count}`` or
    ``None`` (with a printed warning) if scoring failed or the
    ``ultrastar-score`` dependency is unavailable.

    When the installed ``ultrastar_score`` exposes ``detect_pitch_frames``
    (added in 0.2.1), pitch detection runs once and is reused for all three
    difficulties via ``score_song(..., pitch_frames=...)``. Older versions
    fall back to three independent ``score_song`` calls.
    """
    try:
        import ultrastar_score as usm
        from ultrastar_score import Difficulty, score_song
        from ultrastar_score.parser import parse_ultrastar

        song = parse_ultrastar(str(txt_path))
        difficulties = {
            "easy": Difficulty.EASY,
            "medium": Difficulty.MEDIUM,
            "hard": Difficulty.HARD,
        }

        if hasattr(usm, "detect_pitch_frames"):
            pitch_frames = usm.detect_pitch_frames(str(audio_path))
            scores = {
                name: score_song(song, str(audio_path), difficulty=diff, pitch_frames=pitch_frames)
                for name, diff in difficulties.items()
            }
        else:
            scores = {
                name: score_song(song, str(audio_path), difficulty=diff)
                for name, diff in difficulties.items()
            }

        result = {name: round(scores[name].percentage, 2) for name in difficulties}
        result["notes"] = scores["easy"].notes_total
        return result
    except (ImportError, OSError, ValueError, RuntimeError,
            AttributeError, KeyError, TypeError) as e:
        print(f"Warning: scoring skipped for {txt_path.parent.name}: {e}")
        return None


def score_all(pairs: dict[str, tuple[Path, Path]]) -> dict[str, dict[str, float]]:
    """Score every pair in ``pairs``, skipping (with a warning) any failure."""
    scores: dict[str, dict[str, float]] = {}
    for name, (txt_path, audio_path) in pairs.items():
        result = score_song_folder(txt_path, audio_path)
        if result is not None:
            scores[name] = result
    return scores


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class SongDelta:
    """Per-song comparison of current scores against the baseline."""

    name: str
    baseline: dict[str, float]
    current: dict[str, float]
    deltas: dict[str, float] = field(default_factory=dict)
    regressed: bool = False


@dataclass
class ComparisonReport:
    """Full comparison of a scoring run against a baseline."""

    deltas: list[SongDelta]
    new_songs: list[str]
    missing_songs: list[str]
    has_regression: bool


def compare_scores(
    current: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
    tolerance: float,
) -> ComparisonReport:
    """Compare freshly scored songs against a baseline.

    A song "regresses" if any difficulty's score dropped by more than
    ``tolerance`` percentage points. Songs present in only one of the two
    inputs are reported separately and never count as a regression.
    """
    deltas: list[SongDelta] = []
    has_regression = False

    for name in sorted(set(current) & set(baseline)):
        cur = current[name]
        base = baseline[name]
        song_deltas: dict[str, float] = {}
        regressed = False

        for diff in _DIFFICULTIES:
            if diff in cur and diff in base:
                d = cur[diff] - base[diff]
                song_deltas[diff] = d
                if d < -tolerance:
                    regressed = True

        if regressed:
            has_regression = True

        deltas.append(SongDelta(
            name=name, baseline=base, current=cur,
            deltas=song_deltas, regressed=regressed,
        ))

    new_songs = sorted(set(current) - set(baseline))
    missing_songs = sorted(set(baseline) - set(current))

    return ComparisonReport(
        deltas=deltas,
        new_songs=new_songs,
        missing_songs=missing_songs,
        has_regression=has_regression,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_delta(d: float) -> str:
    return f"{d:+.1f}"


def format_table(report: ComparisonReport, tolerance: float) -> str:
    """Render a human-readable comparison table."""
    lines: list[str] = []

    if not report.deltas:
        lines.append("No songs found in both the baseline and the current run.")
    else:
        name_width = max(len(sd.name) for sd in report.deltas)
        name_width = max(name_width, len("Song"))
        header = (
            f"{'Song':<{name_width}}  {'Easy Δ':>8}  {'Medium Δ':>10}  {'Hard Δ':>8}  Status"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for sd in report.deltas:
            easy = _fmt_delta(sd.deltas["easy"]) if "easy" in sd.deltas else "n/a"
            medium = _fmt_delta(sd.deltas["medium"]) if "medium" in sd.deltas else "n/a"
            hard = _fmt_delta(sd.deltas["hard"]) if "hard" in sd.deltas else "n/a"
            status = "REGRESSION" if sd.regressed else "ok"
            lines.append(
                f"{sd.name:<{name_width}}  {easy:>8}  {medium:>10}  {hard:>8}  {status}"
            )

    if report.new_songs:
        lines.append("")
        lines.append("New folders (not in baseline, informational only):")
        for name in report.new_songs:
            lines.append(f"  + {name}")

    if report.missing_songs:
        lines.append("")
        lines.append("Missing folders (in baseline but not found now, informational only):")
        for name in report.missing_songs:
            lines.append(f"  - {name}")

    lines.append("")
    if report.has_regression:
        lines.append(f"RESULT: regression(s) detected (tolerance={tolerance:.2f} pct points)")
    else:
        lines.append(f"RESULT: no regression (tolerance={tolerance:.2f} pct points)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(
        description=(
            "Regression-test the real (ptAKF) game score of already-converted "
            "UltraSinger songs across code changes."
        ),
        epilog=(
            "Example:\n"
            "  python tools/regression_benchmark.py D:/path/to/output "
            "--baseline D:/path/baseline.json --update-baseline\n"
            "  python tools/regression_benchmark.py D:/path/to/output "
            "--baseline D:/path/baseline.json --tolerance 0.5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "songs_dir",
        help="Directory containing one subfolder per converted song "
             "(each with a .txt and a '[Vocals]' audio file)",
    )
    parser.add_argument(
        "--baseline", required=True,
        help="Path to the baseline JSON file (outside the repo; user data)",
    )
    parser.add_argument(
        "--update-baseline", action="store_true",
        help="(Re)score every song and write the baseline file instead of comparing",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.5,
        help="Allowed score drop in percentage points before it counts as a "
             "regression (default: 0.5)",
    )
    args = parser.parse_args(argv)

    root = Path(args.songs_dir)
    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        return 2

    pairs, skip_reasons = find_song_pairs(root)
    for reason in skip_reasons:
        print(f"Warning: {reason}")

    if not pairs:
        print("No song folders with a matching .txt and '[Vocals]' audio file found.")
        return 2

    if args.update_baseline:
        scores = score_all(pairs)
        baseline_path = Path(args.baseline)
        baseline_path.write_text(
            json.dumps(scores, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote baseline for {len(scores)} song(s) to {baseline_path}")
        return 0

    baseline_path = Path(args.baseline)
    if not baseline_path.is_file():
        print(f"Error: baseline file not found: {baseline_path}")
        return 2

    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error: could not read baseline file {baseline_path}: {e}")
        return 2

    current = score_all(pairs)
    report = compare_scores(current, baseline, args.tolerance)
    print(format_table(report, args.tolerance))

    return 1 if report.has_regression else 0


if __name__ == "__main__":
    sys.exit(main())
