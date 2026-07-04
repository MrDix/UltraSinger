"""Find the weakest UltraStar charts in a large generated song library.

Scans a directory tree for UltraSinger output folders — a folder qualifies
when it contains an UltraStar ``*.txt`` chart *and* a separated vocals
audio track (UltraSinger's output convention: filename contains
``[Vocals]``, extension ``.ogg``/``.mp3``/``.wav``). Each chart is scored
against its own vocal track with ``ultrastar-score`` (the same ptAKF
algorithm the games use) at the MEDIUM difficulty only — Easy/Hard are
skipped here purely for speed, since this tool scans whole libraries and
Medium is the most representative middle ground.

Folders that have a chart but *no* isolated vocals track (e.g. original
song rips that only ship the full mix) are skipped and counted, never
scored: scoring a chart against a full instrumental+vocal mix would be
meaningless noise for a ptAKF-based pitch detector, which is built to
listen to an isolated voice. A future ``--separate`` mode could run vocal
separation on demand for such folders, but that is out of scope here —
this tool only rescoring existing UltraSinger outputs.

The scan is resumable: results are appended to a CSV as they come in, and
a re-run reads that CSV first and skips any path already present (whether
it succeeded or errored previously). Use ``--fresh`` to ignore an existing
CSV and rescore everything from scratch.

Usage:
    uv run python tools/library_rescore.py "D:\\UltraStar\\Songs_UltraSinger"
    uv run python tools/library_rescore.py "D:\\UltraStar\\Songs_UltraSinger" --output results.csv
    uv run python tools/library_rescore.py "D:\\UltraStar\\Songs_UltraSinger" --limit 20
    uv run python tools/library_rescore.py "D:\\UltraStar\\Songs_UltraSinger" --fresh
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure stdout/stderr can print Unicode song paths on Windows consoles.
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# UltraSinger's separated-vocals output convention: "<title> [Vocals].<ext>"
_VOCAL_MARKER = "[vocals]"
_VOCAL_EXTENSIONS = (".ogg", ".mp3", ".wav")

CSV_FIELDS = ("pfad", "medium_pct", "notes", "status", "fehler")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@dataclass
class SongEntry:
    """A discovered UltraSinger output folder ready to be scored."""
    dir_path: Path
    txt_path: Path
    vocal_path: Path


@dataclass
class DiscoveryResult:
    entries: list[SongEntry] = field(default_factory=list)
    skipped_no_vocals: int = 0


def _pick_best(candidates: list[str]) -> str:
    """Pick the most plausible file when several match.

    Prefers names without ``[`` (UltraSinger's stem-only chart/vocals
    files carry brackets like ``[Vocals]``/``[Instrumental]``/``[CO]``
    for side files, not the main chart), then the shortest, then
    alphabetically — this is a heuristic tie-breaker, not a hard rule.
    """
    return sorted(candidates, key=lambda f: ("[" in f, len(f), f))[0]


def find_song_dirs(root: Path) -> DiscoveryResult:
    """Recursively find UltraSinger output folders under *root*.

    A folder qualifies when it has at least one ``*.txt`` file AND a
    separated vocals audio file. Folders with a chart but no vocals track
    are counted in ``skipped_no_vocals`` and not scored (see module
    docstring for why). Folders without any ``*.txt`` at all are not
    UltraStar output folders and are silently ignored.
    """
    result = DiscoveryResult()
    for dirpath, _dirnames, filenames in os.walk(root):
        txt_files = [f for f in filenames if f.lower().endswith(".txt")]
        if not txt_files:
            continue

        vocal_files = [
            f for f in filenames
            if _VOCAL_MARKER in f.lower() and f.lower().endswith(_VOCAL_EXTENSIONS)
        ]
        if not vocal_files:
            result.skipped_no_vocals += 1
            continue

        dir_path = Path(dirpath)
        result.entries.append(SongEntry(
            dir_path=dir_path,
            txt_path=dir_path / _pick_best(txt_files),
            vocal_path=dir_path / _pick_best(vocal_files),
        ))

    return result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_medium(txt_path: Path, vocal_path: Path) -> tuple[float, int]:
    """Score a chart against its vocal track at USDX MEDIUM difficulty.

    Returns ``(percentage, note_count)``. Uses the ``detect_pitch_frames``
    + ``score_song(pitch_frames=...)`` fast path when the installed
    ``ultrastar-score`` version offers it (checked via ``hasattr`` so this
    keeps working across library versions that don't), otherwise falls
    back to the plain ``score_song`` call.

    Raises whatever the underlying parse/score call raises — the caller
    is expected to catch broadly and record a CSV error row per song so
    one broken chart or unreadable audio file never aborts a whole
    library scan.
    """
    import ultrastar_score
    from ultrastar_score import Difficulty, score_song
    from ultrastar_score.parser import parse_ultrastar

    song = parse_ultrastar(str(txt_path))
    note_count = len(song.all_notes)

    if hasattr(ultrastar_score, "detect_pitch_frames"):
        pitch_frames = ultrastar_score.detect_pitch_frames(str(vocal_path))
        result = score_song(
            song, str(vocal_path), difficulty=Difficulty.MEDIUM,
            pitch_frames=pitch_frames,
        )
    else:
        result = score_song(song, str(vocal_path), difficulty=Difficulty.MEDIUM)

    return result.percentage, note_count


# ---------------------------------------------------------------------------
# CSV resume
# ---------------------------------------------------------------------------

def load_existing_results(output_path: Path) -> dict[str, dict]:
    """Read a previously written results CSV, keyed by the ``pfad`` column.

    Returns an empty dict when the file doesn't exist (yet) or is empty.
    """
    results: dict[str, dict] = {}
    if not output_path.exists() or output_path.stat().st_size == 0:
        return results

    with output_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("pfad")
            if path:
                results[path] = row
    return results


def append_result(output_path: Path, row: dict, write_header: bool) -> None:
    """Append one result row to the CSV, creating it with a header if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    total_found: int = 0
    skipped_no_vocals: int = 0
    scored_this_run: int = 0
    skipped_resume: int = 0
    errors_this_run: int = 0
    all_rows: dict[str, dict] = field(default_factory=dict)


def run(
    root: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    fresh: bool = False,
    score_fn=score_medium,
    progress_every: int = 10,
    log=print,
) -> RunSummary:
    """Discover, (re)score, and CSV-append results for one library scan.

    Kept separate from ``main()`` so tests can inject a fake ``score_fn``
    without touching real audio decoding.
    """
    discovery = find_song_dirs(root)
    entries = discovery.entries
    if limit is not None:
        entries = entries[:limit]

    if fresh and output_path.exists():
        output_path.unlink()

    existing = {} if fresh else load_existing_results(output_path)
    need_header = not output_path.exists() or output_path.stat().st_size == 0

    summary = RunSummary(
        total_found=len(discovery.entries),
        skipped_no_vocals=discovery.skipped_no_vocals,
        all_rows=dict(existing),
    )

    total = len(entries)
    start_time = time.monotonic()
    scoring_time_total = 0.0

    for i, entry in enumerate(entries, 1):
        key = str(entry.dir_path)

        if key in existing:
            summary.skipped_resume += 1
        else:
            t0 = time.monotonic()
            try:
                pct, notes = score_fn(entry.txt_path, entry.vocal_path)
                row = {
                    "pfad": key, "medium_pct": f"{pct:.1f}",
                    "notes": notes, "status": "ok", "fehler": "",
                }
                summary.scored_this_run += 1
            except Exception as e:  # noqa: BLE001 - one bad song must never abort a library scan
                row = {
                    "pfad": key, "medium_pct": "", "notes": "",
                    "status": "error", "fehler": str(e),
                }
                summary.errors_this_run += 1
            scoring_time_total += time.monotonic() - t0

            append_result(output_path, row, write_header=need_header)
            need_header = False
            existing[key] = row
            summary.all_rows[key] = row

        if i % progress_every == 0 or i == total:
            elapsed = time.monotonic() - start_time
            avg = scoring_time_total / summary.scored_this_run if summary.scored_this_run else 0.0
            eta = avg * (total - i)
            log(
                f"[{i}/{total}] neu={summary.scored_this_run} "
                f"resume-uebersprungen={summary.skipped_resume} "
                f"fehler={summary.errors_this_run} "
                f"elapsed={elapsed:.0f}s ETA~{eta:.0f}s"
            )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(summary: RunSummary, limit_worst: int = 20) -> None:
    ok_rows = [r for r in summary.all_rows.values() if r.get("status") == "ok"]
    ok_rows.sort(key=lambda r: float(r["medium_pct"]))

    print(f"\n=== Top {min(limit_worst, len(ok_rows))} schlechteste Charts (Medium) ===")
    if not ok_rows:
        print("(keine erfolgreich gescorten Songs)")
    for r in ok_rows[:limit_worst]:
        print(f"{float(r['medium_pct']):6.1f}%  notes={r['notes']:<6}  {r['pfad']}")

    error_rows = [r for r in summary.all_rows.values() if r.get("status") == "error"]

    print("\n=== Zusammenfassung ===")
    print(f"Gefunden (mit Vocals):     {summary.total_found}")
    print(f"Uebersprungen (kein Vocal): {summary.skipped_no_vocals}")
    print(f"Neu gescort in diesem Lauf: {summary.scored_this_run}")
    print(f"Resume (bereits vorhanden): {summary.skipped_resume}")
    print(f"Fehler (gesamt in CSV):     {len(error_rows)}")
    if error_rows:
        print("\n--- Fehler (erste 10) ---")
        for r in error_rows[:10]:
            print(f"  {r['pfad']}: {r['fehler']}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Find the weakest UltraStar charts in a generated song library "
            "by re-scoring them (MEDIUM difficulty) against their own "
            "separated vocal track."
        ),
    )
    parser.add_argument("root", help="Root directory to scan recursively")
    parser.add_argument(
        "--output", default="library_rescore_results.csv",
        help="CSV results file (appended to, resumable). Default: %(default)s",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N discovered songs (for test runs)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore any existing --output CSV and rescore everything",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_path = Path(args.output)

    print(f"Scanne {root} ...")
    summary = run(root, output_path, limit=args.limit, fresh=args.fresh)
    _print_report(summary)


if __name__ == "__main__":
    main()
