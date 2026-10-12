"""Triage an UltraStar song library and move broken songs out of the way.

This is a **destructive** tool: songs that fail its checks are moved
(``shutil.move``) into a separate target directory, preserving their
relative folder structure. Safety comes first, throughout:

- Dry-run is the default. Nothing is touched unless ``--apply`` is passed.
- Stage 1 (no GPU) only flags a song when it is *unambiguously* broken:
  unparsable chart, zero notes, missing audio file, or an audio file
  ffprobe can't decode. Anything else passes stage 1 -- "when in doubt,
  keep it" (a corrupt song staying in the library is much cheaper than a
  good song being moved out of it). A file that can't be read right now
  (locked, no permission, vanished) or an ffprobe that fails to run says
  nothing about the song: that is an "error" verdict, never a move.
- Stage 2 (opt-in via ``--stage2``) additionally scores charts against
  their own separated vocals (ptAKF/MEDIUM, same engine as
  ``tools/library_rescore.py``) and flags songs scoring below a
  conservative threshold. Any exception anywhere in stage 2 (separation,
  scoring, timeouts, unreadable audio) results in an "error" verdict --
  never a move. A song is only ever moved because a real score was
  computed and it was below the threshold.
- The move itself has hard guards: ``corrupt_dir`` may not live inside
  ``source_dir`` (that would cause moved songs to be rescanned / an
  infinite loop), the two directories may not be identical, an
  existing destination is never overwritten (the song is left in place
  and logged as an error instead), and a folder that holds other song
  folders is never moved (it would take those songs along; it is logged
  as an error instead).
- Progress is written to a resumable CSV as it happens. A re-run skips
  songs whose previous verdict was terminal (``moved``/``kept``); rows
  that were only "would_move" (dry run) or "error" are re-evaluated,
  since neither represents a settled outcome.

Usage:
    python tools/library_triage.py "D:\\UltraStar\\Songs" "D:\\UltraStar\\_corrupt"
    python tools/library_triage.py "D:\\UltraStar\\Songs" "D:\\UltraStar\\_corrupt" --apply
    python tools/library_triage.py "D:\\UltraStar\\Songs" "D:\\UltraStar\\_corrupt" --stage2 --apply
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure stdout/stderr can print Unicode song paths on Windows consoles.
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Allow "from modules...." imports when this script is run directly (mirrors
# the pattern used by tools/pitch_tracker_benchmark.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_ultrastar import parse_score_song, read_chart_text, require_score_library  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = (".mp3", ".ogg", ".opus", ".m4a", ".wav", ".flac")

# Lenient sniff pattern used only to recognize "this .txt is plausibly an
# UltraStar chart" during folder discovery -- not a validity check.
_NOTE_LINE_RE = re.compile(r"^[:\*FRG]\s+-?\d+\s+\d+\s+-?\d+", re.MULTILINE)

CSV_FIELDS = ("rel_path", "stage1_verdict", "stage2_score", "action", "reason")

DEFAULT_STAGE2_THRESHOLD = 40.0
TERMINAL_ACTIONS = ("moved", "kept")

# Sentinel meaning "auto-detect ffprobe" -- distinct from an explicit None,
# which means "don't check audio decodability at all" (used by tests).
_AUTO = object()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@dataclass
class SongFolder:
    """A directory recognized as an UltraStar song folder."""
    dir_path: Path
    rel_path: str
    txt_path: Path
    # A .txt of the folder that could not be checked while the library was
    # scanned (read error, ...): stage 1 then reports the folder as an
    # error instead of judging it by the other files.
    unchecked_txt: Path | None = None


def _read_text_lenient(path: Path) -> str:
    """Best-effort text decode that never fails on the content, the same way
    the chart parser decodes it (UTF-8 and UTF-16 byte order marks,
    ``#ENCODING``, Windows-1252), so discovery and parser agree on what a
    chart is."""
    return read_chart_text(path)


def _looks_like_ultrastar_txt(path: Path) -> bool:
    """Lenient content sniff: is this .txt plausibly an UltraStar chart?

    Only used to tell song folders apart from "collection" folders that
    happen to contain an unrelated .txt (readme, license, ...). Actual
    validity is judged later by :func:`evaluate_stage1` via the real
    parser -- a chart that fails *that* check still counts as a song
    folder here, so it can be flagged as stage-1-corrupt rather than
    silently ignored. A .txt that can't be read counts as a chart too:
    its folder is then reported as an error instead of being skipped,
    judged by the other files in it, or moved along with a parent folder.
    """
    try:
        text = _read_text_lenient(path)
    except OSError:
        return True
    if not text.strip():
        return False
    upper = text.upper()
    return "#TITLE" in upper or "#BPM" in upper or bool(_NOTE_LINE_RE.search(text))


def _pick_primary_txt(candidates: list[Path]) -> tuple[Path, Path | None]:
    """Pick the chart .txt that parses with the most notes.

    Folders normally have exactly one qualifying .txt; this only matters
    for the rare folder with several. Ties broken alphabetically. If none
    of the candidates parse cleanly, the first (alphabetically) is kept
    so stage 1 still has something to flag as corrupt.

    Returns ``(primary, unchecked)``: *unchecked* is the first candidate
    that could not be checked at all (read error, ...), else None. Such a
    file may be the real chart or no chart at all, so stage 1 must not
    judge the folder this time.
    """
    ordered = sorted(candidates, key=lambda p: p.name)
    best = ordered[0]
    best_notes = -1
    unchecked = None
    for candidate in ordered:
        try:
            song = parse_score_song(candidate)
            notes = len(song.all_notes)
        except ValueError:
            notes = -1
        except Exception:  # noqa: BLE001 - not checkable now; stage 1 reports the folder as an error
            unchecked = unchecked or candidate
            continue
        if notes > best_notes:
            best_notes = notes
            best = candidate
    return best, unchecked


def find_song_folders(source_dir: Path, *, skip_dir: Path | None = None) -> list[SongFolder]:
    """Recursively find UltraStar song folders under *source_dir*.

    A folder qualifies when it directly (non-recursively) contains at
    least one .txt file that plausibly looks like an UltraStar chart
    (see :func:`_looks_like_ultrastar_txt`). Folders with no .txt at all,
    or only unrelated .txt files, are not songs and are ignored.

    *skip_dir*, when given, is pruned from the walk (defense in depth --
    ``run()`` already refuses to start when ``corrupt_dir`` is nested
    inside ``source_dir``, but this keeps discovery itself safe too).
    """
    source_dir = Path(source_dir).resolve()
    skip_dir = skip_dir.resolve() if skip_dir is not None else None

    folders: list[SongFolder] = []
    for dirpath, dirnames, filenames in os.walk(source_dir):
        current = Path(dirpath)

        if skip_dir is not None and (current == skip_dir or skip_dir in current.parents):
            dirnames[:] = []
            continue

        txt_files = [f for f in filenames if f.lower().endswith(".txt")]
        if not txt_files:
            continue

        candidates = [
            current / f for f in txt_files
            if _looks_like_ultrastar_txt(current / f)
        ]
        if not candidates:
            continue

        primary, unchecked = _pick_primary_txt(candidates)
        rel = current.relative_to(source_dir)
        folders.append(SongFolder(dir_path=current, rel_path=str(rel), txt_path=primary,
                                  unchecked_txt=unchecked))

    return folders


def _nested_song_folders(dir_path: Path) -> list[Path]:
    """Song folders anywhere below *dir_path* (not counting itself).

    Uses the same test as :func:`find_song_folders` (a .txt that can't be
    read counts as a chart): a move must never take a song along by
    mistake. Raises OSError if a subfolder can't be listed.
    """
    def _raise(error: OSError) -> None:
        raise error

    nested = []
    for dirpath, _dirnames, filenames in os.walk(dir_path, onerror=_raise):
        current = Path(dirpath)
        if current == dir_path:
            continue
        if any(f.lower().endswith(".txt") and _looks_like_ultrastar_txt(current / f) for f in filenames):
            nested.append(current)
    return nested


# ---------------------------------------------------------------------------
# Stage 1 (no GPU) -- hard corruption only
# ---------------------------------------------------------------------------

@dataclass
class Stage1Result:
    verdict: str  # "ok" | "corrupt" | "error" (could not be checked; never moved)
    reason: str = ""
    audio_path: Path | None = None


def _find_audio_file(dir_path: Path, song) -> Path | None:
    """Locate the song's audio file: the tagged file first, any known
    audio extension in the folder otherwise."""
    tagged = getattr(song, "audio", "") or ""
    if tagged:
        candidate = dir_path / tagged
        if candidate.is_file():
            return candidate

    for f in sorted(dir_path.iterdir()):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
            return f
    return None


def _resolve_ffprobe() -> str | None:
    """Find an ffprobe executable, reusing the project's own resolution
    logic (PATH + configured ffmpeg_path) with a plain PATH lookup as a
    last resort. Returns None if none can be found."""
    try:
        from modules.ffmpeg_helper import get_ffmpeg_and_ffprobe_paths
        _, ffprobe_path = get_ffmpeg_and_ffprobe_paths()
        return ffprobe_path
    except Exception:
        return shutil.which("ffprobe")


def _check_ffprobe(ffprobe_path: str) -> None:
    """Refuse to run with an ffprobe that does not work at all -- every song
    would fail the audio check and be moved. Raises ValueError (like the
    directory guards)."""
    try:
        result = subprocess.run([ffprobe_path, "-version"], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ValueError(f"ffprobe ({ffprobe_path}) does not run: {e}") from e
    if result.returncode != 0:
        raise ValueError(f"ffprobe ({ffprobe_path}) does not run (exit code {result.returncode}).")


class ProbeError(Exception):
    """ffprobe itself failed (could not start, timed out, unreadable output);
    this says nothing about the audio file."""


def probe_audio(path: Path, ffprobe_path: str) -> tuple[bool, str]:
    """Run ffprobe on *path*. Returns ``(ok, reason)``; *reason* is only
    meaningful when *ok* is False.

    Not ok when: ffprobe exits non-zero, there's no audio stream, or the
    reported duration is zero/empty. Raises :class:`ProbeError` when
    ffprobe can't be launched, times out, crashes or is killed, or its
    output isn't parseable JSON.
    """
    cmd = [
        ffprobe_path, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    try:
        # ffprobe writes UTF-8. Decoding it with the console codepage fails on
        # tags like curly quotes and would lose the whole output.
        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=60)
    except subprocess.TimeoutExpired as e:
        raise ProbeError("ffprobe timeout") from e
    except OSError as e:
        raise ProbeError(f"ffprobe launch failed: {e}") from e

    # Killed by a signal (negative on POSIX, e.g. when memory runs out) or
    # crashed (Windows status codes from 0xC0000000): not the file's fault.
    if result.returncode < 0 or result.returncode >= 0xC0000000:
        raise ProbeError(f"ffprobe crashed or was killed (exit {result.returncode})")
    if result.returncode != 0:
        return False, f"ffprobe exit {result.returncode}: {result.stderr.strip()[:300]}"

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise ProbeError("ffprobe output unparseable") from e
    if not isinstance(data, dict):
        raise ProbeError("ffprobe output unparseable")

    streams = data.get("streams") or []
    if not any(s.get("codec_type") == "audio" for s in streams):
        return False, "no audio streams"

    duration_raw = (data.get("format") or {}).get("duration")
    try:
        duration = float(duration_raw) if duration_raw not in (None, "") else 0.0
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        return False, "zero/empty duration"

    return True, ""


def evaluate_stage1(song: SongFolder, *, ffprobe_path: str | None) -> Stage1Result:
    """Stage 1 (no GPU): flag *only* unambiguous corruption.

    Conservative by design -- a song stays ("ok") unless it clearly can't
    be played at all. See the module docstring for the full rationale.

    Note on "completely undecodable text": the underlying parser's own
    encoding fallback chain (utf-8 -> cp1252 -> latin-1 -> utf-8 with
    replacement chars) never actually raises for bad encoding -- byte
    values are always representable in latin-1. A file that is genuinely
    not text (e.g. binary data misnamed ``.txt``) decodes into garbage
    that then simply matches no note lines, which the ``no_notes`` check
    below already catches. Mojibake in an otherwise-valid chart (wrong
    but decodable encoding) is deliberately *not* flagged, per spec: the
    song still plays, only the lyric text looks wrong. An empty ``#GAP``
    is read as 0, like the games do.

    Read errors (a locked or vanished file, no permission), ffprobe
    failures and any other unexpected exception are no evidence of
    corruption: they give an "error" verdict, which never moves the song.
    Only a ValueError -- all the parser raises for a malformed chart (a
    non-numeric ``#BPM``, text it cannot decode) -- means corrupt. The
    folder is not judged either when a .txt of it could not be checked
    while the library was scanned, or when the chart no longer looks like
    one (e.g. emptied while an editor saves it): both are errors that the
    next run checks again.
    """
    if song.unchecked_txt is not None:
        return Stage1Result(
            "error", f"txt_unchecked: {song.unchecked_txt.name} could not be read during the scan")
    if not _looks_like_ultrastar_txt(song.txt_path):
        return Stage1Result("error", f"txt_changed: {song.txt_path.name} no longer looks like a chart")
    try:
        parsed = parse_score_song(song.txt_path)
    except OSError as e:
        return Stage1Result("error", f"txt_unreadable: {e}")
    except ValueError as e:
        return Stage1Result("corrupt", f"unparsable_txt: {e}")
    except Exception as e:  # noqa: BLE001 - e.g. a missing package or no memory: nothing against the chart
        return Stage1Result("error", f"parse_failed: {e!r}")

    if len(parsed.all_notes) == 0:
        return Stage1Result("corrupt", "no_notes")

    try:
        audio_path = _find_audio_file(song.dir_path, parsed)
    except OSError as e:
        return Stage1Result("error", f"folder_unreadable: {e}")
    if audio_path is None:
        return Stage1Result("corrupt", "no_audio")

    if ffprobe_path:
        try:
            with open(audio_path, "rb") as f:
                f.read(1)
            ok, probe_reason = probe_audio(audio_path, ffprobe_path)
        except OSError as e:
            return Stage1Result("error", f"audio_unreadable: {e}", audio_path)
        except ProbeError as e:
            return Stage1Result("error", f"ffprobe_failed: {e}", audio_path)
        if not ok:
            return Stage1Result("corrupt", f"audio_undecodable: {probe_reason}", audio_path)

    return Stage1Result("ok", "", audio_path)


# ---------------------------------------------------------------------------
# Stage 2 (GPU) -- vocal-score quality/sync gate
# ---------------------------------------------------------------------------

def _make_separator(model_name: str, output_dir: str):
    """Instantiate + load an audio-separator model once, for reuse across
    the whole stage-2 pass (loading per song would be far too slow)."""
    from audio_separator.separator import Separator  # type: ignore[import-untyped]

    separator = Separator(
        output_dir=output_dir,
        output_format="WAV",
        sample_rate=44100,
        normalization_threshold=0.9,
    )
    separator.load_model(model_filename=model_name)
    return separator


def _default_separate(audio_path: Path, temp_dir: Path, *, separator) -> None:
    """Default stage-2 separation step: run the shared, pre-loaded
    ``separator`` against one song, writing ``vocals.wav``/``no_vocals.wav``
    into *temp_dir* (same output_dir the separator was constructed with)."""
    separator.separate(
        str(audio_path),
        custom_output_names={"Vocals": "vocals", "Instrumental": "no_vocals"},
    )


def _default_score_medium(txt_path: Path, vocals_path: Path) -> float:
    """Default stage-2 scoring step: ptAKF MEDIUM score, same approach as
    ``tools/library_rescore.py``. A ``#RELATIVE:YES`` chart is scored with
    absolute beats (see :func:`compare_ultrastar.parse_score_song`)."""
    import ultrastar_score
    from ultrastar_score import Difficulty, score_song

    song = parse_score_song(txt_path)
    if hasattr(ultrastar_score, "detect_pitch_frames"):
        pitch_frames = ultrastar_score.detect_pitch_frames(str(vocals_path))
        result = score_song(
            song, str(vocals_path), difficulty=Difficulty.MEDIUM,
            pitch_frames=pitch_frames,
        )
    else:
        result = score_song(song, str(vocals_path), difficulty=Difficulty.MEDIUM)
    return result.percentage


def evaluate_stage2(
    txt_path: Path,
    audio_path: Path,
    *,
    separate_fn,
    score_fn,
    temp_dir: Path,
    threshold: float,
) -> tuple[str, str, float | None]:
    """Stage 2: separate vocals, score MEDIUM, compare to *threshold*.

    Returns ``(verdict, reason, score)`` where verdict is one of
    ``"ok"``, ``"corrupt"``, ``"error"``.

    FAIL-SAFE (critical): *any* exception raised by ``separate_fn`` or
    ``score_fn`` -- separation failure, scoring failure, timeout, unreadable
    audio -- is caught here and reported as ``"error"``, never
    ``"corrupt"``. Callers must never move a song on an "error" verdict:
    a song is only ever moved because a real score was computed and found
    below the threshold. Temp vocals/instrumental files are always removed
    afterwards, success or failure.
    """
    try:
        separate_fn(audio_path, temp_dir)
        vocals_path = temp_dir / "vocals.wav"
        if not vocals_path.is_file():
            raise RuntimeError("separation did not produce vocals.wav")
        pct = score_fn(txt_path, vocals_path)
    except Exception as e:  # noqa: BLE001 - fail-safe: never let this abort or mis-flag as corrupt
        return "error", f"stage2_exception: {e}", None
    finally:
        for name in ("vocals.wav", "no_vocals.wav"):
            f = temp_dir / name
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass

    if pct < threshold:
        return "corrupt", f"stage2_score={pct:.1f}<{threshold:.1f}", pct
    return "ok", "", pct


# ---------------------------------------------------------------------------
# Move logic (the safety-critical part)
# ---------------------------------------------------------------------------

def _validate_dirs(source_dir: Path, corrupt_dir: Path) -> None:
    """Hard guards, checked once up front. Raises ValueError on violation."""
    if source_dir == corrupt_dir:
        raise ValueError(
            f"source_dir und corrupt_dir sind identisch ({source_dir}) - Abbruch."
        )
    try:
        corrupt_dir.relative_to(source_dir)
        nested = True
    except ValueError:
        nested = False
    if nested:
        raise ValueError(
            f"corrupt_dir ({corrupt_dir}) liegt innerhalb von source_dir "
            f"({source_dir}) - verschobene Songs wuerden erneut gescannt "
            "werden. Abbruch."
        )


def move_song(song: SongFolder, corrupt_dir: Path, *, apply: bool) -> tuple[str, str]:
    """Move (or, in dry-run, simulate moving) *song* into *corrupt_dir*,
    preserving its path relative to ``source_dir``. Returns ``(action, reason)``.

    Never overwrites an existing destination (skips + logs instead) and
    never moves ``source_dir`` itself. Never moves a folder that holds
    other song folders either (e.g. a collection folder with a stray
    chart): the songs inside would go along. Such a folder is reported
    as an error, in a dry run too. Move failures (locked file,
    permission denied, ...) are caught and reported, never raised -- one
    bad move must not abort the whole run.
    """
    if song.rel_path in (".", ""):
        return "error", "refusing to move source_dir root itself"

    try:
        nested = _nested_song_folders(song.dir_path)
    except OSError as e:
        return "error", f"not_moved: cannot check for song folders inside: {e}"
    if nested:
        return "error", (
            f"contains_song_folders_skipped: {len(nested)} song folder(s) inside, "
            f"e.g. {min(nested).relative_to(song.dir_path)}"
        )

    dest = corrupt_dir / song.rel_path

    if not apply:
        return "would_move", f"dest={dest}"

    if dest.exists():
        return "error", f"target_exists_skipped: {dest}"

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(song.dir_path), str(dest))
    except OSError as e:
        return "error", f"move_failed: {e}"

    return "moved", ""


# ---------------------------------------------------------------------------
# CSV resume
# ---------------------------------------------------------------------------

def load_existing_results(log_path: Path) -> dict[str, dict]:
    """Read a previous triage CSV, keyed by ``rel_path``. Empty dict if
    the file doesn't exist (yet) or is empty."""
    results: dict[str, dict] = {}
    if not log_path.exists() or log_path.stat().st_size == 0:
        return results

    with log_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("rel_path")
            if key:
                results[key] = row
    return results


def append_result(log_path: Path, row: dict, write_header: bool) -> None:
    """Append one result row to the CSV, creating it with a header if needed."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Per-song orchestration
# ---------------------------------------------------------------------------

def _row(song: SongFolder, stage1_verdict: str, stage2_score: str, action: str, reason: str) -> dict:
    return {
        "rel_path": song.rel_path,
        "stage1_verdict": stage1_verdict,
        "stage2_score": stage2_score,
        "action": action,
        "reason": reason,
    }


def _combine_reasons(a: str, b: str) -> str:
    if a and b:
        return f"{a}; {b}"
    return a or b


def _process_one(
    song: SongFolder,
    *,
    corrupt_dir: Path,
    apply: bool,
    ffprobe_path: str | None,
    stage2: bool,
    stage2_threshold: float,
    separate_fn,
    score_fn,
    temp_dir: Path | None,
) -> dict:
    """Evaluate one song folder end-to-end (stage 1, optionally stage 2,
    then move/skip) and return its CSV row."""
    stage1_result = evaluate_stage1(song, ffprobe_path=ffprobe_path)
    if stage1_result.verdict == "error":
        return _row(song, "error", "", "error", stage1_result.reason)

    is_corrupt = stage1_result.verdict == "corrupt"
    reason = stage1_result.reason
    stage2_score_str = ""
    had_error = False

    if stage1_result.verdict == "ok" and stage2:
        verdict2, reason2, score = evaluate_stage2(
            song.txt_path, stage1_result.audio_path,
            separate_fn=separate_fn, score_fn=score_fn,
            temp_dir=temp_dir, threshold=stage2_threshold,
        )
        if score is not None:
            stage2_score_str = f"{score:.1f}"
        if verdict2 == "corrupt":
            is_corrupt = True
            reason = reason2
        elif verdict2 == "error":
            had_error = True
            reason = reason2

    if had_error:
        return _row(song, stage1_result.verdict, stage2_score_str, "error", reason)

    if is_corrupt:
        action, move_reason = move_song(song, corrupt_dir, apply=apply)
        return _row(song, stage1_result.verdict, stage2_score_str, action, _combine_reasons(reason, move_reason))

    return _row(song, stage1_result.verdict, stage2_score_str, "kept", "")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    total: int = 0
    stage1_corrupt: int = 0
    stage2_fail: int = 0
    kept: int = 0
    moved: int = 0
    would_move: int = 0
    errors: int = 0
    skipped_resume: int = 0
    apply: bool = False
    all_rows: dict[str, dict] = field(default_factory=dict)


def run(
    source_dir: Path,
    corrupt_dir: Path,
    *,
    apply: bool = False,
    stage2: bool = False,
    stage2_threshold: float = DEFAULT_STAGE2_THRESHOLD,
    log_path: Path | None = None,
    limit: int | None = None,
    separator_model: str | None = None,
    progress_every: int = 25,
    log=print,
    ffprobe_path=_AUTO,
    separate_fn=None,
    score_fn=_default_score_medium,
) -> RunSummary:
    """Discover, triage, and (maybe) move songs in one library pass.

    Raises ``ValueError`` if ``corrupt_dir``/``source_dir`` violate the
    safety guards (see :func:`_validate_dirs`), if the optional scoring
    package is missing, or if ffprobe does not work -- callers must not
    catch this and continue; it means the run must not proceed at all.
    """
    source_dir = Path(source_dir).resolve()
    corrupt_dir = Path(corrupt_dir).resolve()
    _validate_dirs(source_dir, corrupt_dir)
    # Every chart is read with the optional score library: without it each
    # one would look unparsable and be moved.
    require_score_library()

    if log_path is None:
        log_path = corrupt_dir / "triage_log.csv"
    log_path = Path(log_path)

    # Moved songs mirror their full path *including* the source folder's own
    # name: source c:\Songs\Foo + song Foo\Party2\X  ->  corrupt\Foo\Party2\X.
    # This keeps songs from different source roots separated under one trash
    # dir and preserves where they came from. The guard/log above stay on the
    # user-given corrupt_dir; only the move destination gets the source name.
    move_dest_root = corrupt_dir / source_dir.name if source_dir.name else corrupt_dir

    resolved_ffprobe = _resolve_ffprobe() if ffprobe_path is _AUTO else ffprobe_path
    if resolved_ffprobe:
        _check_ffprobe(resolved_ffprobe)

    existing = load_existing_results(log_path)
    songs = find_song_folders(source_dir, skip_dir=corrupt_dir)
    if limit is not None:
        songs = songs[:limit]

    need_header = not log_path.exists() or log_path.stat().st_size == 0
    summary = RunSummary(total=len(songs), apply=apply, all_rows=dict(existing))

    temp_dir: Path | None = None
    own_separate_fn = separate_fn
    try:
        if stage2:
            temp_dir = Path(tempfile.mkdtemp(prefix="triage_vocals_"))
            if own_separate_fn is None:
                from modules.Audio.separation import DEFAULT_AUDIO_SEPARATOR_MODEL
                model_name = separator_model or DEFAULT_AUDIO_SEPARATOR_MODEL.value
                separator = _make_separator(model_name, str(temp_dir))
                own_separate_fn = functools.partial(_default_separate, separator=separator)

        total = len(songs)
        start_time = time.monotonic()
        stage2_time_total = 0.0
        stage2_done = 0

        for i, song in enumerate(songs, 1):
            prior = existing.get(song.rel_path)
            if prior and prior.get("action") in TERMINAL_ACTIONS:
                # A song "kept" without ever running stage 2 must be
                # re-checked when stage 2 is now requested -- this is the
                # stage-1-only pass followed by a separate stage-2 pass
                # (both writing the same log). Detect it by an empty
                # stage2_score on the prior "kept" row.
                needs_stage2_recheck = (
                    stage2
                    and prior.get("action") == "kept"
                    and not (prior.get("stage2_score") or "").strip()
                )
                if not needs_stage2_recheck:
                    summary.skipped_resume += 1
                    continue

            t0 = time.monotonic()
            try:
                row = _process_one(
                    song,
                    corrupt_dir=move_dest_root,
                    apply=apply,
                    ffprobe_path=resolved_ffprobe,
                    stage2=stage2,
                    stage2_threshold=stage2_threshold,
                    separate_fn=own_separate_fn,
                    score_fn=score_fn,
                    temp_dir=temp_dir,
                )
            except Exception as e:  # noqa: BLE001 - one song must never abort the run; an error never moves it
                row = _row(song, "error", "", "error", f"unexpected_error: {e!r}")
            if stage2 and row["stage1_verdict"] == "ok":
                stage2_time_total += time.monotonic() - t0
                stage2_done += 1

            append_result(log_path, row, write_header=need_header)
            need_header = False
            existing[song.rel_path] = row
            summary.all_rows[song.rel_path] = row

            action = row["action"]
            if action == "moved":
                summary.moved += 1
            elif action == "would_move":
                summary.would_move += 1
            elif action == "kept":
                summary.kept += 1
            elif action == "error":
                summary.errors += 1

            if row["stage1_verdict"] == "corrupt":
                summary.stage1_corrupt += 1
            elif action in ("moved", "would_move"):
                # stage1 was ok but the song is still being moved -> stage 2 caught it.
                summary.stage2_fail += 1

            if action in ("moved", "would_move"):
                verb = "MOVED" if action == "moved" else "WOULD MOVE"
                log(f"{verb} {song.rel_path} ({row['reason']})")
            elif action == "error":
                log(f"ERROR {song.rel_path} ({row['reason']})")

            if i % progress_every == 0 or i == total:
                elapsed = time.monotonic() - start_time
                avg2 = stage2_time_total / stage2_done if stage2_done else 0.0
                eta = avg2 * (total - i) if stage2 else 0.0
                log(
                    f"[{i}/{total}] korrupt={summary.stage1_corrupt + summary.stage2_fail} "
                    f"behalten={summary.kept} fehler={summary.errors} "
                    f"resume-uebersprungen={summary.skipped_resume} "
                    f"elapsed={elapsed:.0f}s" + (f" ETA~{eta:.0f}s" if stage2 else "")
                )
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(summary: RunSummary) -> None:
    print("\n=== Zusammenfassung ===")
    print(f"Songs gefunden (neu zu pruefen): {summary.total}")
    print(f"Resume uebersprungen:            {summary.skipped_resume}")
    print(f"Stufe-1 korrupt:                 {summary.stage1_corrupt}")
    print(f"Stufe-2 fehlgeschlagen:          {summary.stage2_fail}")
    print(f"Behalten (kept):                 {summary.kept}")
    print(f"Fehler/inconclusive:             {summary.errors}")
    if summary.apply:
        print(f"Verschoben:                      {summary.moved}")
    else:
        print(
            f"\n=== DRY RUN === nichts verschoben, "
            f"{summary.would_move} wuerden verschoben werden."
        )
        print("Mit --apply erneut ausfuehren, um wirklich zu verschieben.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Triage an UltraStar song library: move songs with hard "
            "structural corruption (and, optionally, a failing vocal "
            "score) into a separate target directory. Dry-run by default."
        ),
    )
    parser.add_argument("source_dir", help="Library root to scan recursively")
    parser.add_argument("corrupt_dir", help="Destination for broken songs (must not be inside source_dir)")
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually move songs. Without this flag: dry run, report only.",
    )
    parser.add_argument(
        "--stage2", action="store_true",
        help="Also run the GPU vocal-scoring stage on stage-1 survivors (default: off).",
    )
    parser.add_argument(
        "--stage2-threshold", type=float, default=DEFAULT_STAGE2_THRESHOLD,
        help="Medium-difficulty score percentage below which a song fails stage 2. Default: %(default)s",
    )
    parser.add_argument(
        "--log", default=None,
        help="Progress/verdict CSV, resumable. Default: <corrupt_dir>/triage_log.csv",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N discovered songs (for test runs).",
    )
    parser.add_argument(
        "--separator-model", default=None,
        help="audio-separator model filename for stage 2. Default: the Mel-Band-Roformer "
             "default from src/modules/Audio/separation.py",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    corrupt_dir = Path(args.corrupt_dir)
    log_path = Path(args.log) if args.log else None

    if not args.apply:
        print("=== DRY RUN === (kein --apply -> es wird NICHTS verschoben)")

    print(f"Scanne {source_dir} ...")
    try:
        summary = run(
            source_dir, corrupt_dir,
            apply=args.apply,
            stage2=args.stage2,
            stage2_threshold=args.stage2_threshold,
            log_path=log_path,
            limit=args.limit,
            separator_model=args.separator_model,
        )
    except ValueError as e:
        print(f"FEHLER: {e}", file=sys.stderr)
        sys.exit(1)

    _print_report(summary)


if __name__ == "__main__":
    main()
