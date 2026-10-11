"""Cache folder for the intermediate files of one conversion.

The intermediate files (separated stems, denoised and mono audio, transcription
and pitch data - several hundred MB per song) are written to a folder of their
own outside the song folder. In a song folder that a file-sync client watches,
the client would upload them while the conversion still uses them. Each
conversion gets a new cache folder, as it gets a new song folder. When the
conversion is done, the folder is deleted, or moved into the song folder when
the cache is to be kept.
"""

from __future__ import annotations

import os
import secrets
import shutil
import sys
import time
from pathlib import Path

# Name ending of the cache folders
FOLDER_SUFFIX = ".ultrasinger"
# File in every cache folder made here: only folders with it are ever removed
# as leftovers, also in a folder chosen with --cache_folder. A kept cache loses it.
MARKER = ".ultrasinger-cache"
# A conversion that was cancelled or crashed leaves its cache folder behind. It
# is removed when nothing in it has changed for longer than a conversion takes.
STALE_AFTER_S = 24 * 3600
# Characters of the song name in a folder name (short paths stay below the
# Windows path length limit).
NAME_CHARS = 40


def default_root() -> Path:
    """Per-user folder for the cache folders, outside any synced folder."""
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "UltraSinger" / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ultrasinger" / "cache"


def create(song_name: str, root: str | None = None) -> str:
    """A new, empty cache folder for one conversion of ``song_name`` in ``root``
    (default: :func:`default_root`). Leftovers of earlier conversions there are
    removed first (see :func:`remove_stale_folders`)."""
    base = Path(root) if root else default_root()
    base.mkdir(parents=True, exist_ok=True)
    remove_stale_folders(base)
    name = song_name[:NAME_CHARS].strip(" .")
    prefix = f"{name} " if name else ""
    for _ in range(100):
        folder = base / f"{prefix}{secrets.token_hex(4)}{FOLDER_SUFFIX}"
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        (folder / MARKER).touch()
        return str(folder)
    raise FileExistsError(f"no unused cache folder name in {base}")


def remove_stale_folders(base: Path, now: float | None = None) -> int:
    """Remove the cache folders in ``base`` that conversions left behind: those
    with the marker in which nothing changed for longer than ``STALE_AFTER_S``.
    Returns how many were removed."""
    now = time.time() if now is None else now
    removed = 0
    try:
        children = list(base.iterdir())
    except OSError:
        return 0
    for child in children:
        try:
            if (child.name.endswith(FOLDER_SUFFIX) and child.is_dir() and not child.is_symlink()
                    and (child / MARKER).is_file()
                    and now - _last_change(child) > STALE_AFTER_S):
                _remove(child)
                removed += 1
        except OSError:
            pass  # in use or gone: the next conversion tries again
    return removed


def finish(folder: str, song_folder: str, keep: bool) -> str | None:
    """End of a conversion: delete the cache ``folder``, or with ``keep`` move it
    into ``song_folder`` as ``cache`` (``cache (1)`` ... when that exists).
    Returns where a kept cache is: in the song folder, or still in ``folder``
    when the move failed."""
    if not keep:
        try:
            _remove(Path(folder))
        except OSError:
            pass  # what is left keeps its marker and is removed as a leftover later
        return None
    # A kept cache is no leftover: without the marker it is never removed
    try:
        (Path(folder) / MARKER).unlink(missing_ok=True)
    except OSError:
        pass
    target = _unused_path(os.path.join(song_folder, "cache"))
    try:
        shutil.move(folder, target)
    except OSError:
        return folder
    return target


def _last_change(folder: Path) -> float:
    """Newest modification time of the folder and anything in it."""
    newest = folder.stat().st_mtime
    for dirpath, dirnames, filenames in os.walk(folder):
        for name in dirnames + filenames:
            try:
                newest = max(newest, os.stat(os.path.join(dirpath, name), follow_symlinks=False).st_mtime)
            except OSError:
                pass
    return newest


def _remove(folder: Path) -> None:
    """Delete a cache folder, its marker last: a folder that could not be deleted
    completely is still recognised as a leftover by a later conversion."""
    for child in folder.iterdir():
        if child.name == MARKER:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    shutil.rmtree(folder)


def _unused_path(path: str) -> str:
    """``path``, or ``path (1)``, ``path (2)`` ... when it exists."""
    candidate, number = path, 1
    while os.path.exists(candidate):
        candidate = f"{path} ({number})"
        number += 1
    return candidate
