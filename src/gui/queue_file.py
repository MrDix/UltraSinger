"""The conversion queue as a file.

The GUI keeps the queue in queue.json in the private settings directory: it
saves the queue whenever it changes and brings it back on the next start.
Export Queue and Import Queue use the same format in a file the user picks:

    {"version": 1, "items": [{"input_type": "url", "input_source": "https://...",
      "title": "Artist - Song Title", "video_id": "...", "status": "pending",
      "settings_overrides": {"language_mode": "manual", "language": "de"}}]}

Finished songs are left out; a song that was running or was cancelled is
written as waiting ("pending"), a failed one as "failed". An item is stored by
its source (the video URL or the local file), never by the temporary file its
browser stream is downloaded to while it runs: after a restart that file is
gone, and the item downloads again.

A queue file never holds a secret: of the per-song settings only those of the
per-song settings dialog are written, without API keys and tokens.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path

from .config import _DEFAULTS, _INVALID, _back_up_file, _fit_type, _is_secret_key, config_dir
from .cookie_file import _replace
from .models import QueueItem

logger = logging.getLogger(__name__)

QUEUE_FILE_NAME = "queue.json"
FORMAT_VERSION = 1

# A queue of thousands of songs is far smaller; a bigger file is no queue file
_MAX_FILE_SIZE = 10 * 1024 * 1024

# Songs read from one file: the queue list builds a row for each, and many
# thousands of rows would freeze the GUI (at every start, for the saved queue)
_MAX_SONGS = 1000

# Longer titles are cut (the queue list draws the title of each row)
_MAX_TITLE = 500

# Longer sources are no song (a command line on Windows holds at most this many characters)
_MAX_SOURCE = 32_767

# Settings that apply to all songs: the per-song settings dialog does not show them
_GLOBAL_ONLY = frozenset({
    "output_folder", "cookie_file", "proxy_mode", "proxy_url", "proxy_no_proxy",
    "potoken_auto_start", "potoken_auto_start_node", "potoken_auto_start_docker",
    "potoken_base_url", "llm_providers", "llm_api_base_url", "llm_model",
    "llm_legacy_provider_id", "training_library", "training_workdir",
    "training_exclude", "training_model", "training_epochs",
})

# The settings an item can override (those of the per-song settings dialog),
# without the API keys and tokens
PER_SONG_SETTINGS = frozenset(
    key for key in (set(_DEFAULTS) | {"llm_provider_id"}) - _GLOBAL_ONLY
    if not _is_secret_key(key)
)

# Per-song settings that name a path, a web address or a model. An imported
# file, which may come from someone else, does not set them: a program or
# model it names could run code, and a server would get the API key.
_NOT_IMPORTED = frozenset({
    "ffmpeg_path", "musescore_path", "cache_folder", "segmentation_model",
    "segmentation_model_repo", "whisper_model", "whisper_align_model",
    "audio_separator_model", "demucs_model", "remote_stt_api_base_url",
    "remote_stt_model",
})

# Statuses of a song that has not finished; it waits (again) after a restore
_WAITING = ("pending", "running", "cancelled")

# A video ID of the video platform (letters, digits, "-" and "_")
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def queue_file_path() -> Path:
    """Where the GUI keeps the queue between sessions (in the private settings directory)."""
    return config_dir() / QUEUE_FILE_NAME


def _shortened(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "\u2026"


def _stored_settings(settings, imported: bool = False, problems: list[str] | None = None,
                     where: str = "") -> dict:
    """The per-song settings of ``settings`` that a queue file holds.

    A note on each setting that is left out goes to ``problems``; it names the
    setting, never its value (that could be a secret). A value of a wrong type
    (e.g. hand-edited) is left out: the per-song settings dialog could not show it.
    """
    def skip(reason: str):
        if problems is not None:
            problems.append(f"{where}: {reason}")

    if not isinstance(settings, dict):
        if settings is not None:
            skip("per-song settings ignored (not a list of settings)")
        return {}
    stored = {}
    for key, value in settings.items():
        if key not in PER_SONG_SETTINGS:
            if isinstance(key, str) and _is_secret_key(key):
                skip(f"setting '{_shortened(key)}' ignored "
                     "(API keys and tokens are never kept in a queue file)")
            else:
                skip(f"setting '{_shortened(str(key))}' ignored (not a per-song setting)")
            continue
        if imported and key in _NOT_IMPORTED:
            skip(f"setting '{key}' not imported (it names a path, web address or model; "
                 "your own setting applies)")
            continue
        if value is None:
            continue
        fitted = _fit_type(value, _DEFAULTS.get(key, ""))
        if fitted is _INVALID:
            skip(f"setting '{key}' ignored (wrong type)")
            continue
        stored[key] = fitted
    return stored


def _entry(item: QueueItem) -> dict | None:
    """The queue-file entry of ``item``; None for a finished song."""
    if item.status == "done":
        return None
    return {
        "input_type": item.input_type,
        "input_source": item.input_source,
        "title": item.title,
        "video_id": item.video_id,
        # A song that ran when the queue was saved, or was cancelled, waits again
        "status": "failed" if item.status == "failed" else "pending",
        "settings_overrides": _stored_settings(item.settings_overrides),
    }


def _item(entry, number: int, imported: bool, problems: list[str]) -> QueueItem | None:
    """The queue item of a file entry; None, with a note in ``problems``, for an invalid one."""
    if not isinstance(entry, dict):
        problems.append(f"entry {number} skipped (not a song)")
        return None
    title = entry.get("title")
    title = title[:_MAX_TITLE] if isinstance(title, str) and title.strip() else ""
    where = f"entry {number}" + (f" '{_shortened(title)}'" if title else "")
    input_type = entry.get("input_type")
    source = entry.get("input_source")
    if input_type not in ("url", "file"):
        problems.append(f"{where} skipped (unknown source type)")
        return None
    if (not isinstance(source, str) or not source.strip() or "\x00" in source
            or len(source) > _MAX_SOURCE):
        problems.append(f"{where} skipped (no valid source)")
        return None
    if input_type == "url" and not source.lower().startswith(("https://", "http://")):
        problems.append(f"{where} skipped (the source is not a web address)")
        return None
    if input_type == "file" and not Path(source).is_absolute():
        problems.append(f"{where} skipped (the source is not a full file path)")
        return None
    status = entry.get("status", "pending")
    if status == "done":
        problems.append(f"{where} skipped (the song is finished)")
        return None
    if status not in (*_WAITING, "failed"):
        problems.append(f"{where} skipped (unknown status)")
        return None
    video_id = entry.get("video_id")
    if not (isinstance(video_id, str) and _VIDEO_ID.fullmatch(video_id)):
        # It becomes part of a file name (the download of a browser stream)
        video_id = ""
    return QueueItem(
        input_source=source,
        input_type=input_type,
        title=title or source,
        # An imported song is added to be converted; a restored failed one stays marked
        status="failed" if status == "failed" and not imported else "pending",
        video_id=video_id,
        settings_overrides=_stored_settings(entry.get("settings_overrides"), imported, problems, where),
    )


def write_queue_file(path: Path, items: list[QueueItem]) -> int:
    """Write the songs of ``items`` that are not finished to ``path`` in one step.

    Returns the number of songs written. Raises OSError.
    """
    entries = [entry for entry in map(_entry, items) if entry is not None]
    text = json.dumps({"version": FORMAT_VERSION, "items": entries}, indent=2, ensure_ascii=False)
    path = Path(path)
    # A temporary file renamed over the target: no reader sees half a queue.
    # mkstemp creates it readable only by the owner.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="queue_", suffix=".tmp")
    try:
        # backslashreplace: a lone surrogate (possible in a Windows file name)
        # has no UTF-8 form; written as \udXXX it is a JSON escape that reads back
        with os.fdopen(fd, "w", encoding="utf-8", errors="backslashreplace") as f:
            f.write(text)
        _replace(Path(tmp), path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return len(entries)


def read_queue_file(path: Path, imported: bool = False) -> tuple[list[QueueItem], list[str]]:
    """The songs of the queue file ``path``, and a note on each entry or setting left out.

    ``imported``: the user imports the file (it is not the saved queue). Its
    songs are added as waiting songs, and it does not set the per-song
    settings that name a path, web address or model.

    Raises OSError when the file cannot be read, and ValueError when it is no
    queue file this version can read.
    """
    path = Path(path)
    # utf-8-sig: editors may write a byte order mark
    with open(path, "r", encoding="utf-8-sig") as f:
        if os.fstat(f.fileno()).st_size > _MAX_FILE_SIZE:
            raise ValueError("the file is too large for a queue file")
        try:
            data = json.load(f)
        except RecursionError:
            raise ValueError("the file is nested too deeply") from None
    if not isinstance(data, dict) or "version" not in data:
        raise ValueError("not a queue file")
    version = data["version"]
    if version != FORMAT_VERSION or isinstance(version, bool):
        raise ValueError(f"unsupported queue file version {_shortened(repr(version), 20)} "
                         f"(this version of UltraSinger reads version {FORMAT_VERSION})")
    entries = data.get("items")
    if not isinstance(entries, list):
        raise ValueError("not a queue file (no list of songs)")
    items, problems = [], []
    if len(entries) > _MAX_SONGS:
        problems.append(f"only the first {_MAX_SONGS} songs are read, "
                        f"{len(entries) - _MAX_SONGS} more are left out")
        entries = entries[:_MAX_SONGS]
    for number, entry in enumerate(entries, start=1):
        item = _item(entry, number, imported, problems)
        if item is not None:
            items.append(item)
    for problem in problems:
        logger.warning("Queue file %s: %s", path, problem)
    return items, problems


def save_queue(path: Path, items: list[QueueItem]) -> None:
    """Save the queue for the next start; a failure is logged and the GUI goes on."""
    try:
        write_queue_file(path, items)
    except (OSError, ValueError, TypeError):
        logger.warning("Failed to save the queue to %s", path, exc_info=True)


def _read_saved_queue(path: Path) -> tuple[list[QueueItem], list[str]]:
    """``read_queue_file``, retried for a moment: on Windows a reader that has
    the file open (e.g. a virus scanner) makes it fail."""
    attempts = 0
    while True:
        try:
            return read_queue_file(path)
        except PermissionError:
            attempts += 1
            if attempts == 20:
                raise
            time.sleep(0.05)


def load_saved_queue(path: Path) -> tuple[list[QueueItem], list[str], bool]:
    """The queue the last session saved, a note on everything that did not come
    back, and whether the queue may be saved over the file.

    When something of the file does not come back (it cannot be read, or holds
    entries or settings this version leaves out), the file is copied aside
    first, so nothing in it is lost silently. If that copy fails too, the
    queue must not be saved over the file in this session.
    """
    path = Path(path)
    try:
        items, notes = _read_saved_queue(path)
    except FileNotFoundError:
        return [], [], True
    except (OSError, ValueError) as exc:  # ValueError: damaged, or another format
        logger.warning("Queue file %s cannot be read (%s); starting with an empty queue",
                       path, exc)
        items, notes = [], [f"cannot be read ({exc}); the queue starts empty."]
    if not notes:
        return items, [], True
    backup = _back_up_file(path)
    if backup is None:
        note = "The file could not be copied, so the queue is not saved in this session."
    else:
        note = f"The file as it was is kept in {backup}."
    logger.warning("Queue file %s: %s", path, note)
    return items, notes + [note], backup is not None


def without_duplicates(imported: list[QueueItem],
                       queue: list[QueueItem]) -> tuple[list[QueueItem], int]:
    """The imported songs that do not wait in the queue already, and how many were left out.

    A song waits already when an item that has not finished (pending or
    running) has the same source and the same per-song settings, as far as an
    imported file can set them: re-importing an export adds nothing twice. A
    song the queue has finished, or that failed or was cancelled, is added again.
    """
    def key(item: QueueItem):
        settings = _stored_settings(item.settings_overrides, imported=True)
        return (item.input_type, item.input_source, json.dumps(settings, sort_keys=True))

    waiting = {key(item) for item in queue if item.status in ("pending", "running")}
    added = [item for item in imported if key(item) not in waiting]
    return added, len(imported) - len(added)
