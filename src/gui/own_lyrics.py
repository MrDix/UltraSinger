"""The lyrics a user gives for one song of the queue.

They are a per-song setting (``lyrics_text``). A conversion gets them as a
file (``--lyrics_file``): the text is written into the folder ``own_lyrics``
of the settings folder, under a name made from the text itself, so the same
lyrics are the same file. Files that no conversion has used for a while are
removed.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

from .config import config_dir

logger = logging.getLogger(__name__)

# The files of lyrics older than this are removed when a new one is written
KEEP_DAYS = 30


def own_lyrics_folder() -> Path:
    return config_dir() / "own_lyrics"


def lyrics_sources_folder() -> Path:
    """The folder of the user's lyrics sources (see ``modules.lyrics_sources``)."""
    return config_dir() / "lyrics_sources"


def read_lyrics_text(path: str | Path) -> str:
    """The text of a lyrics file of the user: UTF-8 (also UTF-16 with a byte
    order mark), else the Western European encoding of Windows."""
    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def write_own_lyrics(text: str) -> Path:
    """The file with the lyrics ``text`` for a conversion.

    Raises:
        OSError: The file cannot be written.
    """
    folder = own_lyrics_folder()
    folder.mkdir(parents=True, exist_ok=True)
    _remove_old_files(folder)
    data = text.strip().encode("utf-8")
    path = folder / f"{hashlib.sha256(data).hexdigest()[:16]}.txt"
    if not path.exists():
        path.write_bytes(data)
    else:
        path.touch()  # in use again
    return path


def _remove_old_files(folder: Path) -> None:
    limit = time.time() - KEEP_DAYS * 86400
    for path in folder.glob("*.txt"):
        try:
            if path.stat().st_mtime < limit:
                path.unlink()
        except OSError as error:
            logger.debug("Old lyrics file %s not removed: %s", path.name, error)
