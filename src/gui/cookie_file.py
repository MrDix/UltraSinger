"""Writing, copying and removing the exported cookie file (Netscape format for yt-dlp).

The GUI rewrites the cookie file while conversions and format probes read
it. Every write therefore replaces the file in one step (temporary file +
rename), and all writes, copies and removals take one lock, so no reader
sees a half-written file.

yt-dlp writes its cookies back into the cookie file it was given when it
ends. Each yt-dlp run therefore gets a private copy: it can neither clobber
the shared file nor bring back cookies that a bot-detection reset deleted.
A format probe also gets its PO token in a private yt-dlp option file
instead of on its command line, which other local users can read.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# Name prefixes of the private copies, of the private yt-dlp option files and
# of the temporary files of a write; files a crash left behind are removed on
# the next start (remove_stale_copies)
_COPY_PREFIX = "cookies-copy-"
_OPTIONS_PREFIX = "ytdlp-options-"
_TEMP_PREFIX = ".cookies-"


def _new_private_file(directory: Path, data: bytes, prefix: str, suffix: str) -> Path:
    """A new file in ``directory`` with ``data``, readable only by the owner."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=str(directory), prefix=prefix, suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)


def _replace(source: Path, target: Path) -> None:
    """``os.replace``, retried for a moment: on Windows a reader that has the
    target open (e.g. a virus scanner) makes it fail."""
    for attempt in range(20):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def write_cookie_file(path: str | Path, text: str) -> Path:
    """Replace the cookie file ``path`` with ``text`` in one step."""
    path = Path(path)
    with _lock:
        tmp = _new_private_file(path.parent, text.encode("utf-8"), _TEMP_PREFIX, ".tmp")
        try:
            _replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    return path


def write_private_copy(directory: str | Path, text: str) -> Path:
    """A new private cookie file with ``text`` in ``directory``; the caller removes it."""
    return _new_private_file(Path(directory), text.encode("utf-8"), _COPY_PREFIX, ".txt")


def write_private_options(directory: str | Path, text: str) -> Path:
    """A new private yt-dlp option file (for --config-locations) with ``text``;
    the caller removes it."""
    return _new_private_file(Path(directory), text.encode("utf-8"), _OPTIONS_PREFIX, ".conf")


def copy_cookie_file(path: str | Path) -> Path:
    """A private copy of the cookie file ``path``, next to it; the caller removes it."""
    path = Path(path)
    with _lock:
        data = path.read_bytes()
    return _new_private_file(path.parent, data, _COPY_PREFIX, ".txt")


def remove_cookie_file(path: str | Path) -> None:
    """Delete a cookie file (or a private copy); a missing file is fine."""
    with _lock:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete %s: %s", path, exc)


def remove_stale_copies(directory: str | Path) -> None:
    """Delete the cookie copies an earlier, crashed session left in ``directory``."""
    try:
        leftovers = (list(Path(directory).glob(f"{_COPY_PREFIX}*.txt"))
                     + list(Path(directory).glob(f"{_OPTIONS_PREFIX}*.conf"))
                     + list(Path(directory).glob(f"{_TEMP_PREFIX}*.tmp")))
    except OSError:
        return
    for leftover in leftovers:
        remove_cookie_file(leftover)
