"""Find the uv executable that the install and update scripts use."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def find_uv() -> str | None:
    """Path of the uv executable to run, or None if uv is not installed.

    The install and update scripts install uv into ``~/.local/bin`` and put
    that folder first on PATH. Another uv found earlier on PATH, e.g. an old
    one installed with pip into a Python ``Scripts`` folder, may fail to read
    the project's ``uv.lock`` or rewrite it in an older format, so the copy in
    ``~/.local/bin`` is preferred whenever it exists.
    """
    local = Path.home() / ".local" / "bin" / ("uv.exe" if os.name == "nt" else "uv")
    if local.is_file():
        return str(local)
    return shutil.which("uv")
