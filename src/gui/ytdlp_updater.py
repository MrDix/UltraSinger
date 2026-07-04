"""Check for newer yt-dlp releases and perform a one-click update.

Video platforms frequently change their player/delivery internals; yt-dlp ships
frequent releases to keep up, and an install left un-upgraded for more
than a few months tends to start failing downloads with HTTP 403. This
module only *checks* for a newer release and, on explicit user request,
runs the update — it never updates automatically.

Update mechanism: this project pins dependencies via ``uv`` (see
``pyproject.toml`` / ``uv.lock``), so "updating yt-dlp" means re-locking
and re-syncing the project's virtual environment rather than pip-installing
into it directly. See :func:`run_update` for details and a note on how this
interacts with CUDA developer setups using ``git update-index
--skip-worktree`` on ``pyproject.toml``.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_PYPI_URL = "https://pypi.org/pypi/yt-dlp/json"
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_LOCK_TIMEOUT = 180
_SYNC_TIMEOUT = 300
_MAX_OUTPUT_CHARS = 4000


def get_installed_version() -> str:
    """Return the installed yt-dlp version, or "" if it cannot be determined.

    Fail-open: never raises.
    """
    try:
        return importlib.metadata.version("yt-dlp")
    except importlib.metadata.PackageNotFoundError:
        return ""
    except Exception:  # noqa: BLE001 — fail open, this is a startup check
        logger.debug("Failed to read installed yt-dlp version", exc_info=True)
        return ""


def get_latest_version(timeout: float = 5) -> str:
    """Return the newest yt-dlp version published on PyPI, or "" on failure.

    Fail-open: any network/parse error results in "" rather than raising,
    so a flaky connection never blocks GUI startup.
    """
    try:
        req = urllib.request.Request(
            _PYPI_URL, headers={"User-Agent": "UltraSinger"}
        )
        with urllib.request.urlopen(  # noqa: S310 — fixed https URL, no user input
            req, timeout=timeout
        ) as resp:
            if resp.status != 200:
                return ""
            data = json.loads(resp.read().decode("utf-8", "replace"))
        version = data.get("info", {}).get("version", "")
        return str(version) if version else ""
    except (
        urllib.error.URLError,
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as e:
        logger.debug("Failed to fetch latest yt-dlp version: %s", e)
        return ""


def is_outdated(installed: str, latest: str) -> bool:
    """True if ``latest`` is a parseable version strictly newer than ``installed``.

    Tolerant of empty/unparseable input — returns False rather than raising,
    since either value may be "" when the corresponding lookup failed.
    """
    if not installed or not latest:
        return False
    try:
        from packaging.version import InvalidVersion, Version

        return Version(latest) > Version(installed)
    except InvalidVersion:
        return False


def _run_step(cmd: list[str], cwd: Path, timeout: int) -> tuple[bool, str]:
    """Run one update subprocess step. Returns (ok, short_log_excerpt)."""
    printable = " ".join(cmd)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=_CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        msg = f"$ {printable}\nTimed out after {timeout}s"
        logger.warning("yt-dlp update step timed out: %s", printable)
        return False, msg
    except (OSError, subprocess.SubprocessError) as e:
        msg = f"$ {printable}\nFailed to run: {e}"
        logger.warning("yt-dlp update step failed to launch: %s", e)
        return False, msg

    lines = [f"$ {printable}"]
    stdout_tail = (result.stdout or "").strip().splitlines()[-8:]
    stderr_tail = (result.stderr or "").strip().splitlines()[-8:]
    lines.extend(stdout_tail)
    if result.returncode != 0:
        lines.append(f"[exit code {result.returncode}]")
        lines.extend(stderr_tail)
        return False, "\n".join(lines)
    return True, "\n".join(lines)


def run_update(repo_root: str | Path) -> tuple[bool, str]:
    """Upgrade yt-dlp in the project's uv-managed environment.

    Runs, in ``repo_root``:

      1. ``uv lock --upgrade-package yt-dlp``
      2. ``uv sync --extra gui --extra scoring --extra potoken``

    Important: this **regenerates ``uv.lock``** with a newer yt-dlp pin (and
    whatever transitive changes that implies) — it is the same mechanism the
    install scripts use, not a raw ``pip install --upgrade`` into the venv.
    For CUDA-configured worktrees where ``pyproject.toml`` has been marked
    ``git update-index --skip-worktree`` (see the CUDA install docs), this
    is safe: ``uv lock`` reads whatever ``pyproject.toml`` is on disk right
    now (the skip-worktree, CUDA-pinned version) and rewrites ``uv.lock``
    locally to match it — it does not touch git, does not revert the
    skip-worktree edit, and does not run any git command at all.

    Never raises. Returns ``(ok, combined_output)`` where ``combined_output``
    is a short, human-readable summary (last lines of each step) suitable
    for a console log line or a message box body.
    """
    repo_root = Path(repo_root)

    if shutil.which("uv") is None:
        msg = (
            "'uv' was not found on PATH. Install uv "
            "(https://docs.astral.sh/uv/) and try again, or re-run the "
            "install script for this platform."
        )
        logger.warning("yt-dlp update aborted: %s", msg)
        return False, msg

    steps = [
        (["uv", "lock", "--upgrade-package", "yt-dlp"], _LOCK_TIMEOUT),
        (
            ["uv", "sync", "--extra", "gui", "--extra", "scoring", "--extra", "potoken"],
            _SYNC_TIMEOUT,
        ),
    ]

    output_chunks: list[str] = []
    for cmd, timeout in steps:
        ok, chunk = _run_step(cmd, repo_root, timeout)
        output_chunks.append(chunk)
        if not ok:
            combined = "\n".join(output_chunks)
            return False, combined[-_MAX_OUTPUT_CHARS:]

    output_chunks.append("yt-dlp updated successfully. Restart UltraSinger to use it.")
    combined = "\n".join(output_chunks)
    return True, combined[-_MAX_OUTPUT_CHARS:]
