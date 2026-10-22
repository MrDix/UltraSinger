"""Check for a newer UltraSinger version and start the update script.

UltraSinger is installed as a git checkout and updated with
``install/update.bat`` (Windows) or ``install/update.sh``. On start the GUI
looks in the background whether the branch it runs on has new commits
upstream and, if so, asks whether to update. It never updates by itself.

The check fetches the upstream branch with git: the commits that would come
are then known (their subjects are shown in the question), and a checkout
with commits of its own (a developer's) is never offered an update that
``git pull --ff-only`` would refuse.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
_FETCH_TIMEOUT = 30
_GIT_TIMEOUT = 10
_POLL_SECONDS = 0.2
# At most this many commit subjects are listed in the question
MAX_SUBJECTS = 15
# Set for the update script started from the GUI: it starts the GUI again
# when the update has completed
RESTART_ENV = "ULTRASINGER_RESTART_GUI"


@dataclass
class UpdateInfo:
    """New commits on the upstream branch of the checkout."""

    upstream: str            # e.g. "origin/main"
    upstream_commit: str     # full hash of the upstream branch
    count: int               # number of new commits
    subjects: list[str] = field(default_factory=list)  # newest first, at most MAX_SUBJECTS


def repo_root() -> Path:
    """The folder of the checkout UltraSinger runs from (src/gui/ -> two up)."""
    return Path(__file__).resolve().parents[2]


def update_script(root: Path) -> Path:
    """The update script of the platform."""
    return root / "install" / ("update.bat" if os.name == "nt" else "update.sh")


def _git(root: Path, *args: str, timeout: float = _GIT_TIMEOUT, cancel=None) -> str | None:
    """Output of a git command in ``root``, or None when it fails or is cancelled.

    ``cancel`` (a ``threading.Event``) ends the command early: the GUI
    does not wait for a slow fetch when it closes.
    """
    git = shutil.which("git")
    if git is None:
        return None
    env = dict(os.environ)
    # Never ask for credentials: the check runs unattended (git's own prompt,
    # an askpass program the environment or git config names - an empty
    # GIT_ASKPASS wins over both -, the credential manager's sign-in window and
    # ssh's askpass). The SSH setup of the user is kept; in a session of its
    # own (POSIX), ssh has no terminal to ask on either.
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS"] = ""
    env["SSH_ASKPASS_REQUIRE"] = "never"
    env["GCM_INTERACTIVE"] = "never"
    try:
        proc = subprocess.Popen(
            [git, *args],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=_CREATE_NO_WINDOW,
            start_new_session=os.name != "nt",
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("git %s failed: %s", " ".join(args), e)
        return None
    deadline = time.monotonic() + timeout
    while True:
        try:
            out, err = proc.communicate(timeout=_POLL_SECONDS)
            break
        except subprocess.TimeoutExpired:
            if (cancel is not None and cancel.is_set()) or time.monotonic() > deadline:
                _kill(proc)
                logger.debug("git %s stopped (cancelled or timed out)", " ".join(args))
                return None
    if proc.returncode != 0:
        logger.debug("git %s failed: %s", " ".join(args), (err or "").strip())
        return None
    return out


def _kill(proc: subprocess.Popen) -> None:
    """End a git command together with its helpers (the fetch runs git-remote-https)."""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                           timeout=_GIT_TIMEOUT, creationflags=_CREATE_NO_WINDOW)
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        # The command leads a session of its own (see _git): end its whole group
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    proc.kill()
    try:
        proc.communicate(timeout=_GIT_TIMEOUT)
    except subprocess.TimeoutExpired:
        pass


def check_for_update(root: Path | None = None, cancel=None) -> UpdateInfo | None:
    """New commits on the upstream branch, or None (none, or no check possible).

    None as well for a checkout without git or upstream branch, a failed
    fetch (offline), and a checkout with commits of its own: the update script
    only fast-forwards. ``cancel`` (a ``threading.Event``) stops the check.
    Never raises.
    """
    root = root or repo_root()
    try:
        if not (root / ".git").exists() or not update_script(root).is_file():
            return None
        upstream = _git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", cancel=cancel)
        if not upstream or "/" not in upstream.strip():
            return None
        upstream = upstream.strip()
        remote, branch = upstream.split("/", 1)
        if _git(root, "fetch", "--quiet", "--no-tags", remote, branch, timeout=_FETCH_TIMEOUT,
                cancel=cancel) is None:
            return None
        counts = _git(root, "rev-list", "--left-right", "--count", "HEAD...@{u}", cancel=cancel)
        if counts is None:
            return None
        ahead, behind = (int(n) for n in counts.split())
        if behind == 0 or ahead > 0:
            if ahead > 0:
                logger.debug("No update offered: the checkout has %d commit(s) of its own", ahead)
            return None
        commit = (_git(root, "rev-parse", "@{u}", cancel=cancel) or "").strip()
        log = _git(root, "log", f"--max-count={MAX_SUBJECTS}", "--format=%s", "HEAD..@{u}", cancel=cancel) or ""
        return UpdateInfo(upstream, commit, behind, [s for s in log.splitlines() if s.strip()])
    except Exception:  # noqa: BLE001 — fail open, this is a startup check
        logger.debug("Update check failed", exc_info=True)
        return None


def start_update(root: Path | None = None, restart: bool = True) -> bool:
    """Start the update script in a console window of its own; True if it started.

    Windows only (see :func:`can_start_update`). The GUI quits right after:
    the script stops a running UltraSinger of the checkout anyway, to replace
    its files. With ``restart`` it starts the GUI again when the update has
    completed. Never raises.
    """
    root = root or repo_root()
    script = update_script(root)
    env = dict(os.environ)
    if restart:
        env[RESTART_ENV] = "1"
    # The GUI's own Python environment must not leak into the update's uv calls
    for name in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
        env.pop(name, None)
    # One command line, as cmd.exe reads it: with /s it only removes the outer
    # quotes, so the script's path stays quoted even with & or ^ in it (the
    # quoting of a list of arguments does not protect these from cmd.exe)
    command = f'"{env.get("COMSPEC", "cmd.exe")}" /d /s /c ""{script}""'
    try:
        subprocess.Popen(
            command,
            cwd=str(root),
            env=env,
            creationflags=_CREATE_NEW_CONSOLE,
            close_fds=True,
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning("Failed to start the update", exc_info=True)
        return False
    logger.info("Started the update: %s", script)
    return True


def can_start_update() -> bool:
    """Whether :func:`start_update` can run the script here; elsewhere the user runs it."""
    return os.name == "nt"
