"""Manage a bgutil PO-token provider for yt-dlp full-quality downloads.

YouTube's SABR delivery no longer exposes a capturable media URL or PO
token to the embedded browser (the token lives in the binary POST body,
which Qt's request interceptor cannot read).  The maintained solution is
the ``bgutil-ytdlp-pot-provider`` yt-dlp plugin, which fetches GVS
Proof-of-Origin tokens from a small local provider server.  With the
plugin installed (same venv as yt-dlp) and the server reachable, yt-dlp
transparently obtains PO tokens and restores the full format list.

This module only manages the *server*: it checks whether one is already
running and, when possible, starts one via Docker.  It never blocks the
GUI and fails open — if no provider is available the app keeps working
with the previous (degraded) behaviour.

Provider server options (either works; the plugin auto-detects the
default port 4416):
  * Docker:  docker run -d --rm -p 4416:4416 brainicism/bgutil-ytdlp-pot-provider
  * Node.js: clone the repo, ``npm install`` in ``server/`` and run it.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:4416"
_DOCKER_IMAGE = "brainicism/bgutil-ytdlp-pot-provider"
_CONTAINER_NAME = "ultrasinger-bgutil-pot"
_PING_TIMEOUT = 2.0


@dataclass
class ProviderStatus:
    """Result of ensuring the PO-token provider is available."""

    running: bool                # a provider server responds on base_url
    started_by_us: bool = False  # we launched the Docker container
    base_url: str = DEFAULT_BASE_URL
    detail: str = ""             # human-readable status / setup hint

    @property
    def plugin_installed(self) -> bool:
        return _is_plugin_installed()


def _is_plugin_installed() -> bool:
    """True if the bgutil yt-dlp plugin is importable in this venv."""
    import importlib.util

    return importlib.util.find_spec("yt_dlp_plugins") is not None


def is_provider_running(base_url: str = DEFAULT_BASE_URL) -> bool:
    """Check whether a bgutil provider server responds on ``base_url``."""
    try:
        with urllib.request.urlopen(
            f"{base_url}/ping", timeout=_PING_TIMEOUT
        ) as resp:
            if resp.status != 200:
                return False
            # /ping returns JSON (server_uptime, version, ...); tolerate any
            try:
                json.loads(resp.read().decode("utf-8", "replace") or "{}")
            except ValueError:
                pass
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True, timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _start_docker_provider(base_url: str) -> bool:
    """Best-effort: launch the provider container.  Returns True on start."""
    # Remove a stale container of the same name first (ignore errors).
    try:
        subprocess.run(
            ["docker", "rm", "-f", _CONTAINER_NAME],
            capture_output=True, timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        pass

    port = base_url.rsplit(":", 1)[-1]
    try:
        r = subprocess.run(
            ["docker", "run", "-d", "--rm", "--name", _CONTAINER_NAME,
             "-p", f"{port}:4416", _DOCKER_IMAGE],
            capture_output=True, timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Failed to start bgutil provider via Docker: %s", e)
        return False
    if r.returncode != 0:
        logger.warning(
            "docker run for bgutil provider failed: %s",
            r.stderr.decode("utf-8", "replace").strip()[:300],
        )
        return False
    return True


def _wait_until_running(base_url: str, attempts: int = 15) -> bool:
    """Poll the provider until it responds (container needs a moment)."""
    import time

    for _ in range(attempts):
        if is_provider_running(base_url):
            return True
        time.sleep(1.0)
    return False


_SETUP_HINT = (
    "PO-token provider not available - YouTube downloads may be limited to "
    "360p / blocked (HTTP 403). Start a provider to restore full quality:\n"
    "  Docker: docker run -d --rm -p 4416:4416 "
    "brainicism/bgutil-ytdlp-pot-provider\n"
    "  or see https://github.com/Brainicism/bgutil-ytdlp-pot-provider "
    "for a Node.js setup."
)


def ensure_provider(
    base_url: str = DEFAULT_BASE_URL,
    auto_start_docker: bool = True,
) -> ProviderStatus:
    """Ensure a PO-token provider is reachable.

    Never raises.  If a server already runs, reports it.  Otherwise tries
    to start one via Docker (when available and ``auto_start_docker``),
    else returns a status carrying a setup hint.
    """
    if is_provider_running(base_url):
        return ProviderStatus(
            running=True, base_url=base_url,
            detail="PO-token provider is running - full-quality downloads enabled",
        )

    if auto_start_docker and _docker_available():
        logger.info("Starting bgutil PO-token provider via Docker ...")
        if _start_docker_provider(base_url) and _wait_until_running(base_url):
            return ProviderStatus(
                running=True, started_by_us=True, base_url=base_url,
                detail="Started PO-token provider (Docker) - full-quality "
                       "downloads enabled",
            )
        return ProviderStatus(
            running=False, base_url=base_url,
            detail="Could not start the PO-token provider via Docker. "
                   + _SETUP_HINT,
        )

    return ProviderStatus(running=False, base_url=base_url, detail=_SETUP_HINT)


def stop_provider_if_started(status: ProviderStatus) -> None:
    """Stop the Docker container if this process started it (cleanup)."""
    if not status.started_by_us:
        return
    try:
        subprocess.run(
            ["docker", "stop", _CONTAINER_NAME],
            capture_output=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        logger.info("Stopped bgutil PO-token provider container")
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Failed to stop bgutil provider container: %s", e)
