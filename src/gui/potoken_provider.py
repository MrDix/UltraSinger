"""Manage a bgutil PO-token provider for yt-dlp full-quality downloads.

The video platform's SABR delivery no longer exposes a capturable media URL or PO
token to the embedded browser (the token lives in the binary POST body,
which Qt's request interceptor cannot read).  The maintained solution is
the ``bgutil-ytdlp-pot-provider`` yt-dlp plugin, which fetches GVS
Proof-of-Origin tokens from a small local provider server.  With the
plugin installed (same venv as yt-dlp) and the server reachable, yt-dlp
transparently obtains PO tokens and restores the full format list.

This module only manages the *server*: it checks whether one is already
running and, when possible, starts one.  A local **Node.js** server is
preferred (set up by the install scripts, no Docker needed); Docker is a
fallback when its image is available.  It never blocks the GUI and fails
open — if no provider is available the app keeps working with the
previous (degraded) behaviour.

Provider server options (the plugin auto-detects the default port 4416):
  * Node.js: the install scripts build the provider under ``.potoken/``;
    the GUI launches ``node <server>/build/main.js`` on startup.
  * Docker:  docker run -d --rm -p 4416:4416 brainicism/bgutil-ytdlp-pot-provider
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:4416"
_DEFAULT_PORT = 4416
_DOCKER_IMAGE = "brainicism/bgutil-ytdlp-pot-provider"
_CONTAINER_PREFIX = "ultrasinger-bgutil-pot"
_PING_TIMEOUT = 2.0
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
# Concrete plugin module shipped by bgutil-ytdlp-pot-provider (checking the
# shared ``yt_dlp_plugins`` namespace would match any unrelated yt-dlp plugin)
_PLUGIN_MODULE = "yt_dlp_plugins.extractor.getpot_bgutil"
_PLUGIN_DIST = "bgutil-ytdlp-pot-provider"

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Location the install scripts build the Node provider into (repo-relative).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_NODE_SERVER_ENTRY = (
    _REPO_ROOT / ".potoken" / "bgutil-ytdlp-pot-provider"
    / "server" / "build" / "main.js"
)


@dataclass
class ProviderStatus:
    """Result of ensuring the PO-token provider is available."""

    running: bool                 # a provider server responds on base_url
    started_by_us: bool = False   # we launched the server
    base_url: str = DEFAULT_BASE_URL
    container_id: str = ""        # id of the Docker container we started
    process: object = field(default=None, repr=False)  # Popen of a Node server
    detail: str = ""              # human-readable status / setup hint

    @property
    def plugin_installed(self) -> bool:
        return _is_plugin_installed()


def _is_plugin_installed() -> bool:
    """True if the concrete bgutil yt-dlp plugin is available in this venv."""
    import importlib.metadata
    import importlib.util

    try:
        if importlib.util.find_spec(_PLUGIN_MODULE) is not None:
            return True
    except (ImportError, ValueError):
        pass
    # Fallback: the distribution is installed even if the namespace import
    # cannot be resolved standalone.
    try:
        importlib.metadata.distribution(_PLUGIN_DIST)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _normalize_base_url(base_url: str) -> tuple[str, int]:
    """Validate and normalize ``base_url``.

    Returns ``(normalized_url, port)``.  Only HTTP(S) loopback URLs are
    accepted (the provider is always local); anything else raises
    ``ValueError`` so callers fall back to the default.
    """
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if host not in _LOOPBACK_HOSTS:
        raise ValueError(f"non-loopback host: {host!r}")
    port = parsed.port or _DEFAULT_PORT
    return f"{parsed.scheme}://{host}:{port}", port


def is_provider_running(base_url: str = DEFAULT_BASE_URL) -> bool:
    """Check whether a bgutil provider server responds on ``base_url``."""
    try:
        normalized, _ = _normalize_base_url(base_url)
    except ValueError as e:
        logger.warning("Invalid PO-token provider URL %r: %s", base_url, e)
        return False
    try:
        # Proxy-free opener: this is a loopback URL, and behind a corporate
        # proxy (http_proxy set without a no_proxy exception) the default
        # opener would route the ping through the proxy and always fail.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(  # noqa: S310 — validated loopback URL
            f"{normalized}/ping", timeout=_PING_TIMEOUT
        ) as resp:
            if resp.status != 200:
                return False
            try:
                json.loads(resp.read().decode("utf-8", "replace") or "{}")
            except ValueError:
                pass
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _node_exe() -> str | None:
    """Absolute path to the node executable, or None if not on PATH."""
    return shutil.which("node")


def _provider_log_path() -> Path:
    return _REPO_ROOT / ".potoken" / "provider.log"


def _start_node_provider(node: str, entry: Path) -> subprocess.Popen | None:
    """Launch the local Node provider server.  Returns the Popen or None.

    The server's output goes to ``.potoken/provider.log`` (overwritten per
    start) so a failing launch is diagnosable instead of vanishing into
    DEVNULL — essential on corporate machines where e.g. antivirus or
    policy tooling can break or massively delay the first start.
    """
    log_path = _provider_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
    except OSError:
        log_file = subprocess.DEVNULL
    try:
        proc = subprocess.Popen(
            [node, str(entry)],
            stdout=log_file,
            stderr=subprocess.STDOUT if log_file is not subprocess.DEVNULL
            else subprocess.DEVNULL,
            cwd=str(entry.parent),
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Failed to start bgutil provider via Node: %s", e)
        return None
    finally:
        # Popen duplicated the handle; our Python-side object can close.
        if log_file is not subprocess.DEVNULL:
            log_file.close()
    return proc


def _provider_log_tail(lines: int = 10) -> str:
    """Last lines of the provider log, for failure diagnostics ('' if none)."""
    try:
        text = _provider_log_path().read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.strip().splitlines()[-lines:])


def _docker_exe() -> str | None:
    """Absolute path to the docker executable, or None if not on PATH."""
    return shutil.which("docker")


def _docker_available(docker: str) -> bool:
    try:
        r = subprocess.run(
            [docker, "info"],
            capture_output=True, timeout=8,
            creationflags=_CREATE_NO_WINDOW,
        )
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _start_docker_provider(docker: str, port: int) -> str:
    """Best-effort: launch the provider container.

    Returns the container id on success, "" on failure.  Uses a
    process-unique container name and captures the id so cleanup only ever
    touches the container this process started.
    """
    name = f"{_CONTAINER_PREFIX}-{os.getpid()}"
    try:
        r = subprocess.run(
            [docker, "run", "-d", "--rm", "--name", name,
             "-p", f"{port}:4416", _DOCKER_IMAGE],
            capture_output=True, timeout=60,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Failed to start bgutil provider via Docker: %s", e)
        return ""
    if r.returncode != 0:
        logger.warning(
            "docker run for bgutil provider failed: %s",
            r.stderr.decode("utf-8", "replace").strip()[:300],
        )
        return ""
    return r.stdout.decode("utf-8", "replace").strip()


def _wait_until_running(
    base_url: str, attempts: int = 15,
    cancel: threading.Event | None = None,
    proc: subprocess.Popen | None = None,
) -> bool:
    """Poll the provider until it responds.

    When *proc* is given, a died process aborts the wait immediately
    (crash) instead of polling into the void. The attempt budget must be
    generous for first launches: on corporate machines, antivirus scanning
    of the freshly built node_modules can delay server startup far beyond
    a minute (observed in the field).
    """
    import time

    for _ in range(attempts):
        if cancel is not None and cancel.is_set():
            return False
        if proc is not None and proc.poll() is not None:
            logger.warning(
                "PO-token provider process exited early (code %s)",
                proc.returncode,
            )
            return False
        if is_provider_running(base_url):
            return True
        time.sleep(1.0)
    return False


_SETUP_HINT = (
    "PO-token provider not available - video downloads may be limited to "
    "360p / blocked (HTTP 403). Re-run the install script to set up the "
    "Node.js provider (it needs Node.js from https://nodejs.org installed), "
    "or start one manually - see "
    "https://github.com/Brainicism/bgutil-ytdlp-pot-provider."
)


def ensure_provider(
    base_url: str = DEFAULT_BASE_URL,
    auto_start_node: bool = True,
    auto_start_docker: bool = True,
    node_entry: Path | None = None,
    cancel: threading.Event | None = None,
) -> ProviderStatus:
    """Ensure a PO-token provider is reachable.

    Never raises.  Order: an already-running server → a local Node server
    (preferred, no Docker needed) → Docker → a setup hint.  ``cancel``
    lets the GUI abort a pending start on shutdown.
    """
    try:
        normalized, port = _normalize_base_url(base_url)
    except ValueError:
        normalized, port = DEFAULT_BASE_URL, _DEFAULT_PORT

    if is_provider_running(normalized):
        return ProviderStatus(
            running=True, base_url=normalized,
            detail="PO-token provider is running - full-quality downloads enabled",
        )

    if cancel is not None and cancel.is_set():
        return ProviderStatus(running=False, base_url=normalized, detail=_SETUP_HINT)

    # Preferred: a local Node.js provider server (set up by the installer).
    entry = node_entry or _NODE_SERVER_ENTRY
    node = _node_exe() if auto_start_node else None
    if node and entry.is_file():
        logger.info("Starting bgutil PO-token provider via Node ...")
        proc = _start_node_provider(node, entry)
        # Generous budget (~2 min): first launches on corporate machines are
        # routinely delayed by antivirus scanning of node_modules.
        if proc is not None and _wait_until_running(
                normalized, attempts=120, cancel=cancel, proc=proc):
            return ProviderStatus(
                running=True, started_by_us=True, base_url=normalized,
                process=proc,
                detail="Started PO-token provider (Node.js) - full-quality "
                       "downloads enabled",
            )
        if proc is not None:
            crashed = proc.poll() is not None
            _terminate_process(proc)
            tail = _provider_log_tail()
            if tail:
                logger.warning("PO-token provider output (tail):\n%s", tail)
            return ProviderStatus(
                running=False, base_url=normalized,
                detail=(
                    ("PO-token provider crashed on start"
                     if crashed else
                     "PO-token provider started but did not become ready "
                     "within 2 minutes")
                    + " - video downloads may be limited to 360p / blocked. "
                    f"See {_provider_log_path()} for the server output. "
                    "On a first launch, antivirus scanning can delay startup: "
                    "restarting the app often succeeds."
                ),
            )

    if cancel is not None and cancel.is_set():
        return ProviderStatus(running=False, base_url=normalized, detail=_SETUP_HINT)

    # Fallback: Docker, when available.
    docker = _docker_exe() if auto_start_docker else None
    if docker and _docker_available(docker):
        logger.info("Starting bgutil PO-token provider via Docker ...")
        container_id = _start_docker_provider(docker, port)
        if container_id and _wait_until_running(normalized, cancel=cancel):
            return ProviderStatus(
                running=True, started_by_us=True, base_url=normalized,
                container_id=container_id,
                detail="Started PO-token provider (Docker) - full-quality "
                       "downloads enabled",
            )
        if container_id:
            _stop_container(docker, container_id)

    return ProviderStatus(running=False, base_url=normalized, detail=_SETUP_HINT)


def _terminate_process(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        logger.info("Stopped bgutil PO-token provider (Node.js)")
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Failed to stop Node provider process: %s", e)


def _stop_container(docker: str, container_id: str) -> None:
    try:
        subprocess.run(
            [docker, "stop", container_id],
            capture_output=True, timeout=20,
            creationflags=_CREATE_NO_WINDOW,
        )
        logger.info("Stopped bgutil PO-token provider container")
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Failed to stop bgutil provider container: %s", e)


def stop_provider_if_started(status: ProviderStatus | None) -> None:
    """Stop the provider this process started (Node process or container)."""
    if status is None or not status.started_by_us:
        return
    if status.process is not None:
        _terminate_process(status.process)
        return
    if status.container_id:
        docker = _docker_exe()
        if docker:
            _stop_container(docker, status.container_id)
