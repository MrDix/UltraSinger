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
  * Docker:  docker run -d --rm -p 127.0.0.1:4416:4416 brainicism/bgutil-ytdlp-pot-provider:<version>
    with <version> = the version of the installed plugin.

The plugin refuses a provider of another major version, so a provider is
only used, and a local build only started, when its major version is the
plugin's.  This also keeps releases before 2.0.0, which listen on all network
interfaces (security advisory GHSA-qpv9-8xfj-xx9m), from being started.
"""

from __future__ import annotations

import json
import logging
import os
import re
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
# The provider release the install scripts build (PROVIDER_TAG in
# install/helpers/setup_potoken_provider.*, the plugin version in uv.lock).
# Stands in for the plugin's version when the plugin cannot be read.
_PINNED_PROVIDER_VERSION = "2.0.2"
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
_BOOTSTRAP_TIMEOUT = 900.0  # seconds; npm install on slow disks/AV takes minutes


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


def _plugin_version() -> str | None:
    """Version of the installed bgutil yt-dlp plugin, or None."""
    import importlib.metadata

    try:
        return importlib.metadata.version(_PLUGIN_DIST)
    except importlib.metadata.PackageNotFoundError:
        return None


def _expected_version() -> str:
    """The provider version the plugin works with (the pinned one without it)."""
    return _plugin_version() or _PINNED_PROVIDER_VERSION


def _same_major(version: str | None, expected: str) -> bool:
    """True if ``version`` has the major version of ``expected``.

    The plugin only accepts a provider of its own major version (and none
    that reports no version at all).
    """
    if not version:
        return False
    return version.strip().split(".", 1)[0] == expected.strip().split(".", 1)[0]


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


def _ping(normalized_url: str) -> str | None:
    """Ping the provider on an already normalized loopback URL.

    Returns None when no provider answers, else the version it reports
    ("" when it reports none).
    """
    try:
        # Proxy-free opener: this is a loopback URL, and behind a corporate
        # proxy (http_proxy set without a no_proxy exception) the default
        # opener would route the ping through the proxy and always fail.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(  # noqa: S310 — validated loopback URL
            f"{normalized_url}/ping", timeout=_PING_TIMEOUT
        ) as resp:
            if resp.status != 200:
                return None
            try:
                data = json.loads(resp.read().decode("utf-8", "replace") or "{}")
            except ValueError:
                return ""
            version = data.get("version") if isinstance(data, dict) else None
            return version if isinstance(version, str) else ""
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _confirmed_version(normalized_url: str, attempts: int = 3) -> str | None:
    """Version of a provider that has just answered a ping.

    A missing answer is asked again, so a slow moment (e.g. the load of a
    first start) does not count as a provider of the wrong version.
    """
    import time

    version = _ping(normalized_url)
    for _ in range(attempts - 1):
        if version is not None:
            break
        time.sleep(1.0)
        version = _ping(normalized_url)
    return version


def is_provider_running(base_url: str = DEFAULT_BASE_URL) -> bool:
    """Check whether a bgutil provider server responds on ``base_url``."""
    try:
        normalized, _ = _normalize_base_url(base_url)
    except ValueError as e:
        logger.warning("Invalid PO-token provider URL %r: %s", base_url, e)
        return False
    return _ping(normalized) is not None


def _node_exe() -> str | None:
    """Absolute path to the node executable, or None if not on PATH."""
    return shutil.which("node")


def _setup_script() -> Path:
    """The install helper that fetches and builds the Node provider."""
    name = (
        "setup_potoken_provider.bat" if os.name == "nt"
        else "setup_potoken_provider.sh"
    )
    return _REPO_ROOT / "install" / "helpers" / name


def _setup_log_path() -> Path:
    return _REPO_ROOT / ".potoken" / "setup.log"


_VERSION_IN_BUILD = re.compile(r'\bVERSION\s*=\s*"([^"]+)"')


def _build_version(entry: Path) -> str:
    """Version of the Node provider built at ``entry`` ("" if unknown).

    The server reports the VERSION constant of its compiled utils module;
    the release's server/package.json has the same version.
    """
    try:
        found = _VERSION_IN_BUILD.search(
            (entry.parent / "utils.js").read_text(encoding="utf-8", errors="replace"))
        if found:
            return found.group(1)
    except OSError:
        pass
    try:
        package = json.loads(
            (entry.parent.parent / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    version = package.get("version") if isinstance(package, dict) else None
    return version if isinstance(version, str) else ""


def _node_build_problem(entry: Path, expected: str) -> str:
    """Why the Node provider build at ``entry`` must not be started ("" if it may).

    Its major version must be the plugin's, and its dependencies must be
    installed (npm writes the hidden lockfile; security software sometimes
    deletes node_modules).
    """
    if not entry.is_file():
        return "is missing"
    version = _build_version(entry)
    if not _same_major(version, expected):
        return (f"is version {version}" if version else "has an unknown version") \
            + f", but the plugin is {expected}"
    if not (entry.parent.parent / "node_modules" / ".package-lock.json").is_file():
        return "is incomplete (its dependencies are missing)"
    return ""


def _bootstrap_node_provider(
    entry: Path,
    cancel: threading.Event | None = None,
    timeout: float = _BOOTSTRAP_TIMEOUT,
    on_progress=None,
) -> bool:
    """Build the Node provider via the repo's setup script.

    Covers installs where the provider setup could not finish — typically
    because Node.js was installed during that very run and was not yet
    visible to the same console — and builds that are outdated (another
    major version than the plugin) or incomplete.  Node.js is available
    *now* (the caller checked), so build the provider here instead of asking
    the user to re-run the installer.  The script's output goes to
    ``.potoken/setup.log``.  Returns True when the script succeeded and the
    server entry exists afterwards.  Never raises.
    """
    import time

    script = _setup_script()
    if not script.is_file() or shutil.which("git") is None:
        return False
    if cancel is not None and cancel.is_set():
        return False

    logger.info(
        "PO-token provider is missing or outdated - running %s (output in %s)",
        script.name, _setup_log_path(),
    )
    if on_progress is not None:
        on_progress(
            "PO-token provider is missing or outdated - downloading and "
            "building it now (this can take a few minutes)..."
        )
    # The GUI starts (and warms up) the server itself right afterwards.
    env = dict(os.environ, ULTRASINGER_POTOKEN_SKIP_WARMUP="1")
    if os.name == "nt":
        cmd = ["cmd.exe", "/d", "/c", str(script), "install\\update.bat"]
    else:
        cmd = ["bash", str(script), "./install/update.sh"]

    log_path = _setup_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
    except OSError:
        log_file = subprocess.DEVNULL
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT if log_file is not subprocess.DEVNULL
            else subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            cwd=str(_REPO_ROOT),
            env=env,
            creationflags=_CREATE_NO_WINDOW,
            # its own process group, so a cancel can stop git/npm/tsc too
            start_new_session=os.name != "nt",
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Failed to run the PO-token provider setup: %s", e)
        return False
    finally:
        if log_file is not subprocess.DEVNULL:
            log_file.close()

    deadline = time.monotonic() + timeout
    while proc.poll() is None:
        if (cancel is not None and cancel.is_set()) \
                or time.monotonic() >= deadline:
            _terminate_tree(proc)
            return False
        time.sleep(1.0)

    if proc.returncode == 0 and entry.is_file():
        logger.info("PO-token provider built successfully.")
        return True
    logger.warning(
        "PO-token provider setup did not complete (exit code %s) - see %s",
        proc.returncode, log_path,
    )
    return False


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


def _docker_image() -> str:
    """The provider image in the version of the installed plugin.

    The plugin refuses a provider of another major version, and an image
    without a version would be whatever release is the newest.
    """
    return f"{_DOCKER_IMAGE}:{_expected_version()}"


def _start_docker_provider(docker: str, port: int) -> str:
    """Best-effort: launch the provider container.

    Returns the container id on success, "" on failure.  Uses a
    process-unique container name and captures the id so cleanup only ever
    touches the container this process started.  The port is published on
    the loopback address only, so other machines cannot use the provider.
    """
    name = f"{_CONTAINER_PREFIX}-{os.getpid()}"
    try:
        r = subprocess.run(
            [docker, "run", "-d", "--rm", "--name", name,
             "-p", f"127.0.0.1:{port}:4416", _docker_image()],
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
    "Node.js provider (it needs Node.js 22 or newer from https://nodejs.org), "
    "or start one manually - see "
    "https://github.com/Brainicism/bgutil-ytdlp-pot-provider."
)
_LIMITED = " Until then video downloads may be limited to 360p / blocked."
_REBUILD = (
    " Run install\\update.bat (Windows) or install/update.sh (Linux/macOS) to "
    "rebuild it; this needs Node.js 22 or newer."
)


def _foreign_provider_detail(version: str, expected: str, url: str) -> str:
    """A provider that this app did not start answers with the wrong version."""
    return (
        f"A PO-token provider of version {version or 'unknown'} is running on "
        f"{url}, but the plugin is {expected} and only works with a provider "
        "of the same major version. Usually an earlier UltraSinger session "
        "left it running: end the node process (node.exe on Windows) that "
        "runs from .potoken/bgutil-ytdlp-pot-provider, or stop its Docker "
        f"container ({_CONTAINER_PREFIX}-... in 'docker ps'), then restart "
        "UltraSinger." + _LIMITED
    )


def ensure_provider(
    base_url: str = DEFAULT_BASE_URL,
    auto_start_node: bool = True,
    auto_start_docker: bool = True,
    node_entry: Path | None = None,
    cancel: threading.Event | None = None,
    on_progress=None,
) -> ProviderStatus:
    """Ensure a PO-token provider of the plugin's major version is reachable.

    Never raises.  Order: an already-running server → a local Node server
    (preferred, no Docker needed; built on the fly when the installer could
    not, or when the build is outdated or incomplete) → Docker → a setup
    hint.  A provider of another major version is never used, and a build of
    one never started.  ``cancel`` lets the GUI abort a pending start on
    shutdown; ``on_progress`` (a ``str`` callable) receives human-readable
    notes about long-running steps.
    """
    try:
        normalized, port = _normalize_base_url(base_url)
    except ValueError:
        normalized, port = DEFAULT_BASE_URL, _DEFAULT_PORT
    expected = _expected_version()

    running_version = _ping(normalized)
    if running_version is not None:
        if _same_major(running_version, expected):
            return ProviderStatus(
                running=True, base_url=normalized,
                detail="PO-token provider is running - full-quality downloads enabled",
            )
        # Not started by this app: it blocks the port, and the plugin refuses it.
        logger.warning(
            "PO-token provider %r on %s does not match plugin %s",
            running_version, normalized, expected,
        )
        return ProviderStatus(
            running=False, base_url=normalized,
            detail=_foreign_provider_detail(running_version, expected, normalized),
        )

    if cancel is not None and cancel.is_set():
        return ProviderStatus(running=False, base_url=normalized, detail=_SETUP_HINT)

    # Preferred: a local Node.js provider server (set up by the installer).
    entry = node_entry or _NODE_SERVER_ENTRY
    node = _node_exe() if auto_start_node else None
    node_detail = ""
    if node:
        problem = _node_build_problem(entry, expected)
        if problem:
            # The installer could not finish the provider setup (typically:
            # Node.js was installed during that very run but was not yet
            # visible to the same console), or the build is outdated or
            # incomplete.  Node.js is available now, so self-heal by building
            # the provider here instead of asking for a reinstall.
            _bootstrap_node_provider(entry, cancel=cancel, on_progress=on_progress)
            problem = _node_build_problem(entry, expected)
        if problem and entry.is_file():
            # Never start such a build: the plugin refuses another major
            # version, releases before 2.0.0 listen on all network interfaces,
            # and without its dependencies the server cannot run.
            logger.warning("Local PO-token provider build %s - not started", problem)
            node_detail = (
                f"The local PO-token provider build {problem}, so it is not "
                "started." + _REBUILD + _LIMITED
            )
        elif not problem:
            logger.info("Starting bgutil PO-token provider via Node ...")
            proc = _start_node_provider(node, entry)
            # Generous budget (~2 min): first launches on corporate machines are
            # routinely delayed by antivirus scanning of node_modules.
            if proc is not None and _wait_until_running(
                    normalized, attempts=120, cancel=cancel, proc=proc):
                version = _confirmed_version(normalized)
                if _same_major(version, expected):
                    return ProviderStatus(
                        running=True, started_by_us=True, base_url=normalized,
                        process=proc,
                        detail="Started PO-token provider (Node.js) - full-quality "
                               "downloads enabled",
                    )
                _terminate_process(proc)
                return ProviderStatus(
                    running=False, base_url=normalized,
                    detail=(
                        "The local PO-token provider reported version "
                        f"{version or 'unknown'}, but the plugin is {expected}, "
                        "so it was stopped." + _REBUILD + _LIMITED
                    ),
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
        return ProviderStatus(
            running=False, base_url=normalized, detail=node_detail or _SETUP_HINT)

    # Fallback: Docker, when available.
    docker = _docker_exe() if auto_start_docker else None
    if docker and _docker_available(docker):
        logger.info("Starting bgutil PO-token provider via Docker ...")
        container_id = _start_docker_provider(docker, port)
        if container_id and _wait_until_running(normalized, cancel=cancel):
            version = _confirmed_version(normalized)
            if _same_major(version, expected):
                return ProviderStatus(
                    running=True, started_by_us=True, base_url=normalized,
                    container_id=container_id,
                    detail="Started PO-token provider (Docker) - full-quality "
                           "downloads enabled",
                )
            _stop_container(docker, container_id)
            return ProviderStatus(
                running=False, base_url=normalized,
                detail=(
                    "The PO-token provider container reported version "
                    f"{version or 'unknown'}, but the plugin is {expected}, so "
                    "it was stopped - video downloads may be limited to 360p / "
                    "blocked."
                ),
            )
        if container_id:
            _stop_container(docker, container_id)

    return ProviderStatus(
        running=False, base_url=normalized, detail=node_detail or _SETUP_HINT)


def _terminate_tree(proc: subprocess.Popen) -> None:
    """Stop the setup script together with everything it started.

    Terminating only the script would leave git, npm or tsc running, still
    writing into the provider folder while the next start reads it.
    """
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True, timeout=30,
                creationflags=_CREATE_NO_WINDOW,
            )
        else:
            import signal

            os.killpg(proc.pid, signal.SIGTERM)  # started in its own session
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Failed to stop the PO-token provider setup: %s", e)
    _terminate_process(proc)


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
