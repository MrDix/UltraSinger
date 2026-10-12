"""Tests for the bgutil PO-token provider manager."""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock, patch

import pytest

from src.gui import potoken_provider as pp


def _built_provider(tmp_path, version="2.0.2", dependencies=True):
    """A Node provider build as the install scripts leave it; returns its entry."""
    server = tmp_path / "server"
    (server / "build").mkdir(parents=True, exist_ok=True)
    entry = server / "build" / "main.js"
    entry.write_text("// built", encoding="utf-8")
    (server / "build" / "utils.js").write_text(
        f'export const VERSION = "{version}";\n', encoding="utf-8")
    if dependencies:
        (server / "node_modules").mkdir(exist_ok=True)
        (server / "node_modules" / ".package-lock.json").write_text("{}", encoding="utf-8")
    return entry


@pytest.fixture
def isolated():
    """No real ping of port 4416 and no real setup script; the plugin is 2.0.2."""
    with patch.object(pp, "_ping", return_value=None), \
         patch.object(pp, "_bootstrap_node_provider", return_value=False), \
         patch.object(pp, "_plugin_version", return_value="2.0.2"):
        yield


def _patched_opener(response=None, side_effect=None):
    """Patch the proxy-free opener the ping now uses (build_opener().open)."""
    opener = MagicMock()
    if side_effect is not None:
        opener.open.side_effect = side_effect
    else:
        opener.open.return_value = response
    return patch("urllib.request.build_opener", return_value=opener)


class TestIsProviderRunning:
    def test_returns_true_on_http_200(self):
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = b'{"version":"1.0"}'
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        with _patched_opener(response=resp):
            assert pp.is_provider_running("http://127.0.0.1:4416") is True

    def test_returns_false_on_non_200(self):
        resp = MagicMock()
        resp.status = 503
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        with _patched_opener(response=resp):
            assert pp.is_provider_running() is False

    def test_returns_false_on_connection_error(self):
        with _patched_opener(side_effect=OSError("refused")):
            assert pp.is_provider_running() is False

    def test_ping_uses_proxy_free_opener(self):
        """The loopback ping must bypass any configured proxy."""
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = b"{}"
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        opener = MagicMock()
        opener.open.return_value = resp
        with patch("urllib.request.build_opener",
                   return_value=opener) as bo:
            pp.is_provider_running()
            handler = bo.call_args[0][0]
            import urllib.request as ur
            assert isinstance(handler, ur.ProxyHandler)
            assert handler.proxies == {}


class TestNormalizeBaseUrl:
    def test_default(self):
        url, port = pp._normalize_base_url("http://127.0.0.1:4416")
        assert url == "http://127.0.0.1:4416"
        assert port == 4416

    def test_defaults_port_when_absent(self):
        url, port = pp._normalize_base_url("http://localhost")
        assert port == 4416
        assert url == "http://localhost:4416"

    def test_rejects_non_loopback(self):
        import pytest
        with pytest.raises(ValueError):
            pp._normalize_base_url("http://evil.example.com:4416")

    def test_rejects_non_http_scheme(self):
        import pytest
        with pytest.raises(ValueError):
            pp._normalize_base_url("file:///etc/passwd")


@pytest.mark.usefixtures("isolated")
class TestEnsureProvider:
    def test_already_running_reported(self):
        with patch.object(pp, "_ping", return_value="2.0.2"):
            status = pp.ensure_provider()
        assert status.running is True
        assert status.started_by_us is False

    def test_prefers_node_when_available(self, tmp_path):
        entry = _built_provider(tmp_path)
        proc = MagicMock()
        proc.poll.return_value = None  # still running
        with patch.object(pp, "_ping", side_effect=[None, "2.0.2"]), \
             patch.object(pp, "is_provider_running", return_value=True), \
             patch.object(pp, "_node_exe", return_value="/usr/bin/node"), \
             patch.object(pp, "_start_node_provider", return_value=proc), \
             patch.object(pp, "_docker_exe") as docker_exe:
            status = pp.ensure_provider(node_entry=entry)
        assert status.running is True
        assert status.started_by_us is True
        assert status.process is proc
        docker_exe.assert_not_called()

    def test_falls_back_to_docker_when_no_node(self, tmp_path):
        missing_entry = tmp_path / "nope.js"
        with patch.object(pp, "_ping", side_effect=[None, "2.0.2"]), \
             patch.object(pp, "is_provider_running", return_value=True), \
             patch.object(pp, "_node_exe", return_value=None), \
             patch.object(pp, "_docker_exe", return_value="/usr/bin/docker"), \
             patch.object(pp, "_docker_available", return_value=True), \
             patch.object(pp, "_start_docker_provider", return_value="abc123"):
            status = pp.ensure_provider(node_entry=missing_entry)
        assert status.running is True
        assert status.container_id == "abc123"

    def test_no_provider_returns_setup_hint(self, tmp_path):
        with patch.object(pp, "_node_exe", return_value=None), \
             patch.object(pp, "_docker_exe", return_value=None):
            status = pp.ensure_provider(node_entry=tmp_path / "nope.js")
        assert status.running is False
        assert "install script" in status.detail.lower()
        assert "Node.js 22 or newer" in status.detail

    def test_cancel_before_start_skips_launch(self, tmp_path):
        import threading
        cancel = threading.Event()
        cancel.set()
        with patch.object(pp, "_node_exe") as node_exe:
            status = pp.ensure_provider(
                node_entry=tmp_path / "nope.js", cancel=cancel
            )
        assert status.running is False
        node_exe.assert_not_called()

    def test_auto_start_disabled_skips_launch(self, tmp_path):
        with patch.object(pp, "_node_exe") as node_exe, \
             patch.object(pp, "_docker_exe") as docker_exe:
            status = pp.ensure_provider(
                node_entry=tmp_path / "nope.js",
                auto_start_node=False, auto_start_docker=False,
            )
        assert status.running is False
        node_exe.assert_not_called()
        docker_exe.assert_not_called()

    def test_node_crash_reports_log_hint(self, tmp_path):
        entry = _built_provider(tmp_path)
        proc = MagicMock()
        proc.poll.return_value = 1  # crashed immediately
        proc.returncode = 1
        with patch.object(pp, "is_provider_running", return_value=False), \
             patch.object(pp, "_node_exe", return_value="/usr/bin/node"), \
             patch.object(pp, "_start_node_provider", return_value=proc), \
             patch.object(pp, "_provider_log_tail", return_value="boom"), \
             patch.object(pp, "_docker_exe", return_value=None):
            status = pp.ensure_provider(node_entry=entry)
        assert status.running is False
        assert "crashed" in status.detail
        assert "provider.log" in status.detail

    def test_node_timeout_mentions_antivirus_hint(self, tmp_path):
        entry = _built_provider(tmp_path)
        proc = MagicMock()
        proc.poll.return_value = None  # alive but never ready
        with patch.object(pp, "is_provider_running", return_value=False), \
             patch.object(pp, "_node_exe", return_value="/usr/bin/node"), \
             patch.object(pp, "_start_node_provider", return_value=proc), \
             patch.object(pp, "_provider_log_tail", return_value=""), \
             patch.object(pp, "_docker_exe", return_value=None), \
             patch("time.sleep"):
            status = pp.ensure_provider(node_entry=entry)
        assert status.running is False
        assert "did not become ready" in status.detail
        assert "antivirus" in status.detail


@pytest.mark.usefixtures("isolated")
class TestProviderVersions:
    """The plugin refuses a provider of another major version: never use or start one."""

    def test_reuses_a_running_provider_of_the_plugins_major_version(self):
        with patch.object(pp, "_ping", return_value="2.0.1"):
            status = pp.ensure_provider()
        assert status.running is True

    def test_foreign_provider_of_another_major_version_is_not_used(self, tmp_path):
        with patch.object(pp, "_ping", return_value="1.3.1"), \
             patch.object(pp, "_node_exe") as node_exe:
            status = pp.ensure_provider(node_entry=_built_provider(tmp_path))
        assert status.running is False and status.started_by_us is False
        assert "1.3.1" in status.detail and "2.0.2" in status.detail
        assert "http://127.0.0.1:4416" in status.detail
        node_exe.assert_not_called()

    def test_provider_reporting_no_version_is_not_used(self):
        with patch.object(pp, "_ping", return_value=""):
            status = pp.ensure_provider()
        assert status.running is False
        assert "unknown" in status.detail

    def test_without_the_plugin_the_pinned_release_is_expected(self):
        with patch.object(pp, "_plugin_version", return_value=None), \
             patch.object(pp, "_ping", return_value="1.3.1"):
            status = pp.ensure_provider()
        assert status.running is False
        assert pp._PINNED_PROVIDER_VERSION in status.detail

    def test_outdated_node_build_is_never_started(self, tmp_path):
        """E.g. Node.js 18/20: the update cannot rebuild, the old 1.3.x build stays."""
        entry = _built_provider(tmp_path, version="1.3.1")
        with patch.object(pp, "_node_exe", return_value="/usr/bin/node"), \
             patch.object(pp, "_start_node_provider") as start, \
             patch.object(pp, "_docker_exe", return_value=None):
            status = pp.ensure_provider(node_entry=entry)
        start.assert_not_called()
        pp._bootstrap_node_provider.assert_called_once()  # a rebuild was tried
        assert status.running is False
        assert "1.3.1" in status.detail and "2.0.2" in status.detail
        assert "Node.js 22 or newer" in status.detail and "update" in status.detail

    def test_outdated_node_build_is_rebuilt_and_started(self, tmp_path):
        entry = _built_provider(tmp_path, version="1.3.1")

        def rebuild(e, cancel=None, on_progress=None, **kwargs):
            _built_provider(tmp_path, version="2.0.2")
            return True

        proc = MagicMock()
        proc.poll.return_value = None
        with patch.object(pp, "_bootstrap_node_provider", side_effect=rebuild), \
             patch.object(pp, "_ping", side_effect=[None, "2.0.2"]), \
             patch.object(pp, "is_provider_running", return_value=True), \
             patch.object(pp, "_node_exe", return_value="/usr/bin/node"), \
             patch.object(pp, "_start_node_provider", return_value=proc) as start:
            status = pp.ensure_provider(node_entry=entry)
        start.assert_called_once()
        assert status.running is True and status.started_by_us is True

    def test_outdated_node_build_falls_back_to_docker(self, tmp_path):
        entry = _built_provider(tmp_path, version="1.3.1")
        with patch.object(pp, "_ping", side_effect=[None, "2.0.2"]), \
             patch.object(pp, "is_provider_running", return_value=True), \
             patch.object(pp, "_node_exe", return_value="/usr/bin/node"), \
             patch.object(pp, "_start_node_provider") as start, \
             patch.object(pp, "_docker_exe", return_value="/usr/bin/docker"), \
             patch.object(pp, "_docker_available", return_value=True), \
             patch.object(pp, "_start_docker_provider", return_value="abc123"):
            status = pp.ensure_provider(node_entry=entry)
        start.assert_not_called()
        assert status.running is True and status.container_id == "abc123"

    def test_node_build_without_dependencies_is_not_started(self, tmp_path):
        entry = _built_provider(tmp_path, dependencies=False)
        with patch.object(pp, "_node_exe", return_value="/usr/bin/node"), \
             patch.object(pp, "_start_node_provider") as start, \
             patch.object(pp, "_docker_exe", return_value=None):
            status = pp.ensure_provider(node_entry=entry)
        start.assert_not_called()
        assert status.running is False
        assert "dependencies are missing" in status.detail

    def test_started_node_provider_of_another_major_is_stopped(self, tmp_path):
        entry = _built_provider(tmp_path)
        proc = MagicMock()
        proc.poll.return_value = None
        with patch.object(pp, "_ping", side_effect=[None, "1.3.1"]), \
             patch.object(pp, "is_provider_running", return_value=True), \
             patch.object(pp, "_node_exe", return_value="/usr/bin/node"), \
             patch.object(pp, "_start_node_provider", return_value=proc):
            status = pp.ensure_provider(node_entry=entry)
        proc.terminate.assert_called_once()
        assert status.running is False and status.process is None
        assert "1.3.1" in status.detail

    def test_started_docker_provider_of_another_major_is_stopped(self, tmp_path):
        with patch.object(pp, "_ping", side_effect=[None, "3.0.0"]), \
             patch.object(pp, "is_provider_running", return_value=True), \
             patch.object(pp, "_node_exe", return_value=None), \
             patch.object(pp, "_docker_exe", return_value="/usr/bin/docker"), \
             patch.object(pp, "_docker_available", return_value=True), \
             patch.object(pp, "_start_docker_provider", return_value="abc123"), \
             patch.object(pp, "_stop_container") as stop:
            status = pp.ensure_provider(node_entry=tmp_path / "nope.js")
        stop.assert_called_once_with("/usr/bin/docker", "abc123")
        assert status.running is False and status.container_id == ""
        assert "3.0.0" in status.detail


class TestVersionHelpers:
    def _ping_answer(self, body):
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = body
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        with _patched_opener(response=resp):
            return pp._ping("http://127.0.0.1:4416")

    def test_ping_returns_the_reported_version(self):
        assert self._ping_answer(b'{"server_uptime": 4.5, "version": "2.0.2"}') == "2.0.2"
        assert self._ping_answer(b'{"server_uptime": 4.5}') == ""
        assert self._ping_answer(b"not json") == ""
        with _patched_opener(side_effect=OSError("refused")):
            assert pp._ping("http://127.0.0.1:4416") is None

    def test_build_version_from_the_compiled_server(self, tmp_path):
        entry = _built_provider(tmp_path, version="2.0.2")
        assert pp._build_version(entry) == "2.0.2"

    def test_build_version_falls_back_to_package_json(self, tmp_path):
        entry = _built_provider(tmp_path)
        (entry.parent / "utils.js").unlink()
        (entry.parent.parent / "package.json").write_text('{"version": "1.3.1"}', encoding="utf-8")
        assert pp._build_version(entry) == "1.3.1"
        (entry.parent.parent / "package.json").unlink()
        assert pp._build_version(entry) == ""

    def test_same_major(self):
        assert pp._same_major("2.0.1", "2.0.2")
        assert not pp._same_major("1.3.1", "2.0.2")
        assert not pp._same_major("", "2.0.2")
        assert not pp._same_major(None, "2.0.2")


class TestBootstrapNodeProvider:
    """First-launch self-heal: build the provider when the installer could
    not (e.g. Node.js only became visible after the install run)."""

    def _completed_proc(self, returncode=0):
        proc = MagicMock()
        proc.poll.return_value = returncode
        proc.returncode = returncode
        return proc

    def test_builds_provider_and_reports_success(self, tmp_path):
        entry = tmp_path / "main.js"
        script = tmp_path / "setup.bat"
        script.write_text("rem stub", encoding="utf-8")
        entry.write_text("// built", encoding="utf-8")
        progress = MagicMock()
        with patch.object(pp, "_setup_script", return_value=script), \
             patch.object(pp, "_setup_log_path",
                          return_value=tmp_path / "setup.log"), \
             patch("shutil.which", return_value="/usr/bin/git"), \
             patch("subprocess.Popen",
                   return_value=self._completed_proc(0)) as popen:
            ok = pp._bootstrap_node_provider(entry, on_progress=progress)
        assert ok is True
        progress.assert_called_once()
        # The GUI manages the server itself - the script must skip warm-up.
        env = popen.call_args.kwargs["env"]
        assert env["ULTRASINGER_POTOKEN_SKIP_WARMUP"] == "1"

    def test_returns_false_when_script_missing(self, tmp_path):
        with patch.object(pp, "_setup_script",
                          return_value=tmp_path / "missing.bat"), \
             patch("subprocess.Popen") as popen:
            ok = pp._bootstrap_node_provider(tmp_path / "main.js")
        assert ok is False
        popen.assert_not_called()

    def test_returns_false_when_git_missing(self, tmp_path):
        script = tmp_path / "setup.bat"
        script.write_text("rem stub", encoding="utf-8")
        with patch.object(pp, "_setup_script", return_value=script), \
             patch("shutil.which", return_value=None), \
             patch("subprocess.Popen") as popen:
            ok = pp._bootstrap_node_provider(tmp_path / "main.js")
        assert ok is False
        popen.assert_not_called()

    def test_returns_false_when_entry_still_missing(self, tmp_path):
        script = tmp_path / "setup.bat"
        script.write_text("rem stub", encoding="utf-8")
        with patch.object(pp, "_setup_script", return_value=script), \
             patch.object(pp, "_setup_log_path",
                          return_value=tmp_path / "setup.log"), \
             patch("shutil.which", return_value="/usr/bin/git"), \
             patch("subprocess.Popen",
                   return_value=self._completed_proc(0)):
            ok = pp._bootstrap_node_provider(tmp_path / "never_built.js")
        assert ok is False

    def test_cancel_before_launch_skips(self, tmp_path):
        import threading
        script = tmp_path / "setup.bat"
        script.write_text("rem stub", encoding="utf-8")
        cancel = threading.Event()
        cancel.set()
        with patch.object(pp, "_setup_script", return_value=script), \
             patch("shutil.which", return_value="/usr/bin/git"), \
             patch("subprocess.Popen") as popen:
            ok = pp._bootstrap_node_provider(tmp_path / "main.js",
                                             cancel=cancel)
        assert ok is False
        popen.assert_not_called()

    def test_ensure_provider_bootstraps_when_entry_missing(self, tmp_path):
        """Node available but no built provider: self-heal instead of hint."""
        entry = tmp_path / "server" / "build" / "main.js"

        def fake_bootstrap(e, cancel=None, on_progress=None, **kwargs):
            _built_provider(tmp_path)
            return True

        proc = MagicMock()
        proc.poll.return_value = None
        with patch.object(pp, "_plugin_version", return_value="2.0.2"), \
             patch.object(pp, "_ping", side_effect=[None, "2.0.2"]), \
             patch.object(pp, "is_provider_running", return_value=True), \
             patch.object(pp, "_node_exe", return_value="/usr/bin/node"), \
             patch.object(pp, "_bootstrap_node_provider",
                          side_effect=fake_bootstrap) as bootstrap, \
             patch.object(pp, "_start_node_provider", return_value=proc), \
             patch.object(pp, "_docker_exe") as docker_exe:
            status = pp.ensure_provider(node_entry=entry)
        assert status.running is True
        assert status.started_by_us is True
        bootstrap.assert_called_once()
        docker_exe.assert_not_called()

    def test_ensure_provider_no_bootstrap_when_entry_exists(self, tmp_path):
        entry = _built_provider(tmp_path)
        proc = MagicMock()
        proc.poll.return_value = None
        with patch.object(pp, "_plugin_version", return_value="2.0.2"), \
             patch.object(pp, "_ping", side_effect=[None, "2.0.2"]), \
             patch.object(pp, "is_provider_running", return_value=True), \
             patch.object(pp, "_node_exe", return_value="/usr/bin/node"), \
             patch.object(pp, "_bootstrap_node_provider") as bootstrap, \
             patch.object(pp, "_start_node_provider", return_value=proc):
            status = pp.ensure_provider(node_entry=entry)
        assert status.running is True
        bootstrap.assert_not_called()

    def test_ensure_provider_no_bootstrap_without_node(self, tmp_path):
        with patch.object(pp, "_plugin_version", return_value="2.0.2"), \
             patch.object(pp, "_ping", return_value=None), \
             patch.object(pp, "_node_exe", return_value=None), \
             patch.object(pp, "_bootstrap_node_provider") as bootstrap, \
             patch.object(pp, "_docker_exe", return_value=None):
            status = pp.ensure_provider(node_entry=tmp_path / "nope.js")
        assert status.running is False
        bootstrap.assert_not_called()


class TestStopProvider:
    def test_terminates_node_process(self):
        proc = MagicMock()
        status = pp.ProviderStatus(running=True, started_by_us=True, process=proc)
        pp.stop_provider_if_started(status)
        proc.terminate.assert_called_once()

    def test_stops_docker_container_by_id(self):
        status = pp.ProviderStatus(
            running=True, started_by_us=True, container_id="abc123"
        )
        with patch.object(pp, "_docker_exe", return_value="/usr/bin/docker"), \
             patch("subprocess.run") as run:
            pp.stop_provider_if_started(status)
        run.assert_called_once()
        assert "abc123" in run.call_args[0][0]

    def test_noop_when_not_started_by_us(self):
        status = pp.ProviderStatus(
            running=True, started_by_us=False, container_id="abc123"
        )
        with patch("subprocess.run") as run:
            pp.stop_provider_if_started(status)
        run.assert_not_called()

    def test_noop_when_nothing_to_stop(self):
        status = pp.ProviderStatus(running=False, started_by_us=True)
        with patch("subprocess.run") as run:
            pp.stop_provider_if_started(status)
        run.assert_not_called()


class TestStartDockerProvider:
    def _docker_command(self, version):
        result = MagicMock(returncode=0, stdout=b"abc123\n", stderr=b"")
        with patch("subprocess.run", return_value=result) as run, \
             patch("importlib.metadata.version", version):
            assert pp._start_docker_provider("/usr/bin/docker", 4416) == "abc123"
        return run.call_args[0][0]

    def test_port_is_published_on_loopback_only(self):
        cmd = self._docker_command(MagicMock(return_value="2.0.2"))
        assert cmd[cmd.index("-p") + 1] == "127.0.0.1:4416:4416"

    def test_image_has_the_plugin_version(self):
        """The plugin refuses a provider of another major version."""
        image = "brainicism/bgutil-ytdlp-pot-provider"
        assert self._docker_command(MagicMock(return_value="2.0.2"))[-1] == f"{image}:2.0.2"
        missing = MagicMock(side_effect=importlib.metadata.PackageNotFoundError)
        assert self._docker_command(missing)[-1] == f"{image}:{pp._PINNED_PROVIDER_VERSION}"
