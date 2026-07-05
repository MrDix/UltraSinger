"""Tests for the bgutil PO-token provider manager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.gui import potoken_provider as pp


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


class TestEnsureProvider:
    def test_already_running_reported(self):
        with patch.object(pp, "is_provider_running", return_value=True):
            status = pp.ensure_provider()
        assert status.running is True
        assert status.started_by_us is False

    def test_prefers_node_when_available(self, tmp_path):
        entry = tmp_path / "main.js"
        entry.write_text("// stub", encoding="utf-8")
        proc = object()
        with patch.object(pp, "is_provider_running", side_effect=[False, True]), \
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
        with patch.object(pp, "is_provider_running", side_effect=[False, True]), \
             patch.object(pp, "_node_exe", return_value=None), \
             patch.object(pp, "_docker_exe", return_value="/usr/bin/docker"), \
             patch.object(pp, "_docker_available", return_value=True), \
             patch.object(pp, "_start_docker_provider", return_value="abc123"):
            status = pp.ensure_provider(node_entry=missing_entry)
        assert status.running is True
        assert status.container_id == "abc123"

    def test_no_provider_returns_setup_hint(self, tmp_path):
        with patch.object(pp, "is_provider_running", return_value=False), \
             patch.object(pp, "_node_exe", return_value=None), \
             patch.object(pp, "_docker_exe", return_value=None):
            status = pp.ensure_provider(node_entry=tmp_path / "nope.js")
        assert status.running is False
        assert "install script" in status.detail.lower()

    def test_cancel_before_start_skips_launch(self, tmp_path):
        import threading
        cancel = threading.Event()
        cancel.set()
        with patch.object(pp, "is_provider_running", return_value=False), \
             patch.object(pp, "_node_exe") as node_exe:
            status = pp.ensure_provider(
                node_entry=tmp_path / "nope.js", cancel=cancel
            )
        assert status.running is False
        node_exe.assert_not_called()

    def test_auto_start_disabled_skips_launch(self, tmp_path):
        with patch.object(pp, "is_provider_running", return_value=False), \
             patch.object(pp, "_node_exe") as node_exe, \
             patch.object(pp, "_docker_exe") as docker_exe:
            status = pp.ensure_provider(
                node_entry=tmp_path / "nope.js",
                auto_start_node=False, auto_start_docker=False,
            )
        assert status.running is False
        node_exe.assert_not_called()
        docker_exe.assert_not_called()


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
