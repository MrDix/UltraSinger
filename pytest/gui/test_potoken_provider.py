"""Tests for the bgutil PO-token provider manager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.gui import potoken_provider as pp


class TestIsProviderRunning:
    def test_returns_true_on_http_200(self):
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = b'{"version":"1.0"}'
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        with patch("urllib.request.urlopen", return_value=resp):
            assert pp.is_provider_running("http://127.0.0.1:4416") is True

    def test_returns_false_on_non_200(self):
        resp = MagicMock()
        resp.status = 503
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        with patch("urllib.request.urlopen", return_value=resp):
            assert pp.is_provider_running() is False

    def test_returns_false_on_connection_error(self):
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            assert pp.is_provider_running() is False


class TestEnsureProvider:
    def test_already_running_reported(self):
        with patch.object(pp, "is_provider_running", return_value=True):
            status = pp.ensure_provider()
        assert status.running is True
        assert status.started_by_us is False

    def test_starts_via_docker_when_available(self):
        with patch.object(pp, "is_provider_running", side_effect=[False, True]), \
             patch.object(pp, "_docker_available", return_value=True), \
             patch.object(pp, "_start_docker_provider", return_value=True), \
             patch.object(pp, "_wait_until_running", return_value=True):
            status = pp.ensure_provider(auto_start_docker=True)
        assert status.running is True
        assert status.started_by_us is True

    def test_docker_start_failure_returns_hint(self):
        with patch.object(pp, "is_provider_running", return_value=False), \
             patch.object(pp, "_docker_available", return_value=True), \
             patch.object(pp, "_start_docker_provider", return_value=False):
            status = pp.ensure_provider(auto_start_docker=True)
        assert status.running is False
        assert "provider" in status.detail.lower()

    def test_no_docker_returns_setup_hint(self):
        with patch.object(pp, "is_provider_running", return_value=False), \
             patch.object(pp, "_docker_available", return_value=False):
            status = pp.ensure_provider(auto_start_docker=True)
        assert status.running is False
        assert "docker run" in status.detail

    def test_auto_start_disabled_skips_docker(self):
        with patch.object(pp, "is_provider_running", return_value=False), \
             patch.object(pp, "_docker_available") as docker_check:
            status = pp.ensure_provider(auto_start_docker=False)
        assert status.running is False
        docker_check.assert_not_called()


class TestStopProvider:
    def test_stops_only_when_started_by_us(self):
        status = pp.ProviderStatus(running=True, started_by_us=True)
        with patch("subprocess.run") as run:
            pp.stop_provider_if_started(status)
        run.assert_called_once()

    def test_noop_when_not_started_by_us(self):
        status = pp.ProviderStatus(running=True, started_by_us=False)
        with patch("subprocess.run") as run:
            pp.stop_provider_if_started(status)
        run.assert_not_called()
