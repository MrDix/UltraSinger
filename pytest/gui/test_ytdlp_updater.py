"""Tests for the yt-dlp update checker / one-click updater."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from src.gui import ytdlp_updater as up


class TestGetInstalledVersion:
    def test_returns_version_string(self):
        with patch("importlib.metadata.version", return_value="2026.6.9"):
            assert up.get_installed_version() == "2026.6.9"

    def test_returns_empty_when_not_installed(self):
        import importlib.metadata as md
        with patch(
            "importlib.metadata.version",
            side_effect=md.PackageNotFoundError("yt-dlp"),
        ):
            assert up.get_installed_version() == ""

    def test_fails_open_on_unexpected_error(self):
        with patch("importlib.metadata.version", side_effect=RuntimeError("boom")):
            assert up.get_installed_version() == ""


class TestGetLatestVersion:
    def test_parses_pypi_response(self):
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = b'{"info": {"version": "2026.7.1"}}'
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        with patch("urllib.request.urlopen", return_value=resp):
            assert up.get_latest_version() == "2026.7.1"

    def test_returns_empty_on_non_200(self):
        resp = MagicMock()
        resp.status = 503
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        with patch("urllib.request.urlopen", return_value=resp):
            assert up.get_latest_version() == ""

    def test_returns_empty_on_connection_error(self):
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            assert up.get_latest_version() == ""

    def test_returns_empty_on_malformed_json(self):
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = b"not json"
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        with patch("urllib.request.urlopen", return_value=resp):
            assert up.get_latest_version() == ""

    def test_no_real_network_call(self):
        # Guard against accidentally hitting the real network in CI.
        with patch("urllib.request.urlopen") as urlopen:
            up.get_latest_version()
            urlopen.assert_called_once()
            assert "pypi.org" in urlopen.call_args[0][0].full_url


class TestIsOutdated:
    def test_older_installed_is_outdated(self):
        assert up.is_outdated("2026.1.1", "2026.6.9") is True

    def test_same_version_not_outdated(self):
        assert up.is_outdated("2026.6.9", "2026.6.9") is False

    def test_newer_installed_not_outdated(self):
        assert up.is_outdated("2026.7.1", "2026.6.9") is False

    def test_empty_installed_not_outdated(self):
        assert up.is_outdated("", "2026.6.9") is False

    def test_empty_latest_not_outdated(self):
        assert up.is_outdated("2026.6.9", "") is False

    def test_both_empty_not_outdated(self):
        assert up.is_outdated("", "") is False

    def test_unparseable_versions_not_outdated(self):
        assert up.is_outdated("not-a-version", "also-not-a-version") is False


class TestRunUpdate:
    def test_ok_when_both_steps_succeed(self, tmp_path):
        lock_result = MagicMock(returncode=0, stdout="Resolved 42 packages\n", stderr="")
        sync_result = MagicMock(returncode=0, stdout="Installed 1 package\n", stderr="")
        with patch("shutil.which", return_value="/usr/bin/uv"), \
             patch("subprocess.run", side_effect=[lock_result, sync_result]) as run:
            ok, output = up.run_update(tmp_path)
        assert ok is True
        assert "successfully" in output.lower()
        assert run.call_count == 2
        first_cmd = run.call_args_list[0].args[0]
        second_cmd = run.call_args_list[1].args[0]
        assert first_cmd == ["uv", "lock", "--upgrade-package", "yt-dlp"]
        assert second_cmd == [
            "uv", "sync", "--extra", "gui", "--extra", "scoring", "--extra", "potoken",
        ]

    def test_fails_when_lock_step_fails(self, tmp_path):
        lock_result = MagicMock(returncode=1, stdout="", stderr="resolution failed")
        with patch("shutil.which", return_value="/usr/bin/uv"), \
             patch("subprocess.run", return_value=lock_result) as run:
            ok, output = up.run_update(tmp_path)
        assert ok is False
        assert "resolution failed" in output
        run.assert_called_once()  # sync step must not run after lock fails

    def test_fails_when_sync_step_fails(self, tmp_path):
        lock_result = MagicMock(returncode=0, stdout="ok\n", stderr="")
        sync_result = MagicMock(returncode=1, stdout="", stderr="network error")
        with patch("shutil.which", return_value="/usr/bin/uv"), \
             patch("subprocess.run", side_effect=[lock_result, sync_result]):
            ok, output = up.run_update(tmp_path)
        assert ok is False
        assert "network error" in output

    def test_fails_open_without_uv_on_path(self, tmp_path):
        with patch("shutil.which", return_value=None), \
             patch("subprocess.run") as run:
            ok, output = up.run_update(tmp_path)
        assert ok is False
        assert "uv" in output.lower()
        run.assert_not_called()

    def test_handles_timeout(self, tmp_path):
        with patch("shutil.which", return_value="/usr/bin/uv"), \
             patch(
                 "subprocess.run",
                 side_effect=subprocess.TimeoutExpired(cmd="uv lock", timeout=180),
             ):
            ok, output = up.run_update(tmp_path)
        assert ok is False
        assert "timed out" in output.lower()

    def test_handles_launch_failure(self, tmp_path):
        with patch("shutil.which", return_value="/usr/bin/uv"), \
             patch("subprocess.run", side_effect=OSError("not found")):
            ok, output = up.run_update(tmp_path)
        assert ok is False
        assert "failed to run" in output.lower()

    def test_no_real_subprocess_call_uses_mock(self, tmp_path):
        # Guard against accidentally shelling out to real uv in CI.
        lock_result = MagicMock(returncode=0, stdout="", stderr="")
        sync_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("shutil.which", return_value="/usr/bin/uv"), \
             patch("subprocess.run", side_effect=[lock_result, sync_result]) as run:
            up.run_update(tmp_path)
            for call in run.call_args_list:
                assert call.kwargs.get("cwd") == str(tmp_path)
