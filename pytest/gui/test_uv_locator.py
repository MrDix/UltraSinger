"""Tests for finding the uv executable (the installed uv before other copies on PATH)."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

from src.gui import uv_locator

UV_NAME = "uv.exe" if os.name == "nt" else "uv"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A home folder without uv (Path.home() reads USERPROFILE on Windows, HOME elsewhere)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def _install_uv(home):
    uv = home / ".local" / "bin" / UV_NAME
    uv.parent.mkdir(parents=True)
    uv.write_text("", encoding="utf-8")
    return uv


class TestFindUv:
    def test_prefers_the_installed_uv_over_an_older_one_on_path(self, home):
        installed = _install_uv(home)
        with patch("shutil.which", return_value="/python/Scripts/uv"):
            assert uv_locator.find_uv() == str(installed)

    def test_uses_path_without_an_installed_uv(self, home):
        with patch("shutil.which", return_value="/usr/bin/uv"):
            assert uv_locator.find_uv() == "/usr/bin/uv"

    def test_none_without_any_uv(self, home):
        with patch("shutil.which", return_value=None):
            assert uv_locator.find_uv() is None

    def test_ignores_a_folder_named_like_uv(self, home):
        (home / ".local" / "bin" / UV_NAME).mkdir(parents=True)
        with patch("shutil.which", return_value="/usr/bin/uv"):
            assert uv_locator.find_uv() == "/usr/bin/uv"


class TestConversionCommand:
    @pytest.fixture
    def runner(self):
        pytest.importorskip("PySide6.QtCore", exc_type=ImportError)
        from src.gui import ultrasinger_runner
        return ultrasinger_runner

    def test_runs_the_conversion_with_the_found_uv(self, runner, tmp_path):
        with patch.object(runner, "find_uv", return_value="/home/me/.local/bin/uv"):
            cmd = runner._build_command(tmp_path)
        assert cmd == ["/home/me/.local/bin/uv", "run", "python", str(tmp_path / "src" / "UltraSinger.py")]

    def test_runs_the_conversion_with_this_python_without_uv(self, runner, tmp_path):
        with patch.object(runner, "find_uv", return_value=None):
            cmd = runner._build_command(tmp_path)
        assert cmd == [sys.executable, str(tmp_path / "src" / "UltraSinger.py")]
