"""Looking for a newer UltraSinger version and starting the update script."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from src.gui import app_updater as up

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(cwd: Path, *args: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=cwd, env=env, check=True,
                          capture_output=True, text=True).stdout


def _commit(repo: Path, subject: str) -> None:
    (repo / "file.txt").write_text(subject, encoding="utf-8")
    _git(repo, "add", "file.txt")
    _git(repo, "commit", "-qm", subject)


@pytest.fixture
def checkout(tmp_path):
    """A checkout with install/update scripts, cloned from a remote; returns (checkout, upstream clone)."""
    remote, seed, app = tmp_path / "remote.git", tmp_path / "seed", tmp_path / "app"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(tmp_path, "clone", "-q", str(remote), str(seed))
    _git(seed, "symbolic-ref", "HEAD", "refs/heads/main")
    (seed / "install").mkdir()
    for name in ("update.bat", "update.sh"):
        (seed / "install" / name).write_text("echo\n", encoding="utf-8")
    _git(seed, "add", "install")
    _commit(seed, "First version")
    _git(seed, "push", "-q", "origin", "main")
    _git(tmp_path, "clone", "-q", str(remote), str(app))
    return app, seed


def _push(seed: Path, *subjects: str) -> None:
    for subject in subjects:
        _commit(seed, subject)
    _git(seed, "push", "-q", "origin", "main")


@needs_git
class TestCheckForUpdate:
    def test_up_to_date(self, checkout):
        app, _ = checkout
        assert up.check_for_update(app) is None

    def test_new_commits_upstream(self, checkout):
        app, seed = checkout
        _push(seed, "Second version", "Third version")
        info = up.check_for_update(app)
        assert info is not None
        assert info.upstream == "origin/main"
        assert info.count == 2
        assert info.subjects == ["Third version", "Second version"]
        assert info.upstream_commit == _git(seed, "rev-parse", "HEAD").strip()
        # Only fetched: the checkout itself is unchanged
        assert _git(app, "log", "-1", "--format=%s").strip() == "First version"

    def test_lists_at_most_the_newest_subjects(self, checkout, monkeypatch):
        app, seed = checkout
        monkeypatch.setattr(up, "MAX_SUBJECTS", 2)
        _push(seed, "A", "B", "C")
        info = up.check_for_update(app)
        assert (info.count, info.subjects) == (3, ["C", "B"])

    def test_no_update_for_a_checkout_with_commits_of_its_own(self, checkout):
        """The update script only fast-forwards."""
        app, seed = checkout
        _push(seed, "Upstream change")
        _commit(app, "Local change")
        assert up.check_for_update(app) is None

    def test_no_update_without_an_upstream_branch(self, checkout):
        app, seed = checkout
        _push(seed, "Upstream change")
        _git(app, "checkout", "-q", "-b", "local")
        assert up.check_for_update(app) is None

    def test_no_update_on_a_detached_head(self, checkout):
        app, seed = checkout
        _push(seed, "Upstream change")
        _git(app, "checkout", "-q", "--detach")
        assert up.check_for_update(app) is None

    def test_no_update_when_the_fetch_fails(self, checkout, tmp_path):
        app, seed = checkout
        _push(seed, "Upstream change")
        _git(app, "remote", "set-url", "origin", str(tmp_path / "gone.git"))
        assert up.check_for_update(app) is None

    def test_no_update_without_an_update_script(self, checkout):
        app, seed = checkout
        _push(seed, "Upstream change")
        shutil.rmtree(app / "install")
        assert up.check_for_update(app) is None

    def test_cancelled(self, checkout):
        app, seed = checkout
        _push(seed, "Upstream change")
        cancel = threading.Event()
        cancel.set()
        with patch.object(up, "_POLL_SECONDS", 0):
            assert up.check_for_update(app, cancel=cancel) is None


def test_no_update_outside_a_checkout(tmp_path):
    (tmp_path / "install").mkdir()
    (tmp_path / "install" / "update.bat").write_text("", encoding="utf-8")
    (tmp_path / "install" / "update.sh").write_text("", encoding="utf-8")
    assert up.check_for_update(tmp_path) is None


def test_no_update_without_git(tmp_path):
    (tmp_path / ".git").mkdir()
    with patch.object(up.shutil, "which", return_value=None):
        assert up.check_for_update(tmp_path) is None


def test_repo_root_is_the_checkout():
    assert (up.repo_root() / "src" / "gui" / "app_updater.py").is_file()


class TestStartUpdate:
    def test_runs_the_update_script_in_a_console_of_its_own(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / ".venv"))
        monkeypatch.delenv(up.RESTART_ENV, raising=False)
        with patch.object(up.subprocess, "Popen") as popen:
            assert up.start_update(tmp_path)
        args, kwargs = popen.call_args
        assert args[0][-2:] == ["/c", str(up.update_script(tmp_path))]
        assert kwargs["cwd"] == str(tmp_path)
        assert kwargs["creationflags"] == up._CREATE_NEW_CONSOLE
        assert kwargs["env"][up.RESTART_ENV] == "1"
        # The GUI's environment does not reach the update's uv calls
        assert "VIRTUAL_ENV" not in kwargs["env"]
        assert up.RESTART_ENV not in os.environ

    def test_without_a_restart(self, tmp_path, monkeypatch):
        monkeypatch.delenv(up.RESTART_ENV, raising=False)
        with patch.object(up.subprocess, "Popen") as popen:
            assert up.start_update(tmp_path, restart=False)
        assert up.RESTART_ENV not in popen.call_args.kwargs["env"]

    def test_reports_a_failed_start(self, tmp_path):
        with patch.object(up.subprocess, "Popen", side_effect=OSError("no cmd")):
            assert not up.start_update(tmp_path)


def test_update_bat_starts_the_gui_again_only_after_a_completed_update():
    text = (Path(__file__).resolve().parents[2] / "install" / "update.bat").read_text(encoding="utf-8")
    restart = text.index(f"if defined {up.RESTART_ENV} (")
    # After the pull (the beginning must stay unchanged) and after the success message
    assert text.index("git pull --ff-only") < text.index('echo Update completed.') < restart
    block = text[restart:text.index(")", text.index("exit /b 0", restart))]
    assert "run_gui_on_windows.bat" in block and "exit /b 0" in block
    # A failed update ends at its pause, before the restart
    assert text.index("\n:restore_and_fail\n") > restart
