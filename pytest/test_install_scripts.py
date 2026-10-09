"""Tests for the install / update scripts in install/.

Every script that locks the dependencies must lock the newest yt-dlp (video
platforms change often, so the pinned yt-dlp soon fails to download) and fall
back to the pinned version when that is not possible. update.sh is also run
for real against a throw-away git remote with a stub ``uv``.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INSTALL = REPO / "install"
UPGRADE = "uv lock --upgrade-package yt-dlp"

INSTALL_SCRIPTS = [
    "CPU/windows_cpu.bat", "CPU/linux_cpu.sh", "CPU/macos_cpu.sh",
    "CUDA/windows_cuda_gpu.bat", "CUDA/linux_cuda_gpu.sh",
]
UPDATE_SCRIPTS = ["update.bat", "update.sh"]


def _lines(script: str) -> list[str]:
    return [line.strip() for line in (INSTALL / script).read_text(encoding="utf-8").splitlines()]


def _index(lines: list[str], predicate) -> int:
    return next(i for i, line in enumerate(lines) if predicate(line))


@pytest.mark.parametrize("script", INSTALL_SCRIPTS + UPDATE_SCRIPTS)
def test_locks_the_newest_ytdlp_before_syncing(script):
    lines = _lines(script)
    upgrade = _index(lines, lambda l: UPGRADE in l)
    sync = _index(lines, lambda l: l.startswith("uv sync"))
    assert upgrade < sync


@pytest.mark.parametrize("script", INSTALL_SCRIPTS)
def test_install_falls_back_to_the_pinned_version(script):
    lines = _lines(script)
    upgrade = _index(lines, lambda l: UPGRADE in l)
    sync = _index(lines, lambda l: l.startswith("uv sync"))
    assert "uv lock" in lines[upgrade + 1:sync], "no plain 'uv lock' fallback after the yt-dlp upgrade"
    assert any("Warning: could not upgrade yt-dlp" in l for l in lines[upgrade:sync])


@pytest.mark.parametrize("script", UPDATE_SCRIPTS)
def test_update_discards_a_local_lock_change_before_pulling(script):
    lines = _lines(script)
    discard = _index(lines, lambda l: "git checkout HEAD -- uv.lock" in l)
    pull = _index(lines, lambda l: l.startswith("git pull"))
    assert discard < pull
    assert any("git diff --quiet HEAD -- uv.lock" in l for l in lines[:pull])


# --------------------------------------------------------------------------
# update.sh, run for real
# --------------------------------------------------------------------------

STUB_UV = """#!/bin/bash
echo "$*" >> "$UV_STUB_LOG"
case "$*" in
    "lock --upgrade-package yt-dlp") [ -n "$UV_STUB_FAIL_UPGRADE" ] && exit 1 ;;
    "lock") [ -n "$UV_STUB_FAIL_LOCK" ] && exit 1 ;;
esac
exit 0
"""

# On Windows "bash" may resolve to WSL; set ULTRASINGER_TEST_BASH to a Git Bash
# (e.g. C:\Program Files\Git\bin\bash.exe) to run these tests there.
BASH = os.environ.get("ULTRASINGER_TEST_BASH") or (None if sys.platform == "win32" else shutil.which("bash"))
needs_bash = pytest.mark.skipif(BASH is None or shutil.which("git") is None,
                                reason="runs install/update.sh with bash and git")


def _git(cwd: Path, *args: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=cwd, env=env, check=True,
                          capture_output=True, text=True).stdout


def _commit_lock(repo: Path, content: str) -> None:
    (repo / "uv.lock").write_text(content, encoding="utf-8")
    _git(repo, "commit", "-qam", "lock")
    _git(repo, "push", "-q", "origin", "main")


@pytest.fixture
def install(tmp_path):
    """An "installation" cloned from a remote whose uv.lock has moved on since."""
    remote, seed, app = tmp_path / "remote.git", tmp_path / "seed", tmp_path / "app"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(tmp_path, "clone", "-q", "-c", "core.autocrlf=false", str(remote), str(seed))
    _git(seed, "symbolic-ref", "HEAD", "refs/heads/main")
    (seed / "install").mkdir()
    script = (INSTALL / "update.sh").read_text(encoding="utf-8").replace("\r\n", "\n")
    (seed / "install" / "update.sh").write_text(script, encoding="utf-8", newline="\n")
    (seed / "pyproject.toml").write_text('url = "https://download.pytorch.org/whl/cpu"\n', encoding="utf-8")
    (seed / "uv.lock").write_text("lock v1\n", encoding="utf-8")
    _git(seed, "add", "install/update.sh", "pyproject.toml", "uv.lock")
    _git(seed, "commit", "-qm", "init")
    _git(seed, "push", "-q", "origin", "main")
    _git(tmp_path, "clone", "-q", "-c", "core.autocrlf=false", str(remote), str(app))
    _commit_lock(seed, "lock v2\n")  # upstream changes uv.lock after the install

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(STUB_UV, encoding="utf-8", newline="\n")
    (bin_dir / "uv").chmod(0o755)
    log = tmp_path / "uv.log"
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", HOME=str(tmp_path / "home"),
               UV_STUB_LOG=str(log), UV_LINK_MODE="copy")
    for key in ("UV_STUB_FAIL_UPGRADE", "UV_STUB_FAIL_LOCK"):
        env.pop(key, None)

    def run(**extra):
        result = subprocess.run([BASH, str(app / "install" / "update.sh")], cwd=app, env=dict(env, **extra),
                                capture_output=True, text=True, timeout=120)
        calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return result, calls

    return app, run


@needs_bash
def test_update_cpu_install_replaces_the_local_lock_and_upgrades_ytdlp(install):
    app, run = install
    (app / "uv.lock").write_text("lock with a newer yt-dlp from the app\n", encoding="utf-8")
    result, calls = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Discarding the local changes to uv.lock" in result.stdout
    assert (app / "uv.lock").read_text(encoding="utf-8") == "lock v2\n"  # the pull went through
    assert calls == ["lock --upgrade-package yt-dlp", "sync --extra gui --extra scoring --extra potoken"]


@needs_bash
def test_update_cpu_install_continues_when_the_upgrade_fails(install):
    app, run = install
    result, calls = run(UV_STUB_FAIL_UPGRADE="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Warning: could not upgrade yt-dlp" in result.stdout
    assert "Discarding" not in result.stdout  # nothing local to discard
    assert calls == ["lock --upgrade-package yt-dlp", "sync --extra gui --extra scoring --extra potoken"]


def _make_cuda(app: Path) -> None:
    (app / "pyproject.toml").write_text('url = "https://download.pytorch.org/whl/cu128"\n', encoding="utf-8")
    (app / "uv.lock").write_text("cuda lock with a newer yt-dlp\n", encoding="utf-8")
    _git(app, "update-index", "--skip-worktree", "pyproject.toml", "uv.lock")


def _protected(app: Path) -> bool:
    flags = _git(app, "ls-files", "-v", "pyproject.toml", "uv.lock").split()
    return flags.count("S") == 2


@needs_bash
def test_update_cuda_install_upgrades_ytdlp_and_keeps_the_protection(install):
    app, run = install
    _make_cuda(app)
    result, calls = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "whl/cu128" in (app / "pyproject.toml").read_text(encoding="utf-8")
    assert _protected(app)
    assert calls == ["lock --upgrade-package yt-dlp", "sync --extra gui --extra scoring --extra potoken"]


@needs_bash
def test_update_cuda_install_falls_back_to_a_plain_lock(install):
    app, run = install
    _make_cuda(app)
    result, calls = run(UV_STUB_FAIL_UPGRADE="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == ["lock --upgrade-package yt-dlp", "lock", "sync --extra gui --extra scoring --extra potoken"]
    assert _protected(app)


@needs_bash
def test_update_cuda_install_restores_its_files_when_locking_fails(install):
    app, run = install
    _make_cuda(app)
    result, calls = run(UV_STUB_FAIL_UPGRADE="1", UV_STUB_FAIL_LOCK="1")
    assert result.returncode != 0
    assert "restoring the previous CUDA configuration" in result.stdout
    assert (app / "uv.lock").read_text(encoding="utf-8") == "cuda lock with a newer yt-dlp\n"
    assert "whl/cu128" in (app / "pyproject.toml").read_text(encoding="utf-8")
    assert _protected(app)
    assert calls == ["lock --upgrade-package yt-dlp", "lock"]
