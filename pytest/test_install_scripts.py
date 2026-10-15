"""Tests for the install / update scripts in install/ and the GUI launchers.

Every script that locks the dependencies must lock the newest yt-dlp (video
platforms change often, so the pinned yt-dlp soon fails to download) and fall
back to the pinned version when that is not possible. update.sh, the Linux
install scripts, auto_install.sh and the shell helpers are also run for real
against throw-away git repositories with stubs for uv, nvidia-smi, node and npm;
on Windows, so are update.bat, the Windows install scripts and the batch helper.
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tomllib
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


# The line that puts the uv of the install scripts first on PATH, and the launcher's uv check
LAUNCHERS = {
    "run_gui_on_windows.bat": ('set "PATH=%USERPROFILE%\\.local\\bin;', "where uv"),
    "run_gui_on_linux.sh": ('export PATH="$HOME/.local/bin:', "if ! command -v uv"),
    "run_gui_on_mac.command": ('export PATH="$HOME/.local/bin:', "if ! command -v uv"),
}


@pytest.mark.parametrize("script", sorted(LAUNCHERS))
def test_gui_launcher_prefers_the_installed_uv(script):
    prefer, check = LAUNCHERS[script]
    lines = [line.strip() for line in (REPO / script).read_text(encoding="utf-8").splitlines()]
    assert _index(lines, lambda l: l.startswith(prefer)) < _index(lines, lambda l: l.startswith(check))


@pytest.mark.parametrize("script", UPDATE_SCRIPTS)
def test_update_discards_a_local_lock_change_before_pulling(script):
    lines = _lines(script)
    discard = _index(lines, lambda l: "git checkout HEAD -- uv.lock" in l)
    pull = _index(lines, lambda l: l.startswith("git pull"))
    assert discard < pull
    assert any("git diff --quiet HEAD -- uv.lock" in l for l in lines[:pull])


# cmd.exe reads a running batch file piece by piece: when update.bat pulls a new
# version of itself, the running one carries on at the same byte offset in the
# new file. So everything up to and including the pull line stays as released.
UPDATE_BAT_PULL_LINE = b"\r\ngit pull --ff-only\r\n"
UPDATE_BAT_UNTIL_PULL_SHA256 = "f3a7bcdfbcf2fed45eb03c3d3ea10195469a65f4deee02d2676466bb28ceda59"


def test_update_bat_keeps_its_beginning_up_to_the_pull():
    """An update.bat that pulls this version must carry on right after the pull in it."""
    data = (INSTALL / "update.bat").read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    assert data.count(UPDATE_BAT_PULL_LINE) == 1
    end = data.index(UPDATE_BAT_PULL_LINE) + len(UPDATE_BAT_PULL_LINE)
    assert hashlib.sha256(data[:end]).hexdigest() == UPDATE_BAT_UNTIL_PULL_SHA256, (
        "install/update.bat changed before its pull: an update.bat that pulls this version would carry "
        "on at the wrong place in it - add new steps after the pull (see the comment there)")


WINDOWS_INSTALL_SCRIPTS = ["CPU/windows_cpu.bat", "CUDA/windows_cuda_gpu.bat"]


@pytest.mark.parametrize("script", WINDOWS_INSTALL_SCRIPTS)
def test_windows_install_hands_uv_a_python_version_not_a_path(script):
    """A path read back from Python's output is garbled for non-ASCII user folders (ANSI vs. OEM code page)."""
    text = (INSTALL / script).read_text(encoding="utf-8")
    assert "sys.executable" not in text
    assert 'uv sync --python "!PYTHON_VERSION!"' in text


@pytest.mark.parametrize("script", WINDOWS_INSTALL_SCRIPTS)
def test_windows_install_succeeds_whatever_the_provider_setup_returns(script):
    """The PO-token setup is optional: its exit code must not become the install's exit code."""
    lines = _lines(script)
    provider = _index(lines, lambda l: "setup_potoken_provider.bat" in l)
    fail = _index(lines, lambda l: l == ":restore_and_fail")
    assert "exit /b 0" in lines[provider:fail]


def test_auto_install_bat_works_from_the_repository_root():
    """Started from install\\ (a double-click), it must still find this installation's .venv and processes."""
    lines = _lines("auto_install.bat")
    root = _index(lines, lambda l: l == 'pushd "%~dp0.."')
    assert root < _index(lines, lambda l: "%CD%" in l and not l.startswith("REM"))


def _git_files(*patterns: str) -> list[str]:
    return subprocess.run(["git", "ls-files", "--", *patterns], cwd=REPO, check=True, capture_output=True,
                          text=True).stdout.split()


@pytest.mark.skipif(shutil.which("git") is None, reason="reads the git attributes")
def test_shell_scripts_keep_lf_line_endings_in_every_checkout():
    """bash fails on CR, so a Windows checkout (core.autocrlf) must not convert the shell scripts."""
    scripts = _git_files("*.sh", "*.command")
    assert "install/update.sh" in scripts and "run_gui_on_mac.command" in scripts
    attrs = subprocess.run(["git", "check-attr", "eol", "--", *scripts], cwd=REPO, check=True,
                           capture_output=True, text=True).stdout.splitlines()
    assert attrs and all(line.endswith(": eol: lf") for line in attrs), attrs
    for script in scripts:
        blob = subprocess.run(["git", "show", f"HEAD:{script}"], cwd=REPO, capture_output=True).stdout
        assert b"\r" not in blob, f"{script} has CR line endings in the repository"


UV_INSTALLER = re.compile(r"astral\.sh/uv/(?:([0-9][^/\s]*)/)?install\.(?:sh|ps1)")


def test_the_uv_installer_is_pinned_to_one_version_everywhere():
    """Install scripts, the container image, the notebook and the CI install the same, fixed uv release."""
    versions = {}
    for rel in INSTALL_SCRIPTS + ["../colab/UltraSinger.ipynb", "../run_gui_on_linux.sh", "../run_gui_on_mac.command",
                                  "../.github/workflows/main.yml"]:
        text = (INSTALL / rel).read_text(encoding="utf-8")
        found = UV_INSTALLER.findall(text)
        assert found, f"{rel} does not install uv"
        assert all(found), f"{rel} runs the unpinned uv installer"
        versions[rel] = set(found)
    image = re.findall(r"COPY --from=ghcr\.io/astral-sh/uv:(\S+) ", (REPO / "Dockerfile").read_text(encoding="utf-8"))
    assert image, "the Dockerfile does not install a pinned uv"
    versions["Dockerfile"] = set(image)
    assert len(set().union(*versions.values())) == 1, versions


def test_every_uv_setup_action_in_ci_is_pinned_to_that_release():
    """Every step of every workflow that uses the uv setup action pins the
    release of the install scripts as its "version" (read as YAML, so the
    spelling of a step does not matter)."""
    yaml = pytest.importorskip("yaml")
    release = re.findall(r"COPY --from=ghcr\.io/astral-sh/uv:(\S+) ",
                         (REPO / "Dockerfile").read_text(encoding="utf-8"))[0]
    workflows = REPO / ".github/workflows"
    found = {}
    for workflow in sorted([*workflows.glob("*.yml"), *workflows.glob("*.yaml")]):
        jobs = (yaml.safe_load(workflow.read_text(encoding="utf-8")) or {}).get("jobs") or {}
        for job_name, job in jobs.items():
            for step in (job or {}).get("steps") or []:
                uses = str((step or {}).get("uses") or "")
                if not uses.startswith("astral-sh/setup-uv@"):
                    continue
                version = ((step.get("with") or {}).get("version"))
                assert version is not None, f"{workflow.name} ({job_name}) sets up uv without a pinned version"
                found.setdefault(workflow.name, set()).add(str(version))
    assert "tests.yml" in found, "the test workflow does not set up uv"
    assert set().union(*found.values()) == {release}, (found, release)


def _assignment(text: str, name: str) -> str:
    match = re.search(rf'^\s*(?:set ")?{name}="?([^"\r\n]+)"?\s*$', text, re.MULTILINE)
    assert match, f"{name} not found"
    return match.group(1)


def test_po_token_provider_is_pinned_to_the_plugin_version_in_the_lock():
    """The provider server is built from the release that matches the yt-dlp plugin in uv.lock."""
    sh = (INSTALL / "helpers/setup_potoken_provider.sh").read_text(encoding="utf-8")
    bat = (INSTALL / "helpers/setup_potoken_provider.bat").read_text(encoding="utf-8")
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    plugin = next(p["version"] for p in lock["package"] if p["name"] == "bgutil-ytdlp-pot-provider")
    for name in ("PROVIDER_TAG", "PROVIDER_COMMIT", "NODE_MIN_MAJOR"):
        assert _assignment(sh, name) == _assignment(bat, name), name
    assert _assignment(sh, "PROVIDER_TAG") == plugin
    assert re.fullmatch(r"[0-9a-f]{40}", _assignment(sh, "PROVIDER_COMMIT"))
    # the GUI's stand-in for the plugin version (Docker image, expected provider version)
    gui = (REPO / "src/gui/potoken_provider.py").read_text(encoding="utf-8")
    assert re.search(rf'^_PINNED_PROVIDER_VERSION = "{re.escape(plugin)}"$', gui, re.MULTILINE)
    # the requirement allows exactly the pinned major version
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    requirement = next(r for r in project["project"]["optional-dependencies"]["potoken"]
                       if r.startswith("bgutil-ytdlp-pot-provider"))
    assert requirement == f"bgutil-ytdlp-pot-provider>={plugin},<{int(plugin.split('.')[0]) + 1}"
    # nothing may fetch or build an unpinned provider any more
    for text in (sh, bat):
        assert "pull --ff-only" not in text and "npm install" not in text and "--yes" not in text


def test_container_build_context_leaves_out_local_state():
    """Only an allowlist goes into the image: never cookies, tokens, environments or songs."""
    patterns = [line.strip() for line in (REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    assert patterns[0] == "*"
    included = {p[1:] for p in patterns if p.startswith("!")}
    assert included <= {"pyproject.toml", "uv.lock", "LICENSE", "README.md", "src", "tools"}, included
    assert "**/cookies.txt" in patterns


def test_container_user_has_a_writable_home_for_the_model_caches():
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    user = dockerfile.index("USER 1000:1000")
    for env in ("ENV HOME=/home/ultrasinger", "HF_HOME=", "TORCH_HOME=", "XDG_CACHE_HOME="):
        assert -1 < dockerfile.find(env) < user, env
    assert 'chown -R 1000:1000 /app/UltraSinger "$HOME"' in dockerfile
    assert "/root/.local/bin" not in dockerfile  # not readable for that user


@pytest.mark.parametrize("compose", ["container/compose-cpu.yml", "container/compose-gpu.yml"])
def test_compose_mounts_do_not_create_root_owned_paths(compose):
    yaml = pytest.importorskip("yaml")
    service = yaml.safe_load((REPO / compose).read_text(encoding="utf-8"))["services"]["ultrasinger"]
    binds = [v for v in service["volumes"] if isinstance(v, dict) and v.get("type") == "bind"]
    assert {b["target"] for b in binds} == {"/app/UltraSinger/input", "/app/UltraSinger/output"}
    assert all(b["bind"]["create_host_path"] is False for b in binds)
    assert "model-cache:/home/ultrasinger/.cache" in service["volumes"]
    assert not any("cookies" in str(v) for v in service["volumes"])


# --------------------------------------------------------------------------
# update.sh, run for real
# --------------------------------------------------------------------------

STUB_UV = """#!/bin/bash
echo "$*" >> "$UV_STUB_LOG"
case "$*" in
    "lock --upgrade-package yt-dlp") [ -n "$UV_STUB_FAIL_UPGRADE" ] && exit 1 ;;
    "lock") [ -n "$UV_STUB_FAIL_LOCK" ] && exit 1 ;;
    sync*)
        mkdir -p .venv
        if [ -n "$UV_STUB_LOCK_PARTIAL" ]; then mkdir -p .venv/locked; touch .venv/locked/f; chmod 555 .venv/locked; fi
        if [ -n "$UV_STUB_FAIL_SYNC" ]; then touch .venv/partial; exit 1; fi
        touch .venv/marker-new ;;
esac
exit 0
"""
UV_STUB_SWITCHES = ("UV_STUB_FAIL_UPGRADE", "UV_STUB_FAIL_LOCK", "UV_STUB_FAIL_SYNC", "UV_STUB_LOCK_PARTIAL")

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


def _installation(tmp_path: Path, script: str) -> Path:
    """An "installation" with install/<script>, cloned from a remote whose uv.lock has moved on since."""
    remote, seed, app = tmp_path / "remote.git", tmp_path / "seed", tmp_path / "app"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(tmp_path, "clone", "-q", "-c", "core.autocrlf=false", str(remote), str(seed))
    _git(seed, "symbolic-ref", "HEAD", "refs/heads/main")
    (seed / "install").mkdir()
    text = (INSTALL / script).read_text(encoding="utf-8").replace("\r\n", "\n")
    (seed / "install" / script).write_text(text, encoding="utf-8",
                                           newline="\r\n" if script.endswith(".bat") else "\n")
    (seed / "pyproject.toml").write_text('url = "https://download.pytorch.org/whl/cpu"\n', encoding="utf-8")
    (seed / "uv.lock").write_text("lock v1\n", encoding="utf-8")
    _git(seed, "add", f"install/{script}", "pyproject.toml", "uv.lock")
    _git(seed, "commit", "-qm", "init")
    _git(seed, "push", "-q", "origin", "main")
    _git(tmp_path, "clone", "-q", "-c", "core.autocrlf=false", str(remote), str(app))
    _commit_lock(seed, "lock v2\n")  # upstream changes uv.lock after the install
    return app


def _change_upstream_pyproject(app: Path, extra: str) -> None:
    seed = app.parent / "seed"
    (seed / "pyproject.toml").write_text(f'url = "https://download.pytorch.org/whl/cpu"\n{extra}',
                                         encoding="utf-8")
    _git(seed, "commit", "-qam", "pyproject")
    _git(seed, "push", "-q", "origin", "main")


@pytest.fixture
def install(tmp_path):
    """update.sh in an "installation" whose upstream has moved on; uv is a stub."""
    app = _installation(tmp_path, "update.sh")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(STUB_UV, encoding="utf-8", newline="\n")
    (bin_dir / "uv").chmod(0o755)
    log = tmp_path / "uv.log"
    (tmp_path / "tmp").mkdir()
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", HOME=str(tmp_path / "home"),
               UV_STUB_LOG=str(log), UV_LINK_MODE="copy", TMPDIR=(tmp_path / "tmp").as_posix())
    for key in UV_STUB_SWITCHES:
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


def _make_cuda(app: Path, variant: str = "cu128", extra: str = "") -> None:
    (app / "pyproject.toml").write_text(f'url = "https://download.pytorch.org/whl/{variant}"\n{extra}',
                                        encoding="utf-8")
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


@needs_bash
def test_update_cuda_install_keeps_its_cuda_variant(install):
    app, run = install
    _make_cuda(app, "cu126")
    result, calls = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "whl/cu126" in (app / "pyproject.toml").read_text(encoding="utf-8")
    assert (app / "uv.lock").read_text(encoding="utf-8") == "lock v2\n"  # the pull went through
    assert _protected(app)
    assert "Note: pyproject.toml has local changes" not in result.stdout

def _native(path: str) -> Path:
    """A path printed by the script; Git Bash prints C:/x as /c/x."""
    drive = re.match(r"/([a-zA-Z])/(.*)", path)
    return Path(f"{drive.group(1)}:/{drive.group(2)}") if sys.platform == "win32" and drive else Path(path)


@needs_bash
def test_update_cuda_install_keeps_a_copy_of_other_local_pyproject_changes(install):
    app, run = install
    _make_cuda(app, extra='extra = "mine"\n')
    result, calls = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "extra" not in (app / "pyproject.toml").read_text(encoding="utf-8")
    kept = re.search(r"kept as\s+(\S+pyproject\.toml)", result.stdout)
    assert kept, result.stdout
    assert 'extra = "mine"' in _native(kept.group(1)).read_text(encoding="utf-8")


@needs_bash
def test_update_cuda_install_reports_a_failed_sync(install):
    app, run = install
    _make_cuda(app, "cu126")
    result, calls = run(UV_STUB_FAIL_SYNC="1")
    assert result.returncode != 0
    assert "restoring the previous CUDA configuration" in result.stdout
    assert "The update is incomplete" in result.stdout and "install/update.sh again" in result.stdout
    assert "whl/cu126" in (app / "pyproject.toml").read_text(encoding="utf-8")
    assert _protected(app)


@needs_bash
def test_update_cpu_install_reports_a_failed_sync(install):
    app, run = install
    result, calls = run(UV_STUB_FAIL_SYNC="1")
    assert result.returncode != 0
    assert "The update is incomplete" in result.stdout
    assert calls == ["lock --upgrade-package yt-dlp", "sync --extra gui --extra scoring --extra potoken"]


@needs_bash
def test_update_after_a_failed_update_reports_no_local_changes(install):
    """A failed update puts the pyproject.toml of before its pull back; the next run must not call that an edit."""
    app, run = install
    _make_cuda(app)
    _change_upstream_pyproject(app, 'new = "upstream"\n')
    result, calls = run(UV_STUB_FAIL_SYNC="1")
    assert result.returncode != 0 and "restoring the previous CUDA configuration" in result.stdout
    result, calls = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "local changes besides the CUDA index" not in result.stdout
    pyproject = (app / "pyproject.toml").read_text(encoding="utf-8")
    assert 'new = "upstream"' in pyproject and "whl/cu128" in pyproject


@needs_bash
def test_update_reports_no_incomplete_update_when_the_pull_fails(install, tmp_path):
    app, run = install
    _git(app, "remote", "set-url", "origin", str(tmp_path / "missing.git"))
    result, calls = run()
    assert result.returncode != 0
    assert "The update is incomplete" not in result.stdout
    assert calls == []


# --------------------------------------------------------------------------
# Linux install scripts, run for real
# --------------------------------------------------------------------------

def _write_script(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("\r\n", "\n"), encoding="utf-8", newline="\n")
    path.chmod(0o755)


COMPLETE_MARKER = ".ultrasinger-complete"  # written into .venv once it is fully built


def _venv(app: Path) -> list[str]:
    """The files in .venv, without the completion marker."""
    if not (app / ".venv").is_dir():
        return []
    return sorted(p.name for p in (app / ".venv").iterdir() if p.name != COMPLETE_MARKER)


@pytest.fixture
def linux_install(tmp_path):
    """A CPU installation with a working .venv; the helpers and uv are stubs."""
    app = tmp_path / "app"
    for rel in ("CPU/linux_cpu.sh", "CUDA/linux_cuda_gpu.sh"):
        _write_script(app / "install" / rel, (INSTALL / rel).read_text(encoding="utf-8"))
    _write_script(app / "install/helpers/check_build_tools.sh", "#!/bin/bash\nexit 0\n")
    _write_script(app / "install/helpers/setup_potoken_provider.sh", "#!/bin/bash\necho stub provider\nexit 2\n")
    (app / "pyproject.toml").write_text('url = "https://download.pytorch.org/whl/cpu"\n', encoding="utf-8")
    (app / "uv.lock").write_text("lock cpu\n", encoding="utf-8")
    _git(tmp_path, "init", "-q", str(app))
    _git(app, "add", "-A")
    _git(app, "commit", "-qm", "init")
    (app / ".venv").mkdir()
    (app / ".venv" / "marker-old").write_text("old", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    _write_script(bin_dir / "uv", STUB_UV)
    log = tmp_path / "uv.log"
    (tmp_path / "tmp").mkdir()
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", HOME=str(tmp_path / "home"),
               UV_STUB_LOG=str(log), TMPDIR=(tmp_path / "tmp").as_posix())
    for key in UV_STUB_SWITCHES:
        env.pop(key, None)

    def run(script, **extra):
        log.unlink(missing_ok=True)
        result = subprocess.run([BASH, str(app / "install" / script)], cwd=tmp_path, env=dict(env, **extra),
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
        calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return result, calls

    return app, run


def _make_cuda_install(app: Path) -> None:
    (app / "pyproject.toml").write_text('url = "https://download.pytorch.org/whl/cu128"\n', encoding="utf-8")
    (app / "uv.lock").write_text("lock cuda\n", encoding="utf-8")
    _git(app, "update-index", "--skip-worktree", "pyproject.toml", "uv.lock")


@needs_bash
def test_linux_install_replaces_the_environment_when_it_succeeds(linux_install):
    app, run = linux_install
    result, calls = run("CPU/linux_cpu.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _venv(app) == ["marker-new"]
    assert (app / ".venv" / COMPLETE_MARKER).exists()
    assert not (app / ".venv.old").exists()
    assert calls == ["--version", "lock --upgrade-package yt-dlp", "sync --extra gui --extra scoring --extra potoken"]
    assert "full-quality video downloads are NOT enabled yet" in result.stdout  # the provider is optional


@needs_bash
def test_linux_install_keeps_the_previous_installation_when_locking_fails(linux_install):
    app, run = linux_install
    _make_cuda_install(app)
    result, calls = run("CPU/linux_cpu.sh", UV_STUB_FAIL_UPGRADE="1", UV_STUB_FAIL_LOCK="1")
    assert result.returncode != 0
    assert "restoring the previous installation" in result.stdout
    assert _venv(app) == ["marker-old"]
    assert "whl/cu128" in (app / "pyproject.toml").read_text(encoding="utf-8")
    assert (app / "uv.lock").read_text(encoding="utf-8") == "lock cuda\n"
    assert _protected(app)


@needs_bash
def test_linux_install_keeps_the_previous_environment_when_syncing_fails(linux_install):
    app, run = linux_install
    _make_cuda_install(app)
    result, calls = run("CPU/linux_cpu.sh", UV_STUB_FAIL_SYNC="1")
    assert result.returncode != 0
    assert _venv(app) == ["marker-old"]
    assert not (app / ".venv.old").exists()
    assert "whl/cu128" in (app / "pyproject.toml").read_text(encoding="utf-8")
    assert _protected(app)


@needs_bash
@pytest.mark.skipif(sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs a folder that POSIX permissions keep from being deleted")
def test_linux_install_never_moves_the_previous_environment_into_a_leftover(linux_install):
    """When the half-built .venv cannot be removed, the previous one stays in .venv.old, not inside .venv."""
    app, run = linux_install
    try:
        result, calls = run("CPU/linux_cpu.sh", UV_STUB_FAIL_SYNC="1", UV_STUB_LOCK_PARTIAL="1")
        assert result.returncode != 0
        assert "is kept in" in result.stdout
        assert (app / ".venv.old" / "marker-old").exists()
        assert not (app / ".venv" / ".venv.old").exists()
    finally:
        locked = app / ".venv" / "locked"
        if locked.exists():
            locked.chmod(0o755)


@needs_bash
def test_linux_cuda_install_restores_the_cpu_installation_when_syncing_fails(linux_install):
    app, run = linux_install
    result, calls = run("CUDA/linux_cuda_gpu.sh", UV_STUB_FAIL_SYNC="1")
    assert result.returncode != 0
    assert _venv(app) == ["marker-old"]
    assert "whl/cpu" in (app / "pyproject.toml").read_text(encoding="utf-8")
    assert not _protected(app)


@needs_bash
def test_linux_cuda_install_protects_its_configuration(linux_install):
    app, run = linux_install
    result, calls = run("CUDA/linux_cuda_gpu.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _venv(app) == ["marker-new"]
    assert "whl/cu128" in (app / "pyproject.toml").read_text(encoding="utf-8")
    assert _protected(app)


@needs_bash
def test_linux_install_returns_to_the_previous_environment_after_an_interrupted_sync(linux_install):
    """Killed during the sync: .venv is half built, .venv.old is the working one - keep that one."""
    app, run = linux_install
    (app / ".venv").rename(app / ".venv.old")
    (app / ".venv").mkdir()
    (app / ".venv" / "half-built").write_text("x", encoding="utf-8")
    result, calls = run("CPU/linux_cpu.sh", UV_STUB_FAIL_SYNC="1")
    assert result.returncode != 0
    assert _venv(app) == ["marker-old"]
    assert not (app / ".venv.old").exists()


@needs_bash
def test_linux_install_drops_the_backup_left_by_an_interrupted_clean_up(linux_install):
    """Killed after the sync: .venv is complete (marked), .venv.old is stale - keep .venv."""
    app, run = linux_install
    (app / ".venv" / COMPLETE_MARKER).write_text("", encoding="utf-8")
    (app / ".venv.old").mkdir()
    (app / ".venv.old" / "stale").write_text("x", encoding="utf-8")
    result, calls = run("CPU/linux_cpu.sh", UV_STUB_FAIL_SYNC="1")
    assert result.returncode != 0
    assert _venv(app) == ["marker-old"]
    assert not (app / ".venv.old").exists()


@needs_bash
@pytest.mark.parametrize("script", ["CPU/linux_cpu.sh", "CUDA/linux_cuda_gpu.sh"])
def test_linux_install_removes_its_incomplete_environment_when_there_was_none(linux_install, script):
    """A first installation that fails leaves no half-built .venv behind for the launchers to start."""
    app, run = linux_install
    shutil.rmtree(app / ".venv")
    result, calls = run(script, UV_STUB_FAIL_SYNC="1")
    assert result.returncode != 0
    assert "restoring the previous installation" in result.stdout
    assert not (app / ".venv").exists() and not (app / ".venv.old").exists()
    assert "whl/cpu" in (app / "pyproject.toml").read_text(encoding="utf-8")


def test_macos_install_matches_the_linux_install():
    """macos_cpu.sh is linux_cpu.sh with BSD sed and its own name; keep the two in step."""
    linux = (INSTALL / "CPU/linux_cpu.sh").read_text(encoding="utf-8")
    macos = (INSTALL / "CPU/macos_cpu.sh").read_text(encoding="utf-8")
    expected = linux.replace("sed -i 's|", "sed -i '' 's|").replace('"install/CPU/linux_cpu.sh"',
                                                                     '"install/CPU/macos_cpu.sh"')
    assert macos.replace("\r\n", "\n") == expected.replace("\r\n", "\n")


# --------------------------------------------------------------------------
# install/helpers/check_build_tools.sh, run for real
# --------------------------------------------------------------------------

STUB_FIND_UV = """#!/bin/bash
if [ "$1 $2" = "python find" ] && [ -n "$UV_STUB_PYTHON" ]; then echo "$UV_STUB_PYTHON"; exit 0; fi
exit 2
"""
STUB_PYTHON = """#!/bin/bash
case "$2" in
    *version_info*) echo "3.12" ;;
    *Python.h*) [ -n "$PY_STUB_HEADERS" ] && exit 0; exit 1 ;;
esac
"""


@pytest.fixture
def build_tools(tmp_path):
    bin_dir = tmp_path / "bin"
    _write_script(bin_dir / "uv", STUB_FIND_UV)
    _write_script(tmp_path / "python3.12", STUB_PYTHON)
    script = tmp_path / "check_build_tools.sh"
    _write_script(script, (INSTALL / "helpers/check_build_tools.sh").read_text(encoding="utf-8"))

    def run(compiler=True, python=True, headers=True):
        if compiler:
            _write_script(bin_dir / "cc", "#!/bin/bash\nexit 0\n")
        # only the stubs on PATH, so a real compiler of this machine cannot answer
        env = dict(os.environ, PATH=str(bin_dir))
        env.pop("PY_STUB_HEADERS", None)
        env.pop("UV_STUB_PYTHON", None)
        if python:
            env["UV_STUB_PYTHON"] = (tmp_path / "python3.12").as_posix()
        if headers:
            env["PY_STUB_HEADERS"] = "1"
        return subprocess.run([BASH, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True,
                              timeout=60)

    return run


@needs_bash
def test_build_tools_check_passes_with_compiler_and_headers(build_tools):
    result = build_tools()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ACTION REQUIRED" not in result.stdout


@needs_bash
def test_build_tools_check_asks_for_a_compiler(build_tools):
    result = build_tools(compiler=False)
    assert result.returncode == 1
    assert "need a C compiler." in result.stdout and "build-essential" in result.stdout


@needs_bash
def test_build_tools_check_asks_for_the_python_headers(build_tools):
    result = build_tools(headers=False)
    assert result.returncode == 1
    assert "Python 3.12 headers" in result.stdout and "python3.12-dev" in result.stdout


@needs_bash
def test_build_tools_check_trusts_a_python_uv_still_downloads(build_tools):
    """No suitable Python yet: uv downloads a managed one, which brings its headers."""
    result = build_tools(python=False, headers=False)
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------
# auto_install.sh: which build a GPU gets
# --------------------------------------------------------------------------

STUB_NVIDIA_SMI = """#!/bin/bash
case "$*" in
    *name,memory.total*) echo "NVIDIA Test GPU, 8192" ;;
    *compute_cap*) echo "$SMI_CC" ;;
    "") echo "| NVIDIA-SMI 999.99    Driver Version: 999.99    CUDA Version: $SMI_CUDA |" ;;
esac
"""


@pytest.fixture
def auto_install(tmp_path):
    app = tmp_path / "app"
    _write_script(app / "install/auto_install.sh", (INSTALL / "auto_install.sh").read_text(encoding="utf-8"))
    _write_script(app / "install/helpers/ensure_ffmpeg.sh", "#!/bin/bash\nexit 0\n")
    for rel, build in (("CPU/linux_cpu.sh", "cpu"), ("CPU/macos_cpu.sh", "cpu"), ("CUDA/linux_cuda_gpu.sh", "cuda")):
        _write_script(app / "install" / rel, f"#!/bin/bash\necho 'RAN {build}'\n")
    bin_dir = tmp_path / "bin"
    _write_script(bin_dir / "nvidia-smi", STUB_NVIDIA_SMI)
    _write_script(bin_dir / "uv", "#!/bin/bash\nexit 0\n")

    def run(*args, cc="8.6", cuda="13.0"):
        env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", SMI_CC=cc, SMI_CUDA=cuda,
                   UV_LINK_MODE="copy")
        env.pop("ULTRASINGER_BUILD", None)
        return subprocess.run([BASH, str(app / "install/auto_install.sh"), *args], cwd=tmp_path, env=env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)

    return run


@needs_bash
@pytest.mark.parametrize("cc, cuda", [("8.6", "13.0"), ("7.0", "12.0")])
def test_auto_install_uses_cuda_for_a_suitable_gpu(auto_install, cc, cuda):
    result = auto_install(cc=cc, cuda=cuda)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RAN cuda" in result.stdout


@needs_bash
@pytest.mark.parametrize("cc, cuda, reason", [
    ("5.2", "13.0", "compute capability is below 7.0"),
    # the Linux CUDA 12.8 build of PyTorch has no kernels for compute capability 6.x
    ("6.1", "13.0", "compute capability is below 7.0"),
    ("8.6", "11.4", "does not support CUDA 12"),
])
def test_auto_install_uses_the_cpu_build_for_an_unsuitable_gpu(auto_install, cc, cuda, reason):
    result = auto_install(cc=cc, cuda=cuda)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RAN cpu" in result.stdout
    assert "cannot run the CUDA build" in result.stdout and reason in result.stdout


@needs_bash
def test_auto_install_does_not_hold_unknown_values_against_the_gpu(auto_install):
    """An old nvidia-smi without the compute_cap field answers with an error text."""
    result = auto_install(cc='Field "compute_cap" is not a valid field to query.')
    assert "RAN cuda" in result.stdout


@needs_bash
def test_auto_install_warns_when_cuda_is_forced_on_an_unsuitable_gpu(auto_install):
    result = auto_install("--cuda", cc="5.2")
    assert "RAN cuda" in result.stdout
    assert "the CUDA build is forced, but this GPU cannot run it" in result.stdout


# --------------------------------------------------------------------------
# install/helpers/setup_potoken_provider.sh, run for real against a local
# stand-in of the provider repository
# --------------------------------------------------------------------------

PROVIDER_URL = "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git"
STUB_NODE = """#!/bin/bash
case "$1" in
    -p) echo "$NODE_STUB_MAJOR" ;;
    --version) echo "v$NODE_STUB_MAJOR.0.0" ;;
esac
"""
STUB_NPM = """#!/bin/bash
echo "npm $*" >> "$BUILD_STUB_LOG"
mkdir -p node_modules && touch node_modules/.package-lock.json
"""
STUB_NPX = """#!/bin/bash
echo "npx $*" >> "$BUILD_STUB_LOG"
out=build
while [ $# -gt 0 ]; do [ "$1" = "--outDir" ] && out="$2"; shift; done
mkdir -p "$out" && echo built > "$out/main.js" && echo 'export const VERSION = "x";' > "$out/utils.js"
"""
BUILD_STEPS = ["npm ci --include=dev --no-audit --no-fund", "npx --no -- tsc --outDir build.new"]


def _commit_of(repo: Path, ref: str) -> str:
    return _git(repo, "rev-parse", ref).strip()


def _provider_releases(tmp_path: Path) -> tuple[Path, Path]:
    """Releases 1.0.0 and 2.0.0 of a provider stand-in: its source and a bare repository."""
    src = tmp_path / "provider-src"
    (src / "server" / "src").mkdir(parents=True)
    _git(tmp_path, "init", "-q", str(src))
    for version in ("1.0.0", "2.0.0"):
        (src / "server" / "src" / "main.ts").write_text(f"// {version}\n", encoding="utf-8")
        _git(src, "add", "-A")
        _git(src, "commit", "-qm", version)
        _git(src, "tag", version)
    remote = tmp_path / "provider.git"
    _git(tmp_path, "clone", "-q", "--bare", str(src), str(remote))
    return src, remote


@pytest.fixture
def provider(tmp_path):
    """Releases 1.0.0 and 2.0.0 of a provider stand-in, served from a local repository."""
    src, remote = _provider_releases(tmp_path)
    app = tmp_path / "app"
    _git(tmp_path, "init", "-q", str(app))  # the provider lives inside the UltraSinger checkout
    script = app / "install/helpers/setup_potoken_provider.sh"
    pinned = {"PROVIDER_TAG": "2.0.0", "PROVIDER_COMMIT": _commit_of(src, "2.0.0")}

    def install_helper(**pin):
        text = (INSTALL / "helpers/setup_potoken_provider.sh").read_text(encoding="utf-8")
        for name, value in dict(pinned, **pin).items():
            text = re.sub(rf"^{name}=.*$", f'{name}="{value}"', text, count=1, flags=re.MULTILINE)
        _write_script(script, text)

    bin_dir = tmp_path / "bin"
    for name, stub in (("node", STUB_NODE), ("npm", STUB_NPM), ("npx", STUB_NPX)):
        _write_script(bin_dir / name, stub)
    log = tmp_path / "build.log"
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", NODE_STUB_MAJOR="24",
               BUILD_STUB_LOG=str(log), ULTRASINGER_POTOKEN_SKIP_WARMUP="1",
               # fetch the stand-in instead of the real repository
               GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0=f"url.{remote.as_uri()}.insteadOf",
               GIT_CONFIG_VALUE_0=PROVIDER_URL)

    def run(**extra):
        log.unlink(missing_ok=True)
        result = subprocess.run([BASH, str(script), "install/update.sh"], cwd=app, env=dict(env, **extra),
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
        steps = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return result, steps

    install_helper()
    checkout = app / ".potoken" / "bgutil-ytdlp-pot-provider"
    return {"src": src, "remote": remote, "checkout": checkout, "run": run, "install_helper": install_helper}


@needs_bash
def test_provider_is_built_from_the_pinned_release(provider):
    result, steps = provider["run"]()
    assert result.returncode == 0, result.stdout + result.stderr
    checkout = provider["checkout"]
    assert _commit_of(checkout, "HEAD") == _commit_of(provider["src"], "2.0.0")
    assert steps == BUILD_STEPS
    assert (checkout / "server/build/.ultrasinger-commit").read_text(encoding="utf-8").strip() \
        == _commit_of(provider["src"], "2.0.0")
    assert not (checkout / "server/build.new").exists()


@needs_bash
def test_provider_checkout_of_another_release_is_moved_and_rebuilt(provider, tmp_path):
    checkout = provider["checkout"]
    _git(tmp_path, "clone", "-q", "--depth", "1", "--branch", "1.0.0", provider["remote"].as_uri(), str(checkout))
    (checkout / "server/src/main.ts").write_text("// local patch\n", encoding="utf-8")
    (checkout / "server/build").mkdir()
    (checkout / "server/build/main.js").write_text("old release\n", encoding="utf-8")
    (checkout / "server/tsconfig.tsbuildinfo").write_text("{}", encoding="utf-8")
    result, steps = provider["run"]()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "No such file" not in result.stderr  # an old build has no stamp yet
    assert _commit_of(checkout, "HEAD") == _commit_of(provider["src"], "2.0.0")
    assert (checkout / "server/src/main.ts").read_text(encoding="utf-8") == "// 2.0.0\n"
    assert (checkout / "server/build/main.js").read_text(encoding="utf-8") == "built\n"
    assert not (checkout / "server/tsconfig.tsbuildinfo").exists()  # a full, not an incremental build
    assert not (checkout / "server/build.old").exists()
    assert steps == BUILD_STEPS


@needs_bash
def test_provider_already_built_from_the_pinned_release_is_kept(provider):
    provider["run"]()
    result, steps = provider["run"]()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "already built" in result.stdout
    assert steps == []


@needs_bash
def test_provider_is_not_built_when_the_tag_points_elsewhere(provider):
    """A moved tag (the commit does not match the pinned one) must not be built."""
    provider["install_helper"](PROVIDER_COMMIT=_commit_of(provider["src"], "1.0.0"))
    result, steps = provider["run"]()
    assert result.returncode == 3
    assert "Could not download the provider source" in result.stdout
    assert steps == []
    assert not (provider["checkout"] / "server/build").exists()


@needs_bash
def test_provider_needs_a_recent_node(provider):
    result, steps = provider["run"](NODE_STUB_MAJOR="18")
    assert result.returncode == 2
    assert "Node.js 22 or newer is needed" in result.stdout and "found v18.0.0" in result.stdout
    assert steps == []
    assert not provider["checkout"].exists()


@needs_bash
def test_provider_folder_without_git_metadata_is_downloaded_again(provider):
    """git must never run in a provider folder without .git: it would act on the parent repository."""
    checkout = provider["checkout"]
    (checkout / "server").mkdir(parents=True)
    (checkout / "server/leftover").write_text("x", encoding="utf-8")
    result, steps = provider["run"]()
    assert result.returncode == 0, result.stdout + result.stderr
    assert _commit_of(checkout, "HEAD") == _commit_of(provider["src"], "2.0.0")
    assert not (checkout / "server/leftover").exists()
    assert not checkout.with_name(checkout.name + ".new").exists()


@needs_bash
def test_provider_folder_without_git_metadata_survives_a_failed_download(provider, tmp_path):
    """The folder (e.g. restored from a backup without .git) is only replaced by a verified download."""
    checkout = provider["checkout"]
    (checkout / "server/build").mkdir(parents=True)
    (checkout / "server/build/main.js").write_text("working build\n", encoding="utf-8")
    offline = f"url.{(tmp_path / 'unreachable.git').as_uri()}.insteadOf"
    result, steps = provider["run"](GIT_CONFIG_KEY_0=offline)
    assert result.returncode == 3
    assert (checkout / "server/build/main.js").read_text(encoding="utf-8") == "working build\n"
    assert not checkout.with_name(checkout.name + ".new").exists()
    assert steps == []


@needs_bash
@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to rename a folder in use")
def test_provider_folder_in_use_is_kept_whole(provider):
    """A virus scanner or a running provider holds a file: rename fails, nothing is half deleted."""
    checkout = provider["checkout"]
    (checkout / "server/build").mkdir(parents=True)
    (checkout / "server/build/main.js").write_text("working build\n", encoding="utf-8")
    (checkout / "server/build/utils.js").write_text("// utils\n", encoding="utf-8")
    with open(checkout / "server/build/main.js", "rb"):
        result, steps = provider["run"]()
    assert result.returncode == 3
    assert "is in use" in result.stdout and "Could not download" not in result.stdout
    assert (checkout / "server/build/main.js").read_text(encoding="utf-8") == "working build\n"
    assert (checkout / "server/build/utils.js").exists()
    for leftover in (".new", ".old"):
        assert not checkout.with_name(checkout.name + leftover).exists()


LEFTOVERS = (".new", ".old")  # next to the provider folder; build.new/build.old inside server/


def _make_leftovers(checkout: Path) -> None:
    """What a run stopped half way leaves behind."""
    for folder in [checkout.with_name(checkout.name + s) for s in LEFTOVERS] \
            + [checkout / "server" / "build.new", checkout / "server" / "build.old"]:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "leftover").write_text("x", encoding="utf-8")


def _leftovers(checkout: Path) -> list[str]:
    """Temporary folders of the helper that must not stay behind."""
    return [s for s in LEFTOVERS if checkout.with_name(checkout.name + s).exists()] \
        + [b for b in ("build.new", "build.old") if (checkout / "server" / b).exists()]


@needs_bash
def test_provider_leftovers_of_an_interrupted_run_are_removed(provider):
    provider["run"]()
    _make_leftovers(provider["checkout"])
    result, steps = provider["run"]()
    assert result.returncode == 0 and "already built" in result.stdout
    assert _leftovers(provider["checkout"]) == []


@needs_bash
@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to rename a folder in use")
def test_provider_build_in_use_is_kept_whole(provider):
    """E.g. the running provider works in build: the rebuild stops before anything is deleted."""
    provider["run"]()
    checkout = provider["checkout"]
    shutil.rmtree(checkout / "server/node_modules")  # forces a rebuild
    with open(checkout / "server/build/main.js", "rb"):
        result, steps = provider["run"]()
    assert result.returncode == 3
    assert "build folder" in result.stdout and "is in use" in result.stdout
    assert "did not complete" not in result.stdout
    assert steps == []  # npm ci never started
    assert (checkout / "server/build/main.js").read_text(encoding="utf-8") == "built\n"
    assert (checkout / "server/build/.ultrasinger-commit").exists()
    assert _leftovers(checkout) == []


@needs_bash
def test_provider_with_missing_dependencies_is_rebuilt(provider):
    """Security software may delete node_modules: build and stamp alone are not enough."""
    provider["run"]()
    shutil.rmtree(provider["checkout"] / "server/node_modules")
    result, steps = provider["run"]()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "already built" not in result.stdout
    assert steps == BUILD_STEPS


@needs_bash
def test_provider_build_without_its_version_file_is_rebuilt(provider):
    """The GUI reads the version from build/utils.js and asks for a rebuild without it."""
    provider["run"]()
    (provider["checkout"] / "server/build/utils.js").unlink()
    result, steps = provider["run"]()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "already built" not in result.stdout
    assert steps == BUILD_STEPS
    assert (provider["checkout"] / "server/build/utils.js").exists()


# --------------------------------------------------------------------------
# install/helpers/setup_potoken_provider.bat, run for real with cmd.exe
# against the same stand-in (Windows only)
# --------------------------------------------------------------------------

needs_cmd = pytest.mark.skipif(sys.platform != "win32" or shutil.which("git") is None,
                               reason="runs the Windows batch helper with cmd.exe and git")
STUB_NODE_CMD = (
    '@echo off\r\n'
    'if "%~1"=="-p" (echo %NODE_STUB_MAJOR%& exit /b 0)\r\n'
    'if "%~1"=="--version" (echo v%NODE_STUB_MAJOR%.0.0& exit /b 0)\r\n'
    'exit /b 0\r\n'
)
STUB_NPM_CMD = (
    '@echo off\r\n'
    '>>"%BUILD_STUB_LOG%" echo npm %*\r\n'
    'if not exist node_modules mkdir node_modules\r\n'
    'type nul > node_modules\\.package-lock.json\r\n'
    'exit /b 0\r\n'
)
STUB_NPX_CMD = (
    '@echo off\r\n'
    '>>"%BUILD_STUB_LOG%" echo npx %*\r\n'
    'set "OUT=build"\r\n'
    ':args\r\n'
    'if "%~1"=="" goto built\r\n'
    'if "%~1"=="--outDir" set "OUT=%~2"\r\n'
    'shift\r\n'
    'goto args\r\n'
    ':built\r\n'
    'if not exist "%OUT%" mkdir "%OUT%"\r\n'
    '>"%OUT%\\main.js" echo built\r\n'
    '>"%OUT%\\utils.js" echo export const VERSION = "x";\r\n'
    'exit /b 0\r\n'
)


@pytest.fixture
def bat_provider(tmp_path):
    """The batch helper, pinned to release 2.0.0 of the stand-in, with node/npm/npx stubs."""
    src, remote = _provider_releases(tmp_path)
    app = tmp_path / "app"
    _git(tmp_path, "init", "-q", str(app))  # the provider lives inside the UltraSinger checkout
    script = app / "install" / "helpers" / "setup_potoken_provider.bat"
    text = (INSTALL / "helpers/setup_potoken_provider.bat").read_text(encoding="utf-8")
    text = re.sub(r'set "PROVIDER_TAG=[^"]*"', 'set "PROVIDER_TAG=2.0.0"', text)
    text = re.sub(r'set "PROVIDER_COMMIT=[^"]*"', f'set "PROVIDER_COMMIT={_commit_of(src, "2.0.0")}"', text)
    script.parent.mkdir(parents=True)
    script.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, stub in (("node.cmd", STUB_NODE_CMD), ("npm.cmd", STUB_NPM_CMD), ("npx.cmd", STUB_NPX_CMD)):
        (bin_dir / name).write_bytes(stub.encode("ascii"))
    log = tmp_path / "build.log"
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", NODE_STUB_MAJOR="24",
               BUILD_STUB_LOG=str(log), ULTRASINGER_POTOKEN_SKIP_WARMUP="1",
               GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0=f"url.{remote.as_uri()}.insteadOf",
               GIT_CONFIG_VALUE_0=PROVIDER_URL)

    def run(**extra):
        log.unlink(missing_ok=True)
        result = subprocess.run(["cmd.exe", "/d", "/c", str(script), "install\\update.bat"], cwd=app,
                                env=dict(env, **extra), stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=120)
        steps = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return result, steps

    checkout = app / ".potoken" / "bgutil-ytdlp-pot-provider"
    return {"src": src, "checkout": checkout, "run": run,
            "offline": f"url.{(tmp_path / 'unreachable.git').as_uri()}.insteadOf"}


@needs_cmd
def test_bat_provider_is_built_from_the_pinned_release_and_then_kept(bat_provider):
    result, steps = bat_provider["run"]()
    assert result.returncode == 0, result.stdout + result.stderr
    checkout = bat_provider["checkout"]
    assert _commit_of(checkout, "HEAD") == _commit_of(bat_provider["src"], "2.0.0")
    assert steps == BUILD_STEPS
    assert (checkout / "server/build/main.js").exists()
    assert (checkout / "server/build/.ultrasinger-commit").read_text(encoding="utf-8").strip() \
        == _commit_of(bat_provider["src"], "2.0.0")
    assert _leftovers(checkout) == []
    result, steps = bat_provider["run"]()
    assert result.returncode == 0 and "already built" in result.stdout and steps == []


@needs_cmd
def test_bat_provider_with_missing_dependencies_is_rebuilt(bat_provider):
    bat_provider["run"]()
    shutil.rmtree(bat_provider["checkout"] / "server/node_modules")
    result, steps = bat_provider["run"]()
    assert result.returncode == 0, result.stdout + result.stderr
    assert steps == BUILD_STEPS


@needs_cmd
def test_bat_provider_folder_without_git_metadata(bat_provider):
    """Kept when the download fails, replaced (by renaming) when it works."""
    checkout = bat_provider["checkout"]
    (checkout / "server/build").mkdir(parents=True)
    (checkout / "server/build/main.js").write_text("working build\n", encoding="utf-8")
    result, steps = bat_provider["run"](GIT_CONFIG_KEY_0=bat_provider["offline"])
    assert result.returncode == 3 and "Could not download" in result.stdout
    assert (checkout / "server/build/main.js").read_text(encoding="utf-8") == "working build\n"
    assert _leftovers(checkout) == [] and steps == []
    result, steps = bat_provider["run"]()
    assert result.returncode == 0, result.stdout + result.stderr
    assert _commit_of(checkout, "HEAD") == _commit_of(bat_provider["src"], "2.0.0")
    assert _leftovers(checkout) == []


@needs_cmd
def test_bat_provider_folder_in_use_is_kept_whole(bat_provider):
    """A held file makes the rename fail as a whole: nothing is half deleted, and the advice fits."""
    checkout = bat_provider["checkout"]
    (checkout / "server/build").mkdir(parents=True)
    (checkout / "server/build/main.js").write_text("working build\n", encoding="utf-8")
    (checkout / "server/build/utils.js").write_text("// utils\n", encoding="utf-8")
    with open(checkout / "server/build/main.js", "rb"):
        result, steps = bat_provider["run"]()
    assert result.returncode == 3
    assert "is in use" in result.stdout and "Could not download" not in result.stdout
    assert (checkout / "server/build/main.js").read_text(encoding="utf-8") == "working build\n"
    assert (checkout / "server/build/utils.js").exists()
    assert _leftovers(checkout) == []


@needs_cmd
def test_bat_provider_leftovers_of_an_interrupted_run_are_removed(bat_provider):
    bat_provider["run"]()
    _make_leftovers(bat_provider["checkout"])
    result, steps = bat_provider["run"]()
    assert result.returncode == 0 and "already built" in result.stdout
    assert _leftovers(bat_provider["checkout"]) == []


@needs_cmd
def test_bat_provider_build_in_use_is_kept_whole(bat_provider):
    """E.g. the running provider works in build: the rebuild stops before anything is deleted."""
    bat_provider["run"]()
    checkout = bat_provider["checkout"]
    shutil.rmtree(checkout / "server/node_modules")  # forces a rebuild
    with open(checkout / "server/build/main.js", "rb"):
        result, steps = bat_provider["run"]()
    assert result.returncode == 3
    assert "build folder" in result.stdout and "is in use" in result.stdout
    assert "did not complete" not in result.stdout
    assert steps == []  # npm ci never started
    assert (checkout / "server/build/main.js").exists()
    assert (checkout / "server/build/.ultrasinger-commit").exists()
    assert _leftovers(checkout) == []


@needs_cmd
def test_bat_provider_needs_a_recent_node(bat_provider):
    result, steps = bat_provider["run"](NODE_STUB_MAJOR="18")
    assert result.returncode == 2
    assert "Node.js 22 or newer is needed" in result.stdout and "found v18.0.0" in result.stdout
    assert steps == [] and not bat_provider["checkout"].exists()


# --------------------------------------------------------------------------
# update.bat and the Windows install scripts, run for real with cmd.exe
# (Windows only). uv is a small compiled program here: a batch file called
# without "call" would end the script that calls it.
# --------------------------------------------------------------------------

STUB_UV_CS = r"""
using System; using System.IO;
class Uv {
    static bool On(string name) { return !string.IsNullOrEmpty(Environment.GetEnvironmentVariable(name)); }
    static int Main(string[] args) {
        string call = string.Join(" ", args);
        File.AppendAllText(Environment.GetEnvironmentVariable("UV_STUB_LOG"), call + "\n");
        if (call == "lock --upgrade-package yt-dlp" && On("UV_STUB_FAIL_UPGRADE")) return 1;
        if (call == "lock" && On("UV_STUB_FAIL_LOCK")) return 1;
        if (call.StartsWith("sync")) {
            Directory.CreateDirectory(".venv");
            if (On("UV_STUB_FAIL_SYNC")) { File.WriteAllText(@".venv\partial", ""); return 1; }
            File.WriteAllText(@".venv\marker-new", "");
        }
        return 0;
    }
}
"""


@pytest.fixture(scope="session")
def stub_uv_exe(tmp_path_factory):
    """A folder with a uv.exe that behaves like STUB_UV, built with the C# compiler that comes with Windows."""
    csc = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Microsoft.NET" / "Framework64" / "v4.0.30319" / "csc.exe"
    if not csc.is_file():
        pytest.skip("needs the C# compiler that comes with Windows (csc.exe) to build a stand-in for uv")
    bin_dir = tmp_path_factory.mktemp("uv_stub")
    (bin_dir / "uv.cs").write_text(STUB_UV_CS, encoding="utf-8")
    subprocess.run([str(csc), "/nologo", f"/out:{bin_dir / 'uv.exe'}", str(bin_dir / "uv.cs")], check=True,
                   capture_output=True)
    return bin_dir


def _bat_runner(tmp_path: Path, bin_dir: Path, cwd: Path):
    """Runs batch scripts with the stubs first on PATH and a home without the real uv in it."""
    log = tmp_path / "uv.log"
    for name in ("home", "tmp"):
        (tmp_path / name).mkdir(exist_ok=True)
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", USERPROFILE=str(tmp_path / "home"),
               TEMP=str(tmp_path / "tmp"), TMP=str(tmp_path / "tmp"), UV_STUB_LOG=str(log), UV_LINK_MODE="copy")
    for key in UV_STUB_SWITCHES:
        env.pop(key, None)

    def run(script: Path, **extra):
        log.unlink(missing_ok=True)
        result = subprocess.run(["cmd.exe", "/d", "/c", str(script)], cwd=cwd, env=dict(env, **extra),
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=120)
        calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return result, calls

    return run


@pytest.fixture
def bat_update(tmp_path, stub_uv_exe):
    """update.bat in an "installation" whose upstream has moved on; uv is a stub."""
    app = _installation(tmp_path, "update.bat")
    run = _bat_runner(tmp_path, stub_uv_exe, app)
    return app, lambda **extra: run(app / "install" / "update.bat", **extra)


@needs_cmd
def test_bat_update_cuda_install_keeps_its_cuda_variant(bat_update):
    app, run = bat_update
    _make_cuda(app, "cu126")
    _change_upstream_pyproject(app, 'new = "upstream"\n')
    result, calls = run()
    assert result.returncode == 0, result.stdout + result.stderr
    pyproject = (app / "pyproject.toml").read_text(encoding="utf-8")
    assert "whl/cu126" in pyproject and 'new = "upstream"' in pyproject
    assert (app / "uv.lock").read_text(encoding="utf-8") == "lock v2\n"  # the pull went through
    assert _protected(app)
    assert "local changes besides the CUDA index" not in result.stdout
    assert calls == ["lock --upgrade-package yt-dlp", "sync --extra gui --extra scoring --extra potoken"]


@needs_cmd
def test_bat_update_cuda_install_keeps_a_copy_of_other_local_pyproject_changes(bat_update):
    app, run = bat_update
    _make_cuda(app, extra='extra = "mine"\n')
    result, calls = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "extra" not in (app / "pyproject.toml").read_text(encoding="utf-8")
    kept = re.search(r"kept as\s+(\S+pyproject\.toml)", result.stdout)
    assert kept, result.stdout
    assert 'extra = "mine"' in Path(kept.group(1)).read_text(encoding="utf-8")


@needs_cmd
@pytest.mark.parametrize("more_upstream", [False, True], ids=["up-to-date", "new-commits"])
def test_bat_update_after_a_failed_update_reports_no_local_changes(bat_update, more_upstream):
    """A failed update puts the pyproject.toml of before its pull back; the next run must not call that an edit."""
    app, run = bat_update
    _make_cuda(app, "cu126")
    _change_upstream_pyproject(app, 'new = "upstream"\n')
    result, calls = run(UV_STUB_FAIL_SYNC="1")
    assert result.returncode == 1
    assert "restoring the previous CUDA configuration" in result.stdout
    assert "The update is incomplete" in result.stdout and "install\\update.bat again" in result.stdout
    if more_upstream:
        _change_upstream_pyproject(app, 'new = "upstream"\nnewer = "upstream"\n')
    result, calls = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "local changes besides the CUDA index" not in result.stdout
    pyproject = (app / "pyproject.toml").read_text(encoding="utf-8")
    assert "whl/cu126" in pyproject and 'new = "upstream"' in pyproject
    assert ('newer = "upstream"' in pyproject) == more_upstream
    assert _protected(app)


@needs_cmd
def test_bat_update_reports_no_incomplete_update_when_the_pull_fails(bat_update, tmp_path):
    app, run = bat_update
    _make_cuda(app)
    _git(app, "remote", "set-url", "origin", str(tmp_path / "missing.git"))
    result, calls = run(PULLED="1", KEEP_BACKUP="1")  # not even when the environment has these variables
    assert result.returncode == 1
    assert "restoring the previous CUDA configuration" in result.stdout
    assert "The update is incomplete" not in result.stdout
    assert calls == []
    assert _protected(app) and "whl/cu128" in (app / "pyproject.toml").read_text(encoding="utf-8")
    assert not list((tmp_path / "tmp").glob("ultrasinger_update_*"))  # no copy kept


@needs_cmd
def test_bat_update_carries_on_right_after_the_pull_in_the_pulled_file(bat_update):
    """The running update.bat continues at its byte offset in the file its pull replaced (see the hash test)."""
    app, run = bat_update
    marker = "MARKER - THE PULLED UPDATE.BAT GOES ON HERE"
    seed = app.parent / "seed"
    data = (seed / "install" / "update.bat").read_bytes()
    end = data.index(UPDATE_BAT_PULL_LINE) + len(UPDATE_BAT_PULL_LINE)
    (seed / "install" / "update.bat").write_bytes(data[:end] + f"echo {marker}\r\n".encode() + data[end:])
    _git(seed, "commit", "-qam", "update.bat")
    _git(seed, "push", "-q", "origin", "main")
    _make_cuda(app, "cu126")
    result, calls = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count(marker) == 1, result.stdout
    assert result.stdout.count("Pulling latest changes") == 1 and result.stdout.count("Update completed.") == 1
    assert "whl/cu126" in (app / "pyproject.toml").read_text(encoding="utf-8")
    assert calls == ["lock --upgrade-package yt-dlp", "sync --extra gui --extra scoring --extra potoken"]
    # cmd.exe prints its errors to stderr; apart from them, only git's fetch and checkout notes end up there
    unexpected = [line for line in result.stderr.splitlines()
                  if line.strip() and not line.startswith(("From ", "Updated ")) and " -> " not in line]
    assert unexpected == [], result.stderr


@pytest.fixture
def bat_install(tmp_path, stub_uv_exe):
    """A CPU installation with a working .venv for the Windows install scripts; uv and the provider are stubs."""
    app = tmp_path / "app"
    for rel in WINDOWS_INSTALL_SCRIPTS:
        script = app / "install" / rel
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_bytes((INSTALL / rel).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    helper = app / "install" / "helpers" / "setup_potoken_provider.bat"
    helper.parent.mkdir(parents=True)
    helper.write_bytes(b"@echo off\r\necho stub provider\r\nexit /b 2\r\n")
    (app / "pyproject.toml").write_text('url = "https://download.pytorch.org/whl/cpu"\n', encoding="utf-8")
    (app / "uv.lock").write_text("lock cpu\n", encoding="utf-8")
    _git(tmp_path, "init", "-q", str(app))
    _git(app, "add", "-A")
    _git(app, "commit", "-qm", "init")
    (app / ".venv").mkdir()
    (app / ".venv" / "marker-old").write_text("old", encoding="utf-8")
    run = _bat_runner(tmp_path, stub_uv_exe, tmp_path)
    return app, lambda script, **extra: run(app / "install" / script, **extra)


@needs_cmd
@pytest.mark.parametrize("script", WINDOWS_INSTALL_SCRIPTS)
def test_bat_install_removes_its_incomplete_environment_when_there_was_none(bat_install, script):
    """A first installation that fails leaves no half-built .venv behind for the launcher to start."""
    app, run = bat_install
    shutil.rmtree(app / ".venv")
    result, calls = run(script, UV_STUB_FAIL_SYNC="1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "restoring the previous installation" in result.stdout
    assert not (app / ".venv").exists() and not (app / ".venv.old").exists()
    assert "whl/cpu" in (app / "pyproject.toml").read_text(encoding="utf-8")


@needs_cmd
def test_bat_install_keeps_the_previous_environment_when_syncing_fails(bat_install):
    """The environment put back is not taken for an incomplete one of this run."""
    app, run = bat_install
    result, calls = run("CPU/windows_cpu.bat", UV_STUB_FAIL_SYNC="1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert _venv(app) == ["marker-old"]
    assert not (app / ".venv.old").exists()


@needs_cmd
def test_bat_install_names_a_leftover_backup_that_is_in_use(bat_install):
    """A stale .venv.old that cannot be removed is reported as such, not as .venv being in use."""
    app, run = bat_install
    (app / ".venv" / COMPLETE_MARKER).write_text("", encoding="utf-8")
    (app / ".venv.old").mkdir()
    with open(app / ".venv.old" / "held", "wb"):
        result, calls = run("CPU/windows_cpu.bat")
    assert result.returncode == 1
    assert ".venv.old, the backup of an environment left behind" in result.stdout
    assert "(.venv) is in use" not in result.stdout
    assert _venv(app) == ["marker-old"] and (app / ".venv" / COMPLETE_MARKER).exists()
    assert not any(call.startswith("sync") for call in calls)
