"""Tests for the install / update scripts in install/ and the GUI launchers.

Every script that locks the dependencies must lock the newest yt-dlp (video
platforms change often, so the pinned yt-dlp soon fails to download) and fall
back to the pinned version when that is not possible. update.sh and the
PO-token provider helper are also run for real against throw-away git
repositories with stubs for uv, node and npm.
"""

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


def _write_script(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("\r\n", "\n"), encoding="utf-8", newline="\n")
    path.chmod(0o755)


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
mkdir -p "$out" && echo built > "$out/main.js"
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


@needs_bash
def test_provider_with_missing_dependencies_is_rebuilt(provider):
    """Security software may delete node_modules: build and stamp alone are not enough."""
    provider["run"]()
    shutil.rmtree(provider["checkout"] / "server/node_modules")
    result, steps = provider["run"]()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "already built" not in result.stdout
    assert steps == BUILD_STEPS


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


def _leftovers(checkout: Path) -> list[str]:
    """Temporary folders of the helper that must not stay behind."""
    return [s for s in (".new", ".old") if checkout.with_name(checkout.name + s).exists()] \
        + (["build.new"] if (checkout / "server" / "build.new").exists() else [])


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
def test_bat_provider_needs_a_recent_node(bat_provider):
    result, steps = bat_provider["run"](NODE_STUB_MAJOR="18")
    assert result.returncode == 2
    assert "Node.js 22 or newer is needed" in result.stdout and "found v18.0.0" in result.stdout
    assert steps == [] and not bat_provider["checkout"].exists()
