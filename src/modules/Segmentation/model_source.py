"""Fetch a segmentation model from a GitHub repository (public or private).

A repository is given as ``owner/repo`` or ``owner/repo/path/to/model.pt``
(default file: ``segmentation.pt``). The file is downloaded through the
GitHub REST API - with an access token for private repositories - and cached
locally; it is only downloaded again when its content (git blob SHA) changed.
Without network access the cached copy is used.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, gold_highlighted

DEFAULT_MODEL_FILE = "segmentation.pt"
API_ROOT = "https://api.github.com"
TIMEOUT_S = 60
TOKEN_ENV = "ULTRASINGER_MODEL_TOKEN"

_SPEC_RE = re.compile(r"^(?:https?://github\.com/)?([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/(.+))?/?$")


def parse_repo_spec(spec: str) -> tuple[str, str, str]:
    """``owner/repo[/path]`` (or a github.com URL) -> ``(owner, repo, path)``."""
    m = _SPEC_RE.match((spec or "").strip())
    if not m:
        raise ValueError(f"not a repository reference: {spec!r} (expected owner/repo[/path])")
    owner, repo, path = m.group(1), m.group(2), (m.group(3) or DEFAULT_MODEL_FILE).strip("/")
    path = re.sub(r"^(?:blob|tree|raw)/[^/]+/", "", path)  # tolerate pasted browser URLs
    return owner, repo, path


def cache_dir() -> Path:
    """Per-user cache folder for downloaded models."""
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"]) / "UltraSinger"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ultrasinger"
    return base / "models"


def _request(url: str, token: str | None, accept: str) -> urllib.request.Request:
    headers = {"Accept": accept, "User-Agent": "UltraSinger", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def _remote_sha(owner: str, repo: str, path: str, token: str | None) -> str:
    url = f"{API_ROOT}/repos/{owner}/{repo}/contents/{urllib.parse.quote(path)}"
    with urllib.request.urlopen(_request(url, token, "application/vnd.github+json"), timeout=TIMEOUT_S) as r:
        meta = json.loads(r.read().decode("utf-8"))
    if not isinstance(meta, dict) or meta.get("type") != "file":
        raise ValueError(f"{path} is not a file in {owner}/{repo}")
    return str(meta["sha"])


def _download(owner: str, repo: str, path: str, token: str | None, target: Path) -> None:
    url = f"{API_ROOT}/repos/{owner}/{repo}/contents/{urllib.parse.quote(path)}"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, \
                urllib.request.urlopen(_request(url, token, "application/vnd.github.raw"), timeout=TIMEOUT_S) as r:
            while chunk := r.read(1 << 20):
                out.write(chunk)
        os.replace(tmp, target)  # atomic: never leave a half-written model behind
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _describe_error(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        if e.code in (401, 403):
            return f"HTTP {e.code} - access denied (check the access token)"
        if e.code == 404:
            return "HTTP 404 - repository or file not found (private repositories need an access token)"
        return f"HTTP {e.code}"
    return str(e) or type(e).__name__


def fetch_model(spec: str, token: str | None = None, cache: Path | None = None) -> str | None:
    """Local path of the (cached) model from repository ``spec``, or ``None``.

    Downloads the file when it is not cached yet or its SHA changed. On any
    network or access error the cached copy is used if there is one.
    """
    token = token or os.environ.get(TOKEN_ENV) or None
    try:
        owner, repo, path = parse_repo_spec(spec)
    except ValueError as e:
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} {e}")
        return None
    folder = (cache or cache_dir()) / f"{owner}__{repo}"
    target = folder / path.replace("/", "__")
    sha_file = target.with_name(target.name + ".sha")
    cached_sha = sha_file.read_text(encoding="utf-8").strip() if sha_file.exists() and target.exists() else None
    try:
        sha = _remote_sha(owner, repo, path, token)
        if sha != cached_sha:
            print(f"{ULTRASINGER_HEAD} Downloading segmentation model {blue_highlighted(f'{owner}/{repo}/{path}')}")
            _download(owner, repo, path, token, target)
            sha_file.write_text(sha, encoding="utf-8")
        return str(target)
    except Exception as e:  # noqa: BLE001 - fall back to the cache, never abort the conversion
        if cached_sha is not None:
            print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} could not check {owner}/{repo} for a newer "
                  f"model ({_describe_error(e)}) - using the cached copy")
            return str(target)
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} could not download the segmentation model "
              f"from {owner}/{repo} ({_describe_error(e)})")
        return None
