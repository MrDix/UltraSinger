"""Tests for downloading segmentation models from a GitHub repository (model_source).

GitHub is simulated: urllib.request.urlopen is replaced by a fake that serves
file metadata (with a SHA) and raw content.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from modules.Segmentation import model_source as ms


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class FakeGitHub:
    def __init__(self, sha="sha1", content=b"MODEL-1", fail=None):
        self.sha, self.content, self.fail = sha, content, fail
        self.requests = []

    def urlopen(self, req, timeout=None):
        self.requests.append(req)
        if self.fail:
            raise self.fail
        if req.get_header("Accept") == "application/vnd.github.raw":
            return _Resp(self.content)
        return _Resp(json.dumps({"type": "file", "sha": self.sha}).encode())

    def downloads(self):
        return sum(1 for r in self.requests if r.get_header("Accept") == "application/vnd.github.raw")


@pytest.fixture
def github(monkeypatch):
    fake = FakeGitHub()
    monkeypatch.setattr(ms.urllib.request, "urlopen", fake.urlopen)
    monkeypatch.delenv(ms.TOKEN_ENV, raising=False)
    return fake


class TestParseSpec:
    @pytest.mark.parametrize("spec, expected", [
        ("owner/repo", ("owner", "repo", "segmentation.pt")),
        ("owner/repo/models/v2.pt", ("owner", "repo", "models/v2.pt")),
        ("https://github.com/owner/repo", ("owner", "repo", "segmentation.pt")),
        ("https://github.com/owner/repo/blob/main/models/v2.pt", ("owner", "repo", "models/v2.pt")),
        (" owner/repo.git ", ("owner", "repo", "segmentation.pt")),
    ])
    def test_valid(self, spec, expected):
        assert ms.parse_repo_spec(spec) == expected

    @pytest.mark.parametrize("spec", ["", "noslash", "a b/c"])
    def test_invalid(self, spec):
        with pytest.raises(ValueError):
            ms.parse_repo_spec(spec)


class TestFetch:
    def test_downloads_and_caches(self, github, tmp_path):
        path = ms.fetch_model("owner/repo", cache=tmp_path)
        assert open(path, "rb").read() == b"MODEL-1"
        assert github.downloads() == 1
        # second call: same SHA -> no new download
        assert ms.fetch_model("owner/repo", cache=tmp_path) == path
        assert github.downloads() == 1

    def test_new_version_is_downloaded(self, github, tmp_path):
        path = ms.fetch_model("owner/repo", cache=tmp_path)
        github.sha, github.content = "sha2", b"MODEL-2"
        assert ms.fetch_model("owner/repo", cache=tmp_path) == path
        assert open(path, "rb").read() == b"MODEL-2"
        assert github.downloads() == 2

    def test_offline_uses_cache(self, github, tmp_path, capsys):
        path = ms.fetch_model("owner/repo", cache=tmp_path)
        github.fail = urllib.error.URLError("no network")
        assert ms.fetch_model("owner/repo", cache=tmp_path) == path
        assert "using the cached copy" in capsys.readouterr().out

    def test_error_without_cache_returns_none(self, github, tmp_path, capsys):
        github.fail = urllib.error.HTTPError("u", 404, "Not Found", {}, None)
        assert ms.fetch_model("owner/repo", cache=tmp_path) is None
        assert "access token" in capsys.readouterr().out

    def test_token_sent_as_bearer(self, github, tmp_path):
        ms.fetch_model("owner/repo", token="secret-token", cache=tmp_path)
        assert all(r.get_header("Authorization") == "Bearer secret-token" for r in github.requests)

    def test_token_from_environment(self, github, tmp_path, monkeypatch):
        monkeypatch.setenv(ms.TOKEN_ENV, "env-token")
        ms.fetch_model("owner/repo", cache=tmp_path)
        assert github.requests[0].get_header("Authorization") == "Bearer env-token"

    def test_no_token_no_header(self, github, tmp_path):
        ms.fetch_model("owner/repo", cache=tmp_path)
        assert github.requests[0].get_header("Authorization") is None

    def test_failed_download_leaves_no_partial_file(self, github, tmp_path, monkeypatch):
        def broken(req, timeout=None):
            if req.get_header("Accept") == "application/vnd.github.raw":
                raise urllib.error.URLError("connection reset")
            return _Resp(json.dumps({"type": "file", "sha": "s"}).encode())
        monkeypatch.setattr(ms.urllib.request, "urlopen", broken)
        assert ms.fetch_model("owner/repo", cache=tmp_path) is None
        assert not [p for p in tmp_path.rglob("*") if p.is_file()]

    def test_directory_instead_of_file(self, github, tmp_path, monkeypatch):
        monkeypatch.setattr(ms.urllib.request, "urlopen",
                            lambda req, timeout=None: _Resp(json.dumps([{"name": "x"}]).encode()))
        assert ms.fetch_model("owner/repo/models", cache=tmp_path) is None

    def test_invalid_spec(self, tmp_path):
        assert ms.fetch_model("not a repo", cache=tmp_path) is None

    def test_cache_dir_is_per_user(self):
        assert ms.cache_dir().name == "models"
