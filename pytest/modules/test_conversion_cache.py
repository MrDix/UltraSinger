"""Tests for the cache folder of a conversion (modules.conversion_cache) and its use in UltraSinger.py."""

import os
import time
from pathlib import Path

import pytest

import src.UltraSinger as us
from src.UltraSinger import init_settings

cc = us.conversion_cache  # the module object UltraSinger.py uses

_DAY_AGO = time.time() - cc.STALE_AFTER_S - 60


def _age(path: Path) -> None:
    os.utime(path, (_DAY_AGO, _DAY_AGO))


class TestDefaultRoot:
    def test_windows_local_app_data(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cc.sys, "platform", "win32")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert cc.default_root() == tmp_path / "UltraSinger" / "cache"

    def test_xdg_cache_home_elsewhere(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cc.sys, "platform", "linux")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert cc.default_root() == tmp_path / "ultrasinger" / "cache"


class TestCreate:
    def test_new_folder_per_conversion_named_after_the_song(self, tmp_path):
        a = cc.create("Artist - Title", str(tmp_path))
        b = cc.create("Artist - Title", str(tmp_path))
        assert a != b and os.path.isdir(a) and os.path.isdir(b)
        assert Path(a).parent == tmp_path
        assert Path(a).name.startswith("Artist - Title ") and Path(a).name.endswith(cc.FOLDER_SUFFIX)

    def test_long_song_names_are_cut(self, tmp_path):
        name = Path(cc.create("x" * 200, str(tmp_path))).name
        assert name.startswith("x" * cc.NAME_CHARS + " ") and len(name) == cc.NAME_CHARS + 9 + len(cc.FOLDER_SUFFIX)

    def test_default_root_is_made(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cc, "default_root", lambda: tmp_path / "local" / "cache")
        assert Path(cc.create("Song")).parent == tmp_path / "local" / "cache"

    def test_leftovers_are_removed_first(self, tmp_path):
        old = tmp_path / f"Old 0a1b2c3d{cc.FOLDER_SUFFIX}"
        old.mkdir()
        _age(old)
        cc.create("Song", str(tmp_path))
        assert not old.exists()


class TestLeftovers:
    def test_only_old_cache_folders_are_removed(self, tmp_path):
        old = tmp_path / f"Old 0a1b2c3d{cc.FOLDER_SUFFIX}"
        fresh = tmp_path / f"Fresh 0a1b2c3e{cc.FOLDER_SUFFIX}"
        other = tmp_path / "Holiday photos"
        for folder in (old, fresh, other):
            (folder / "separated").mkdir(parents=True)
            (folder / "separated" / "vocals.wav").write_bytes(b"v")
        a_file = tmp_path / f"notes{cc.FOLDER_SUFFIX}"
        a_file.write_text("x")
        for path in (old, other, a_file):
            _age(path)
        assert cc.remove_stale_folders(tmp_path) == 1
        assert not old.exists()
        assert fresh.exists() and other.exists() and a_file.exists()

    def test_missing_root(self, tmp_path):
        assert cc.remove_stale_folders(tmp_path / "missing") == 0


class TestFinish:
    @staticmethod
    def _cache(tmp_path):
        folder = Path(cc.create("Song", str(tmp_path / "root")))
        (folder / "separated").mkdir()
        (folder / "separated" / "vocals.wav").write_bytes(b"v")
        (folder / "pitch.json").write_text("{}")
        song = tmp_path / "Song"
        song.mkdir()
        return folder, song

    def test_deleted_without_keep(self, tmp_path):
        folder, song = self._cache(tmp_path)
        assert cc.finish(str(folder), str(song), keep=False) is None
        assert not folder.exists() and not (song / "cache").exists()

    def test_kept_cache_moves_into_the_song_folder(self, tmp_path):
        folder, song = self._cache(tmp_path)
        assert cc.finish(str(folder), str(song), keep=True) == str(song / "cache")
        assert not folder.exists()
        assert (song / "cache" / "separated" / "vocals.wav").read_bytes() == b"v"
        assert (song / "cache" / "pitch.json").exists()

    def test_kept_cache_stays_when_the_song_folder_has_one(self, tmp_path):
        folder, song = self._cache(tmp_path)
        (song / "cache").mkdir()
        assert cc.finish(str(folder), str(song), keep=True) == str(folder)
        assert (folder / "pitch.json").exists()


class TestConversion:
    """The cache folder of a conversion: outside the song folder, deleted or kept at the end."""

    @pytest.fixture
    def song(self, tmp_path, monkeypatch):
        settings = init_settings(["-i", "test.mp3", "--cache_folder", str(tmp_path / "root")])
        song = tmp_path / "out" / "Artist - Title"
        song.mkdir(parents=True)
        monkeypatch.setattr(settings, "output_folder_path", str(song))
        monkeypatch.setattr(settings, "keep_cache", False)
        monkeypatch.setattr(settings, "cache_override_path", None)
        yield settings, song
        init_settings(["-i", "test.mp3"])  # reset the shared settings

    def test_cache_folder_is_outside_the_song_folder(self, song, tmp_path, capsys):
        settings, folder = song
        cache = Path(us.setup_cache_folder())
        assert cache.parent == tmp_path / "root" and cache.name.startswith("Artist - Title ")
        assert str(cache) in capsys.readouterr().out
        us.finish_cache_folder(str(cache))
        assert not cache.exists() and not (folder / "cache").exists()

    def test_kept_cache_ends_in_the_song_folder(self, song, monkeypatch):
        settings, folder = song
        monkeypatch.setattr(settings, "keep_cache", True)
        cache = Path(us.setup_cache_folder())
        (cache / "pitch.json").write_text("{}")
        us.finish_cache_folder(str(cache))
        assert (folder / "cache" / "pitch.json").exists() and not cache.exists()

    def test_song_folder_when_no_cache_folder_can_be_made(self, song, monkeypatch, capsys):
        settings, folder = song

        def fail(song_name, root=None):
            raise PermissionError(13, "Permission denied")
        monkeypatch.setattr(cc, "create", fail)
        cache = us.setup_cache_folder()
        assert cache == str(folder / "cache") and "Permission denied" in capsys.readouterr().out
        os.makedirs(cache)
        us.finish_cache_folder(cache)
        assert not os.path.exists(cache)

    def test_fixed_cache_folder_is_used_as_is(self, song, monkeypatch, tmp_path):
        settings, _ = song
        fixed = tmp_path / "fixed"
        fixed.mkdir()
        monkeypatch.setattr(settings, "cache_override_path", str(fixed))
        assert us.setup_cache_folder() == str(fixed)
        monkeypatch.setattr(settings, "keep_cache", True)
        us.finish_cache_folder(str(fixed))
        assert fixed.exists()
        monkeypatch.setattr(settings, "keep_cache", False)
        us.finish_cache_folder(str(fixed))
        assert not fixed.exists()


class TestCacheFolderFlag:
    def test_flag_and_reset(self):
        assert init_settings(["-i", "test.mp3", "--cache_folder", "D:/cache"]).cache_folder == "D:/cache"
        assert init_settings(["-i", "test.mp3"]).cache_folder is None

    def test_class_default(self):
        from src.Settings import Settings
        assert Settings().cache_folder is None
