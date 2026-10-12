"""Tests for the cache folder of a conversion (modules.conversion_cache) and its use in UltraSinger.py."""

import os
import shutil
import sys
import time
from pathlib import Path

import pytest

import src.UltraSinger as us
from src.UltraSinger import init_settings

cc = us.conversion_cache  # the module object UltraSinger.py uses

_DAY_AGO = time.time() - cc.STALE_AFTER_S - 60


def _age(path: Path) -> None:
    """Make ``path`` and everything in it look a day old."""
    for p in [*path.rglob("*"), path] if path.is_dir() else [path]:
        os.utime(p, (_DAY_AGO, _DAY_AGO))


def _make_dir_link(link: Path, target: Path) -> None:
    """A link to a folder: a junction on Windows (no rights needed), else a symbolic link."""
    if sys.platform == "win32":
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def _leftover(root: Path, name: str, marked: bool = True) -> Path:
    """A cache folder as a cancelled conversion leaves it (made a day ago)."""
    folder = root / f"{name} 0a1b2c3d{cc.FOLDER_SUFFIX}"
    (folder / "separated").mkdir(parents=True)
    (folder / "separated" / "vocals.wav").write_bytes(b"v")
    if marked:
        (folder / cc.MARKER).touch()
    _age(folder)
    return folder


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
        assert (Path(a) / cc.MARKER).is_file()  # marks it as made by UltraSinger

    def test_long_song_names_are_cut(self, tmp_path):
        name = Path(cc.create("x" * 200, str(tmp_path))).name
        assert name.startswith("x" * cc.NAME_CHARS + " ") and len(name) == cc.NAME_CHARS + 9 + len(cc.FOLDER_SUFFIX)

    def test_default_root_is_made(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cc, "default_root", lambda: tmp_path / "local" / "cache")
        assert Path(cc.create("Song")).parent == tmp_path / "local" / "cache"

    def test_leftovers_are_removed_first(self, tmp_path):
        old = _leftover(tmp_path, "Old")
        cc.create("Song", str(tmp_path))
        assert not old.exists()


class TestLeftovers:
    def test_only_old_cache_folders_are_removed(self, tmp_path):
        fresh = Path(cc.create("Fresh", str(tmp_path)))  # (cleans up itself: make the leftover after it)
        old = _leftover(tmp_path, "Old")
        other = tmp_path / "Holiday photos"
        (other / "2024").mkdir(parents=True)
        _age(other)
        a_file = tmp_path / f"notes{cc.FOLDER_SUFFIX}"
        a_file.write_text("x")
        _age(a_file)
        assert cc.remove_stale_folders(tmp_path) == 1
        assert not old.exists()
        assert fresh.exists() and other.exists() and a_file.exists()

    def test_folder_without_marker_is_not_removed(self, tmp_path):
        # Same name ending, but not made by UltraSinger (e.g. in a chosen --cache_folder)
        foreign = _leftover(tmp_path, "Foreign", marked=False)
        assert cc.remove_stale_folders(tmp_path) == 0 and foreign.exists()

    def test_recent_change_deep_inside_keeps_the_folder(self, tmp_path):
        # A long conversion writes into a subfolder: the folder's own time stays old
        busy = _leftover(tmp_path, "Busy")
        (busy / "separated" / "vocals.wav").write_bytes(b"newer")
        os.utime(busy, (_DAY_AGO, _DAY_AGO))
        assert cc.remove_stale_folders(tmp_path) == 0 and busy.exists()

    def test_missing_root(self, tmp_path):
        assert cc.remove_stale_folders(tmp_path / "missing") == 0

    def test_linked_folder_is_not_entered(self, tmp_path):
        # A link (junction on Windows) named like a cache folder, to a folder with the marker
        target = tmp_path / "elsewhere"
        target.mkdir()
        (target / cc.MARKER).touch()
        (target / "precious.txt").write_text("x")
        root = tmp_path / "root"
        root.mkdir()
        link = root / f"Linked 0a1b2c3d{cc.FOLDER_SUFFIX}"
        _make_dir_link(link, target)
        _age(target)
        assert cc.remove_stale_folders(root) == 0
        assert (target / "precious.txt").exists() and (target / cc.MARKER).exists()

    def test_link_inside_a_leftover_is_removed_not_followed(self, tmp_path):
        target = tmp_path / "elsewhere"
        target.mkdir()
        (target / "precious.txt").write_text("x")
        old = _leftover(tmp_path / "root", "Old")
        _make_dir_link(old / "link", target)
        _age(old)
        assert cc.remove_stale_folders(tmp_path / "root") == 1
        assert not old.exists() and (target / "precious.txt").exists()


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

    def test_kept_cache_is_never_a_leftover(self, tmp_path):
        folder, song = self._cache(tmp_path)
        kept = Path(cc.finish(str(folder), str(song), keep=True))
        assert not (kept / cc.MARKER).exists()

    def test_kept_cache_gets_an_unused_name(self, tmp_path):
        folder, song = self._cache(tmp_path)
        (song / "cache").mkdir()
        (song / "cache (1)").mkdir()
        assert cc.finish(str(folder), str(song), keep=True) == str(song / "cache (2)")
        assert (song / "cache (2)" / "pitch.json").exists() and not folder.exists()

    def test_kept_cache_that_cannot_be_moved_stays_and_is_not_removed(self, tmp_path, monkeypatch):
        folder, song = self._cache(tmp_path)

        def fail(src, dst):
            raise PermissionError(13, "Permission denied")
        monkeypatch.setattr(cc.shutil, "move", fail)
        assert cc.finish(str(folder), str(song), keep=True) == str(folder)
        _age(folder)
        assert cc.remove_stale_folders(folder.parent) == 0 and (folder / "pitch.json").exists()

    def test_kept_cache_made_in_the_song_folder_still_moves_to_cache(self, tmp_path):
        # --cache_folder pointing at the song folder: the marked cache folder is made there
        song = tmp_path / "Song"
        folder = Path(cc.create("Song", str(song)))
        (folder / "pitch.json").write_text("{}")
        assert folder.parent == song
        assert cc.finish(str(folder), str(song), keep=True) == str(song / "cache")
        assert (song / "cache" / "pitch.json").exists() and not (song / "cache" / cc.MARKER).exists()
        assert not folder.exists()

    def test_marker_outlives_a_partial_delete(self, tmp_path, monkeypatch):
        folder, song = self._cache(tmp_path)
        rmtree = shutil.rmtree

        def in_use(path, *args, **kwargs):
            if os.path.basename(path) == "separated":
                raise PermissionError(13, "in use")
            return rmtree(path, *args, **kwargs)
        monkeypatch.setattr(cc.shutil, "rmtree", in_use)
        assert cc.finish(str(folder), str(song), keep=False) is None
        # what is left is still marked, so a later conversion removes it
        assert (folder / cc.MARKER).is_file() and (folder / "separated").exists()
        monkeypatch.setattr(cc.shutil, "rmtree", rmtree)
        _age(folder)
        assert cc.remove_stale_folders(folder.parent) == 1 and not folder.exists()


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
        assert cache == str(folder / "cache") and os.path.isdir(cache)
        assert "Permission denied" in capsys.readouterr().out
        us.finish_cache_folder(cache)
        assert not os.path.exists(cache)

    def test_fallback_never_takes_an_existing_folder(self, song, monkeypatch):
        settings, folder = song
        (folder / "cache").mkdir()
        (folder / "cache" / "kept.json").write_text("{}")  # e.g. a cache kept before

        def fail(song_name, root=None):
            raise PermissionError(13, "Permission denied")
        monkeypatch.setattr(cc, "create", fail)
        cache = us.setup_cache_folder()
        assert cache == str(folder / "cache (1)")
        us.finish_cache_folder(cache)
        assert not os.path.exists(cache) and (folder / "cache" / "kept.json").exists()

    def test_kept_fallback_cache_stays_in_place(self, song, monkeypatch):
        settings, folder = song
        monkeypatch.setattr(settings, "keep_cache", True)

        def fail(song_name, root=None):
            raise PermissionError(13, "Permission denied")
        monkeypatch.setattr(cc, "create", fail)
        cache = us.setup_cache_folder()
        (Path(cache) / "pitch.json").write_text("{}")
        us.finish_cache_folder(cache)
        assert (folder / "cache" / "pitch.json").exists() and not (folder / "cache (1)").exists()

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


class TestContentKey:
    def test_key_follows_the_content_not_the_name(self, tmp_path):
        a, b, c = tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "c.wav"
        a.write_bytes(b"vocal")
        b.write_bytes(b"vocal")
        c.write_bytes(b"vocal, denoised")
        assert cc.content_key(str(a)) == cc.content_key(str(b)) != cc.content_key(str(c))
        assert len(cc.content_key(str(a))) == cc.KEY_CHARS


class TestCachedSteps:
    """A cached step result is only reused for the audio it was made from (cache folder used again)."""

    def test_pitch_data(self, tmp_path, monkeypatch):
        calls = []

        def pitch(path):
            calls.append(path)
            return us.PitchedData([0.0], [440.0], [0.9])
        monkeypatch.setattr(us, "get_pitch_with_file", pitch)
        monkeypatch.setattr(us.settings, "pitcher", "swiftf0")
        monkeypatch.setattr(us.settings, "skip_cache_pitch_detection", False)
        audio = tmp_path / "Artist - Title_mute.wav"
        paths = us.ProcessDataPaths(processing_audio_path=str(audio), cache_folder_path=str(tmp_path))
        audio.write_bytes(b"vocal A")
        us.pitch_audio(paths)
        us.pitch_audio(paths)
        assert len(calls) == 1
        audio.write_bytes(b"vocal B")  # e.g. separated with another model
        us.pitch_audio(paths)
        assert len(calls) == 2

    def test_denoised_vocal(self, tmp_path, monkeypatch):
        song = tmp_path / "Artist - Title.ogg"
        song.write_bytes(b"mix")
        stems = tmp_path / "stems"
        stems.mkdir()
        vocals = stems / "vocals.wav"
        monkeypatch.setattr(us, "separate_vocal_from_audio", lambda *args, **kwargs: str(stems))
        denoised = []
        monkeypatch.setattr(us, "denoise_vocal_audio",
                            lambda input_path, output_path, **kwargs: denoised.append(output_path))
        monkeypatch.setattr(us, "convert_audio_to_mono_wav", lambda *args: None)
        monkeypatch.setattr(us, "mute_no_singing_parts", lambda *args: None)
        process_data = us.ProcessData(basename="Artist - Title")
        process_data.process_data_paths.cache_folder_path = str(tmp_path / "cache")
        process_data.process_data_paths.audio_output_file_path = str(song)
        for content, use_separated_vocal in ((b"vocals", True), (b"vocals", False),
                                             (b"vocals of another model", True), (b"vocals", True)):
            vocals.write_bytes(content)
            monkeypatch.setattr(us.settings, "use_separated_vocal", use_separated_vocal)
            us.CreateProcessAudio(process_data)
        names = [os.path.basename(path) for path in denoised]
        assert names[0].endswith(f"_{cc.content_key(str(vocals))}.wav")
        assert names[1].endswith(f"_{cc.content_key(str(song))}.wav")  # the unseparated audio
        assert len(set(names)) == 3 and names[3] == names[0]


class TestCacheFolderFlag:
    def test_flag_and_reset(self):
        assert init_settings(["-i", "test.mp3", "--cache_folder", "D:/cache"]).cache_folder == "D:/cache"
        assert init_settings(["-i", "test.mp3"]).cache_folder is None

    def test_class_default(self):
        from src.Settings import Settings
        assert Settings().cache_folder is None
