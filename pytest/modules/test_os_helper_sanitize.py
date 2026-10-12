"""Tests for output folder name sanitizing (modules.os_helper.sanitize_filename)."""

from __future__ import annotations

import os
import re

import pytest

from modules import os_helper
from modules.os_helper import MAX_NAME_CHARS, get_unused_song_output_dir, sanitize_filename


class TestSanitizeFilename:
    @pytest.mark.parametrize("raw, expected", [
        ("Artist - Title", "Artist - Title"),
        ("Artist - Title (Is It Love?)", "Artist - Title (Is It Love)"),
        ("Artist - Part One / Part Two", "Artist - Part One - Part Two"),
        ('Artist - "Quoted": Title', "Artist - Quoted Title"),
        ("Artist - A<B>C", "Artist - A(B)C"),
        ("Artist - A\\B|C*D", "Artist - A-B-C-D"),
        ("Artist - Title...", "Artist - Title"),
        ("Artist - What ?", "Artist - What"),
        ("Artist - Title. ", "Artist - Title"),
    ])
    def test_replacements(self, raw, expected):
        assert sanitize_filename(raw) == expected

    def test_result_is_creatable_folder(self, tmp_path):
        name = sanitize_filename("Artist - Part One / Part Two (Why?) ")
        target = tmp_path / name
        os_helper.create_folder(str(target))
        assert target.is_dir()
        assert target.name == name


class TestNameLength:
    """Long song names are shortened, so the paths made from them stay below the Windows
    limit (the path lengths themselves: test_separation.py, TestPathLength)."""

    def test_short_name_is_kept(self):
        name = "Artist - " + "x" * (MAX_NAME_CHARS - 9)
        assert sanitize_filename(name) == name

    def test_long_name_is_shortened_at_a_word(self):
        short = sanitize_filename("Artist - " + "Word " * 30)
        assert len(short) <= MAX_NAME_CHARS
        assert re.fullmatch(r"Artist - (Word )+[0-9a-f]{6}", short)

    def test_shortened_alike_and_kept_apart(self):
        first = "Artist - " + "Long Title " * 8 + "Part One"
        second = "Artist - " + "Long Title " * 8 + "Part Two"
        assert sanitize_filename(first) == sanitize_filename(first)
        assert sanitize_filename(first) != sanitize_filename(second)
        assert sanitize_filename(first)[:-6] == sanitize_filename(second)[:-6]

    @pytest.mark.parametrize("cut", ["Artist - Title -", "Artist - Title (", "Artist - Title."])
    def test_no_dangling_separator_or_trailing_period(self, cut):
        # Shortened right after ``cut``: the hash and a space take 7 characters
        name = cut + "x" * 100
        assert re.fullmatch(r"Artist - Title [0-9a-f]{6}", os_helper.cap_name_length(name, len(cut) + 7))

    def test_result_is_creatable_folder(self, tmp_path):
        name = sanitize_filename("Artist - " + "Long Title (Version) " * 10)
        target = tmp_path / name
        os_helper.create_folder(str(target))
        assert target.is_dir() and target.name == name

    def test_characters_outside_the_bmp_count_twice(self):
        # Windows counts the path length in UTF-16 units, where an emoji takes two
        note = "\U0001F3B5"
        assert sanitize_filename("Artist - " + note * 25) == "Artist - " + note * 25  # 59 units
        short = sanitize_filename("Artist - " + note * 26)  # 61 units in 35 characters
        assert os_helper.utf16_len(short) <= MAX_NAME_CHARS
        assert re.fullmatch(f"Artist - {note}+ [0-9a-f]{{6}}", short)

    def test_unpaired_surrogate(self):
        # A file name on Windows can hold one: shortened without an error
        short = os_helper.cap_name_length("Artist - \ud800" + "x" * 100)
        assert os_helper.utf16_len(short) <= MAX_NAME_CHARS and short.startswith("Artist - \ud800x")

    def test_utf16_prefix_keeps_whole_characters(self):
        assert os_helper.utf16_prefix("ab\U0001F3B5c", 3) == "ab"
        assert os_helper.utf16_prefix("ab\U0001F3B5c", 4) == "ab\U0001F3B5"
        assert os_helper.utf16_prefix("abc", 10) == "abc"


class TestLongInputFileName:
    """A local input file is copied under the song's name at once: under its own,
    longer name, the path in the song folder could exceed the Windows path limit."""

    def test_copied_under_the_song_name(self, tmp_path, monkeypatch):
        import src.UltraSinger as us
        from src.modules.musicbrainz_client import SongInfo

        source = tmp_path / ("Artist - " + "Long Title " * 8 + ".mp3")
        source.write_bytes(b"ID3")
        targets = []
        copy = os_helper.copy

        def recorded_copy(original, target):
            targets.append(target)
            copy(original, target)
        monkeypatch.setattr(os_helper, "copy", recorded_copy)
        monkeypatch.setattr(us, "search_musicbrainz", lambda title, artist: SongInfo(title="Title", artist="Artist"))
        monkeypatch.setattr(us, "is_video_file", lambda path: False)
        monkeypatch.setattr(us.settings, "input_file_path", str(source))
        monkeypatch.setattr(us.settings, "output_folder_path", str(tmp_path / "out"))
        _, folder, audio_path, _ = us.infos_from_audio_video_input_file()
        assert targets == [audio_path] == [os.path.join(folder, "Artist - Title.mp3")]
        assert os.listdir(folder) == ["Artist - Title.mp3"]


class TestUnusedSongOutputDir:
    def test_unused_folder_is_kept(self, tmp_path):
        assert get_unused_song_output_dir(str(tmp_path / "Artist - Title")) == str(tmp_path / "Artist - Title")

    def test_numbered_after_the_used_ones(self, tmp_path):
        (tmp_path / "Artist - Title").mkdir()
        (tmp_path / "Artist - Title (1)").mkdir()
        assert get_unused_song_output_dir(str(tmp_path / "Artist - Title")) == str(tmp_path / "Artist - Title (2)")

    def test_only_the_song_folder_name_changes(self, tmp_path):
        # A numbered output folder, and a song name that holds a number itself
        songs = tmp_path / "Songs (1)"
        for name in ("Title (1)", "Title (1) (1)"):
            (songs / name).mkdir(parents=True)
        assert get_unused_song_output_dir(str(songs / "Title (1)")) == str(songs / "Title (1) (2)")
