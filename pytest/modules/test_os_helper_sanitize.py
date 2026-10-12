"""Tests for output folder name sanitizing (modules.os_helper.sanitize_filename)."""

from __future__ import annotations

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
