"""Tests for output folder name sanitizing (modules.os_helper.sanitize_filename)."""

from __future__ import annotations

import pytest

from modules import os_helper
from modules.os_helper import sanitize_filename


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
