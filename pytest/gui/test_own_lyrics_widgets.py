"""Round-trip tests for the user's own lyrics of a song and the lyrics sources folder.

Per-song settings dialog (ConversionSettingsForm with per_song) -> collect_config()
-> UltraSingerRunner.build_args() -> UltraSinger.py CLI option parsing (init_settings)
-> Settings field.
"""

import os
import time
import unittest
from pathlib import Path

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.gui import own_lyrics
from src.gui.config import _DEFAULTS, config_dir
from src.gui.settings_dialog import PerSongSettingsDialog
from src.gui.settings_tab import ConversionSettingsForm
from src.gui.ultrasinger_runner import UltraSingerRunner
from src.UltraSinger import init_settings

_app = QApplication.instance() or QApplication([])

LYRICS = "first line of the song\nsecond line of it"


def _cli(args: list[str]) -> list[str]:
    return args[args.index("-i"):] if "-i" in args else ["-i", "test.mp3"] + args


class TestOwnLyricsOfASong(unittest.TestCase):

    def test_no_lyrics_by_default_and_no_option(self):
        self.assertEqual(_DEFAULTS["lyrics_text"], "")
        config = ConversionSettingsForm({}, per_song=True).collect_config()
        self.assertEqual(config["lyrics_text"], "")
        args = UltraSingerRunner().build_args(config, "test.mp3")
        self.assertNotIn("--lyrics_file", args)
        self.assertIsNone(init_settings(_cli(args)).lyrics_file)

    def test_pasted_lyrics_reach_the_conversion_as_a_file(self):
        form = ConversionSettingsForm({}, per_song=True)
        form._lyrics_text.setPlainText(f"\n{LYRICS}\n\n")
        config = form.collect_config()
        self.assertEqual(config["lyrics_text"], LYRICS)
        args = UltraSingerRunner().build_args(config, "test.mp3")
        path = Path(args[args.index("--lyrics_file") + 1])
        self.assertEqual(path.parent, config_dir() / "own_lyrics")
        self.assertEqual(path.read_text(encoding="utf-8"), LYRICS)
        self.assertEqual(init_settings(_cli(args)).lyrics_file, str(path))
        init_settings(["-i", "test.mp3"])  # reset the shared settings

    def test_only_the_settings_of_one_song_have_the_field(self):
        form = ConversionSettingsForm({"lyrics_text": LYRICS})
        self.assertIsNone(form._lyrics_text)
        # the value of the settings is kept as it is: the dialog of a song compares with it
        self.assertEqual(form.collect_config()["lyrics_text"], LYRICS)
        self.assertEqual(ConversionSettingsForm({}).collect_config()["lyrics_text"], "")

    def test_the_lyrics_are_an_override_of_the_song(self):
        dialog = PerSongSettingsDialog(dict(_DEFAULTS), {}, [])
        self.assertEqual(dialog.get_overrides().get("lyrics_text"), None)
        dialog._form._lyrics_text.setPlainText(LYRICS)
        self.assertEqual(dialog.get_overrides()["lyrics_text"], LYRICS)
        again = PerSongSettingsDialog(dict(_DEFAULTS), {"lyrics_text": LYRICS}, [])
        self.assertEqual(again._form._lyrics_text.toPlainText(), LYRICS)

    def test_a_file_is_loaded_into_the_field(self):
        form = ConversionSettingsForm({}, per_song=True)
        path = config_dir() / "some lyrics.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes("Gr\u00fc\u00dfe\nzweite Zeile".encode("cp1252"))
        from unittest.mock import patch
        with patch("src.gui.settings_tab.QFileDialog.getOpenFileName", return_value=(str(path), "")):
            form._load_lyrics_file()
        self.assertEqual(form._lyrics_text.toPlainText(), "Gr\u00fc\u00dfe\nzweite Zeile")
        # no file chosen: the field stays
        with patch("src.gui.settings_tab.QFileDialog.getOpenFileName", return_value=("", "")):
            form._load_lyrics_file()
        self.assertEqual(form._lyrics_text.toPlainText(), "Gr\u00fc\u00dfe\nzweite Zeile")


class TestOwnLyricsFiles(unittest.TestCase):

    def test_the_same_lyrics_are_the_same_file(self):
        first = own_lyrics.write_own_lyrics(LYRICS)
        self.assertEqual(own_lyrics.write_own_lyrics(f"  {LYRICS}\n"), first)
        self.assertNotEqual(own_lyrics.write_own_lyrics(LYRICS + "\nthird line"), first)
        self.assertEqual(len(list(own_lyrics.own_lyrics_folder().glob("*.txt"))), 2)

    def test_files_no_conversion_used_for_long_are_removed(self):
        old = own_lyrics.write_own_lyrics("old lyrics")
        long_ago = time.time() - (own_lyrics.KEEP_DAYS + 1) * 86400
        os.utime(old, (long_ago, long_ago))
        kept = own_lyrics.write_own_lyrics("lyrics of last week")
        last_week = time.time() - 7 * 86400
        os.utime(kept, (last_week, last_week))
        new = own_lyrics.write_own_lyrics("new lyrics")
        self.assertFalse(old.exists())
        self.assertTrue(kept.exists() and new.exists())
        # lyrics that are used again are new again
        own_lyrics.write_own_lyrics("lyrics of last week")
        self.assertGreater(kept.stat().st_mtime, time.time() - 60)


class TestLyricsSourcesFolder(unittest.TestCase):

    def test_the_folder_of_the_settings_is_the_one_the_conversion_asks(self):
        from src.modules import lyrics_sources
        self.assertEqual(own_lyrics.lyrics_sources_folder(), config_dir() / "lyrics_sources")
        self.assertEqual(lyrics_sources.default_sources_dir(), own_lyrics.lyrics_sources_folder())

    def test_open_folder_makes_it(self):
        from unittest.mock import patch
        form = ConversionSettingsForm({})
        with patch("src.gui.settings_tab.QDesktopServices.openUrl") as opened:
            form._open_lyrics_sources_folder()
        self.assertTrue(own_lyrics.lyrics_sources_folder().is_dir())
        opened.assert_called_once()

    def test_the_command_line_names_another_folder(self):
        settings = init_settings(["-i", "test.mp3", "--lyrics_sources_dir", "D:/my sources"])
        self.assertEqual(settings.lyrics_sources_dir, "D:/my sources")
        self.assertIsNone(init_settings(["-i", "test.mp3"]).lyrics_sources_dir)


if __name__ == "__main__":
    unittest.main()
