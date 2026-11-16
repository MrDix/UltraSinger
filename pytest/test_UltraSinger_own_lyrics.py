"""Tests for the user's own lyrics of a song in UltraSinger.py: the lyrics file
of the option or next to the input file, used instead of a lookup."""

import contextlib
import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import src.UltraSinger as us
from src.UltraSinger import init_settings

LYRICS = "hello world\nsecond line"


def _syllable(word, start, end):
    return us.TranscribedData(word=word, start=start, end=end, confidence=0.9)


class TestLyricsFileOption(unittest.TestCase):
    def test_the_options(self):
        settings = init_settings(["-i", "test.mp3", "--lyrics_file", "D:/lyrics/mine.lrc",
                                  "--lyrics_sources_dir", "D:/my sources"])
        self.assertEqual((settings.lyrics_file, settings.lyrics_sources_dir), ("D:/lyrics/mine.lrc", "D:/my sources"))
        settings = init_settings(["-i", "test.mp3"])  # (also resets the shared settings)
        self.assertEqual((settings.lyrics_file, settings.lyrics_sources_dir), (None, None))

    def test_the_file_of_the_option_must_hold_lyrics(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "mine.txt")
            with open(path, "w", encoding="utf-8") as file:
                file.write(LYRICS)
            with patch.object(us.settings, "lyrics_file", path), patch.object(us.settings, "ignore_audio", False):
                us._check_lyrics_options()
                self.assertEqual(us._own_lyrics_file(), path)
            for name, content in (("none.txt", None), ("empty.txt", ""), ("tags.lrc", "[ar:Band]\n[ti:Song]\n"),
                                  ("chart.txt", "#TITLE:Song\n#BPM:300\n: 0 4 60 word\nE\n")):
                broken = os.path.join(folder, name)
                if content is not None:
                    with open(broken, "w", encoding="utf-8") as file:
                        file.write(content)
                out = io.StringIO()
                with self.subTest(name=name), patch.object(us.settings, "lyrics_file", broken), \
                        contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as stop:
                    us._check_lyrics_options()
                self.assertEqual(stop.exception.code, 1)
                self.assertIn("cannot be used", out.getvalue())

    def test_a_file_that_cannot_be_used_ends_the_run_before_anything_is_downloaded(self):
        with tempfile.TemporaryDirectory() as folder:
            out = io.StringIO()
            with patch.object(us.settings, "lyrics_file", os.path.join(folder, "none.txt")), \
                    patch.object(us, "InitProcessData", side_effect=AssertionError("the input was prepared")), \
                    contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as stop:
                us.run()
        self.assertEqual(stop.exception.code, 1)
        self.assertIn("cannot be used", out.getvalue())

    def test_a_folder_of_lyrics_sources_that_is_not_there_is_told(self):
        with tempfile.TemporaryDirectory() as folder:
            for sources_dir, told in ((os.path.join(folder, "Lyrcs"), True), (folder, False), (None, False)):
                out = io.StringIO()
                with self.subTest(sources_dir=sources_dir), patch.object(us.settings, "lyrics_file", None), \
                        patch.object(us.settings, "lyrics_sources_dir", sources_dir), contextlib.redirect_stdout(out):
                    us._check_lyrics_options()
                self.assertEqual("does not exist" in out.getvalue(), told)

    def test_the_re_pitch_of_a_chart_keeps_its_lyrics(self):
        """No lyrics file is used (or reported) when only the pitches of an UltraStar txt are made anew."""
        with tempfile.TemporaryDirectory() as folder:
            chart = os.path.join(folder, "Band - Song.txt")
            with open(chart, "w", encoding="utf-8") as file:
                file.write("#TITLE:Song\n#BPM:300\n: 0 4 60 word\nE\n")
            lyrics = os.path.join(folder, "Band - Song.lrc")
            with open(lyrics, "w", encoding="utf-8") as file:
                file.write("[00:01.00] hello world")
            with patch.object(us.settings, "ignore_audio", True), patch.object(us.settings, "input_file_path", chart):
                with patch.object(us.settings, "lyrics_file", None):
                    self.assertIsNone(us._own_lyrics_file())
                with patch.object(us.settings, "lyrics_file", lyrics):
                    self.assertIsNone(us._own_lyrics_file())

    def test_the_file_next_to_a_local_input_file(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = os.path.join(folder, "Band - Song.mp3")
            open(audio, "wb").close()
            with patch.object(us.settings, "lyrics_file", None), patch.object(us.settings, "input_file_path", audio), \
                    patch.object(us.settings, "ignore_audio", False):
                self.assertIsNone(us._own_lyrics_file())
                sidecar = os.path.join(folder, "Band - Song.lrc")
                with open(sidecar, "w", encoding="utf-8") as file:
                    file.write("[00:01.00] hello world")
                self.assertEqual(us._own_lyrics_file(), sidecar)
            # a link has no file next to it, and neither has a file that is not there
            for source in ("https://example.com/watch?v=abc", os.path.join(folder, "gone.mp3"), ""):
                with self.subTest(source=source), patch.object(us.settings, "lyrics_file", None), \
                        patch.object(us.settings, "input_file_path", source):
                    self.assertIsNone(us._own_lyrics_file())


class TestLookupWithOwnLyrics(unittest.TestCase):
    def test_lyrics_are_wanted_with_a_file_whatever_else(self):
        data = SimpleNamespace(lyrics_file="mine.txt", media_info=SimpleNamespace(artist="", title=""))
        with patch.object(us.settings, "lyrics_lookup", False):
            self.assertTrue(us._lyrics_wanted(data))
            data.lyrics_file = None
            self.assertFalse(us._lyrics_wanted(data))
        named = SimpleNamespace(lyrics_file=None, media_info=SimpleNamespace(artist="Artist", title="Song"))
        with patch.object(us.settings, "lyrics_lookup", True):
            self.assertTrue(us._lyrics_wanted(named))
            self.assertFalse(us._lyrics_wanted(data))

    def test_the_lookup_gets_the_file_and_the_settings(self):
        data = SimpleNamespace(lyrics_file="mine.txt", media_info=SimpleNamespace(artist="Artist", title="Song"),
                               process_data_paths=SimpleNamespace(whisper_audio_path="audio.wav"))
        with patch("modules.lyrics_sources.find_song_lyrics", return_value="lyrics") as find, \
                patch.object(us, "audio_length", return_value=200.0), \
                patch.object(us.settings, "lyrics_lookup", False), \
                patch.object(us.settings, "lyrics_sources_dir", "D:/sources"):
            self.assertEqual(us._lookup_lyrics(data), "lyrics")
        find.assert_called_once_with("Artist", "Song", duration=200.0, lyrics_file="mine.txt",
                                     sources_dir="D:/sources", lookup=False)

    def test_the_default_folder_of_the_lyrics_sources(self):
        from modules import lyrics_sources
        data = SimpleNamespace(lyrics_file=None, media_info=SimpleNamespace(artist="Artist", title="Song"),
                               process_data_paths=SimpleNamespace(whisper_audio_path="audio.wav"))
        with patch("modules.lyrics_sources.find_song_lyrics", return_value=None) as find, \
                patch.object(us, "audio_length", return_value=None), \
                patch.object(us.settings, "lyrics_sources_dir", None):
            us._lookup_lyrics(data)
        self.assertEqual(find.call_args.kwargs["sources_dir"], lyrics_sources.default_sources_dir())

    def test_your_own_lyrics_are_used_in_whatever_language(self):
        """Found lyrics in another language than the singing are another version of the song; the
        lyrics the user gave for the song are its lyrics."""
        with patch.object(us, "check_lyrics_language", return_value=(False, "de")):
            self.assertEqual(us._lyrics_language_check(SimpleNamespace(trusted=True), "text", "en", True),
                             (True, "de"))
            self.assertEqual(us._lyrics_language_check(SimpleNamespace(trusted=False), "text", "en", True),
                             (False, "de"))
            self.assertEqual(us._lyrics_language_check(SimpleNamespace(), "text", "en", True), (False, "de"))

    def test_the_transcription_is_corrected_by_the_file_without_a_lookup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "mine.txt")
            with open(path, "w", encoding="utf-8") as file:
                file.write(LYRICS)
            process_data = SimpleNamespace(
                process_data_paths=SimpleNamespace(cache_folder_path="cache", whisper_audio_path="audio.wav"),
                media_info=SimpleNamespace(artist="", title="", language="en"),
                transcribed_data=[], synced_lyrics=None, plain_lyrics=None, check_transcription=None,
                lyrics_heard=None, lyrics_instead=None, lyrics_found=None, lyrics_source=None, lyrics_file=path,
            )
            transcription = SimpleNamespace(detected_language="en",
                                            transcribed_data=[_syllable("hello ", 1.0, 1.3)])
            out = io.StringIO()
            with patch.object(us, "transcribe_audio", return_value=transcription), \
                    patch.object(us, "audio_length", return_value=200.0), \
                    patch("modules.Speech_Recognition.lyrics_corrector.correct_transcription_from_lyrics",
                          side_effect=lambda data, lyrics: (data, None)) as correct, \
                    patch.object(us, "remove_silence_from_transcription_data", side_effect=lambda path, data: data), \
                    patch("modules.lrclib_client.search_lyrics") as search, \
                    patch.object(us.settings, "lyrics_lookup", False), \
                    patch.object(us.settings, "llm_correct_lyrics", False), \
                    patch.object(us.settings, "hyphenation", False), \
                    patch.object(us.settings, "language", None), \
                    contextlib.redirect_stdout(out):
                us.TranscribeAudio(process_data)
        search.assert_not_called()
        self.assertEqual(correct.call_args.args[1], LYRICS)
        self.assertEqual((process_data.plain_lyrics, process_data.lyrics_source), (LYRICS, "your lyrics file"))
        self.assertIn("Found lyrics in your file", out.getvalue())


class TestInfoFile(unittest.TestCase):
    @staticmethod
    def _info(sources_dir=None, **kwargs):
        with tempfile.TemporaryDirectory() as folder, patch.object(us.settings, "lyrics_sources_dir", sources_dir):
            us._write_settings_info_file(folder, None, None, **kwargs)
            with open(os.path.join(folder, "ultrasinger_parameter.info"), encoding="utf-8") as file:
                return file.read()

    def test_the_lyrics_file_and_the_sources_folder(self):
        text = self._info()
        self.assertIn("  Lyrics file:              none\n", text)
        self.assertIn("  Lyrics sources folder:    default\n", text)
        text = self._info(sources_dir="D:/my sources", lyrics_file="D:/lyrics/mine.lrc",
                          lyrics_source="your lyrics file")
        self.assertIn("  Lyrics file:              D:/lyrics/mine.lrc\n", text)
        self.assertIn("  Lyrics sources folder:    D:/my sources\n", text)
        self.assertIn("your lyrics file synced lyrics:", text)


if __name__ == "__main__":
    unittest.main()
