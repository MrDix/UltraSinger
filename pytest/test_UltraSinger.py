"""Tests for CLI option parsing in UltraSinger.py (init_settings).

Note: `init_settings` mutates and returns the module-level `settings`
singleton rather than a fresh instance, so these tests only assert on the
effect of flags they pass themselves; default values (i.e. what a flag
looks like when *absent*) are checked against a fresh `Settings()` instance
instead of relying on the shared singleton being untouched by other tests.
"""

import unittest

from src.Settings import Settings
from src.UltraSinger import init_settings


class TestCreateAudioChunksFlag(unittest.TestCase):
    """Regression test: --create_audio_chunks must actually enable the setting.

    getopt yields an empty string as `arg` for no-value long options, so a
    handler that does `settings.create_audio_chunks = arg` would always be
    falsy. The flag must set the setting to True explicitly (like --plot /
    --keep_cache do).
    """

    def test_create_audio_chunks_flag_sets_true(self):
        settings = init_settings(["-i", "test.mp3", "--create_audio_chunks"])
        self.assertTrue(settings.create_audio_chunks)
        self.assertIs(settings.create_audio_chunks, True)

    def test_create_audio_chunks_defaults_false(self):
        self.assertFalse(Settings().create_audio_chunks)


class TestDisableMidiFlag(unittest.TestCase):
    """--disable_midi should turn off MIDI creation (enabled by default)."""

    def test_disable_midi_flag_sets_false(self):
        settings = init_settings(["-i", "test.mp3", "--disable_midi"])
        self.assertFalse(settings.create_midi)

    def test_midi_defaults_true(self):
        self.assertTrue(Settings().create_midi)


class TestNoMetadataTagsFlag(unittest.TestCase):
    """--no_metadata_tags should disable ID3/Vorbis tag writing."""

    def test_no_metadata_tags_flag_sets_false(self):
        settings = init_settings(["-i", "test.mp3", "--no_metadata_tags"])
        self.assertFalse(settings.write_metadata_tags)

    def test_metadata_tags_defaults_true(self):
        self.assertTrue(Settings().write_metadata_tags)


class TestRemoteSttTimeoutFlag(unittest.TestCase):
    """--remote_stt_timeout should parse to an int number of seconds."""

    def test_remote_stt_timeout_parses_int_value(self):
        settings = init_settings(["-i", "test.mp3", "--remote_stt_timeout", "45"])
        self.assertEqual(settings.remote_stt_timeout, 45)
        self.assertIsInstance(settings.remote_stt_timeout, int)

    def test_remote_stt_timeout_parses_float_string(self):
        settings = init_settings(["-i", "test.mp3", "--remote_stt_timeout", "90.0"])
        self.assertEqual(settings.remote_stt_timeout, 90)

    def test_remote_stt_timeout_defaults_120(self):
        self.assertEqual(Settings().remote_stt_timeout, 120)


class TestIgnoreAudioFlag(unittest.TestCase):
    """--ignore_audio is a pre-existing (previously undocumented) flag."""

    def test_ignore_audio_flag_sets_true(self):
        settings = init_settings(["-i", "test.mp3", "--ignore_audio"])
        self.assertTrue(settings.ignore_audio)


if __name__ == "__main__":
    unittest.main()
