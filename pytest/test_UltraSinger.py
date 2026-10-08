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


class TestChartStyleResolution(unittest.TestCase):
    """--chart_style drives the ptAKF refit; explicit refit flags override it."""

    def test_default_is_singable_refit_off(self):
        self.assertEqual(Settings().chart_style, "singable")
        settings = init_settings(["-i", "test.mp3"])
        self.assertEqual(settings.chart_style, "singable")
        self.assertFalse(settings.ptakf_refit)

    def test_score_style_enables_refit(self):
        settings = init_settings(["-i", "test.mp3", "--chart_style", "score"])
        self.assertEqual(settings.chart_style, "score")
        self.assertTrue(settings.ptakf_refit)

    def test_singable_style_disables_refit(self):
        settings = init_settings(["-i", "test.mp3", "--chart_style", "singable"])
        self.assertFalse(settings.ptakf_refit)

    def test_explicit_ptakf_refit_overrides_singable_default(self):
        settings = init_settings(["-i", "test.mp3", "--ptakf_refit"])
        self.assertTrue(settings.ptakf_refit)

    def test_explicit_disable_overrides_score_style(self):
        settings = init_settings(
            ["-i", "test.mp3", "--chart_style", "score", "--disable_ptakf_refit"])
        self.assertFalse(settings.ptakf_refit)

    def test_unknown_style_falls_back_to_singable(self):
        settings = init_settings(["-i", "test.mp3", "--chart_style", "bogus"])
        self.assertEqual(settings.chart_style, "singable")
        self.assertFalse(settings.ptakf_refit)

    def test_no_leak_across_calls(self):
        # init_settings mutates a shared singleton; a prior score run must not
        # leak chart_style/refit state into a later default run.
        init_settings(["-i", "test.mp3", "--chart_style", "score"])
        settings = init_settings(["-i", "test.mp3"])
        self.assertEqual(settings.chart_style, "singable")
        self.assertFalse(settings.ptakf_refit)

    def test_class_default_ptakf_refit_matches_singable(self):
        # A fresh Settings() (read before init_settings resolves) must agree
        # with the singable default (refit off).
        self.assertFalse(Settings().ptakf_refit)


class TestSegmentationModelFlag(unittest.TestCase):
    """--segmentation_model takes a model file path; missing files abort early."""

    def test_existing_file_is_used(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.pt")
            open(path, "wb").close()
            settings = init_settings(["-i", "test.mp3", "--segmentation_model", path])
            self.assertEqual(settings.segmentation_model, path)

    def test_missing_file_is_accepted_for_fallback(self):
        # No early exit: the pipeline step warns and keeps the word-based notes.
        settings = init_settings(["-i", "test.mp3", "--segmentation_model", "does/not/exist.pt"])
        self.assertEqual(settings.segmentation_model, "does/not/exist.pt")

    def test_no_leak_across_calls(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.pt")
            open(path, "wb").close()
            init_settings(["-i", "test.mp3", "--segmentation_model", path])
            settings = init_settings(["-i", "test.mp3"])
            self.assertIsNone(settings.segmentation_model)

    def test_class_default_is_none(self):
        self.assertIsNone(Settings().segmentation_model)


class TestSegmentationModelRepoFlags(unittest.TestCase):
    """--segmentation_model_repo / --segmentation_model_token parsing and per-run reset."""

    def test_repo_and_token(self):
        settings = init_settings(["-i", "test.mp3", "--segmentation_model_repo", "owner/repo",
                                  "--segmentation_model_token", "tok"])
        self.assertEqual(settings.segmentation_model_repo, "owner/repo")
        self.assertEqual(settings.segmentation_model_token, "tok")

    def test_model_flag_not_confused_with_repo_flag(self):
        settings = init_settings(["-i", "test.mp3", "--segmentation_model", "m.pt",
                                  "--segmentation_model_repo", "owner/repo"])
        self.assertEqual(settings.segmentation_model, "m.pt")
        self.assertEqual(settings.segmentation_model_repo, "owner/repo")

    def test_no_leak_across_calls(self):
        init_settings(["-i", "test.mp3", "--segmentation_model_repo", "owner/repo",
                       "--segmentation_model_token", "tok"])
        settings = init_settings(["-i", "test.mp3"])
        self.assertIsNone(settings.segmentation_model_repo)
        self.assertIsNone(settings.segmentation_model_token)
class TestLeadVocalPitchFlag(unittest.TestCase):
    def test_default_on(self):
        self.assertTrue(Settings().lead_vocal_pitch)
        self.assertTrue(init_settings(["-i", "test.mp3"]).lead_vocal_pitch)

    def test_disable_and_reset(self):
        self.assertFalse(init_settings(["-i", "test.mp3", "--disable_lead_vocal_pitch"]).lead_vocal_pitch)
        self.assertTrue(init_settings(["-i", "test.mp3"]).lead_vocal_pitch)


class TestTranscribeAudioLyricsLanguage(unittest.TestCase):
    """Synced and plain lyrics from the lookup are checked against the sung language on their own."""

    GERMAN = ("[00:01.00] wir sind heute hier und wir singen, denn es ist ein Tag, an dem die Sonne "
              "scheint und wir sind nicht allein, und es ist schön, dass du da bist, und wir bleiben "
              "noch, bis die Nacht kommt und es dunkel ist, und dann gehen wir nach Haus")
    ENGLISH = ("we are here today and we sing, because it is a day on which the sun is shining and we "
               "are not alone, and it is good that you are here, and we stay until the night comes and "
               "it is dark, and then we go home to the place where we are from")

    def _run(self, synced, plain):
        from types import SimpleNamespace
        from unittest.mock import patch

        import src.UltraSinger as us

        process_data = SimpleNamespace(
            process_data_paths=SimpleNamespace(cache_folder_path="cache", whisper_audio_path="audio.wav"),
            media_info=SimpleNamespace(artist="Artist", title="Title", language=None),
            transcribed_data=[], synced_lyrics=None, plain_lyrics=None,
        )
        transcription = SimpleNamespace(detected_language="de", transcribed_data=[])
        lyrics = SimpleNamespace(synced_lyrics=synced, plain_lyrics=plain)
        with patch.object(us, "transcribe_audio", return_value=transcription), \
                patch.object(us, "remove_silence_from_transcription_data", side_effect=lambda path, data: data), \
                patch("modules.lrclib_client.search_lyrics", return_value=lyrics), \
                patch("modules.Speech_Recognition.lyrics_corrector.correct_transcription_from_lyrics",
                      return_value=([], None)) as correct, \
                patch.object(us.settings, "lyrics_lookup", True), \
                patch.object(us.settings, "llm_correct_lyrics", False), \
                patch.object(us.settings, "hyphenation", False), \
                patch.object(us.settings, "language", None):
            us.TranscribeAudio(process_data)
        return process_data, correct

    def test_plain_lyrics_in_another_language_are_not_used(self):
        process_data, correct = self._run(self.GERMAN, self.ENGLISH)
        self.assertEqual(process_data.synced_lyrics, self.GERMAN)
        self.assertIsNone(process_data.plain_lyrics)
        correct.assert_not_called()

    def test_matching_lyrics_are_used(self):
        plain = self.GERMAN.replace("[00:01.00] ", "")
        process_data, correct = self._run(self.GERMAN, plain)
        self.assertEqual(process_data.plain_lyrics, plain)
        correct.assert_called_once()
