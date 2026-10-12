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
class TestLinebreakTimingFlag(unittest.TestCase):
    def test_default_yass(self):
        self.assertEqual(Settings().linebreak_timing, "yass")
        self.assertEqual(init_settings(["-i", "test.mp3"]).linebreak_timing, "yass")

    def test_values_and_reset(self):
        for value in ("usdx", "after_last_note", "yass"):
            self.assertEqual(init_settings(["-i", "test.mp3", "--linebreak_timing", value]).linebreak_timing, value)
        init_settings(["-i", "test.mp3", "--linebreak_timing", "usdx"])
        self.assertEqual(init_settings(["-i", "test.mp3"]).linebreak_timing, "yass")

    def test_unknown_value_stops(self):
        with self.assertRaises(SystemExit):
            init_settings(["-i", "test.mp3", "--linebreak_timing", "sometimes"])
        init_settings(["-i", "test.mp3"])  # reset the shared settings


class TestTranscriberFlag(unittest.TestCase):
    """--transcriber chooses the transcriber; a Whisper model alone chooses Whisper."""

    def test_default_auto(self):
        self.assertEqual(init_settings(["-i", "test.mp3"]).transcriber, "auto")

    def test_values(self):
        for value in ("auto", "qwen3", "whisper"):
            with self.subTest(value=value):
                self.assertEqual(init_settings(["-i", "test.mp3", "--transcriber", value]).transcriber, value)
        init_settings(["-i", "test.mp3"])

    def test_whisper_model_alone_chooses_whisper(self):
        self.assertEqual(init_settings(["-i", "test.mp3", "--whisper", "small"]).transcriber, "whisper")

    def test_transcriber_wins_in_any_order(self):
        for argv in (["--whisper", "small", "--transcriber", "auto"], ["--transcriber", "auto", "--whisper", "small"]):
            with self.subTest(argv=argv):
                settings = init_settings(["-i", "test.mp3", *argv])
                self.assertEqual((settings.transcriber, settings.whisper_model.value), ("auto", "small"))
        init_settings(["-i", "test.mp3"])

    def test_unknown_value(self):
        with self.assertRaises(SystemExit):
            init_settings(["-i", "test.mp3", "--transcriber", "other"])
        init_settings(["-i", "test.mp3"])


class TestTranscribeAudioDispatch(unittest.TestCase):
    """transcribe_audio runs the chosen transcriber, caches under its name and falls back to Whisper."""

    QWEN3 = None  # TranscriberChoice("qwen3", ...), set in setUp

    def setUp(self):
        from src.modules.Speech_Recognition.qwen3_asr import TranscriberChoice
        self.QWEN3 = TranscriberChoice("qwen3", 2, "6.0 GB of GPU memory free")

    def _run(self, choice, requested="auto", qwen_error=None, cached=None, force_whisper_cpu=False):
        import io
        import os
        import tempfile
        from contextlib import redirect_stdout
        from unittest.mock import patch

        import src.UltraSinger as us
        from src.modules.Speech_Recognition.TranscriptionResult import TranscriptionResult

        result = TranscriptionResult([], "en")
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as cache, \
                patch.object(us, "choose_transcriber", return_value=choice) as choose, \
                patch.object(us, "transcribe_with_qwen3", return_value=result, side_effect=qwen_error) as qwen, \
                patch.object(us, "transcribe_with_whisper", return_value=result) as whisper, \
                patch.object(us, "release_gpu_memory") as release, \
                patch.object(us.settings, "transcriber", requested), \
                patch.object(us.settings, "force_whisper_cpu", force_whisper_cpu), \
                patch.object(us.settings, "skip_cache_transcription", False), \
                redirect_stdout(out):
            if cached:
                with open(os.path.join(cache, cached), "w", encoding="utf-8") as file:
                    file.write(TranscriptionResult([], "de").to_json())
            transcribed = us.transcribe_audio(cache, "audio.wav")
            files = os.listdir(cache)
        return {"choose": choose, "qwen": qwen, "whisper": whisper, "release": release, "files": files,
                "used": us.settings.transcriber_used, "out": out.getvalue(), "result": transcribed}

    def test_qwen3(self):
        run = self._run(self.QWEN3)
        run["qwen"].assert_called_once()
        run["whisper"].assert_not_called()
        self.assertTrue(run["files"][0].startswith("qwen3_1.7B_"))
        self.assertEqual(run["used"], "qwen3 (Qwen/Qwen3-ASR-1.7B)")

    def test_whisper_fallback_keeps_its_cache_name(self):
        from src.modules.Speech_Recognition.qwen3_asr import TranscriberChoice
        run = self._run(TranscriberChoice("whisper", reason="Qwen3-ASR needs a GPU"))
        run["qwen"].assert_not_called()
        run["whisper"].assert_called_once()
        self.assertTrue(run["files"][0].startswith("whisper_"))
        self.assertTrue(run["used"].startswith("whisper ("))

    def test_note_only_when_qwen3_was_asked_for(self):
        from src.modules.Speech_Recognition.qwen3_asr import TranscriberChoice
        choice = TranscriberChoice("whisper", reason="Qwen3-ASR needs a GPU")
        auto = self._run(choice)["out"]
        self.assertIn("Transcribing with Whisper: Qwen3-ASR needs a GPU", auto)
        self.assertNotIn("Note:", auto)
        self.assertIn("Note:", self._run(choice, requested="qwen3")["out"])
        self.assertNotIn("Whisper:", self._run(TranscriberChoice("whisper", reason="selected"), "whisper")["out"])

    def test_whisper_takes_over_when_qwen3_fails(self):
        # the class as the pipeline imports it (src.modules... would be another class object)
        from src.UltraSinger import QwenFailed
        for requested in ("auto", "qwen3"):
            with self.subTest(requested=requested):
                run = self._run(self.QWEN3, requested,
                                qwen_error=QwenFailed("Qwen3-ASR failed (RuntimeError: CUDA out of memory.)"))
                run["whisper"].assert_called_once()
                # what the failed attempt left on the GPU is handed back before Whisper loads
                run["release"].assert_called_once()
                self.assertIn("Note:", run["out"])
                self.assertIn("Qwen3-ASR failed (RuntimeError: CUDA out of memory.)", run["out"])
                self.assertTrue(run["used"].startswith("whisper ("))
                self.assertFalse(any(name.startswith("qwen3_") for name in run["files"]))

    def test_other_errors_are_passed_on(self):
        # e.g. the alignment failing: Whisper would run the same alignment
        with self.assertRaises(RuntimeError):
            self._run(self.QWEN3, qwen_error=RuntimeError("alignment failed"))

    def test_whisper_takes_over_for_an_unknown_song_language(self):
        # the class as the pipeline imports it (src.modules... would be another class object)
        from src.UltraSinger import QwenUnavailable
        run = self._run(self.QWEN3, qwen_error=QwenUnavailable("Qwen3-ASR does not transcribe the language uk"))
        run["whisper"].assert_called_once()
        self.assertIn("Transcribing with Whisper: Qwen3-ASR does not transcribe the language uk", run["out"])

    def test_cached_qwen3_transcription_needs_no_gpu(self):
        from src.modules.Speech_Recognition.qwen3_asr import TranscriberChoice
        run = self._run(TranscriberChoice("whisper", reason="only 1.0 GB of GPU memory free"),
                        cached="qwen3_1.7B_None_None_p30_aligned.json")
        run["choose"].assert_not_called()
        run["qwen"].assert_not_called()
        run["whisper"].assert_not_called()
        self.assertEqual((run["result"].detected_language, run["used"]), ("de", "qwen3 (Qwen/Qwen3-ASR-1.7B)"))
        # Whisper, when asked for, does not take it
        self.assertEqual(self._run(TranscriberChoice("whisper", reason="selected"), "whisper",
                                   cached="qwen3_1.7B_None_None_p30_aligned.json")["whisper"].call_count, 1)

    def test_force_whisper_cpu(self):
        run = self._run(self.QWEN3, force_whisper_cpu=True)
        run["choose"].assert_not_called()
        run["qwen"].assert_not_called()
        run["whisper"].assert_called_once()
        self.assertIn("--force_whisper_cpu keeps the transcription on the CPU", run["out"])


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


class TestAudioLength(unittest.TestCase):
    """The lyrics lookup gets the length of the audio to pick the fitting version."""

    def test_length_of_a_wav(self):
        import os
        import tempfile

        import numpy as np
        import soundfile as sf

        from src.UltraSinger import audio_length
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audio.wav")
            sf.write(path, np.zeros(16000 * 3), 16000)
            self.assertAlmostEqual(audio_length(path), 3.0)

    def test_unreadable_file(self):
        from src.UltraSinger import audio_length
        self.assertIsNone(audio_length("no such file.wav"))


class TestTimingPassesWithModel(unittest.TestCase):
    """Timing refinement and GAP sweep are skipped for notes from the segmentation model."""

    def test_truth_table(self):
        from src.UltraSinger import timing_pass_wanted
        self.assertTrue(timing_pass_wanted(True, False))
        self.assertFalse(timing_pass_wanted(True, True))
        self.assertFalse(timing_pass_wanted(False, False))
        self.assertFalse(timing_pass_wanted(False, True))


class TestRefineGapFlag(unittest.TestCase):
    def test_default_on(self):
        self.assertTrue(Settings().refine_gap)
        self.assertTrue(init_settings(["-i", "test.mp3"]).refine_gap)

    def test_disable_and_reset(self):
        settings = init_settings(["-i", "test.mp3", "--disable_refine_gap"])
        self.assertFalse(settings.refine_gap)
        self.assertTrue(init_settings(["-i", "test.mp3"]).refine_gap)


class TestOctaveConsistencyFlags(unittest.TestCase):
    """By default the octave consistency pass runs for model notes only; the flags set it for all notes."""

    def test_truth_table(self):
        from src.UltraSinger import octave_consistency_wanted
        self.assertTrue(octave_consistency_wanted(None, True))
        self.assertFalse(octave_consistency_wanted(None, False))
        for model_segmentation_used in (True, False):
            self.assertTrue(octave_consistency_wanted(True, model_segmentation_used))
            self.assertFalse(octave_consistency_wanted(False, model_segmentation_used))

    def test_flags_and_reset(self):
        self.assertIsNone(Settings().octave_consistency)
        self.assertTrue(init_settings(["-i", "test.mp3", "--octave_consistency"]).octave_consistency)
        self.assertIs(init_settings(["-i", "test.mp3", "--disable_octave_consistency"]).octave_consistency, False)
        self.assertIsNone(init_settings(["-i", "test.mp3"]).octave_consistency)


class TestOctaveInfoLines(unittest.TestCase):
    """The info file lists both octave options and whether model notes got the consistency pass by default."""

    def _info(self, octave_consistency=None, **kwargs):
        import os
        import tempfile
        from unittest.mock import patch

        import src.UltraSinger as us
        with tempfile.TemporaryDirectory() as d, \
                patch.object(us.settings, "octave_consistency", octave_consistency), \
                patch.object(us.settings, "octave_snap", False):
            us._write_settings_info_file(d, None, None, **kwargs)
            with open(os.path.join(d, "ultrasinger_parameter.info"), encoding="utf-8") as f:
                return f.read()

    def test_model_notes(self):
        text = self._info(model_segmentation_used=True)
        self.assertIn("Octave consistency:       True (default for the segmentation-model notes)", text)
        self.assertIn("Octave spike snap:        False", text)

    def test_word_based_notes(self):
        self.assertIn("Octave consistency:       False\n", self._info())

    def test_disabled(self):
        self.assertIn("Octave consistency:       False\n", self._info(False, model_segmentation_used=True))

    def test_all_notes(self):
        self.assertIn("Octave consistency:       True\n", self._info(True))
