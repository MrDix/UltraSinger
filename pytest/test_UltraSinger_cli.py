"""Tests for the command line of UltraSinger.py: option errors, the per-run
settings reset, and the interactive mode."""

import contextlib
import io
import os
import unittest
from unittest.mock import patch

import src.UltraSinger as us
from src.UltraSinger import init_settings


def _stop(argv):
    """init_settings for options it must refuse: (exit code, printed text)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), unittest.TestCase().assertRaises(SystemExit) as stop:
        init_settings(argv)
    init_settings(["-i", "test.mp3"])  # reset the shared settings
    return stop.exception.code, out.getvalue()


class TestOptionErrors(unittest.TestCase):
    """Wrong options end with a one-line error and exit code 1, not a traceback."""

    def test_unknown_option(self):
        code, out = _stop(["-i", "test.mp3", "--no_such_option"])
        self.assertEqual(code, 1)
        self.assertIn("option --no_such_option not recognized", out)

    def test_missing_value(self):
        code, out = _stop(["-i", "test.mp3", "--bpm"])
        self.assertEqual(code, 1)
        self.assertIn("--bpm requires argument", out)

    def test_no_number(self):
        for argv, expected in ((["--vad_onset", "abc"], "--vad_onset needs a number"),
                               (["--whisper_batch_size", "2.5"], "--whisper_batch_size needs a whole number"),
                               (["--llm_retry_max", "x"], "--llm_retry_max needs a whole number"),
                               (["--refine_hit_ratio", ""], "--refine_hit_ratio needs a number"),
                               (["--remote_stt_timeout", "inf"], "--remote_stt_timeout needs a number"),
                               (["--refine_timing_threshold", "nan"], "--refine_timing_threshold needs a number"),
                               (["--bpm", "inf"], "--bpm must be a positive number")):
            with self.subTest(argv=argv):
                code, out = _stop(["-i", "test.mp3", *argv])
                self.assertEqual(code, 1)
                self.assertIn(expected, out)

    def test_numbers_still_parse(self):
        settings = init_settings(["-i", "test.mp3", "--vad_onset", "0.5", "--whisper_batch_size", "4",
                                  "--remote_stt_timeout", "90.0", "--llm_retry_max", "1" + "0" * 400])
        self.assertEqual((settings.vad_onset, settings.whisper_batch_size, settings.remote_stt_timeout),
                         (0.5, 4, 90))
        self.assertEqual(settings.llm_retry_max, 10 ** 400)  # too long for a float, still a whole number
        init_settings(["-i", "test.mp3"])

    def test_unknown_whisper_model_is_an_error(self):
        code, _ = _stop(["-i", "test.mp3", "--whisper", "no-such-model"])
        self.assertEqual(code, 1)


class TestInputKindOptions(unittest.TestCase):
    def test_http_link_gets_the_default_output_folder(self):
        settings = init_settings(["-i", "http://example.invalid/watch?v=1"])
        self.assertEqual(settings.output_folder_path, os.path.join(os.getcwd(), "output"))
        init_settings(["-i", "test.mp3"])

    def test_is_url(self):
        self.assertTrue(us.is_url("HTTPS://example.invalid/v"))
        self.assertTrue(us.is_url("http://example.invalid/v"))
        self.assertFalse(us.is_url("https_song.mp3"))


class TestSettingsResetPerRun(unittest.TestCase):
    """settings is a reused singleton: no option of an earlier run carries over."""

    def test_options_of_an_earlier_run_do_not_carry_over(self):
        init_settings(["-i", "test.mp3", "--ignore_audio", "--bpm", "100", "--octave", "1", "--plot",
                       "--language", "de", "--disable_separation", "--detect_freestyle"])
        settings = init_settings(["-i", "test.mp3"])
        self.assertFalse(settings.ignore_audio)
        self.assertIsNone(settings.bpm_override)
        self.assertIsNone(settings.octave_shift)
        self.assertFalse(settings.create_plot)
        self.assertIsNone(settings.language)
        self.assertTrue(settings.use_separated_vocal)
        self.assertFalse(settings.detect_growl)

    def test_values_set_during_a_run_do_not_carry_over(self):
        settings = init_settings(["-i", "test.mp3"])
        settings.input_file_is_ultrastar_txt = True
        settings.transcriber_used = "whisper (large-v2)"
        settings = init_settings(["-i", "test.mp3"])
        self.assertFalse(settings.input_file_is_ultrastar_txt)
        self.assertIsNone(settings.transcriber_used)


class _Console:
    """A console that answers the interactive questions by their topic."""

    def __init__(self, answers):
        self.answers = answers

    def input(self, prompt):
        for topic, answer in self.answers.items():
            if topic in prompt:
                return answer
        return ""

    def print(self, *args, **kwargs):
        pass


class TestInteractiveMode(unittest.TestCase):
    def _ask(self, force_cpu):
        from modules import init_interactive_mode
        from Settings import Settings

        settings = Settings()
        answers = {"input file": "song.mp3", "additional options": "y",
                   "Force [green]CPU usage": force_cpu, "ffmpeg": "C:/tools/ffmpeg/bin"}
        with patch.object(init_interactive_mode, "Console", lambda: _Console(answers)), \
                patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            init_interactive_mode.init_settings_interactive(settings)
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        return settings, cuda_visible

    def test_answers_are_kept(self):
        settings, _ = self._ask(force_cpu="n")
        self.assertEqual(settings.input_file_path, "song.mp3")
        self.assertEqual(settings.user_ffmpeg_path, "C:/tools/ffmpeg/bin")
        self.assertFalse(settings.force_cpu)

    def test_force_cpu(self):
        settings, cuda_visible = self._ask(force_cpu="y")
        self.assertTrue(settings.force_cpu)
        self.assertEqual(cuda_visible, "-1")

    def test_requirements_are_checked_after_the_questions(self):
        calls = []
        with patch.object(us, "print_version"), \
                patch("modules.proxy_setup.setup_proxy_environment"), \
                patch.object(us, "init_settings_interactive", side_effect=lambda s: calls.append("interactive")), \
                patch.object(us, "check_requirements", side_effect=lambda: calls.append("requirements")), \
                patch.object(us, "run", side_effect=lambda: calls.append("run")), \
                self.assertRaises(SystemExit):
            us.main(["--interactive"])
        init_settings(["-i", "test.mp3"])
        self.assertEqual(calls, ["interactive", "requirements", "run"])

    def test_forced_cpu_skips_the_gpu_check(self):
        with patch.object(us.settings, "force_cpu", True), \
                patch.object(us.settings, "pytorch_device", "cpu"), \
                patch.object(us.settings, "whisper_batch_size", 4), \
                patch.object(us, "check_gpu_support", side_effect=AssertionError("GPU check with a forced CPU")), \
                patch.object(us, "is_ffmpeg_available", return_value=True), \
                patch.object(us, "get_ffmpeg_and_ffprobe_paths", return_value=("ffmpeg", "ffprobe")), \
                contextlib.redirect_stdout(io.StringIO()):
            us.check_requirements()
            self.assertEqual(us.settings.pytorch_device, "cpu")


if __name__ == "__main__":
    unittest.main()
