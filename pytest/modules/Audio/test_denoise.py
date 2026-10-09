"""Tests for denoise.py"""

import os
import shutil
import unittest

import numpy as np
import pytest
import soundfile as sf

from src.modules.Audio.denoise import denoise_vocal_audio, filter_delay, remove_filter_delay


def _vocal_like(sample_rate: int, seconds: float, seed: int = 0) -> np.ndarray:
    """Harmonic phrases with pauses and a little noise, roughly like a sung line."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    y = np.zeros_like(t)
    start = 0.3
    while start < seconds - 0.5:
        length = rng.uniform(0.3, 0.9)
        f0 = rng.uniform(150, 450)
        sel = (t >= start) & (t < start + length)
        tt = t[sel] - start
        env = np.sin(np.pi * tt / length) ** 2
        y[sel] += env * sum(np.sin(2 * np.pi * f0 * k * tt) / k for k in range(1, 5))
        start += length + rng.uniform(0.1, 0.4)
    return (0.3 * y + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


def _lag(original: np.ndarray, other: np.ndarray, max_lag: int) -> int:
    """Lag of ``other`` against ``original`` in [-max_lag, max_lag] samples."""
    from scipy.signal import correlate

    c = correlate(other, original, mode="full", method="fft")
    zero = len(original) - 1
    return int(np.argmax(c[zero - max_lag:zero + max_lag + 1])) - max_lag


class FilterDelayTest(unittest.TestCase):
    def test_finds_the_delay(self):
        sr = 16000
        x = _vocal_like(sr, 12)
        noise = 0.01 * np.random.default_rng(1).standard_normal(len(x))
        y = (0.7 * np.concatenate([np.zeros(400), x[:-400]]) + noise).astype(np.float32)
        self.assertEqual(filter_delay(x, y, sr), 400)

    def test_no_delay(self):
        sr = 16000
        x = _vocal_like(sr, 8)
        self.assertEqual(filter_delay(x, 0.8 * x, sr), 0)

    def test_unrelated_audio_is_not_shifted(self):
        sr = 16000
        rng = np.random.default_rng(2)
        self.assertEqual(filter_delay(rng.standard_normal(sr * 5), rng.standard_normal(sr * 5), sr), 0)

    def test_too_short_or_silent(self):
        sr = 16000
        self.assertEqual(filter_delay(np.ones(100), np.ones(100), sr), 0)
        self.assertEqual(filter_delay(np.zeros(sr * 3), np.zeros(sr * 3), sr), 0)


class RemoveFilterDelayTest(unittest.TestCase):
    def test_rewrites_the_filtered_file_in_line(self):
        import tempfile

        sr = 16000
        x = _vocal_like(sr, 10)
        stereo = np.stack([x, 0.5 * x], axis=1)
        delayed = np.concatenate([np.zeros((400, 2), np.float32), stereo[:-400]])
        with tempfile.TemporaryDirectory() as tmp:
            original, filtered = os.path.join(tmp, "in.wav"), os.path.join(tmp, "out.wav")
            sf.write(original, stereo, sr, subtype="PCM_16")
            sf.write(filtered, delayed, sr, subtype="PCM_16")
            self.assertAlmostEqual(remove_filter_delay(original, filtered), 0.025)
            out, out_sr = sf.read(filtered, always_2d=True, dtype="float32")
            self.assertEqual((out_sr, out.shape, sf.info(filtered).subtype), (sr, stereo.shape, "PCM_16"))
            np.testing.assert_allclose(out[:-400], stereo[:-400], atol=1e-4)
            self.assertFalse(out[-400:].any())  # the end is padded, the length kept

    def test_unreadable_original_leaves_the_file_alone(self):
        import tempfile

        sr = 16000
        x = _vocal_like(sr, 4)
        with tempfile.TemporaryDirectory() as tmp:
            filtered = os.path.join(tmp, "out.wav")
            sf.write(filtered, x, sr, subtype="PCM_16")
            before = open(filtered, "rb").read()
            self.assertEqual(remove_filter_delay(os.path.join(tmp, "missing.wav"), filtered), 0.0)
            self.assertEqual(open(filtered, "rb").read(), before)

    def test_failed_write_keeps_the_file_and_leaves_no_temp_file(self):
        import tempfile
        from unittest.mock import patch

        sr = 16000
        x = _vocal_like(sr, 6)
        delayed = np.concatenate([np.zeros(400, np.float32), x[:-400]])
        with tempfile.TemporaryDirectory() as tmp:
            original, filtered = os.path.join(tmp, "in.wav"), os.path.join(tmp, "out.wav")
            sf.write(original, x, sr, subtype="PCM_16")
            sf.write(filtered, delayed, sr, subtype="PCM_16")
            before = open(filtered, "rb").read()
            with patch("src.modules.Audio.denoise.sf.write", side_effect=OSError("disk full")):
                self.assertEqual(remove_filter_delay(original, filtered), 0.0)
            self.assertEqual(open(filtered, "rb").read(), before)
            self.assertEqual(sorted(os.listdir(tmp)), ["in.wav", "out.wav"])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
class DenoiseAlignmentTest(unittest.TestCase):
    def test_denoised_vocal_is_in_line_with_the_input(self):
        import tempfile

        sr = 44100
        x = _vocal_like(sr, 8, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            src, out = os.path.join(tmp, "vocals.wav"), os.path.join(tmp, "denoised.wav")
            sf.write(src, np.stack([x, x], axis=1), sr, subtype="PCM_16")
            denoise_vocal_audio(src, out)
            y, _ = sf.read(out, always_2d=True, dtype="float32")
            self.assertEqual(len(y), len(x))
            self.assertEqual(_lag(x, y.mean(axis=1), max_lag=int(0.06 * sr)), 0)


class DenoiseTest(unittest.TestCase):
    @pytest.mark.skip(reason="Skipping this FUNCTION level test, can be used for manual tests")
    def test_ffmpeg_reduce_noise(self):
        # Arrange
        test_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(test_dir + "/../../..")
        test_file_abs_path = os.path.abspath(root_dir + "/test_input/vocals.wav")
        test_file_name = os.path.basename(test_file_abs_path)
        test_output = test_dir + "/test_output"

        # Act
        denoise_vocal_audio(test_file_abs_path, test_output + "/output_" + test_file_name)


if __name__ == "__main__":
    unittest.main()
