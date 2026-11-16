"""Tests for pitcher.py"""

import os
import unittest
import src.modules.Pitcher.Pitcher as test_subject
import pytest
from src.modules.plot import plot


class PitcherTest(unittest.TestCase):
    @pytest.mark.skip(reason="Skipping this FUNCTION level test, can be used for manual tests")
    def test_get_pitch_with_file(self):
        # Arrange
        test_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(test_dir + "/../../..")
        # test_file_abs_path = os.path.abspath(root_dir + "/test_input/audio_denoised.wav")
        test_file_abs_path = os.path.abspath(root_dir + "/test_input/test_denoised.wav")
        test_output = root_dir + "/test_output"

        # Act
        pitched_data = test_subject.get_pitch_with_file(test_file_abs_path)
        # test_subject.get_pitch_with_file(test_file_abs_path)
        plot(pitched_data, test_output, title="pitching test")
        print("done")


class TestNormalizedMono(unittest.TestCase):
    """The samples of a WAV file reach the pitch detection as float32 in -1 ... 1."""

    def test_integer_samples_are_scaled(self):
        import numpy as np
        mono = test_subject.normalized_mono(np.array([0, 16384, -32768], dtype=np.int16))
        self.assertEqual(mono.dtype, np.float32)
        np.testing.assert_allclose(mono, [0.0, 0.5, -1.0])
        np.testing.assert_allclose(
            test_subject.normalized_mono(np.array([0, 2 ** 30], dtype=np.int32)), [0.0, 0.5])
        np.testing.assert_allclose(
            test_subject.normalized_mono(np.array([128, 255, 0], dtype=np.uint8)), [0.0, 127 / 128, -1.0])

    def test_stereo_integer_samples_are_scaled_too(self):
        # the average of two integer channels is a float array: it must not pass for scaled samples
        import numpy as np
        stereo = np.array([[16384, 16384], [32767, 0], [-32768, -32768]], dtype=np.int16)
        mono = test_subject.normalized_mono(stereo)
        self.assertEqual((mono.dtype, mono.shape), (np.float32, (3,)))
        np.testing.assert_allclose(mono, [0.5, 32767 / 65536, -1.0], rtol=1e-6)

    def test_float_samples_stay_as_they_are(self):
        import numpy as np
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype.__name__):
                mono = test_subject.normalized_mono(np.array([[0.5, 0.25], [-1.0, 1.0]], dtype=dtype))
                self.assertEqual(mono.dtype, np.float32)
                np.testing.assert_allclose(mono, [0.375, 0.0])

    def test_a_stereo_file_is_pitched_like_a_mono_file(self):
        # the pitch detection gets the same samples for the same sound, whatever the channel count
        import numpy as np
        from unittest import mock
        samples = (np.sin(np.arange(1600) * 0.1) * 12000).astype(np.int16)
        heard = []

        def detect(audio, sample_rate):
            heard.append(audio)
            return "track"

        for audio in (samples, np.stack([samples, samples], axis=1)):
            with mock.patch.object(test_subject.wavfile, "read", return_value=(16000, audio)), \
                    mock.patch.object(test_subject, "get_pitch_with_swift_f0", side_effect=detect):
                self.assertEqual(test_subject.get_pitch_with_file("stem.wav"), "track")
        np.testing.assert_allclose(heard[0], heard[1], rtol=1e-6)
        self.assertLessEqual(float(np.abs(heard[1]).max()), 1.0)


if __name__ == "__main__":
    unittest.main()
