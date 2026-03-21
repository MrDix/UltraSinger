"""Tests for the vocal separation module."""

import unittest
from unittest.mock import MagicMock, patch

from modules.Audio.separation import (
    AudioSeparatorModel,
    DEFAULT_AUDIO_SEPARATOR_MODEL,
    DemucsModel,
    SeparatorBackend,
    separate_vocal_from_audio,
)


class TestSeparatorBackendEnum(unittest.TestCase):
    def test_values(self):
        self.assertEqual(SeparatorBackend.DEMUCS.value, "demucs")
        self.assertEqual(SeparatorBackend.AUDIO_SEPARATOR.value, "audio_separator")

    def test_from_string(self):
        self.assertEqual(SeparatorBackend("demucs"), SeparatorBackend.DEMUCS)
        self.assertEqual(SeparatorBackend("audio_separator"), SeparatorBackend.AUDIO_SEPARATOR)


class TestAudioSeparatorModelEnum(unittest.TestCase):
    def test_best_model(self):
        self.assertIn("roformer", AudioSeparatorModel.BS_ROFORMER.value)

    def test_default(self):
        self.assertEqual(DEFAULT_AUDIO_SEPARATOR_MODEL, AudioSeparatorModel.BS_ROFORMER)

    def test_all_have_extensions(self):
        """All model filenames should have a file extension."""
        for m in AudioSeparatorModel:
            self.assertTrue(
                m.value.endswith((".ckpt", ".onnx", ".pth")),
                f"{m.name} value '{m.value}' has no recognized extension",
            )


class TestDemucsModelEnum(unittest.TestCase):
    def test_default(self):
        self.assertEqual(DemucsModel.HTDEMUCS.value, "htdemucs")

    def test_all_models_accessible(self):
        """Ensure all enum members have string values."""
        for m in DemucsModel:
            self.assertIsInstance(m.value, str)


class TestSeparateVocalFromAudio(unittest.TestCase):
    """Test the orchestrator function with mocked backends."""

    @patch("modules.Audio.separation._separate_with_demucs")
    @patch("modules.Audio.separation.check_file_exists", return_value=False)
    def test_demucs_backend_calls_demucs(self, _mock_exists, mock_demucs):
        separate_vocal_from_audio(
            cache_folder_path="/cache",
            audio_output_file_path="/audio/song.wav",
            use_separated_vocal=True,
            create_karaoke=False,
            pytorch_device="cpu",
            model=DemucsModel.HTDEMUCS,
            backend=SeparatorBackend.DEMUCS,
        )
        mock_demucs.assert_called_once()

    @patch("modules.Audio.separation._separate_with_audio_separator")
    @patch("modules.Audio.separation.check_file_exists", return_value=False)
    @patch("os.makedirs")
    def test_audio_separator_backend(self, _mock_mkdir, _mock_exists, mock_as):
        separate_vocal_from_audio(
            cache_folder_path="/cache",
            audio_output_file_path="/audio/song.wav",
            use_separated_vocal=True,
            create_karaoke=False,
            pytorch_device="cpu",
            model=AudioSeparatorModel.BS_ROFORMER,
            backend=SeparatorBackend.AUDIO_SEPARATOR,
        )
        mock_as.assert_called_once()

    @patch("modules.Audio.separation._separate_with_demucs")
    @patch("modules.Audio.separation.check_file_exists", return_value=True)
    def test_cache_hit_skips_separation(self, _mock_exists, mock_demucs):
        separate_vocal_from_audio(
            cache_folder_path="/cache",
            audio_output_file_path="/audio/song.wav",
            use_separated_vocal=True,
            create_karaoke=False,
            pytorch_device="cpu",
            model=DemucsModel.HTDEMUCS,
            backend=SeparatorBackend.DEMUCS,
        )
        mock_demucs.assert_not_called()

    @patch("modules.Audio.separation._separate_with_demucs")
    @patch("modules.Audio.separation.check_file_exists", return_value=False)
    def test_separation_disabled_skips(self, _mock_exists, mock_demucs):
        separate_vocal_from_audio(
            cache_folder_path="/cache",
            audio_output_file_path="/audio/song.wav",
            use_separated_vocal=False,
            create_karaoke=False,
            pytorch_device="cpu",
            model=DemucsModel.HTDEMUCS,
            backend=SeparatorBackend.DEMUCS,
        )
        mock_demucs.assert_not_called()

    @patch("modules.Audio.separation._separate_with_audio_separator")
    @patch("modules.Audio.separation.check_file_exists", return_value=False)
    @patch("os.makedirs")
    def test_string_model_name(self, _mock_mkdir, _mock_exists, mock_as):
        """Users can pass arbitrary model filenames as strings."""
        result = separate_vocal_from_audio(
            cache_folder_path="/cache",
            audio_output_file_path="/audio/song.wav",
            use_separated_vocal=True,
            create_karaoke=False,
            pytorch_device="cpu",
            model="custom_model_v2.ckpt",
            backend=SeparatorBackend.AUDIO_SEPARATOR,
        )
        mock_as.assert_called_once()
        # Path should contain the model name
        self.assertIn("custom_model_v2.ckpt", result)

    def test_return_path_structure(self):
        """Return path should follow cache/separated/model/basename pattern."""
        with patch("modules.Audio.separation.check_file_exists", return_value=True):
            path = separate_vocal_from_audio(
                cache_folder_path="/cache",
                audio_output_file_path="/audio/my_song.wav",
                use_separated_vocal=True,
                create_karaoke=False,
                pytorch_device="cpu",
                model=DemucsModel.HTDEMUCS,
                backend=SeparatorBackend.DEMUCS,
            )
        # Should end with model/basename
        self.assertTrue(path.endswith("htdemucs/my_song") or
                       path.endswith("htdemucs\\my_song"))


if __name__ == "__main__":
    unittest.main()
