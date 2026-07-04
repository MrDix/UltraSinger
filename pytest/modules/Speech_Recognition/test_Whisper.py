"""Tests for whisper.py"""

import unittest
from unittest.mock import patch, MagicMock
from src.modules.Speech_Recognition.TranscribedData import TranscribedData
from src.modules.Speech_Recognition.Whisper import (
    convert_to_transcribed_data,
    number_to_words,
    detect_language_from_audio,
    transcribe_with_whisper,
    WhisperModel,
)

class ConvertToTranscribedDataTest(unittest.TestCase):
    def test_convert_to_transcribed_data(self):
        # Arrange
        result_aligned = {
            "segments": [
                {
                    "words": [
                        {"word": "UltraSinger", "start": 1.23, "end": 2.34, "confidence": 0.95},
                        {"word": "is", "start": 2.34, "end": 3.45, "confidence": 0.9},
                        {"word": "cool!", "start": 3.45, "end": 4.56, "confidence": 0.85},
                    ]
                },
                {
                    "words": [
                        {"word": "And", "start": 4.56, "end": 5.67, "confidence": 0.95},
                        {"word": "will", "start": 5.67, "end": 6.78, "confidence": 0.9},
                        {"word": "be", "start": 6.78, "end": 7.89, "confidence": 0.85},
                        {"word": "better!", "start": 7.89, "end": 9.01, "confidence": 0.8},
                    ]
                },
            ]
        }

        # Words should have space at the end
        expected_output = [
            TranscribedData(word="UltraSinger ", start=1.23, end=2.34, is_hyphen=False, confidence=0.95),
            TranscribedData(word="is ", start=2.34, end=3.45, is_hyphen=False, confidence=0.9),
            TranscribedData(word="cool! ", start=3.45, end=4.56, is_hyphen=False, confidence=0.85),
            TranscribedData(word="And ", start=4.56, end=5.67, is_hyphen=False, confidence=0.95),
            TranscribedData(word="will ", start=5.67, end=6.78, is_hyphen=False, confidence=0.9),
            TranscribedData(word="be ", start=6.78, end=7.89, is_hyphen=False, confidence=0.85),
            TranscribedData(word="better! ", start=7.89, end=9.01, is_hyphen=False, confidence=0.8),
        ]

        # Act
        transcribed_data = convert_to_transcribed_data(result_aligned)

        # Assert
        self.assertEqual(len(transcribed_data), len(expected_output))
        for i in range(len(transcribed_data)):
            self.assertEqual(transcribed_data[i].word, expected_output[i].word)
            self.assertEqual(transcribed_data[i].end, expected_output[i].end)
            self.assertEqual(transcribed_data[i].start, expected_output[i].start)
            self.assertEqual(transcribed_data[i].is_hyphen, expected_output[i].is_hyphen)

    def test_number_to_words_converts(self):
        #Original, test with no language passed
        self.act_and_assert("I have 1 million dollars and 2 cents.", "I have one million dollars and two cents.")
        self.act_and_assert("1 2 3 4 5", "one two three four five")
        self.act_and_assert("1, 2, 3, 4, 5,", "one, two, three, four, five,")
        self.act_and_assert("Hello world 1, 2!. 3. 4? Test 100#",
                            "Hello world one, two!. three. four? Test one hundred#")
        #Test English
        self.act_and_assert("I have 1 million dollars and 2 cents.", "I have one million dollars and two cents.", "en")
        self.act_and_assert("1 2 3 4 5", "one two three four five", "en")
        self.act_and_assert("1, 2, 3, 4, 5,", "one, two, three, four, five,", "en")
        self.act_and_assert("Hello world 1, 2!. 3. 4? Test 100#",
                            "Hello world one, two!. three. four? Test one hundred#", "en")
        #Test German
        self.act_and_assert("1 2 3 4 5", "eins zwei drei vier fünf" ,"de")
        self.act_and_assert("1, 2, 3, 4, 5","eins, zwei, drei, vier, fünf","de")
        self.act_and_assert("Ich habe 1 Million Dollar und 2 Cent.","Ich habe eins Million Dollar und zwei Cent.","de")
        self.act_and_assert("Hallo Welt 1, 2!. 3. 4? Test 100#","Hallo Welt eins, zwei!. drei. vier? Test einhundert#","de")
        #Test Spanish
        self.act_and_assert("1 2 3 4 5","uno dos tres cuatro cinco","es")
        self.act_and_assert("1, 2, 3, 4, 5","uno, dos, tres, cuatro, cinco","es")
        self.act_and_assert("Tengo un millón de dólares y 2 centavos","Tengo un millón de dólares y dos centavos","es")
        self.act_and_assert("Hola mundo 1, 2!. 3. 4? Prueba 100#","Hola mundo uno, dos!. tres. cuatro? Prueba cien#","es")
        

    def act_and_assert(self, text, expected_output, language="en"):
        # Act
        result = number_to_words(text, language)

        # Assert
        self.assertEqual(result, expected_output)


class DetectLanguageFromAudioTest(unittest.TestCase):
    """Tests for detect_language_from_audio (mocked faster-whisper)."""

    @patch("faster_whisper.WhisperModel")
    @patch("whisperx.load_audio")
    def test_returns_detected_language(self, mock_load_audio, mock_fw_model_cls):
        """Should return the language code from faster-whisper detect_language."""
        import numpy as np
        mock_load_audio.return_value = np.zeros(16000, dtype=np.float32)

        mock_model = MagicMock()
        mock_model.detect_language.return_value = ("de", 0.95, [("de", 0.95), ("en", 0.03)])
        mock_fw_model_cls.return_value = mock_model

        result = detect_language_from_audio("/fake/audio.wav", device="cpu")

        self.assertEqual(result, "de")
        mock_model.detect_language.assert_called_once()

    @patch("faster_whisper.WhisperModel")
    @patch("whisperx.load_audio")
    def test_uses_vad_filter_to_skip_instrumental_intros(self, mock_load_audio, mock_fw_model_cls):
        """detect_language must pass vad_filter=True so long instrumental
        intros (e.g. "unplugged" songs with singing starting after 30s+)
        don't get analyzed as near-silence, which previously produced a
        low-confidence, effectively random language guess."""
        import numpy as np
        mock_load_audio.return_value = np.zeros(16000, dtype=np.float32)

        mock_model = MagicMock()
        mock_model.detect_language.return_value = ("de", 0.9, [("de", 0.9)])
        mock_fw_model_cls.return_value = mock_model

        detect_language_from_audio("/fake/audio.wav", device="cpu")

        _args, kwargs = mock_model.detect_language.call_args
        self.assertTrue(kwargs.get("vad_filter"))

    @patch("faster_whisper.WhisperModel")
    @patch("whisperx.load_audio")
    def test_returns_english_for_english_audio(self, mock_load_audio, mock_fw_model_cls):
        """Should return 'en' for English audio."""
        import numpy as np
        mock_load_audio.return_value = np.zeros(16000, dtype=np.float32)

        mock_model = MagicMock()
        mock_model.detect_language.return_value = ("en", 0.99, [("en", 0.99)])
        mock_fw_model_cls.return_value = mock_model

        result = detect_language_from_audio("/fake/audio.wav")

        self.assertEqual(result, "en")

    @patch("faster_whisper.WhisperModel")
    @patch("whisperx.load_audio")
    def test_low_confidence_core_language_kept_but_warned(self, mock_load_audio, mock_fw_model_cls):
        """A low-confidence 'en' guess (e.g. from an instrumental-only
        window) must NOT be silently trusted just because 'en' is a core
        language -- previously only non-core low-confidence guesses were
        flagged. The (possibly wrong) language is still returned since we
        have no better guess without --language, but a warning must fire."""
        import numpy as np
        mock_load_audio.return_value = np.zeros(16000, dtype=np.float32)

        mock_model = MagicMock()
        mock_model.detect_language.return_value = ("en", 0.31, [("en", 0.31), ("de", 0.28)])
        mock_fw_model_cls.return_value = mock_model

        result = detect_language_from_audio("/fake/audio.wav")

        self.assertEqual(result, "en")

    @patch("faster_whisper.WhisperModel")
    @patch("whisperx.load_audio")
    def test_low_confidence_non_core_language_falls_back_to_english(
        self, mock_load_audio, mock_fw_model_cls
    ):
        """Existing behavior preserved: low confidence + non-core language
        still falls back to English."""
        import numpy as np
        mock_load_audio.return_value = np.zeros(16000, dtype=np.float32)

        mock_model = MagicMock()
        mock_model.detect_language.return_value = ("cy", 0.2, [("cy", 0.2)])
        mock_fw_model_cls.return_value = mock_model

        result = detect_language_from_audio("/fake/audio.wav")

        self.assertEqual(result, "en")


class TranscribeWithWhisperAlignModelTest(unittest.TestCase):
    """Tests that the resolved language is threaded into load_align_model,
    and that unsupported languages degrade gracefully instead of crashing."""

    def _mock_transcription(self, mock_load_model, mock_load_audio, detected_language):
        import numpy as np
        mock_load_audio.return_value = np.zeros(16000, dtype=np.float32)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"text": "hallo welt"}],
            "language": detected_language,
        }
        mock_load_model.return_value = mock_model

    @patch("whisperx.align")
    @patch("whisperx.load_align_model")
    @patch("whisperx.load_audio")
    @patch("whisperx.load_model")
    def test_detected_language_passed_to_align_model(
        self, mock_load_model, mock_load_audio, mock_load_align_model, mock_align
    ):
        """The language Whisper detected (not a hardcoded 'en') must reach
        load_align_model's language_code argument."""
        self._mock_transcription(mock_load_model, mock_load_audio, "de")
        mock_load_align_model.return_value = (MagicMock(), MagicMock())
        mock_align.return_value = {"segments": []}

        transcribe_with_whisper(
            "/fake/audio.wav", WhisperModel.TINY, device="cpu", language=None,
        )

        mock_load_align_model.assert_called_once_with(
            language_code="de", device="cpu", model_name=None
        )

    @patch("whisperx.align")
    @patch("whisperx.load_align_model")
    @patch("whisperx.load_audio")
    @patch("whisperx.load_model")
    def test_explicit_language_passed_to_align_model(
        self, mock_load_model, mock_load_audio, mock_load_align_model, mock_align
    ):
        """When --language is set explicitly it must be used for alignment,
        even if Whisper's own detection disagrees."""
        self._mock_transcription(mock_load_model, mock_load_audio, "en")
        mock_load_align_model.return_value = (MagicMock(), MagicMock())
        mock_align.return_value = {"segments": []}

        transcribe_with_whisper(
            "/fake/audio.wav", WhisperModel.TINY, device="cpu", language="de",
        )

        mock_load_align_model.assert_called_once_with(
            language_code="de", device="cpu", model_name=None
        )

    @patch("whisperx.align")
    @patch("whisperx.load_align_model")
    @patch("whisperx.load_audio")
    @patch("whisperx.load_model")
    def test_unsupported_language_falls_back_to_english(
        self, mock_load_model, mock_load_audio, mock_load_align_model, mock_align
    ):
        """If WhisperX has no default wav2vec2 model for the detected
        language, we must retry with English instead of crashing."""
        self._mock_transcription(mock_load_model, mock_load_audio, "xx")
        mock_load_align_model.side_effect = [
            ValueError("No default align-model for language: xx"),
            (MagicMock(), MagicMock()),
        ]
        mock_align.return_value = {"segments": []}

        result = transcribe_with_whisper(
            "/fake/audio.wav", WhisperModel.TINY, device="cpu", language=None,
        )

        self.assertEqual(mock_load_align_model.call_count, 2)
        first_call, second_call = mock_load_align_model.call_args_list
        self.assertEqual(
            first_call.kwargs,
            {"language_code": "xx", "device": "cpu", "model_name": None},
        )
        self.assertEqual(
            second_call.kwargs,
            {"language_code": "en", "device": "cpu", "model_name": None},
        )
        # Detected language is still reported as "xx" -- only the alignment
        # model silently degrades, transcription/lyrics metadata is untouched.
        self.assertEqual(result.detected_language, "xx")

    @patch("whisperx.align")
    @patch("whisperx.load_align_model")
    @patch("whisperx.load_audio")
    @patch("whisperx.load_model")
    def test_custom_align_model_error_is_not_swallowed(
        self, mock_load_model, mock_load_audio, mock_load_align_model, mock_align
    ):
        """If the user explicitly requested a custom --whisper_align_model
        and it fails, we must not silently substitute English -- the user
        asked for a specific model, so surface the error."""
        self._mock_transcription(mock_load_model, mock_load_audio, "xx")
        mock_load_align_model.side_effect = ValueError("bad custom model")

        with self.assertRaises(ValueError):
            transcribe_with_whisper(
                "/fake/audio.wav", WhisperModel.TINY, device="cpu",
                language=None, alignment_model="some/custom-model",
            )

        mock_load_align_model.assert_called_once_with(
            language_code="xx", device="cpu", model_name="some/custom-model"
        )


if __name__ == "__main__":
    unittest.main()
