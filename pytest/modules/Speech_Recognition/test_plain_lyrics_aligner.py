"""Tests for the plain lyrics alignment pipeline."""

import unittest
from unittest.mock import patch, MagicMock

from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Speech_Recognition.reference_lyrics_aligner import (
    create_midi_segments_from_plain_lyrics,
)


def _make_pitched_data(duration: float = 10.0, step: float = 0.016) -> PitchedData:
    """Create synthetic PitchedData with A4 pitch throughout."""
    n = int(duration / step)
    return PitchedData(
        times=[i * step for i in range(n)],
        frequencies=[440.0] * n,
        confidence=[0.9] * n,
    )


class TestCreateMidiSegmentsFromPlainLyrics(unittest.TestCase):
    def test_empty_lyrics_returns_empty(self):
        pd = _make_pitched_data()
        result = create_midi_segments_from_plain_lyrics("", "audio.wav", "en", pd)
        self.assertEqual(result, [])

    def test_whitespace_only_returns_empty(self):
        pd = _make_pitched_data()
        result = create_midi_segments_from_plain_lyrics("   \n   ", "audio.wav", "en", pd)
        self.assertEqual(result, [])

    def test_none_lyrics_returns_empty(self):
        pd = _make_pitched_data()
        result = create_midi_segments_from_plain_lyrics(None, "audio.wav", "en", pd)
        self.assertEqual(result, [])

    @patch("src.modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio")
    @patch("src.modules.Speech_Recognition.reference_lyrics_aligner.librosa")
    def test_creates_segments_from_aligned_words(self, mock_librosa, mock_align):
        """Verify that plain lyrics create MidiSegments when alignment succeeds."""
        mock_librosa.get_duration.return_value = 10.0
        mock_align.return_value = [
            {"word": "hello", "start": 1.0, "end": 1.5},
            {"word": "world", "start": 2.0, "end": 2.5},
        ]

        pd = _make_pitched_data()
        result = create_midi_segments_from_plain_lyrics(
            "Hello world", "audio.wav", "en", pd, melisma_split=False,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].word, "hello ")
        self.assertEqual(result[1].word, "world")
        self.assertAlmostEqual(result[0].start, 1.0)
        self.assertAlmostEqual(result[1].start, 2.0)

    @patch("src.modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio")
    @patch("src.modules.Speech_Recognition.reference_lyrics_aligner.librosa")
    def test_multiline_lyrics_joined(self, mock_librosa, mock_align):
        """Verify multiline lyrics are joined into a single text."""
        mock_librosa.get_duration.return_value = 10.0
        mock_align.return_value = [
            {"word": "line", "start": 0.5, "end": 1.0},
            {"word": "one", "start": 1.0, "end": 1.5},
            {"word": "line", "start": 2.0, "end": 2.5},
            {"word": "two", "start": 2.5, "end": 3.0},
        ]

        pd = _make_pitched_data()
        result = create_midi_segments_from_plain_lyrics(
            "Line one\nLine two", "audio.wav", "en", pd, melisma_split=False,
        )

        self.assertEqual(len(result), 4)
        # Check align was called with joined text
        call_args = mock_align.call_args[0][0]
        self.assertEqual(call_args[0]["text"], "Line one Line two")

    @patch("src.modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio")
    @patch("src.modules.Speech_Recognition.reference_lyrics_aligner.librosa")
    def test_alignment_failure_returns_empty(self, mock_librosa, mock_align):
        """When alignment returns no words, return empty list."""
        mock_librosa.get_duration.return_value = 10.0
        mock_align.return_value = []

        pd = _make_pitched_data()
        result = create_midi_segments_from_plain_lyrics(
            "Hello world", "audio.wav", "en", pd,
        )
        self.assertEqual(result, [])

    @patch("src.modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio")
    @patch("src.modules.Speech_Recognition.reference_lyrics_aligner.librosa")
    def test_crlf_normalized(self, mock_librosa, mock_align):
        """Windows-style line breaks should be handled."""
        mock_librosa.get_duration.return_value = 10.0
        mock_align.return_value = [
            {"word": "hello", "start": 0.5, "end": 1.0},
        ]

        pd = _make_pitched_data()
        result = create_midi_segments_from_plain_lyrics(
            "hello\r\nworld", "audio.wav", "en", pd, melisma_split=False,
        )

        # Check text was joined with CRLF normalized
        call_args = mock_align.call_args[0][0]
        self.assertNotIn("\r", call_args[0]["text"])


if __name__ == "__main__":
    unittest.main()
