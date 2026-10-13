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
        result, _ = create_midi_segments_from_plain_lyrics("", "audio.wav", "en", pd)
        self.assertEqual(result, [])

    def test_whitespace_only_returns_empty(self):
        pd = _make_pitched_data()
        result, _ = create_midi_segments_from_plain_lyrics("   \n   ", "audio.wav", "en", pd)
        self.assertEqual(result, [])

    def test_none_lyrics_returns_empty(self):
        pd = _make_pitched_data()
        result, _ = create_midi_segments_from_plain_lyrics(None, "audio.wav", "en", pd)
        self.assertEqual(result, [])

    @patch("src.modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio")
    def test_creates_segments_from_aligned_words(self, mock_align):
        """Verify that plain lyrics create MidiSegments when alignment succeeds."""
        mock_align.return_value = [
            {"word": "hello", "start": 1.0, "end": 1.5},
            {"word": "world", "start": 2.0, "end": 2.5},
        ]

        pd = _make_pitched_data()
        result, _ = create_midi_segments_from_plain_lyrics(
            "Hello world", "audio.wav", "en", pd, melisma_split=False,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].word, "hello ")
        self.assertEqual(result[1].word, "world")
        self.assertAlmostEqual(result[0].start, 1.0)
        self.assertAlmostEqual(result[1].start, 2.0)

    @patch("src.modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio")
    def test_each_line_is_a_segment(self, mock_align):
        """Each lyric line is a segment of its own; without words of a
        transcription, the segments have no times."""
        mock_align.return_value = [
            {"word": "Line", "start": 0.5, "end": 1.0, "backing": False, "line_end": False},
            {"word": "one", "start": 1.0, "end": 1.5, "backing": False, "line_end": True},
            {"word": "Line", "start": 2.0, "end": 2.5, "backing": False, "line_end": False},
            {"word": "two", "start": 2.5, "end": 3.0, "backing": False, "line_end": True},
        ]

        pd = _make_pitched_data()
        result, placement = create_midi_segments_from_plain_lyrics(
            "Line one\nLine two", "audio.wav", "en", pd, melisma_split=False,
        )

        self.assertEqual(len(result), 4)
        segments = mock_align.call_args[0][0]
        self.assertEqual([s["text"] for s in segments], ["Line one", "Line two"])
        self.assertIsNone(mock_align.call_args.kwargs["line_time"])
        self.assertFalse(placement.line_by_line)
        # a line break after the first line only
        self.assertEqual([s.line_break_after for s in result], [False, True, False, False])

    @patch("src.modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio")
    def test_line_times_from_the_transcription(self, mock_align):
        """Enough lines anchored by the transcription's words: the segments get
        their times and are aligned as synced lyrics."""
        mock_align.return_value = [{"word": "one", "start": 1.0, "end": 1.5}]
        lines = [f"line number{i}" for i in range(10)]
        words = [(f"number{i}", 10.0 + 4 * i, 10.3 + 4 * i) for i in range(10)]

        pd = _make_pitched_data()
        _, placement = create_midi_segments_from_plain_lyrics(
            "\n".join(lines), "audio.wav", "en", pd, melisma_split=False, transcription_words=words,
        )

        segments = mock_align.call_args[0][0]
        self.assertEqual(mock_align.call_args.kwargs["line_time"], "time from the transcription")
        # each line starts one word (0.35 s) before its second word
        for i, segment in enumerate(segments):
            self.assertAlmostEqual(segment["start"], 10.0 + 4 * i - 0.35)
        self.assertEqual((placement.lines, placement.anchored, placement.line_by_line), (10, 10, True))

    @patch("src.modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio")
    def test_alignment_failure_returns_empty(self, mock_align):
        """When alignment returns no words, return empty list."""
        mock_align.return_value = []

        pd = _make_pitched_data()
        result, _ = create_midi_segments_from_plain_lyrics(
            "Hello world", "audio.wav", "en", pd,
        )
        self.assertEqual(result, [])

    @patch("src.modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio")
    def test_crlf_normalized(self, mock_align):
        """Windows-style line breaks should be handled."""
        mock_align.return_value = [
            {"word": "hello", "start": 0.5, "end": 1.0},
        ]

        pd = _make_pitched_data()
        create_midi_segments_from_plain_lyrics(
            "hello\r\nworld", "audio.wav", "en", pd, melisma_split=False,
        )

        segments = mock_align.call_args[0][0]
        self.assertEqual([s["text"] for s in segments], ["hello", "world"])


if __name__ == "__main__":
    unittest.main()
