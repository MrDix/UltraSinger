"""Tests for the plain lyrics alignment pipeline."""

import unittest
from unittest.mock import patch, MagicMock

from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Speech_Recognition.reference_lyrics_aligner import (
    create_midi_segments_from_plain_lyrics,
)


def _word(text, start, end, line_end=False):
    return {"word": text, "start": start, "end": end, "backing": False, "line_end": line_end}


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


def _line(i, start):
    """The three words of line ``i``, 0.5 s apart from ``start`` on, as (text, start s, end s)."""
    return [(f"w{i}{part}", start + 0.5 * k, start + 0.5 * k + 0.4) for k, part in enumerate("abc")]


class TestPassagesWithoutLyrics(unittest.TestCase):
    """The passages of the transcription that the lyrics do not have are sung with its words; lines of
    the lyrics that it does not have are left out where their time falls into such a passage."""

    # eight sung lines, four before and four after a passage; the line "never sung here" is not
    # sung: its time from the words falls between them, into the passage
    SUNG = [_line(i, 1.0 + 2.0 * i) for i in range(4)] + [_line(i, 25.0 + 2.0 * (i - 4)) for i in range(4, 8)]
    LYRICS = "\n".join([" ".join(w for w, _, _ in line) for line in SUNG[:4]] + ["never sung here"]
                       + [" ".join(w for w, _, _ in line) for line in SUNG[4:]])
    ALIGNED = [_word(w, start, end, k == 2) for line in SUNG for k, (w, start, end) in enumerate(line)]
    FOREIGN = [("la", 12.0, 12.4), ("di", 12.5, 12.9), ("da,", 13.0, 13.4), ("doo", 13.5, 13.9),
               ("sing", 15.0, 15.4), ("it", 15.5, 15.9), ("so", 18.0, 18.3), ("long", 18.4, 18.7),
               ("my", 18.8, 19.0), ("friend", 19.1, 19.6)]
    HEARD = [w for line in SUNG[:4] for w in line] + FOREIGN + [w for line in SUNG[4:] for w in line]
    LYRIC_WORDS = [w for line in SUNG for w, _, _ in line]

    @staticmethod
    def _sung(duration=40.0):
        """Voiced throughout."""
        pd = _make_pitched_data(duration)
        pd.confidence = [0.95] * len(pd.times)
        return pd

    def _run(self, heard, pd=None, aligned=None, **kwargs):
        with patch("src.modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio",
                   return_value=[dict(w) for w in (aligned or self.ALIGNED)]) as align:
            result, placement = create_midi_segments_from_plain_lyrics(
                self.LYRICS, "audio.wav", "en", pd or self._sung(), melisma_split=False,
                transcription_words=heard, **kwargs)
        return result, placement, align

    def test_words_of_the_passage_between_the_lyrics(self):
        result, placement, align = self._run(self.HEARD)
        foreign = [w for w, _, _ in self.FOREIGN]
        self.assertEqual([s.word.strip() for s in result], self.LYRIC_WORDS[:12] + foreign + self.LYRIC_WORDS[12:])
        self.assertTrue(placement.line_by_line)
        self.assertEqual((placement.passages, placement.passage_words, placement.left_out), (1, 10, 1))
        self.assertIn("1 sung passage without lyrics with the transcription's words (10 words, 1 line of the "
                      "lyrics left out there)", str(placement))
        # the line that is not sung is not aligned, and nothing of the lyrics goes into the passage
        self.assertEqual(len(align.call_args[0][0]), 8)
        self.assertNotIn("never sung here", [seg["text"] for seg in align.call_args[0][0]])
        self.assertEqual(align.call_args.kwargs["silent_spans"], [(12.0, 19.6)])
        # line breaks: after the lyrics line before the passage, at the passage's pauses and its end
        breaks = [s.word.strip() for s in result if s.line_break_after]
        self.assertEqual(breaks[3:7], ["w3c", "doo", "it", "friend"])

    def test_words_of_the_lyrics_in_a_passage(self):
        # a word starting in the passage is left out; one that runs into it ends where it starts
        aligned = [dict(w) for w in self.ALIGNED]
        aligned[11]["end"] = 24.0
        aligned.insert(12, _word("stray", 13.0, 13.4))
        result, _, _ = self._run(self.HEARD, aligned=aligned)
        words = [s.word.strip() for s in result]
        self.assertNotIn("stray", words)
        self.assertEqual(result[words.index("w3c")].end, 12.0)

    def test_few_words_are_no_passage(self):
        heard = [w for w in self.HEARD if not 13.0 <= w[1] < 20.0]
        result, placement, align = self._run(heard)
        self.assertEqual([s.word.strip() for s in result], self.LYRIC_WORDS)
        self.assertEqual((placement.passages, placement.left_out), (0, 0))
        self.assertEqual(len(align.call_args[0][0]), 9)
        self.assertEqual(align.call_args.kwargs["silent_spans"], [])

    def test_words_in_a_pause_are_no_passage(self):
        pd = self._sung()
        # nothing voiced from 11 to 21 s: the transcriber heard words in a pause
        pd.confidence = [0.0 if 11.0 <= t <= 21.0 else 0.95 for t in pd.times]
        result, placement, _ = self._run(self.HEARD, pd)
        self.assertEqual(len(result), 24)
        self.assertEqual((placement.passages, placement.left_out), (0, 0))

    def test_not_for_the_text_of_these_words(self):
        result, placement, align = self._run(self.HEARD, passages_from_words=False)
        self.assertEqual(len(result), 24)
        self.assertEqual(placement.passages, 0)
        self.assertEqual(len(align.call_args[0][0]), 9)

    def test_not_when_the_words_place_too_few_lines(self):
        # the transcription has one line of the lyrics and other words: no measure of what they lack
        heard = self.SUNG[0] + [(f"x{n}", 10.0 + 0.5 * n, 10.4 + 0.5 * n) for n in range(12)]
        result, placement, align = self._run(heard)
        self.assertFalse(placement.line_by_line)
        self.assertEqual((placement.passages, placement.left_out), (0, 0))
        self.assertEqual(len(result), 24)
        self.assertEqual(len(align.call_args[0][0]), 9)

    def test_after_the_end_of_the_lyrics(self):
        tail = [("and", 34.0, 34.3), ("then", 34.4, 34.7), ("it", 34.8, 35.0), ("ends", 35.1, 35.6),
                ("right", 35.7, 35.9), ("here", 36.0, 36.4), ("for", 36.5, 36.7), ("good", 36.8, 37.2)]
        result, placement, _ = self._run(self.HEARD + tail)
        self.assertEqual([s.word.strip() for s in result][-8:], [w for w, _, _ in tail])
        self.assertEqual(placement.passages, 2)
