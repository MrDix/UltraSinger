"""Tests for the plain lyrics alignment pipeline."""

import unittest
from unittest.mock import patch, MagicMock

from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Speech_Recognition.reference_lyrics_aligner import (
    create_midi_segments_from_checked_synced_lyrics,
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

    # a sentence the transcriber made up where nothing is sung: 11 words in 1.1 s
    MADE_UP = [(f"m{n}", 39.0 + 0.1 * n, 39.06 + 0.1 * n) for n in range(11)]

    def test_made_up_words_are_no_passage(self):
        result, placement, _ = self._run(self.HEARD + self.MADE_UP)
        self.assertEqual(placement.passages, 1)
        self.assertNotIn("m0", [s.word.strip() for s in result])

    def test_made_up_line_of_a_passage_is_left_out(self):
        # a sung line, a pause, then the made-up words: the sung line stays a passage
        tail = [(f"t{n}", 34.0 + 0.5 * n, 34.4 + 0.5 * n) for n in range(8)]
        result, placement, _ = self._run(self.HEARD + tail + self.MADE_UP)
        words = [s.word.strip() for s in result]
        self.assertEqual(placement.passages, 2)
        self.assertEqual(words[-8:], [w for w, _, _ in tail])
        self.assertNotIn("m0", words)

    def test_rest_of_a_made_up_passage_too_short(self):
        # four sung words before the made-up words are no passage
        sung = [(f"s{n}", 34.0 + 0.5 * n, 34.4 + 0.5 * n) for n in range(4)]
        result, placement, _ = self._run(self.HEARD + sung + self.MADE_UP)
        self.assertEqual(placement.passages, 1)
        self.assertNotIn("s3", [s.word.strip() for s in result])

    def test_bracketed_words_the_transcription_has_get_notes(self):
        # the words of line 5 in brackets: heard, so sung by the lead; "(oh yeah)" is not heard
        lyrics = self.LYRICS.replace("w5a w5b w5c", "(w5a w5b w5c)").replace("w7c", "w7c (oh yeah)")
        with patch("src.modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio",
                   return_value=[dict(w) for w in self.ALIGNED]) as align:
            _, placement = create_midi_segments_from_plain_lyrics(
                lyrics, "audio.wav", "en", self._sung(), melisma_split=False, transcription_words=self.HEARD)
        texts = [seg["text"] for seg in align.call_args[0][0]]
        self.assertIn("w5a w5b w5c", texts)
        self.assertIn("w7a w7b w7c \x00oh \x00yeah", texts)
        self.assertEqual(placement.backing_sung, 3)
        self.assertIn("3 bracketed words heard in the transcription, sung with notes", str(placement))


def _lrc(lines):
    """LRC text of (time s, text) lines."""
    return "\n".join(f"[{int(t // 60):02d}:{t % 60:05.2f}] {text}" for t, text in lines)


class TestCheckedSyncedLyrics(unittest.TestCase):
    """Synced lyrics whose times the transcription's words check: a line sung elsewhere moves there,
    a passage the lyrics lack is sung with the transcribed words, a line not sung there is left out."""

    SUNG = TestPassagesWithoutLyrics.SUNG
    HEARD = TestPassagesWithoutLyrics.HEARD
    ALIGNED = TestPassagesWithoutLyrics.ALIGNED
    FOREIGN = TestPassagesWithoutLyrics.FOREIGN
    LYRIC_WORDS = TestPassagesWithoutLyrics.LYRIC_WORDS

    def _lines(self, early=0.0):
        """The sung lines at their times; "never sung here" at 15 s, in the passage; the line after the
        passage ``early`` s early."""
        lines = [(line[0][1] - (early if i == 4 else 0.0), " ".join(w for w, _, _ in line))
                 for i, line in enumerate(self.SUNG)]
        return lines[:4] + [(15.0, "never sung here")] + lines[4:]

    def _run(self, lrc_lines, heard):
        with patch("src.modules.Speech_Recognition.reference_lyrics_aligner.librosa") as librosa, \
                patch("src.modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio",
                      return_value=[dict(w) for w in self.ALIGNED]) as align:
            librosa.get_duration.return_value = 40.0
            result, check = create_midi_segments_from_checked_synced_lyrics(
                _lrc(lrc_lines), "audio.wav", "en", TestPassagesWithoutLyrics._sung(), heard,
                melisma_split=False)
        return result, check, align

    def test_line_moves_and_passage_is_sung(self):
        result, check, align = self._run(self._lines(early=6.0), self.HEARD)
        self.assertTrue(check.checked)
        self.assertEqual((check.lines, check.confirmed, check.moved), (9, 8, 1))
        self.assertEqual((check.passages, check.passage_words, check.left_out), (1, 10, 1))
        self.assertEqual(str(check), "8 of 9 lines confirmed by the transcription's words; 1 moved to where they "
                                     "are sung, 1 sung passage without lyrics with the transcription's words "
                                     "(10 words), 1 line of the lyrics left out there")
        segments = align.call_args[0][0]
        self.assertNotIn("never sung here", [seg["text"] for seg in segments])
        # the line after the passage at the time it is sung, not 6 s earlier
        self.assertAlmostEqual(segments[4]["start"], self.SUNG[4][0][1])
        self.assertEqual(align.call_args.kwargs["silent_spans"], [(12.0, 19.6)])
        foreign = [w for w, _, _ in self.FOREIGN]
        self.assertEqual([s.word.strip() for s in result], self.LYRIC_WORDS[:12] + foreign + self.LYRIC_WORDS[12:])

    def test_bracketed_line_the_transcription_has_gets_notes(self):
        lrc_lines = self._lines()
        lrc_lines[2] = (lrc_lines[2][0], "(" + lrc_lines[2][1] + ")")
        lrc_lines.append((36.0, "(oh yeah)"))
        _, check, align = self._run(lrc_lines, self.HEARD)
        texts = [seg["text"] for seg in align.call_args[0][0]]
        self.assertEqual(texts[2], "w2a w2b w2c")
        self.assertEqual(texts[-1], "\x00oh \x00yeah")
        self.assertEqual(check.backing_sung, 3)
        self.assertIn("3 bracketed words heard in the transcription, sung with notes", str(check))

    def test_too_few_confirmed_lines_keep_their_times(self):
        heard = [w for line in self.SUNG[:4] for w in line]  # four lines: too few
        lrc_lines = self._lines(early=6.0)
        result, check, align = self._run(lrc_lines, heard)
        self.assertFalse(check.checked)
        self.assertEqual(str(check), "not checked (too few of the 9 lines confirmed by the transcription's words)")
        segments = align.call_args[0][0]
        self.assertEqual([seg["start"] for seg in segments], [t for t, _ in lrc_lines])
        self.assertEqual(align.call_args.kwargs["silent_spans"], [])
        self.assertEqual(len(result), 24)

    def test_failed_alignment(self):
        with patch("src.modules.Speech_Recognition.reference_lyrics_aligner.librosa") as librosa, \
                patch("src.modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio",
                      return_value=[]):
            librosa.get_duration.return_value = 40.0
            result, check = create_midi_segments_from_checked_synced_lyrics(
                _lrc(self._lines()), "audio.wav", "en", TestPassagesWithoutLyrics._sung(), self.HEARD)
        self.assertEqual(result, [])
        self.assertTrue(check.checked)

    def test_nothing_to_align(self):
        with patch("src.modules.Speech_Recognition.reference_lyrics_aligner.librosa") as librosa, \
                patch("src.modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio") as align:
            librosa.get_duration.return_value = 40.0
            result, check = create_midi_segments_from_checked_synced_lyrics(
                "[00:01.00] (x2)", "audio.wav", "en", TestPassagesWithoutLyrics._sung(), self.HEARD)
        self.assertEqual((result, check.lines), ([], 0))
        align.assert_not_called()
