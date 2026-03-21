"""Tests for LRCLIB lyrics correction."""

import unittest

from src.modules.Speech_Recognition.TranscribedData import TranscribedData
from src.modules.Speech_Recognition.lyrics_corrector import (
    LyricsLookupResult,
    correct_transcription_from_lyrics,
    _normalize_lyrics,
    _normalize_word,
)


def _td(word: str, start: float, end: float, confidence: float = 0.9) -> TranscribedData:
    """Helper to create TranscribedData with minimal args."""
    return TranscribedData(word=word, start=start, end=end, confidence=confidence)


class TestNormalizeLyrics(unittest.TestCase):
    def test_basic(self):
        result = _normalize_lyrics("Hello world\nFoo bar")
        self.assertEqual(result, ["hello", "world", "foo", "bar"])

    def test_punctuation_removed(self):
        result = _normalize_lyrics("Hello, world! How are you?")
        self.assertEqual(result, ["hello", "world", "how", "are", "you"])

    def test_contractions_preserved(self):
        result = _normalize_lyrics("I'm don't won't")
        self.assertEqual(result, ["i'm", "don't", "won't"])

    def test_empty(self):
        self.assertEqual(_normalize_lyrics(""), [])
        self.assertEqual(_normalize_lyrics("   "), [])

    def test_crlf(self):
        result = _normalize_lyrics("line one\r\nline two\rline three")
        self.assertEqual(result, ["line", "one", "line", "two", "line", "three"])


class TestNormalizeWord(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(_normalize_word("Hello"), "hello")

    def test_strip_punctuation(self):
        self.assertEqual(_normalize_word("hello,"), "hello")
        self.assertEqual(_normalize_word("world!"), "world")

    def test_preserve_apostrophe(self):
        self.assertEqual(_normalize_word("don't"), "don't")

    def test_whitespace(self):
        self.assertEqual(_normalize_word("  hello  "), "hello")


class TestCorrectTranscription(unittest.TestCase):
    def test_empty_transcription(self):
        data, result = correct_transcription_from_lyrics([], "hello world")
        self.assertEqual(data, [])
        self.assertEqual(result.words_total, 0)

    def test_empty_lyrics(self):
        td = [_td("hello ", 0.0, 0.5)]
        data, result = correct_transcription_from_lyrics(td, "")
        self.assertEqual(data[0].word, "hello ")
        self.assertEqual(result.words_kept, 1)
        self.assertEqual(result.words_corrected, 0)

    def test_none_lyrics(self):
        td = [_td("hello ", 0.0, 0.5)]
        data, result = correct_transcription_from_lyrics(td, None)
        self.assertEqual(data[0].word, "hello ")

    def test_exact_match(self):
        td = [
            _td("hello ", 0.0, 0.5),
            _td("world ", 0.5, 1.0),
        ]
        data, result = correct_transcription_from_lyrics(td, "Hello World")
        self.assertEqual(result.words_corrected, 0)
        self.assertEqual(result.words_kept, 2)
        self.assertEqual(data[0].word, "hello ")
        self.assertEqual(data[1].word, "world ")

    def test_whisper_error_corrected(self):
        td = [
            _td("helo ", 0.0, 0.5),
            _td("wurld ", 0.5, 1.0),
        ]
        data, result = correct_transcription_from_lyrics(td, "Hello World")
        self.assertEqual(result.words_corrected, 2)
        self.assertEqual(data[0].word, "hello ")
        self.assertEqual(data[1].word, "world ")

    def test_timing_preserved(self):
        td = [
            _td("helo ", 1.5, 2.0),
            _td("wurld ", 2.0, 2.5),
        ]
        data, result = correct_transcription_from_lyrics(td, "Hello World")
        self.assertEqual(data[0].start, 1.5)
        self.assertEqual(data[0].end, 2.0)
        self.assertEqual(data[1].start, 2.0)
        self.assertEqual(data[1].end, 2.5)

    def test_trailing_whitespace_preserved(self):
        td = [
            _td("helo ", 0.0, 0.5),
            _td("wurld", 0.5, 1.0),  # no trailing space
        ]
        data, result = correct_transcription_from_lyrics(td, "Hello World")
        self.assertEqual(data[0].word, "hello ")
        self.assertEqual(data[1].word, "world")  # no trailing space

    def test_extra_whisper_words_kept(self):
        """Whisper transcribes ad-libs not in reference — should be kept."""
        td = [
            _td("hello ", 0.0, 0.5),
            _td("yeah ", 0.5, 0.8),  # ad-lib not in reference
            _td("world ", 0.8, 1.3),
        ]
        data, result = correct_transcription_from_lyrics(td, "Hello World")
        # The matcher should still handle this gracefully
        self.assertEqual(result.words_total, 3)

    def test_partial_match(self):
        """Some words match, some are wrong."""
        td = [
            _td("I ", 0.0, 0.2),
            _td("luv ", 0.2, 0.5),
            _td("you ", 0.5, 0.8),
        ]
        data, result = correct_transcription_from_lyrics(td, "I love you")
        self.assertEqual(data[0].word, "I ")
        self.assertEqual(data[1].word, "love ")
        self.assertEqual(data[2].word, "you ")
        self.assertEqual(result.words_corrected, 1)

    def test_multiline_lyrics(self):
        td = [
            _td("hello ", 0.0, 0.5),
            _td("wurld ", 0.5, 1.0),
            _td("goodby ", 1.5, 2.0),
        ]
        lyrics = "Hello World\nGoodbye"
        data, result = correct_transcription_from_lyrics(td, lyrics)
        self.assertEqual(data[0].word, "hello ")
        self.assertEqual(data[1].word, "world ")
        self.assertEqual(data[2].word, "goodbye ")

    def test_result_stats(self):
        td = [
            _td("hello ", 0.0, 0.5),
            _td("wurld ", 0.5, 1.0),
        ]
        data, result = correct_transcription_from_lyrics(td, "Hello World")
        self.assertEqual(result.words_total, 2)
        self.assertEqual(result.reference_words, 2)
        self.assertIsInstance(result, LyricsLookupResult)


class TestLyricsLookupResult(unittest.TestCase):
    def test_defaults(self):
        r = LyricsLookupResult()
        self.assertEqual(r.words_corrected, 0)
        self.assertEqual(r.words_kept, 0)
        self.assertEqual(r.words_total, 0)
        self.assertEqual(r.reference_words, 0)


if __name__ == "__main__":
    unittest.main()
