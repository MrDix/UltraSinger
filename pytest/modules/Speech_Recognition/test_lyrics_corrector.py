"""Tests for LRCLIB lyrics correction."""

import unittest

from src.modules.Speech_Recognition.TranscribedData import TranscribedData
from src.modules.Speech_Recognition.lyrics_corrector import (
    LyricsLookupResult,
    correct_transcription_from_lyrics,
    _normalize_word,
    _parse_reference_lyrics,
    _tokenize_with_parens,
)


def _td(word: str, start: float, end: float, confidence: float = 0.9) -> TranscribedData:
    """Helper to create TranscribedData with minimal args."""
    return TranscribedData(word=word, start=start, end=end, confidence=confidence)


# --- _tokenize_with_parens ---

class TestTokenizeWithParens(unittest.TestCase):
    def test_no_parens(self):
        result = _tokenize_with_parens("hello world")
        self.assertEqual(result, [("hello", False), ("world", False)])

    def test_single_parens(self):
        result = _tokenize_with_parens("hello (oh yeah) world")
        self.assertEqual(result, [
            ("hello", False), ("oh", True), ("yeah", True), ("world", False)
        ])

    def test_brackets(self):
        result = _tokenize_with_parens("[ad-lib] test")
        self.assertEqual(result, [("ad-lib", True), ("test", False)])

    def test_all_paren(self):
        result = _tokenize_with_parens("(ooh ooh ooh)")
        self.assertEqual(result, [("ooh", True), ("ooh", True), ("ooh", True)])

    def test_mixed(self):
        result = _tokenize_with_parens("verse text (backing) more text [spoken]")
        self.assertEqual(result, [
            ("verse", False), ("text", False),
            ("backing", True),
            ("more", False), ("text", False),
            ("spoken", True),
        ])

    def test_empty(self):
        self.assertEqual(_tokenize_with_parens(""), [])

    def test_empty_parens(self):
        result = _tokenize_with_parens("hello () world")
        self.assertEqual(result, [("hello", False), ("world", False)])


# --- _parse_reference_lyrics ---

class TestParseReferenceLyrics(unittest.TestCase):
    def test_basic_lines(self):
        lyrics = "Hello world\nGoodbye moon"
        ref_words = _parse_reference_lyrics(lyrics)
        self.assertEqual(len(ref_words), 4)
        self.assertEqual(ref_words[0].normalized, "hello")
        self.assertEqual(ref_words[1].normalized, "world")
        self.assertEqual(ref_words[2].normalized, "goodbye")
        self.assertEqual(ref_words[3].normalized, "moon")

    def test_linebreaks_set(self):
        lyrics = "Hello world\nGoodbye moon"
        ref_words = _parse_reference_lyrics(lyrics)
        # "world" is end of first line, not last line → linebreak
        self.assertTrue(ref_words[1].line_break_after)
        # "moon" is last word → no linebreak
        self.assertFalse(ref_words[3].line_break_after)

    def test_empty_line_linebreak(self):
        lyrics = "Hello world\n\nGoodbye moon"
        ref_words = _parse_reference_lyrics(lyrics)
        # "world" should have line_break_after from the non-empty line ending
        # AND the empty line doubles as a paragraph break
        self.assertTrue(ref_words[1].line_break_after)

    def test_freestyle_parens(self):
        lyrics = "Hello (ooh) world"
        ref_words = _parse_reference_lyrics(lyrics)
        self.assertEqual(len(ref_words), 3)
        self.assertFalse(ref_words[0].is_freestyle)  # Hello
        self.assertTrue(ref_words[1].is_freestyle)    # ooh
        self.assertFalse(ref_words[2].is_freestyle)   # world

    def test_freestyle_brackets(self):
        lyrics = "Hello [ad-lib] world"
        ref_words = _parse_reference_lyrics(lyrics)
        self.assertTrue(ref_words[1].is_freestyle)

    def test_multi_word_parens(self):
        lyrics = "Hello (oh yeah baby) world"
        ref_words = _parse_reference_lyrics(lyrics)
        self.assertEqual(len(ref_words), 5)
        self.assertFalse(ref_words[0].is_freestyle)
        self.assertTrue(ref_words[1].is_freestyle)   # oh
        self.assertTrue(ref_words[2].is_freestyle)   # yeah
        self.assertTrue(ref_words[3].is_freestyle)   # baby
        self.assertFalse(ref_words[4].is_freestyle)  # world

    def test_no_trailing_linebreak(self):
        lyrics = "Only one line"
        ref_words = _parse_reference_lyrics(lyrics)
        self.assertFalse(ref_words[-1].line_break_after)

    def test_empty_lyrics(self):
        self.assertEqual(_parse_reference_lyrics(""), [])
        self.assertEqual(_parse_reference_lyrics("   \n   "), [])

    def test_preserves_original_text(self):
        lyrics = "Don't stop"
        ref_words = _parse_reference_lyrics(lyrics)
        self.assertEqual(ref_words[0].original, "Don't")
        self.assertEqual(ref_words[0].normalized, "don't")


# --- correct_transcription_from_lyrics ---

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
        data, _ = correct_transcription_from_lyrics(td, None)
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
        self.assertEqual(data[0].word, "Hello ")
        self.assertEqual(data[1].word, "World ")

    def test_timing_preserved(self):
        td = [
            _td("helo ", 1.5, 2.0),
            _td("wurld ", 2.0, 2.5),
        ]
        data, _ = correct_transcription_from_lyrics(td, "Hello World")
        self.assertEqual(data[0].start, 1.5)
        self.assertEqual(data[0].end, 2.0)
        self.assertEqual(data[1].start, 2.0)
        self.assertEqual(data[1].end, 2.5)

    def test_trailing_whitespace_preserved(self):
        td = [
            _td("helo ", 0.0, 0.5),
            _td("wurld", 0.5, 1.0),  # no trailing space
        ]
        data, _ = correct_transcription_from_lyrics(td, "Hello World")
        self.assertEqual(data[0].word, "Hello ")
        self.assertEqual(data[1].word, "World")  # no trailing space

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
        self.assertEqual(data[1].word, "World ")
        self.assertEqual(data[2].word, "Goodbye ")

    def test_result_stats(self):
        td = [
            _td("hello ", 0.0, 0.5),
            _td("wurld ", 0.5, 1.0),
        ]
        data, result = correct_transcription_from_lyrics(td, "Hello World")
        self.assertEqual(result.words_total, 2)
        self.assertEqual(result.reference_words, 2)
        self.assertIsInstance(result, LyricsLookupResult)

    # --- Freestyle (parentheses) ---

    def test_freestyle_flag_set(self):
        """Parenthesized words in reference should set is_freestyle."""
        td = [
            _td("hello ", 0.0, 0.5),
            _td("ooh ", 0.5, 1.0),
            _td("world ", 1.0, 1.5),
        ]
        data, result = correct_transcription_from_lyrics(
            td, "Hello (ooh) world"
        )
        self.assertFalse(data[0].is_freestyle)
        self.assertTrue(data[1].is_freestyle)
        self.assertFalse(data[2].is_freestyle)
        self.assertEqual(result.freestyle_words, 1)

    def test_freestyle_brackets(self):
        """Bracketed words in reference should set is_freestyle."""
        td = [
            _td("hello ", 0.0, 0.5),
            _td("yeah ", 0.5, 1.0),
            _td("world ", 1.0, 1.5),
        ]
        data, _ = correct_transcription_from_lyrics(
            td, "Hello [yeah] world"
        )
        self.assertTrue(data[1].is_freestyle)

    def test_freestyle_multi_word(self):
        """Multi-word parenthesized region."""
        td = [
            _td("oh ", 0.0, 0.3),
            _td("yeah ", 0.3, 0.6),
            _td("baby ", 0.6, 1.0),
        ]
        data, result = correct_transcription_from_lyrics(
            td, "(oh yeah baby)"
        )
        self.assertTrue(all(d.is_freestyle for d in data))
        self.assertEqual(result.freestyle_words, 3)

    def test_freestyle_not_overcorrected(self):
        """Freestyle words should NOT be changed by post-processing
        (this test verifies the is_freestyle flag survives correction)."""
        td = [
            _td("sing ", 0.0, 0.5),
            _td("ooh ", 0.5, 1.0),
        ]
        data, _ = correct_transcription_from_lyrics(
            td, "sing (ooh)"
        )
        self.assertEqual(data[1].word, "ooh ")
        self.assertTrue(data[1].is_freestyle)

    # --- Linebreaks ---

    def test_linebreak_flag_set(self):
        """Line breaks in reference should set line_break_after."""
        td = [
            _td("hello ", 0.0, 0.5),
            _td("world ", 0.5, 1.0),
            _td("goodbye ", 1.5, 2.0),
            _td("moon ", 2.0, 2.5),
        ]
        data, result = correct_transcription_from_lyrics(
            td, "Hello world\nGoodbye moon"
        )
        # "world" is end of first line → linebreak
        self.assertTrue(data[1].line_break_after)
        # "moon" is last word → no linebreak
        self.assertFalse(data[3].line_break_after)
        self.assertGreater(result.linebreaks_applied, 0)

    def test_no_linebreak_on_single_line(self):
        td = [_td("hello ", 0.0, 0.5), _td("world ", 0.5, 1.0)]
        data, result = correct_transcription_from_lyrics(td, "Hello world")
        self.assertFalse(data[0].line_break_after)
        self.assertFalse(data[1].line_break_after)
        self.assertEqual(result.linebreaks_applied, 0)

    def test_empty_line_paragraph_break(self):
        """Empty lines should produce linebreaks too."""
        td = [
            _td("verse ", 0.0, 0.5),
            _td("one ", 0.5, 1.0),
            _td("verse ", 2.0, 2.5),
            _td("two ", 2.5, 3.0),
        ]
        data, _ = correct_transcription_from_lyrics(
            td, "verse one\n\nverse two"
        )
        # "one" should have linebreak (end of first verse)
        self.assertTrue(data[1].line_break_after)

    # --- Combined ---

    def test_freestyle_and_linebreak(self):
        """Freestyle and linebreak flags should work together."""
        td = [
            _td("hello ", 0.0, 0.5),
            _td("ooh ", 0.5, 1.0),
            _td("world ", 1.0, 1.5),
        ]
        data, _ = correct_transcription_from_lyrics(
            td, "Hello (ooh)\nworld"
        )
        self.assertTrue(data[1].is_freestyle)
        self.assertTrue(data[1].line_break_after)
        self.assertFalse(data[2].line_break_after)  # last word


class TestLyricsLookupResult(unittest.TestCase):
    def test_defaults(self):
        r = LyricsLookupResult()
        self.assertEqual(r.words_corrected, 0)
        self.assertEqual(r.words_kept, 0)
        self.assertEqual(r.words_total, 0)
        self.assertEqual(r.reference_words, 0)
        self.assertEqual(r.freestyle_words, 0)
        self.assertEqual(r.linebreaks_applied, 0)


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


if __name__ == "__main__":
    unittest.main()
