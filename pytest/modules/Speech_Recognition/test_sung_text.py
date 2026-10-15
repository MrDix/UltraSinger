"""Which written words are sung: punctuation and symbols alone are no notes."""

import pytest

from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Speech_Recognition.sung_text import is_closing_punctuation, is_sung, sung_words


@pytest.mark.parametrize("word", ["la", "I'm", "1990", "&", "+", "100%", "♥", "rockin'", "Ça"])
def test_words_with_something_to_sing(word):
    assert is_sung(word)


@pytest.mark.parametrize("word", ["♪", "♪♪", "-", "--", "...", "'", "?", "!", "¿", "«", "", " "])
def test_words_with_nothing_to_sing(word):
    assert not is_sung(word)


@pytest.mark.parametrize("word", ["?", "!", ",", "...", "…", "'", "\"", "»", "?! ", " ."])
def test_closing_punctuation(word):
    assert is_closing_punctuation(word)


@pytest.mark.parametrize("word", ["-", "♪", "¿", "«", "(", "a?", ""])
def test_no_closing_punctuation(word):
    assert not is_closing_punctuation(word)


def _word(text, start, end):
    return TranscribedData(word=text, start=start, end=end, confidence=0.9)


def test_transcription_without_music_notes_and_dashes():
    words = [_word("♪ ", 0.0, 0.1), _word("Hello ", 1.0, 1.4), _word("- ", 1.5, 1.6), _word("you ", 1.7, 2.0),
             _word("& ", 2.1, 2.2), _word("me ", 2.3, 2.6), _word("♪ ", 2.7, 2.8)]
    kept = sung_words(words)
    assert [w.word for w in kept] == ["Hello ", "you ", "& ", "me "]
    assert [(w.start, w.end) for w in kept] == [(1.0, 1.4), (1.7, 2.0), (2.1, 2.2), (2.3, 2.6)]


def test_closing_punctuation_joins_the_word_before_its_space():
    """French puts a space before "?" and "!": the mark goes to the word, its time is not used."""
    words = [_word("Pourquoi ", 1.0, 1.5), _word("? ", 1.6, 1.7), _word("rockin ", 2.0, 2.4), _word("' ", 2.5, 2.6),
             _word("Allez", 3.0, 3.3), _word("!", 3.4, 3.5)]
    kept = sung_words(words)
    assert [w.word for w in kept] == ["Pourquoi? ", "rockin' ", "Allez!"]
    assert [(w.start, w.end) for w in kept] == [(1.0, 1.5), (2.0, 2.4), (3.0, 3.3)]
    assert all(w.confidence == 0.9 for w in kept)


def test_punctuation_before_the_first_word_is_left_out():
    assert [w.word for w in sung_words([_word("... ", 0.0, 0.1), _word("? ", 0.2, 0.3), _word("La ", 1.0, 1.2)])] \
        == ["La "]


def test_the_given_words_are_not_changed():
    words = [_word("Hey ", 1.0, 1.2), _word("! ", 1.3, 1.4)]
    sung_words(words)
    assert [w.word for w in words] == ["Hey ", "! "]


def test_nothing_to_sing_at_all():
    assert sung_words([_word("♪ ", 0.0, 1.0), _word("- ", 1.0, 2.0)]) == []
    assert sung_words([]) == []
