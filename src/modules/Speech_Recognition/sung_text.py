"""Which written words are sung, in lyrics and in transcriptions alike.

A word of punctuation or symbols alone is no note of its own. Closing
punctuation belongs to the word before it ("Hello ." is sung as "Hello."),
other symbols are not sung: dashes, and the music note that a transcriber
writes for an instrumental part ("♪"). Symbols that are read as words ("&",
"+", "%", "♥") stay.
"""

from __future__ import annotations

import re
from dataclasses import replace

from modules.Speech_Recognition.TranscribedData import TranscribedData

_WORD_CHAR_RE = re.compile(r"\w")
# Punctuation that closes the word before it: "Hello .", "hey \"", "oh ..."
_CLOSING_PUNCTUATION_RE = re.compile(r"[.,;:!?…\"'”’»]+")
# Symbols that are sung as words: "you & me" ("and"), "1 + 1", "100 %", "I ♥ you"
_SUNG_SYMBOL_RE = re.compile(r"[&+%$€£¥#@=°♥❤♡]")


def is_sung(word: str) -> bool:
    """Whether ``word`` holds something to sing: a letter, a digit or a
    symbol that is read as a word."""
    return bool(_WORD_CHAR_RE.search(word) or _SUNG_SYMBOL_RE.search(word))


def is_closing_punctuation(word: str) -> bool:
    """Whether ``word`` (spaces around it aside) is only punctuation that
    closes the word before it."""
    return _CLOSING_PUNCTUATION_RE.fullmatch(word.strip()) is not None


def sung_words(words: list[TranscribedData]) -> list[TranscribedData]:
    """The words of a transcription without those that hold nothing to sing.

    Closing punctuation standing alone joins the word before it, before the
    space that ends that word (French puts a space before ``?`` and ``!``);
    other words without anything to sing are left out. The given words are
    not changed.
    """
    kept: list[TranscribedData] = []
    for word in words:
        if is_sung(word.word):
            kept.append(word)
        elif kept and is_closing_punctuation(word.word):
            before = kept[-1].word
            text = before.rstrip()
            kept[-1] = replace(kept[-1], word=text + word.word.strip() + before[len(text):])
    return kept
