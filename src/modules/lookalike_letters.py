"""Cyrillic and Greek letters that stand in for the Latin letters they look like.

Lyrics databases hold Latin-script lyrics in which such letters replace the
Latin ones (the "o" of "love" written as the Cyrillic letter): to the eye
the same text, to every comparison another one. It matches no word of a
transcription, its language is not recognised, and the letters end up in
the chart.
"""

from __future__ import annotations

import re

# The look-alikes (written by their code points: they cannot be told from
# the Latin letters in an editor) and the Latin letters they stand for.
# Capital and small letter are both in it where both look like the Latin
# ones; the small letters of the Cyrillic В К М Н Т and of the Greek Β Ε Ζ
# Η Μ Τ do not (they look like small capitals).
_LOOKALIKES = str.maketrans(
    # Cyrillic small letters а е о р с х у, і ј ѕ of Ukrainian, Serbian and
    # Macedonian, and һ ԁ ԛ ԝ ӏ ѵ
    "\u0430\u0435\u043e\u0440\u0441\u0445\u0443\u0456\u0458\u0455\u04bb\u0501\u051b\u051d\u04cf\u0475"
    # Cyrillic capitals А В Е К М Н О Р С Т Х У, І Ј Ѕ, and Ԛ Ԝ Ӏ Ѵ Һ Ԁ
    "\u0410\u0412\u0415\u041a\u041c\u041d\u041e\u0420\u0421\u0422\u0425\u0423\u0406\u0408\u0405"
    "\u051a\u051c\u04c0\u0474\u04ba\u0500"
    # Greek small letters α ι κ ν ο ρ υ χ, and ϲ ϳ
    "\u03b1\u03b9\u03ba\u03bd\u03bf\u03c1\u03c5\u03c7\u03f2\u03f3"
    # Greek capitals Α Β Ε Ζ Η Ι Κ Μ Ν Ο Ρ Τ Υ Χ, and Ϲ Ϳ
    "\u0391\u0392\u0395\u0396\u0397\u0399\u039a\u039c\u039d\u039f\u03a1\u03a4\u03a5\u03a7\u03f9\u037f",
    "aeopcxyijshdqwlv"
    "ABEKMHOPCTXYIJS"
    "QWIVHD"
    "aikvopuxcj"
    "ABEZHIKMNOPTYXCJ",
)
# A word: letters, also joined by an apostrophe ("don't")
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
# A text is in Cyrillic or Greek script itself when more than this share of
# its Cyrillic and Greek letters are no look-alikes. In Russian, Ukrainian,
# Bulgarian and Greek text they are a third and more (б, г, д, ж, и, л, п
# ...; γ, δ, θ, λ, π, σ ...), in capitals alone still about a fifth; in
# Latin text written with look-alikes none are.
REAL_SCRIPT_SHARE = 0.1
# Words made of look-alikes only are folded when at least this many words of
# the text mix Latin letters with look-alikes: that shows a text written
# with them. One or two such words are slips of a keyboard layout (of 429
# lyrics looked at, 4 had two of them and no word of look-alikes only, the
# ones written with look-alikes 353 among 410 words); a real word of the
# other script beside them stays what it is.
MIXED_WORDS = 3
# ... and when at most this share of the letters of the text are not Latin
# (with the look-alikes folded in the words that have Latin letters)
OTHER_SCRIPT_SHARE = 0.2


def _is_latin(letter: str) -> bool:
    return letter.isalpha() and (letter < "\u0250" or "\u1e00" <= letter <= "\u1eff")


def _is_cyrillic_or_greek(letter: str) -> bool:
    return letter.isalpha() and ("\u0370" <= letter <= "\u052f" or "\u1f00" <= letter <= "\u1fff")


def fold_lookalike_letters(text: str) -> str:
    """The text with the Cyrillic and Greek letters that stand in for Latin
    ones replaced by these (the Cyrillic letter in place of the "o" of
    "love" by the Latin "o").

    Only Latin text written with look-alikes is changed; a text in Cyrillic
    or Greek script stays as it is:

    - A text with more than ``REAL_SCRIPT_SHARE`` of its Cyrillic and Greek
      letters being no look-alikes is in that script (Russian, Ukrainian,
      Bulgarian, Greek ...): nothing is changed, also not in a word that has
      a Latin letter among them.
    - Else the look-alikes are folded in every word that has Latin letters
      too and no other Cyrillic or Greek letter.
    - A word made of look-alikes only (one that reads as "a", "I" or "yes")
      can be a word of either script. It is folded when the text is a Latin
      one written with look-alikes: at least ``MIXED_WORDS`` of its words
      mix Latin letters with look-alikes, and at most
      ``OTHER_SCRIPT_SHARE`` of its letters are then not Latin. Without
      such mixed words nothing shows whose word it is: a short Russian or
      Greek word in an English text stays as it is.

    Each letter is replaced by one letter: the time tags and the lines of
    synced lyrics, and all other characters, stay where they are.
    """
    words = []  # (start, end, Latin letters, look-alikes, other Cyrillic or Greek letters)
    for match in _WORD_RE.finditer(text or ""):
        word = match.group(0)
        folded = word.translate(_LOOKALIKES)
        words.append((match.start(), match.end(),
                      sum(1 for letter in word if _is_latin(letter)),
                      sum(1 for letter, latin in zip(word, folded) if letter != latin),
                      sum(1 for letter in folded if _is_cyrillic_or_greek(letter))))
    lookalikes = sum(word[3] for word in words)
    others = sum(word[4] for word in words)
    if not lookalikes or others > REAL_SCRIPT_SHARE * (lookalikes + others):
        return text
    letters = sum(1 for letter in text if letter.isalpha())
    # The words that mix Latin letters with look-alikes, and the Latin letters once these are folded
    mixed = [word for word in words if word[2] and word[3] and not word[4]]
    latin = sum(word[2] for word in words) + sum(word[3] for word in mixed)
    latin_text = len(mixed) >= MIXED_WORDS and letters - latin <= OTHER_SCRIPT_SHARE * letters
    parts, done = [], 0
    for start, end, word_latin, word_lookalikes, word_others in words:
        if word_lookalikes and not word_others and (word_latin or latin_text):
            parts += [text[done:start], text[start:end].translate(_LOOKALIKES)]
            done = end
    parts.append(text[done:])
    return "".join(parts)
