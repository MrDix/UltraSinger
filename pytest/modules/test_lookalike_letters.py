"""Tests for folding Cyrillic and Greek letters that stand in for Latin ones."""

import unicodedata
import unittest

from src.modules.lookalike_letters import _LOOKALIKES, fold_lookalike_letters

# Latin letters and a Cyrillic look-alike of each (by code point: they cannot be told apart here)
_CYRILLIC = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440", "c": "\u0441", "x": "\u0445",
             "y": "\u0443", "i": "\u0456", "s": "\u0455", "A": "\u0410", "T": "\u0422", "H": "\u041d",
             "I": "\u0406"}
_GREEK = {"o": "\u03bf", "v": "\u03bd", "i": "\u03b9", "A": "\u0391", "T": "\u03a4", "E": "\u0395"}


def _disguised(text: str, letters: str = "aeo", lookalikes: dict = _CYRILLIC) -> str:
    """The text with these Latin letters replaced by their look-alikes."""
    return "".join(lookalikes[letter] if letter in letters else letter for letter in text)


class TestFoldLookalikeLetters(unittest.TestCase):
    def test_words_that_mix_latin_letters_and_lookalikes(self):
        text = "One two three, over the ocean we come home again"
        disguised = _disguised(text)
        self.assertNotEqual(disguised, text)
        self.assertEqual(fold_lookalike_letters(disguised), text)

    def test_capitals_and_rarer_lookalikes(self):
        text = "The Hills Are Table Land, explicitly mixing syllables In A Tiny Hat"
        self.assertEqual(fold_lookalike_letters(_disguised(text, letters="aeopcxyisATHI")), text)

    def test_greek_lookalikes(self):
        text = "Even over The Top, five voices Are in tune"
        self.assertEqual(fold_lookalike_letters(_disguised(text, letters="oviATE", lookalikes=_GREEK)), text)

    def test_words_of_lookalikes_only_in_a_latin_text(self):
        # "a" and "o" alone, "see" and "yes" with every letter replaced
        text = "I found a way, o my friend, to see the light and tell them yes"
        words = text.split()
        for n in (2, 4, 8, 14):
            words[n] = _disguised(words[n], letters="aeosy")
        disguised = " ".join(words)
        self.assertEqual(sum(a != b for a, b in zip(disguised, text)), 8)
        self.assertEqual(fold_lookalike_letters(disguised), text)

    def test_a_text_in_cyrillic_script_stays(self):
        for text in (
                # Russian, Ukrainian, Bulgarian
                "Это просто придуманный текст для проверки, в нём нет ничего особенного",
                "Це просто вигаданий текст для перевірки, у ньому немає нічого особливого",
                "Това е просто измислен текст за проверка, в него няма нищо особено",
                "ЭТО ПРОСТО ПРИДУМАННЫЙ ТЕКСТ ДЛЯ ПРОВЕРКИ, В НЁМ НЕТ НИЧЕГО ОСОБЕННОГО",
                # with words in Latin letters, and with a Latin letter ("o") in a Cyrillic word
                "Это просто придуманный текст для проверки, one two three",
                "ЭТО ПРОСТО ПРИДУМАННЫЙ ТЕКСТ, one two three",
                "Это пр" + "o" + "сто придуманный текст для проверки"):
            with self.subTest(text=text):
                self.assertEqual(fold_lookalike_letters(text), text)

    def test_a_text_in_greek_script_stays(self):
        for text in ("Αυτό είναι ένα φανταστικό κείμενο για δοκιμή, τίποτα το ιδιαίτερο",
                     "ΑΥΤΟ ΕΙΝΑΙ ΕΝΑ ΦΑΝΤΑΣΤΙΚΟ ΚΕΙΜΕΝΟ ΓΙΑ ΔΟΚΙΜΗ, ΤΙΠΟΤΑ ΤΟ ΙΔΙΑΙΤΕΡΟ",
                     "Αυτό είναι ένα φανταστικό κείμενο για δοκιμή, one two three"):
            with self.subTest(text=text):
                self.assertEqual(fold_lookalike_letters(text), text)

    def test_words_of_lookalikes_alone_stay(self):
        """They can be words of either script: without Latin text around them, nothing tells."""
        for text in ("\u0430 \u043e", "\u043e\u0441\u0430", "\u0391\u03a4\u0395"):
            with self.subTest(text=text):
                self.assertEqual(fold_lookalike_letters(text), text)

    def test_a_cyrillic_word_in_a_latin_text_stays(self):
        """Folded are only words whose Cyrillic letters are all look-alikes."""
        text = "One two three, over the ocean we come home, " * 3
        self.assertEqual(fold_lookalike_letters(_disguised(text) + "привет"), text + "привет")

    def test_few_lookalikes_beside_cyrillic_words_stay(self):
        """Most Cyrillic letters of the text are no look-alikes: it is not taken for a disguised one."""
        text = _disguised("One two", letters="e") + " three, over the ocean: это просто придуманный текст"
        self.assertEqual(fold_lookalike_letters(text), text)

    def test_time_tags_and_lines_stay(self):
        synced = "[ar:Some Band]\n[00:01.00] One two <00:01.50>three\n[00:03.00]\n[00:04.00][00:20.00] Over the ocean\n"
        folded = fold_lookalike_letters(_disguised(synced))
        self.assertEqual(folded, synced)
        self.assertEqual(len(folded), len(_disguised(synced)))

    def test_ordinary_text_stays(self):
        for text in ("One two three, over the ocean we come home again",
                     "Ein ausgedachter Text: über die Brücke, Straße für Straße",
                     "Un texte inventé, déjà là où le cœur s'en va",
                     "[00:01.00] One two three\n[00:03.00] 一二三\n[00:05.00] ひとつ ふたつ",
                     "", "   \n"):
            with self.subTest(text=text):
                self.assertEqual(fold_lookalike_letters(text), text)
        self.assertIsNone(fold_lookalike_letters(None))

    def test_the_lookalikes_are_letters_of_the_other_scripts(self):
        for lookalike, latin in _LOOKALIKES.items():
            name = unicodedata.name(chr(lookalike))
            with self.subTest(letter=name):
                self.assertTrue(name.startswith(("CYRILLIC", "GREEK")))
                self.assertTrue(chr(latin).isascii() and chr(latin).isalpha())
                self.assertEqual(chr(lookalike).isupper(), chr(latin).isupper())


if __name__ == "__main__":
    unittest.main()
