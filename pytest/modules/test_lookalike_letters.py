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

    @staticmethod
    def _with_words_disguised(text: str, whole: tuple, partly: tuple) -> str:
        """The text with the words at ``whole`` made of look-alikes only, and one letter of those at ``partly``."""
        words = text.split()
        for n in whole:
            words[n] = _disguised(words[n], letters="aeosy")
        for n in partly:
            words[n] = _disguised(words[n], letters="eo")
        return " ".join(words)

    def test_words_of_lookalikes_only_in_a_text_written_with_lookalikes(self):
        # "a" and "o" alone, "see" and "yes" with every letter replaced; "found", "friend" and "tell" with one
        text = "I found a way, o my friend, to see the light and tell them yes"
        disguised = self._with_words_disguised(text, whole=(2, 4, 8, 14), partly=(1, 6, 12))
        self.assertEqual(sum(a != b for a, b in zip(disguised, text)), 11)
        self.assertEqual(fold_lookalike_letters(disguised), text)

    def test_words_of_lookalikes_only_stay_without_enough_words_that_mix_the_scripts(self):
        """One or two words with a look-alike are slips: they are folded, a word of look-alikes only is not."""
        text = "I found a way, o my friend, to see the light and tell them yes"
        for partly in ((), (1,), (1, 6)):
            with self.subTest(mixed_words=len(partly)):
                disguised = self._with_words_disguised(text, whole=(2, 4, 8, 14), partly=partly)
                self.assertEqual(fold_lookalike_letters(disguised),
                                 self._with_words_disguised(text, whole=(2, 4, 8, 14), partly=()))

    def test_a_word_of_the_other_script_made_of_lookalikes_stays(self):
        """A Russian interjection of three letters that all look like Latin ones, in an English text."""
        interjection = "".join(map(chr, (0x0423, 0x0440, 0x0430)))
        text = ("One two three, over the ocean we come home\nFour five six, the river is wide\n"
                f"Seven eight nine, {interjection}! we sing tonight\nTen more lines and the song is done")
        self.assertEqual(fold_lookalike_letters(text), text)

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

    def test_capital_and_small_letter_are_both_lookalikes(self):
        """Except for the capitals whose small letter looks like a small capital, not like a Latin letter:
        the Cyrillic Ve, Ka, Em, En, Te and the Greek Beta, Epsilon, Zeta, Eta, Mu, Tau."""
        alone = set()
        for lookalike in _LOOKALIKES:
            letter = chr(lookalike)
            other_case = letter.lower() if letter.isupper() else letter.upper()
            if other_case != letter and ord(other_case) not in _LOOKALIKES:
                alone.add(lookalike)
        self.assertEqual(alone, {0x0412, 0x041A, 0x041C, 0x041D, 0x0422,
                                 0x0392, 0x0395, 0x0396, 0x0397, 0x039C, 0x03A4})
        for capital, small, latin in ((0x04BA, 0x04BB, "Hh"), (0x0500, 0x0501, "Dd")):
            self.assertEqual((chr(capital) + chr(small)).translate(_LOOKALIKES), latin)


if __name__ == "__main__":
    unittest.main()
