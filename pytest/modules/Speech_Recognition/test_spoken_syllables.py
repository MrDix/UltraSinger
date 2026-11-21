"""Tests for the syllables a written word is spoken with (modules.Speech_Recognition.spoken_syllables)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from modules.Speech_Recognition import spoken_syllables as sp
from modules.Speech_Recognition.spoken_syllables import SpokenSyllables, split_word


def cut(word: str, count: int, **kwargs) -> str:
    return "-".join(split_word(word, count, **kwargs))


class TestSplitWord:
    @pytest.mark.parametrize("word, count, expected", [
        ("only", 2, "on-ly"), ("baby", 2, "ba-by"), ("never", 2, "ne-ver"), ("city", 2, "ci-ty"),
        ("away", 2, "a-way"), ("over", 2, "o-ver"), ("again", 2, "a-gain"), ("gonna", 2, "gon-na"),
        ("really", 2, "real-ly"), ("happy", 2, "hap-py"), ("broken", 2, "bro-ken"), ("crazy", 2, "cra-zy"),
        ("america", 4, "a-me-ri-ca"), ("nananana", 4, "na-na-na-na"), ("Abend", 2, "A-bend"),
    ])
    def test_cut_between_the_vowel_groups(self, word, count, expected):
        assert cut(word, count) == expected

    @pytest.mark.parametrize("word, count, expected", [
        ("hundred", 2, "hun-dred"), ("pumpkin", 2, "pump-kin"), ("monster", 2, "mon-ster"),
        ("angry", 2, "an-gry"), ("kingdom", 2, "king-dom"), ("mother", 2, "mo-ther"),
        ("nowhere", 2, "no-where"), ("nothing", 2, "no-thing"),
    ])
    def test_consonants_that_belong_together_stay_together(self, word, count, expected):
        assert cut(word, count) == expected

    @pytest.mark.parametrize("word, expected", [
        ("little", "lit-tle"), ("people", "peo-ple"), ("table", "ta-ble"), ("apple", "ap-ple"),
        ("single", "sin-gle"), ("castle", "cas-tle"), ("settled", "set-tled"), ("apples", "ap-ples"),
    ])
    def test_a_consonant_goes_with_the_l_of_an_ending(self, word, expected):
        assert cut(word, 2) == expected

    @pytest.mark.parametrize("word, count, expected", [
        ("lovely", 2, "love-ly"), ("lonely", 2, "lone-ly"), ("alone", 2, "a-lone"), ("desire", 2, "de-sire"),
        ("sometimes", 2, "some-times"), ("somewhere", 2, "some-where"), ("every", 2, "eve-ry"),
        ("everything", 3, "eve-ry-thing"), ("everyone", 3, "eve-ry-one"), ("someone", 2, "some-one"),
        ("anyone", 3, "a-ny-one"),
    ])
    def test_a_silent_e_is_no_syllable(self, word, count, expected):
        """More vowel groups than syllables: the "e" after one consonant joins the syllable before it."""
        assert cut(word, count) == expected

    def test_a_word_is_not_cut_into_more_parts_than_it_is_spoken_with(self):
        assert cut("love", 1) == "love"
        assert cut("loved", 1) == "loved"
        assert cut("sometimes", 1) == "sometimes"
        assert cut("wanted", 2) == "wan-ted"

    @pytest.mark.parametrize("word, count, expected", [
        ("radio", 3, "ra-di-o"), ("video", 3, "vi-de-o"), ("material", 4, "ma-te-ri-al"),
        ("suicidal", 4, "su-i-ci-dal"), ("cruel", 2, "cru-el"), ("poem", 2, "po-em"),
        ("going", 2, "go-ing"), ("being", 2, "be-ing"), ("seeing", 2, "see-ing"), ("goin", 2, "go-in"),
        ("crying", 2, "cry-ing"), ("saying", 2, "say-ing"), ("día", 2, "dí-a"), ("país", 2, "pa-ís"),
    ])
    def test_a_vowel_group_is_cut_where_the_word_has_more_vowel_sounds(self, word, count, expected):
        assert cut(word, count) == expected

    def test_a_vowel_group_is_not_cut_without_need(self):
        """The spoken count is the limit: "ea", "ou" and "io" are one sound in these words."""
        assert cut("real", 1) == "real"
        assert cut("nation", 2) == "na-tion"
        assert cut("idea", 2) == "i-dea"
        # a pair that is no known place for a cut stays, whatever the count says
        assert cut("hour", 2) == "hour"
        assert cut("yeah", 2) == "yeah"

    @pytest.mark.parametrize("word, count, tail, expected", [
        ("realize", 3, 1, "rea-lize"), ("experience", 4, 2, "ex-pe-ri-ence"), ("create", 2, 1, "create"),
        ("realized", 3, 2, "rea-lized"), ("alone", 2, 1, "a-lone"),
    ])
    def test_a_last_e_that_is_not_spoken_makes_no_syllable(self, word, count, tail, expected):
        """The word ends in more consonant sounds than it has letters after the "e"."""
        assert cut(word, count, tail=tail) == expected

    @pytest.mark.parametrize("word, count, tail, expected", [
        ("maybe", 2, 0, "may-be"), ("wanted", 2, 1, "wan-ted"), ("little", 2, 0, "lit-tle"),
        ("settled", 2, 1, "set-tled"), ("roses", 2, 1, "ro-ses"), ("Liebe", 2, 0, "Lie-be"),
    ])
    def test_a_last_e_that_is_spoken_is_a_syllable(self, word, count, tail, expected):
        assert cut(word, count, tail=tail) == expected

    def test_without_the_sounds_at_the_end_a_last_e_counts_as_it_is_written(self):
        assert cut("realize", 3) == "rea-li-ze"
        # with them, the hyphenation's two parts stay: the word cannot be cut into more
        assert split_word("realize", 3, ["real", "ize"], tail=1) == ["real", "ize"]

    @pytest.mark.parametrize("word, expected", [
        ("singing", "sing-ing"), ("ringing", "ring-ing"), ("hanging", "hang-ing"), ("bringin", "bring-in"),
        ("finger", "fin-ger"), ("engine", "en-gine"), ("angry", "an-gry"),
    ])
    def test_ng_stays_together_before_the_ending_ing(self, word, expected):
        assert cut(word, 2) == expected

    def test_ng_before_an_ending_with_an_apostrophe(self):
        assert cut("singin'", 2) == "sing-in'"
        assert cut("bringin’", 2) == "bring-in’"

    def test_ng_of_two_sounds_is_cut_between_them(self):
        """A soft g: "plunging" is n and a sound of its own, not the one sound of "singing"."""
        assert cut("plunging", 2, ng=False) == "plun-ging"
        assert cut("plunging", 2, ng=True) == "plung-ing"
        assert cut("singing", 2, ng=None) == "sing-ing"

    def test_no_cut_inside_a_vowel_group_when_not_asked_for(self):
        assert cut("radio", 3, inner=False) == "ra-dio"
        assert cut("going", 2, inner=False) == "going"
        assert cut("america", 4, inner=False) == "a-me-ri-ca"

    def test_y_is_a_consonant_before_a_vowel_at_the_start_and_between_vowels(self):
        assert cut("beyond", 2) == "be-yond"
        assert cut("yellow", 2) == "yel-low"
        assert cut("maybe", 2) == "may-be"

    def test_the_parts_of_the_hyphenation_are_kept_when_they_are_enough(self):
        assert split_word("beautiful", 3, ["beau", "ti", "ful"]) == ["beau", "ti", "ful"]
        # also when the word is spoken with fewer syllables than they are
        assert split_word("science", 1, ["sci", "ence"]) == ["sci", "ence"]
        assert split_word("different", 2, ["dif", "fer", "ent"]) == ["dif", "fer", "ent"]

    def test_too_few_parts_of_the_hyphenation_are_cut_further(self):
        counts = {"every": 2, "one": 1, "any": 2, "body": 2}
        assert split_word("everyone", 3, ["every", "one"], counts.get) == ["eve", "ry", "one"]
        assert split_word("anybody", 4, ["any", "body"], counts.get) == ["a", "ny", "bo", "dy"]

    def test_the_whole_word_is_cut_anew_when_its_parts_do_not_add_up(self):
        # "rial" alone is spoken with one syllable by this count: the parts make 3, the word has 4
        counts = {"ma": 1, "te": 1, "rial": 1}
        assert split_word("material", 4, ["ma", "te", "rial"], counts.get) == ["ma", "te", "ri", "al"]
        # without a count for the parts
        assert split_word("everyone", 3, ["every", "one"]) == ["eve", "ry", "one"]

    def test_parts_that_cannot_be_more_stay(self):
        assert split_word("rhythm", 2) == ["rhythm"]
        assert split_word("mmm", 3) == ["mmm"]
        assert split_word("TV", 2) == ["TV"]
        assert split_word("radio", 3, ["ra", "dio"], inner=False) == ["ra", "dio"]

    def test_a_part_at_an_apostrophe_is_no_syllable_of_its_own(self):
        assert split_word("you're", 1, ["you", "'re"]) == ["you're"]
        assert split_word("we’ll", 1, ["we", "’ll"]) == ["we’ll"]
        # spoken with two syllables: it stays
        assert split_word("o'clock", 2, ["o'", "clock"]) == ["o'", "clock"]
        assert split_word("you're", 2, ["you", "'re"]) == ["you", "'re"]

    def test_without_a_count_nothing_changes(self):
        assert split_word("only", None) == ["only"]
        assert split_word("only", 0) == ["only"]
        assert split_word("you're", None, ["you", "'re"]) == ["you", "'re"]

    def test_parts_that_are_not_the_word_are_not_used(self):
        assert split_word("only", 2, ["on", "lie"]) == ["on", "ly"]

    def test_a_very_long_text_is_left_alone(self):
        word = "na" * (sp.MAX_WORD_LETTERS // 2 + 1)
        assert split_word(word, 30) == [word]

    @pytest.mark.parametrize("word", ["only", "everyone", "Straße", "İstanbul", "naïve", "o'clock", "x", "aeiou",
                                      "yyyy", "queue", "l'amour", "rock'n'roll", "ÄÖÜ", "ŉ"])
    def test_the_parts_are_always_the_word(self, word):
        for count in range(1, 7):
            for inner in (True, False):
                parts = split_word(word, count, inner=inner)
                assert "".join(parts) == word
                assert all(parts)
                assert len(parts) <= max(count, 1)


class _Backend:
    """Stands in for eSpeak NG: the phonemes of the words it knows."""

    def __init__(self, phonemes: dict[str, str]):
        self.phonemes = phonemes
        self.asked: list[list[str]] = []

    def phonemize(self, words, separator=None, strip=True):
        self.asked.append(list(words))
        return [self.phonemes[word] for word in words]


def _counter(phonemes: dict[str, str]) -> tuple[SpokenSyllables, _Backend]:
    backend = _Backend(phonemes)
    with patch("modules.Speech_Recognition.phoneme_word_starts._espeak_backend", return_value=backend):
        return SpokenSyllables("en-us"), backend


class TestSpokenSyllables:
    def test_one_syllable_for_each_vowel_sound(self):
        spoken, _ = _counter({"baby": "b eɪ b i", "love": "l ʌ v", "radio": "ɹ eɪ d ɪ oʊ", "fire": "f aɪɚ",
                              "little": "l ɪ ɾ əl", "amour": "a m u ʁ", "rien": "ʁ i ɛ̃"})
        assert [spoken.count(word) for word in ("baby", "love", "radio", "fire", "little", "amour", "rien")] == \
            [2, 1, 3, 1, 2, 2, 2]

    def test_a_syllabic_consonant_is_a_syllable(self):
        spoken, _ = _counter({"button": "b ʌ ʔ n̩"})
        assert spoken.count("button") == 2

    def test_several_words_of_an_answer_count_together(self):
        spoken, _ = _counter({"o'clock": "ə | k l ɑː k"})
        assert spoken.count("o'clock") == 2

    def test_no_count_without_a_vowel_sound_or_a_letter(self):
        spoken, backend = _counter({"shh": "ʃ"})
        assert spoken.count("shh") is None
        assert spoken.count("...") is None
        assert spoken.count("24") is None
        assert spoken.count("") is None
        assert spoken.count("x" * (sp.MAX_WORD_LETTERS + 1)) is None
        assert backend.asked == [["shh"]]

    def test_the_words_are_asked_once_and_together(self):
        spoken, backend = _counter({"baby": "b eɪ b i", "love": "l ʌ v", "only": "oʊ n l i"})
        spoken.count_many(["love", "baby", "love", "..."])
        spoken.count_many(["baby", "only"])
        assert backend.asked == [["baby", "love"], ["only"]]
        assert (spoken.count("baby"), spoken.count("love"), spoken.count("only")) == (2, 1, 2)
        assert len(backend.asked) == 2

    def test_the_consonant_sounds_a_word_ends_with(self):
        spoken, _ = _counter({"realize": "ɹ iː ə l aɪ z", "maybe": "m eɪ b i", "times": "t aɪ m z",
                              "little": "l ɪ ɾ əl", "shh": "ʃ"})
        assert [spoken.tail(word) for word in ("realize", "maybe", "times", "little")] == [1, 0, 2, 0]
        assert spoken.tail("shh") is None and spoken.tail("...") is None

    def test_the_one_sound_of_ng_before_the_last_vowel_sound(self):
        spoken, _ = _counter({"singing": "s ɪ ŋ ɪ ŋ", "plunging": "p l ʌ n dʒ ɪ ŋ", "bringin": "b ɹ ɪ ŋ ɪ n",
                              "king": "k ɪ ŋ", "shh": "ʃ"})
        assert [spoken.ng(word) for word in ("singing", "plunging", "bringin", "king")] == [True, False, True, False]
        assert spoken.ng("shh") is None

    def test_words_are_asked_in_small_letters(self):
        """A short word in capitals would be read letter by letter ("IT" as "I T")."""
        spoken, backend = _counter({"it": "ɪ t", "baby": "b eɪ b i"})
        assert (spoken.count("IT"), spoken.count("Baby")) == (1, 2)
        assert backend.asked == [["it"], ["baby"]]

    def test_an_answer_that_does_not_fit_the_words_gives_no_count(self):
        spoken, backend = _counter({})
        backend.phonemize = lambda words, separator=None, strip=True: ["b eɪ b i"]  # one answer for two words
        spoken.count_many(["baby", "love"])
        assert spoken.count("baby") is None and spoken.count("love") is None


class TestSpokenSyllablesOfALanguage:
    def test_no_counts_for_a_language_without_a_voice(self):
        with patch("modules.Speech_Recognition.phoneme_word_starts.espeak_voice", return_value=None):
            assert sp.spoken_syllables("xx") is None

    def test_no_counts_when_espeak_cannot_be_used(self):
        with patch("modules.Speech_Recognition.phoneme_word_starts.espeak_voice", side_effect=ValueError("path")):
            assert sp.spoken_syllables("en") is None
        with patch("modules.Speech_Recognition.phoneme_word_starts.espeak_voice", return_value="en-us"), \
                patch("modules.Speech_Recognition.phoneme_word_starts._espeak_backend",
                      side_effect=RuntimeError("no library")):
            assert sp.spoken_syllables("en") is None

    def test_the_language_code_is_cut_to_two_letters(self):
        with patch("modules.Speech_Recognition.phoneme_word_starts.espeak_voice", return_value=None) as voice:
            sp.spoken_syllables("EN-us")
            sp.spoken_syllables(None)
        assert [call.args[0] for call in voice.call_args_list] == ["en", ""]

    def test_with_espeak(self):
        pytest.importorskip("phonemizer")
        pytest.importorskip("espeakng_loader")
        spoken = sp.spoken_syllables("en")
        if spoken is None:
            pytest.skip("eSpeak NG cannot be used here")
        spoken.count_many(["only", "baby", "love", "radio", "everyone"])
        assert [spoken.count(word) for word in ("only", "baby", "love", "radio", "everyone")] == [2, 2, 1, 3, 3]
        assert (spoken.tail("realize"), spoken.tail("maybe")) == (1, 0)
        assert (spoken.ng("singing"), spoken.ng("plunging")) == (True, False)
        assert split_word("realize", spoken.count("realize"), tail=spoken.tail("realize")) == ["rea", "lize"]
        assert split_word("only", spoken.count("only")) == ["on", "ly"]
        assert split_word("everyone", spoken.count("everyone"), ["every", "one"], spoken.count) == ["eve", "ry", "one"]
