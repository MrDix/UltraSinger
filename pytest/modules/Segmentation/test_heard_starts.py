"""Tests for word starts where a transcription heard the words (modules.Segmentation.heard_starts)."""

from __future__ import annotations

import pytest

from modules.Segmentation import heard_starts as hs
from modules.Segmentation.heard_starts import heard_starts, pair_words, start_where_heard, word_key, words_of
from modules.Segmentation.lyrics import Syllable

EARLIER = (hs.MIN_DIFFERENCE_MS + hs.MAX_DIFFERENCE_MS) / 2  # a difference inside the range of the rule


def _words(*words):
    """Syllables of whole words: (text, start ms, end ms[, line start])."""
    return [Syllable(text + " ", float(a), float(b), bool(rest and rest[0])) for text, a, b, *rest in words]


def _heard(*words):
    """Transcribed words: (text, start ms) -> (text, start s, end s)."""
    return [(text, at / 1000.0, at / 1000.0 + 0.3) for text, at in words]


class TestWordKey:
    @pytest.mark.parametrize("text, key", [("Don't ", "dont"), ("(Baby,", "baby"), ("ÜBER", "uber"), ("...", ""),
                                           ("naïve", "naive"), ("rock-n-roll", "rocknroll")])
    def test_case_accents_and_punctuation_do_not_count(self, text, key):
        assert word_key(text) == key


class TestWordsOf:
    def test_a_space_or_a_line_start_ends_a_word(self):
        syllables = [Syllable("on", 0, 100, True), Syllable("ly ", 100, 200), Syllable("you ", 300, 400),
                     Syllable("ba", 500, 600), Syllable("by", 600, 700), Syllable("oh ", 800, 900, True)]
        assert words_of(syllables) == [(0, 2), (2, 3), (3, 5), (5, 6)]
        assert words_of([]) == []


class TestPairWords:
    def test_equal_words_in_order(self):
        pairs = pair_words(["i", "love", "you", "so"], [0, 500, 1000, 1500], ["i", "love", "me", "so"], [0, 480, 900, 1550])
        assert pairs == [(0, 0), (1, 1), (3, 3)]

    def test_a_repeated_word_is_paired_with_the_occurrence_near_in_time(self):
        """The chorus comes three times in the transcription, once in this part of the lyrics."""
        heard_keys = ["oh", "baby", "oh", "baby", "oh", "baby"]
        heard_at = [1000, 1500, 31000, 31500, 61000, 61500]
        assert pair_words(["oh", "baby"], [30900, 31400], heard_keys, heard_at) == [(0, 2), (1, 3)]
        assert pair_words(["oh", "baby"], [60000, 60600], heard_keys, heard_at) == [(0, 4), (1, 5)]

    def test_two_pairs_count_more_than_one_near_in_time(self):
        # "b" could pair near in time with the first heard "b", but then "a" had no partner
        assert pair_words(["a", "b"], [1000, 1100], ["b", "a", "b"], [1050, 5000, 9000]) == [(0, 1), (1, 2)]

    def test_near_pairs_count_more_than_a_few_more_far_away(self):
        """Three words beside where they are sung, not four of them a minute later."""
        keys, starts = ["x", "a", "b", "c"], [0, 1000, 2000, 3000]
        heard_keys = ["a", "b", "c", "x", "a", "b", "c"]
        heard_at = [1000, 2000, 3000, 60000, 61000, 62000, 63000]
        assert pair_words(keys, starts, heard_keys, heard_at) == [(1, 0), (2, 1), (3, 2)]

    def test_words_without_a_key_pair_with_nothing(self):
        assert pair_words(["", "you"], [0, 500], ["", "you"], [0, 500]) == [(1, 1)]

    def test_nothing_to_pair(self):
        assert pair_words([], [], ["a"], [0]) == []
        assert pair_words(["a"], [0], [], []) == []

    def test_too_many_words_are_not_paired(self, monkeypatch):
        monkeypatch.setattr(hs, "MAX_PAIRING_CELLS", 3)
        assert pair_words(["a", "b"], [0, 1], ["a", "b"], [0, 1]) == []
        assert pair_words(["a"], [0], ["a", "b", "c"], [0, 1, 2]) == [(0, 0)]


class TestHeardStarts:
    def test_a_word_heard_earlier_where_a_note_begins_starts_there(self):
        syllables = _words(("your", 1000, 2000), ("eyes", 4000, 5000), ("shine", 6000, 6500))
        at = 4000 - EARLIER
        moved = heard_starts(syllables, _heard(("your", 1010), ("eyes", at), ("shine", 6020)), [1000, at + 20, 6000])
        assert moved == {1: at}

    def test_not_without_a_note_at_the_heard_start(self):
        syllables = _words(("your", 1000, 2000), ("eyes", 4000, 5000))
        at = 4000 - EARLIER
        heard = _heard(("your", 1010), ("eyes", at))
        assert heard_starts(syllables, heard, [1000, at + hs.NOTE_AT_START_MS + 1]) == {}
        assert heard_starts(syllables, heard, [1000, at - hs.NOTE_AT_START_MS]) == {1: at}
        assert heard_starts(syllables, heard, []) == {}

    def test_not_with_a_note_at_the_aligned_start(self):
        """A note at both starts: the alignment is right about as often as the transcription."""
        syllables = _words(("your", 1000, 2000), ("eyes", 4000, 5000))
        at = 4000 - EARLIER
        heard = _heard(("your", 1010), ("eyes", at))
        assert heard_starts(syllables, heard, [1000, at, 4000 + hs.NOTE_AT_START_MS]) == {}
        assert heard_starts(syllables, heard, [1000, at, 4000 - hs.NOTE_AT_START_MS]) == {}
        assert heard_starts(syllables, heard, [1000, at, 4000 + hs.NOTE_AT_START_MS + 1]) == {1: at}

    def test_not_for_a_small_difference(self):
        syllables = _words(("your", 1000, 2000), ("eyes", 4000, 5000))
        at = 4000 - hs.MIN_DIFFERENCE_MS
        assert heard_starts(syllables, _heard(("eyes", at)), [1000, at]) == {}
        assert heard_starts(syllables, _heard(("eyes", at - 1)), [1000, at - 1]) == {1: at - 1}

    def test_not_for_a_word_heard_far_away(self):
        """Another occurrence of the word, or a slip of the transcription."""
        syllables = _words(("your", 1000, 2000), ("eyes", 14000, 15000))
        at = 14000 - hs.MAX_DIFFERENCE_MS - 1
        assert heard_starts(syllables, _heard(("eyes", at)), [1000, at]) == {}
        assert heard_starts(syllables, _heard(("eyes", at + 1)), [1000, at + 1]) == {1: at + 1}

    def test_a_word_heard_later_starts_there_as_well(self):
        syllables = _words(("your", 1000, 2000), ("eyes", 4000, 5000), ("shine", 9000, 9500))
        later = 4000 + hs.MAX_DIFFERENCE_MS
        assert heard_starts(syllables, _heard(("eyes", 4300)), [1000, 4300]) == {1: 4300.0}
        assert heard_starts(syllables, _heard(("eyes", later)), [1000, later]) == {1: later}
        assert heard_starts(syllables, _heard(("eyes", later + 1)), [1000, later + 1]) == {}

    def test_a_word_does_not_pass_the_words_around_it(self, monkeypatch):
        monkeypatch.setattr(hs, "MAX_DIFFERENCE_MS", 5000.0)
        syllables = _words(("your", 1000, 2000), ("eyes", 4000, 5000), ("shine", 6000, 6500))
        notes = [900, 1100, 5900, 6200]
        # heard before the start of the word before it, and just after it
        assert heard_starts(syllables, _heard(("eyes", 900)), notes) == {}
        assert heard_starts(syllables, _heard(("eyes", 1100)), notes) == {1: 1100.0}
        # heard after the start of the word after it, and just before it
        assert heard_starts(syllables, _heard(("eyes", 6200)), notes) == {}
        assert heard_starts(syllables, _heard(("eyes", 5900)), notes) == {1: 5900.0}
        # the word before it has moved: its new start counts
        syllables = _words(("your", 1000, 2000), ("eyes", 4000, 5000), ("shine", 6000, 6500))
        heard = _heard(("your", 3000), ("eyes", 2900))
        assert heard_starts(syllables, heard, [2900, 3000]) == {0: 3000.0}

    def test_a_word_does_not_pass_a_syllable_of_the_word_before(self):
        """Heard after the start of the word before it, but before its last syllable."""
        syllables = [Syllable("hel", 1000, 1500, True), Syllable("lo ", 1500, 3000), Syllable("world ", 3200, 4000)]
        heard = [("hello", 1.0, 1.4), ("world", 1.3, 1.8)]
        assert heard_starts(syllables, heard, [1000, 1300, 2600]) == {}
        assert heard_starts(syllables, [("hello", 1.0, 1.4), ("world", 1.6, 1.8)], [1000, 1600, 2600]) == {2: 1600.0}

    def test_the_last_syllable_of_a_word_stays_before_the_word_after(self):
        """A word heard later is pressed towards its end: its syllables stay before the next word."""
        syllables = [Syllable("on", 1000, 1500, True), Syllable("ly ", 1500, 3000), Syllable("you ", 2500, 4000)]
        # ("only" 1000-3000 runs into "you" at 2500 in the alignment already)
        assert heard_starts(syllables, [("only", 2.2, 2.4)], [2200]) == {}   # "ly" would start at 2600
        assert heard_starts(syllables, [("only", 1.9, 2.4)], [1900]) == {0: 1900.0}  # "ly" at 2450

    def test_several_words_of_a_line(self):
        syllables = _words(("and", 1000, 1300), ("your", 3000, 3400), ("eyes", 4000, 5000), ("shine", 6000, 6500))
        heard = _heard(("and", 1020), ("your", 2500), ("eyes", 3400), ("shine", 6010))
        assert heard_starts(syllables, heard, [1000, 2500, 3400, 6000]) == {1: 2500.0, 2: 3400.0}

    def test_the_index_is_the_first_syllable_of_the_word(self):
        syllables = [Syllable("oh ", 0, 300, True), Syllable("ba", 2000, 2300), Syllable("by ", 2300, 2600)]
        at = 2000 - EARLIER
        assert heard_starts(syllables, _heard(("Baby,", at)), [0, at]) == {1: at}

    def test_nothing_without_words(self):
        assert heard_starts([], _heard(("a", 0)), [0]) == {}
        assert heard_starts(_words(("a", 0, 100)), [], [0]) == {}
        assert heard_starts(_words(("a", 0, 100)), [("...", 0.0, 0.1)], [0]) == {}


class TestStartWhereHeard:
    def test_the_word_starts_where_it_was_heard(self):
        syllables = _words(("your", 1000, 2000), ("eyes", 4000, 5000))
        at = 4000 - EARLIER
        out, count = start_where_heard(syllables, _heard(("eyes", at)), [1000, at])
        assert count == 1
        assert (out[1].text, out[1].start_ms, out[1].end_ms) == ("eyes ", at, 5000)
        assert out[0] == syllables[0]
        assert syllables[1].start_ms == 4000  # the given syllables are not changed

    def test_the_syllables_of_the_word_share_its_new_time_by_their_letters(self):
        syllables = [Syllable("oh ", 0, 300, True), Syllable("ba", 2000, 2300), Syllable("by ", 2300, 2600, False, ("b", "y "))]
        at = 2000 - EARLIER
        out, count = start_where_heard(syllables, _heard(("baby", at)), [0, at])
        assert count == 1
        middle = at + (2600 - at) / 2
        assert [(s.text, s.start_ms, s.end_ms) for s in out[1:]] == [("ba", at, middle), ("by ", middle, 2600)]
        assert out[2].parts == ("b", "y ") and not out[1].line_start

    def test_a_word_heard_after_its_end_keeps_a_short_time(self):
        syllables = _words(("your", 1000, 1200), ("eyes", 9000, 9500))
        out, count = start_where_heard(syllables, _heard(("your", 2000)), [2000])
        assert count == 1 and (out[0].start_ms, out[0].end_ms) == (2000.0, 2000.0 + hs.MIN_WORD_MS)

    def test_nothing_to_move(self):
        syllables = _words(("your", 1000, 2000), ("eyes", 4000, 5000))
        for heard in (None, [], _heard(("your", 1010), ("eyes", 4020))):
            out, count = start_where_heard(syllables, heard, [1000, 4000])
            assert out is syllables and count == 0
