"""Tests for the syllables a word is spoken with on the notes of the model (modules.Segmentation.lyrics).

The hyphenation gives the syllables that are placed (as before); a syllable on
several notes that is spoken with several syllables is spread over them.
"""

from __future__ import annotations

import pytest

from modules.Midi.MidiSegment import MidiSegment
from modules.Segmentation import lyrics as seg_lyrics
from modules.Segmentation.decode import PredictedNote
from modules.Segmentation.lyrics import Syllable, place_lyrics, spread_parts, syllables_from_segments

COUNTS = {"only": 2, "baby": 2, "love": 1, "radio": 3, "everyone": 3, "every": 2, "one": 1, "you're": 1,
          "dancing": 2, "hello": 2, "material": 4, "ma": 1, "te": 1, "rial": 1, "mio": 2, "o'clock": 2}


class _Hyphenator:
    """Stands in for the hyphenation dictionary."""

    SPLITS = {"everyone": ["every", "one"], "you're": ["you", "'re"], "dancing": ["danc", "ing"],
              "hello": ["hel", "lo"], "material": ["ma", "te", "rial"], "o'clock": ["o'", "clock"]}

    def syllables(self, word):
        return self.SPLITS.get(word.lower(), [word])


class _Spoken:
    """Stands in for the syllable counts of eSpeak NG."""

    def __init__(self, counts=None):
        self.counts = COUNTS if counts is None else counts
        self.asked: list[str] = []

    def count_many(self, words):
        self.asked.extend(words)

    def count(self, word):
        return self.counts.get(word.lower())


@pytest.fixture
def spoken(monkeypatch):
    counter = _Spoken()
    monkeypatch.setattr(seg_lyrics, "_hyphenator", lambda language: _Hyphenator())
    monkeypatch.setattr(seg_lyrics, "spoken_syllables", lambda language: counter)
    return counter


def _seg(word, start, end, line_break=False):
    s = MidiSegment("C4", start, end, word)
    s.line_break_after = line_break
    return s


def _notes(*spans):
    return [PredictedNote(a, b, 60) for a, b in spans]


def _syllables(*words):
    """Each word for 1 s, one after the other."""
    return syllables_from_segments([_seg(word, k, k + 1.0) for k, word in enumerate(words)], "en")


class TestSpokenSyllablesOfAWord:
    def test_a_word_the_hyphenation_leaves_whole_carries_its_spoken_syllables(self, spoken):
        syl = _syllables("only ", "love ")
        assert [(s.text, s.parts) for s in syl] == [("only ", ("on", "ly ")), ("love ", ())]
        assert (syl[0].start_ms, syl[0].end_ms) == (0, 1000)

    def test_the_syllables_of_the_hyphenation_are_placed_as_before(self, spoken):
        """Their texts and times are what they were: only ``parts`` is new."""
        syl = _syllables("hello ", "dancing, ")
        assert [(s.text, s.parts) for s in syl] == [("hel", ()), ("lo ", ()), ("danc", ()), ("ing, ", ())]
        assert [s.start_ms for s in syl] == pytest.approx([0, 600, 1000, 1500])

    def test_a_syllable_of_the_hyphenation_is_cut_further(self, spoken):
        syl = _syllables("everyone ")
        assert [(s.text, s.parts) for s in syl] == [("every", ("eve", "ry")), ("one ", ())]

    def test_other_cuts_than_the_hyphenation_s_make_the_word_one_syllable(self, spoken):
        # "ma-te-rial" by the hyphenation, "ma-te-ri-al" spoken: each part ends where a spoken one ends
        syl = _syllables("material ")
        assert [(s.text, s.parts) for s in syl] == [("ma", ()), ("te", ()), ("rial ", ("ri", "al "))]
        # "mate-rial" would not: the word is one syllable with its spoken ones
        _Hyphenator.SPLITS["material"] = ["mat", "erial"]
        try:
            syl = _syllables("material ")
        finally:
            _Hyphenator.SPLITS["material"] = ["ma", "te", "rial"]
        assert [(s.text, s.parts) for s in syl] == [("material ", ("ma", "te", "ri", "al "))]

    def test_a_part_at_an_apostrophe_is_no_syllable(self, spoken):
        """The hyphenation cuts "you're" in two; it is spoken, and now placed, as one."""
        syl = _syllables("you're ", "o'clock ")
        assert [(s.text, s.parts) for s in syl] == [("you're ", ()), ("o'", ()), ("clock ", ())]

    def test_punctuation_and_spaces_stay_at_the_edges(self, spoken):
        syl = syllables_from_segments([_seg(" (only, ", 0.0, 1.0), _seg("\"Baby\" ", 1.0, 2.0)], "en")
        assert [(s.text, s.parts) for s in syl] == [(" (only, ", (" (on", "ly, ")), ("\"Baby\" ", ("\"Ba", "by\" "))]

    def test_the_pieces_of_a_hyphen_chain_each_have_their_syllables(self, spoken):
        syl = _syllables("ba-by-baby, ")
        assert [(s.text, s.parts) for s in syl] == [("ba-", ()), ("by-", ()), ("baby, ", ("ba", "by, "))]

    def test_all_words_are_looked_up_in_one_go(self, spoken):
        _syllables("only ", "love-baby ", "(everyone) ", "... ")
        assert sorted(spoken.asked) == ["baby", "everyone", "love", "only"]

    def test_no_vowel_group_is_cut_in_a_language_it_is_not_measured_for(self, spoken):
        english = syllables_from_segments([_seg("radio ", 0, 1), _seg("mio ", 1, 2)], "en")
        italian = syllables_from_segments([_seg("radio ", 0, 1), _seg("mio ", 1, 2)], "it")
        assert [s.parts for s in english] == [("ra", "di", "o "), ("mi", "o ")]
        assert [s.parts for s in italian] == [("ra", "dio "), ()]

    def test_without_splitting_every_word_stays_whole(self, spoken):
        """Hyphenation switched off: no syllables of the hyphenation, none of the spoken word."""
        syl = syllables_from_segments([_seg("only ", 0, 1), _seg("hello ", 1, 2, line_break=True),
                                       _seg("~ ", 2, 2.5), _seg("you're ", 3, 4)], "en", split_words=False)
        assert [(s.text, s.parts, s.start_ms, s.end_ms, s.line_start) for s in syl] == [
            ("only ", (), 0, 1000, True), ("hello ", (), 1000, 2500, False), ("you're ", (), 3000, 4000, True)]
        assert spoken.asked == []

    def test_without_counts_the_hyphenation_alone(self, monkeypatch):
        monkeypatch.setattr(seg_lyrics, "_hyphenator", lambda language: _Hyphenator())
        monkeypatch.setattr(seg_lyrics, "spoken_syllables", lambda language: None)
        syl = _syllables("only ", "hello ", "you're ")
        assert [(s.text, s.parts) for s in syl] == [("only ", ()), ("hel", ()), ("lo ", ()), ("you", ()), ("'re ", ())]

    def test_counts_that_fail_leave_the_hyphenation(self, monkeypatch):
        class Broken(_Spoken):
            def count_many(self, words):
                raise RuntimeError("eSpeak NG stopped")

        monkeypatch.setattr(seg_lyrics, "_hyphenator", lambda language: _Hyphenator())
        monkeypatch.setattr(seg_lyrics, "spoken_syllables", lambda language: Broken())
        assert [(s.text, s.parts) for s in _syllables("only ", "hello ")] == [("only ", ()), ("hel", ()), ("lo ", ())]

        class BrokenLater(_Spoken):
            def count(self, word):
                raise RuntimeError("eSpeak NG stopped")

        monkeypatch.setattr(seg_lyrics, "spoken_syllables", lambda language: BrokenLater())
        assert [(s.text, s.parts) for s in _syllables("only ", "hello ")] == [("only ", ()), ("hel", ()), ("lo ", ())]

    def test_without_a_hyphenation_the_spoken_syllables_alone(self, monkeypatch):
        monkeypatch.setattr(seg_lyrics, "_hyphenator", lambda language: None)
        monkeypatch.setattr(seg_lyrics, "spoken_syllables", lambda language: _Spoken())
        syl = _syllables("dancing ", "only ")
        assert [(s.text, s.parts) for s in syl] == [("dancing ", ("dan", "cing ")), ("only ", ("on", "ly "))]


class TestSpreadParts:
    def test_one_part_for_each_note(self):
        assert spread_parts(("on", "ly "), [0, 500], 1000) == ["on", "ly "]
        assert spread_parts(("ra", "di", "o "), [0, 300, 600], 900) == ["ra", "di", "o "]

    def test_more_notes_than_parts_the_note_nearest_to_where_the_part_is_expected(self):
        # "ly" is expected at the middle of the time (half of the letters)
        assert spread_parts(("on", "ly"), [0, 450, 800], 1000) == ["on", "ly", "~"]
        assert spread_parts(("on", "ly"), [0, 100, 600], 1000) == ["on", "~", "ly"]
        # a run of short notes, then a held one
        assert spread_parts(("Ma", "ma"), [0, 100, 200, 300, 400], 2000) == ["Ma", "~", "~", "~", "ma"]

    def test_the_parts_keep_their_order(self):
        texts = spread_parts(("ma", "te", "ri", "al"), [0, 100, 200, 300, 400, 500, 600, 700], 800)
        assert [t for t in texts if t != "~"] == ["ma", "te", "ri", "al"]
        assert texts[0] == "ma"

    def test_fewer_notes_than_parts_the_other_parts_join_the_one_before(self):
        assert spread_parts(("ra", "di", "o "), [0, 300], 900) == ["ra", "dio "]
        assert spread_parts(("ra", "di", "o "), [0, 700], 900) == ["radi", "o "]
        assert spread_parts(("ma", "te", "ri", "al"), [0, 400, 600], 800) == ["mate", "ri", "al"]

    def test_one_note_or_one_part(self):
        assert spread_parts(("on", "ly "), [0], 500) == ["only "]
        assert spread_parts(("love ",), [0, 300], 500) == ["love ", "~"]

    def test_all_parts_are_there_once(self):
        for notes in range(1, 9):
            for count in range(2, 6):
                parts = tuple(f"p{k}" for k in range(count))
                texts = spread_parts(parts, [k * 130.0 for k in range(notes)], notes * 130.0 + 400)
                assert len(texts) == notes
                assert "".join(t for t in texts if t != "~") == "".join(parts)
                assert texts[0].startswith("p0")

    def test_with_very_many_notes_each_part_takes_the_nearest_note_that_is_left(self, monkeypatch):
        starts = [k * 100.0 for k in range(20)]
        best = spread_parts(("a", "b", "c"), starts, 2000)
        monkeypatch.setattr(seg_lyrics, "MAX_SPREADS", 10)
        texts = spread_parts(("a", "b", "c"), starts, 2000)
        # "b" is expected at a third of the time, "c" at two thirds
        assert [k for k, text in enumerate(texts) if text != "~"] == [0, 7, 13]
        assert texts == best
        # the last parts keep a note each when the nearest ones are at the end
        monkeypatch.setattr(seg_lyrics, "MAX_SPREADS", 0)
        late = spread_parts(("a", "b", "c", "d"), [0.0, 10.0, 20.0, 1900.0, 1950.0], 2000)
        assert late == ["a", "~", "b", "c", "d"]


class TestSpokenSyllablesOnNotes:
    def test_a_word_on_two_notes_shows_its_two_syllables(self):
        syl = [Syllable("only ", 0, 1000, parts=("on", "ly ")), Syllable("love ", 1200, 1600)]
        segs = place_lyrics(_notes((0.0, 0.5), (0.5, 1.0), (1.2, 1.6)), syl)
        assert [s.word for s in segs] == ["on", "ly ", "love "]

    def test_a_word_on_one_note_stays_whole(self):
        syl = [Syllable("only ", 0, 1000, parts=("on", "ly ")), Syllable("love ", 1200, 1600)]
        segs = place_lyrics(_notes((0.0, 1.0), (1.2, 1.6)), syl)
        assert [s.word for s in segs] == ["only ", "love "]

    def test_the_space_of_the_word_goes_to_its_last_note(self):
        syl = [Syllable("only ", 0, 1500, parts=("on", "ly ")), Syllable("love ", 2000, 2400)]
        segs = place_lyrics(_notes((0.0, 0.5), (0.7, 1.0), (1.0, 1.5), (2.0, 2.4)), syl)
        assert [s.word for s in segs] == ["on", "ly", "~ ", "love "]

    def test_the_notes_and_the_other_words_are_what_they_were(self):
        """Only the texts of a word's own notes change: the same notes, lines and words as without parts."""
        notes = _notes((0.0, 0.4), (0.4, 0.9), (1.0, 1.3), (1.3, 1.6), (1.6, 2.0), (2.5, 2.9))
        plain = [Syllable("only ", 0, 900, True), Syllable("radio ", 1000, 2000), Syllable("love ", 2500, 2900, True)]
        cut = [Syllable("only ", 0, 900, True, ("on", "ly ")), Syllable("radio ", 1000, 2000, False, ("ra", "di", "o ")),
               Syllable("love ", 2500, 2900, True)]
        before, after = place_lyrics(notes, plain), place_lyrics(notes, cut)
        assert [s.word for s in before] == ["only", "~ ", "radio", "~", "~ ", "love "]
        assert [s.word for s in after] == ["on", "ly ", "ra", "di", "o ", "love "]
        assert [(s.start, s.end, s.note, s.line_break_after) for s in after] == \
            [(s.start, s.end, s.note, s.line_break_after) for s in before]

    def test_a_note_that_also_carries_another_syllable_is_left_alone(self):
        """A syllable without a note of its own joined the note: its text is no longer the one of ``parts``."""
        syl = [Syllable("oh ", 0, 100), Syllable("only ", 150, 1000, parts=("on", "ly ")),
               Syllable("you ", 1900, 2000), Syllable("love ", 2100, 2500)]
        segs = place_lyrics(_notes((0.1, 0.5), (0.5, 1.0), (2.1, 2.5)), syl)
        words = [s.word for s in segs]
        assert "".join(words).replace("~", "") == "oh only you love "
        assert not any(word in ("on", "ly ", "ly") for word in words)

    def test_from_the_words_to_the_notes(self, spoken):
        segments = [_seg("Only ", 0.0, 0.9), _seg("everyone ", 1.0, 2.2, line_break=True), _seg("you're ", 3.0, 3.6)]
        syl = syllables_from_segments(segments, "en")
        notes = _notes((0.0, 0.4), (0.45, 0.9), (1.0, 1.3), (1.35, 1.7), (1.75, 2.2), (3.0, 3.3), (3.3, 3.6))
        segs = place_lyrics(notes, syl)
        assert [s.word for s in segs] == ["On", "ly ", "eve", "ry", "one ", "you're", "~ "]
        assert [s.line_break_after for s in segs] == [False, False, False, False, True, False, False]
