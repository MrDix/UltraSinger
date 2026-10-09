"""Tests for the golden_notes.py module."""

import unittest

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Ultrastar.golden_notes import mark_golden_notes


def _seg(word, start, end, note_type=":"):
    return MidiSegment(note="C4", start=start, end=end, word=word, note_type=note_type)


def _golden(segments):
    return [i for i, seg in enumerate(segments) if seg.note_type == "*"]


class TestMarkGoldenNotes(unittest.TestCase):
    def test_empty_list(self):
        result = mark_golden_notes([], bpm=120.0)
        self.assertEqual(result, [])

    def test_no_candidates_when_all_too_short(self):
        segments = [_seg(f"w{i} ", i * 0.5, i * 0.5 + 0.1) for i in range(100)]
        result = mark_golden_notes(segments, bpm=120.0)
        self.assertEqual(_golden(result), [])

    def test_no_candidates_returns_same_segments_unmarked(self):
        segments = [_seg("hi ", 0.0, 0.05, note_type="F")]
        result = mark_golden_notes(segments, bpm=120.0)
        self.assertEqual(result[0].note_type, "F")

    def test_marks_ten_notes_by_default(self):
        segments = [_seg(f"w{i} ", i * 1.0, i * 1.0 + 0.6) for i in range(100)]
        result = mark_golden_notes(segments, bpm=120.0)
        self.assertEqual(len(_golden(result)), 10)

    def test_count_parameter(self):
        segments = [_seg(f"w{i} ", i * 1.0, i * 1.0 + 0.6) for i in range(100)]
        result = mark_golden_notes(segments, bpm=120.0, count=4)
        self.assertEqual(len(_golden(result)), 4)

    def test_max_fraction_limits_short_songs(self):
        # 20 scorable notes: at most 15% (3) become golden, not 10.
        segments = [_seg(f"w{i} ", i * 1.0, i * 1.0 + 0.6) for i in range(20)]
        result = mark_golden_notes(segments, bpm=120.0)
        self.assertEqual(len(_golden(result)), int(20 * 0.15))

    def test_only_normal_notes_become_golden_not_freestyle_or_rap(self):
        segments = [
            _seg("verse ", 0.0, 1.0, note_type=":"),
            _seg("growl ", 1.0, 2.0, note_type="F"),   # freestyle: never golden
            _seg("flow ", 2.0, 3.0, note_type="R"),    # rap: never golden
            _seg("~", 3.0, 4.0, note_type=":"),        # a held continuation is eligible
        ]
        result = mark_golden_notes(segments, bpm=120.0, max_fraction=1.0)
        self.assertEqual([seg.note_type for seg in result], ["*", "F", "R", "*"])

    def test_longest_notes_are_chosen(self):
        durations = [0.3] * 40
        for i in (3, 11, 27):
            durations[i] = 1.5
        segments = [_seg(f"w{i} ", i * 2.0, i * 2.0 + d) for i, d in enumerate(durations)]
        result = mark_golden_notes(segments, bpm=120.0, count=3, max_fraction=1.0)
        self.assertEqual(_golden(result), [3, 11, 27])

    def test_at_most_three_per_tenth_of_the_song(self):
        # The ten longest notes all sit in the last tenth of the song; only three
        # of them may be golden, the rest come from the other parts of the song.
        segments = [_seg(f"w{i} ", i * 1.0, i * 1.0 + 0.5) for i in range(90)]
        segments += [_seg(f"end{i} ", 90.0 + i, 90.0 + i + 0.9) for i in range(10)]
        result = mark_golden_notes(segments, bpm=120.0)
        golden = _golden(result)
        self.assertEqual(len(golden), 10)
        self.assertEqual(sum(1 for i in golden if i >= 90), 3)

    def test_ties_prefer_the_earlier_note(self):
        segments = [_seg(f"w{i} ", i * 1.0, i * 1.0 + 0.6) for i in range(100)]
        result = mark_golden_notes(segments, bpm=120.0, count=1, max_fraction=1.0)
        self.assertEqual(_golden(result), [0])


if __name__ == "__main__":
    unittest.main()
