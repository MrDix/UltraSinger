"""Tests for the golden_notes.py module."""

import unittest

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Ultrastar.golden_notes import mark_golden_notes


def _seg(word, start, end, note_type=":"):
    return MidiSegment(note="C4", start=start, end=end, word=word, note_type=note_type)


class TestMarkGoldenNotes(unittest.TestCase):
    def test_empty_list(self):
        result = mark_golden_notes([], bpm=120.0)
        self.assertEqual(result, [])

    def test_no_candidates_when_all_too_short(self):
        segments = [_seg(f"w{i} ", i * 0.5, i * 0.5 + 0.1) for i in range(10)]
        result = mark_golden_notes(segments, bpm=120.0, min_duration_ms=350.0)
        self.assertTrue(all(seg.note_type == ":" for seg in result))

    def test_no_candidates_when_all_continuations(self):
        # Long enough, but every note is a tilde continuation -> not eligible.
        segments = [_seg("~", i * 1.0, i * 1.0 + 0.6) for i in range(10)]
        result = mark_golden_notes(segments, bpm=120.0, min_duration_ms=350.0)
        self.assertTrue(all(seg.note_type == ":" for seg in result))

    def test_no_candidates_returns_same_segments_unmarked(self):
        segments = [_seg("hi ", 0.0, 0.05, note_type="F")]
        result = mark_golden_notes(segments, bpm=120.0)
        self.assertEqual(result[0].note_type, "F")

    def test_max_fraction_limits_golden_count(self):
        # 20 eligible, long, real-syllable notes.
        segments = [_seg(f"w{i} ", i * 1.0, i * 1.0 + 0.6) for i in range(20)]
        result = mark_golden_notes(segments, bpm=120.0, max_fraction=0.15,
                                    min_duration_ms=350.0)
        golden_count = sum(1 for seg in result if seg.note_type == "*")
        self.assertEqual(golden_count, int(20 * 0.15))

    def test_only_normal_notes_become_golden_not_freestyle_rap_or_continuation(self):
        segments = [
            _seg("verse ", 0.0, 1.0, note_type=":"),   # eligible
            _seg("growl ", 1.0, 2.0, note_type="F"),   # freestyle: never golden
            _seg("flow ", 2.0, 3.0, note_type="R"),    # rap: never golden
            _seg("~", 3.0, 4.0, note_type=":"),        # continuation: never golden
        ]
        # max_fraction generous enough to select the only eligible note.
        result = mark_golden_notes(segments, bpm=120.0, max_fraction=1.0,
                                    min_duration_ms=350.0)
        self.assertEqual(result[0].note_type, "*")
        self.assertEqual(result[1].note_type, "F")
        self.assertEqual(result[2].note_type, "R")
        self.assertEqual(result[3].note_type, ":")

    def test_longest_note_preferred_within_a_chunk(self):
        # Only one golden slot (max_fraction picks exactly 1), two candidates
        # of different durations -> the longer one must be chosen.
        segments = [
            _seg("short ", 0.0, 0.5, note_type=":"),   # 500 ms
            _seg("long ", 1.0, 3.0, note_type=":"),    # 2000 ms
        ]
        result = mark_golden_notes(segments, bpm=120.0, max_fraction=0.5,
                                    min_duration_ms=350.0)
        golden = [seg for seg in result if seg.note_type == "*"]
        self.assertEqual(len(golden), 1)
        self.assertEqual(golden[0].word, "long ")

    def test_golden_notes_spread_across_song_not_clustered(self):
        # 30 eligible notes in song order; ask for 3 golden notes and check
        # they come from different thirds of the song rather than all being
        # the 3 globally-longest notes bunched together.
        segments = []
        for i in range(30):
            # Make note 5 (early), 15 (middle) and 25 (late) noticeably longer
            # than their neighbours so each chunk has an obvious pick.
            duration = 2.0 if i in (5, 15, 25) else 0.5
            segments.append(_seg(f"w{i} ", i * 3.0, i * 3.0 + duration))

        result = mark_golden_notes(segments, bpm=120.0, max_fraction=0.1,
                                    min_duration_ms=350.0)
        golden_indices = [i for i, seg in enumerate(result) if seg.note_type == "*"]
        self.assertEqual(golden_indices, [5, 15, 25])


if __name__ == "__main__":
    unittest.main()
