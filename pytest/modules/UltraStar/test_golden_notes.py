"""Tests for the golden_notes.py module."""

import unittest

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Ultrastar.golden_notes import mark_golden_notes


def _seg(word, start, end, note_type=":", note="C4"):
    return MidiSegment(note=note, start=start, end=end, word=word, note_type=note_type)


def _golden(segments):
    return [i for i, seg in enumerate(segments) if seg.note_type == "*"]


def _track(*spans):
    """Pitch track with 10 ms frames: spans of (start s, end s, frequency Hz, confidence)."""
    times, freqs, conf = [], [], []
    for start, end, freq, c in spans:
        t = start
        while t < end:
            times.append(round(t, 3)); freqs.append(freq); conf.append(c)
            t += 0.01
    return PitchedData(times, freqs, conf)


C4, D_SHARP4, C5 = 261.63, 311.13, 523.25


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

    def test_a_note_of_exactly_the_minimum_duration_is_eligible(self):
        # 1.2 - 1.0 is a hair under 0.2 in floating point.
        segments = [_seg("short ", 0.0, 0.1), _seg("held ", 1.0, 1.2)]
        result = mark_golden_notes(segments, bpm=120.0, count=1, max_fraction=1.0)
        self.assertEqual(_golden(result), [1])

    def test_a_second_run_adds_nothing(self):
        segments = [_seg(f"w{i} ", i * 1.0, i * 1.0 + 0.6) for i in range(100)]
        mark_golden_notes(segments, bpm=120.0)
        first = _golden(segments)
        result = mark_golden_notes(segments, bpm=120.0)
        self.assertEqual(_golden(result), first)

    def test_existing_golden_notes_count_towards_the_limits(self):
        # Two golden notes already sit in the first tenth: one more fits there,
        # and only eight more are needed for the ten.
        segments = [_seg(f"w{i} ", i * 1.0, i * 1.0 + 0.6) for i in range(100)]
        segments[0].note_type = "*"
        segments[1].note_type = "G"
        result = mark_golden_notes(segments, bpm=120.0)
        marked = [i for i, seg in enumerate(result) if seg.note_type in ("*", "G")]
        self.assertEqual(len(marked), 10)
        self.assertEqual(sum(1 for i in marked if i < 10), 3)


class TestPitchLock(unittest.TestCase):
    """A shorter note the singing stays on beats a longer one it misses."""

    @staticmethod
    def _two_notes():
        # Separate phrases, same pitch: only duration and pitch lock differ.
        return [_seg("long ", 0.0, 1.0), _seg("held ", 3.0, 3.8)]

    def test_without_pitch_track_the_longer_note_wins(self):
        result = mark_golden_notes(self._two_notes(), bpm=120.0, count=1, max_fraction=1.0)
        self.assertEqual(_golden(result), [0])

    def test_note_sung_on_pitch_wins(self):
        track = _track((0.0, 1.0, D_SHARP4, 0.9), (3.0, 3.8, C4, 0.9))
        result = mark_golden_notes(self._two_notes(), bpm=120.0, count=1, max_fraction=1.0,
                                   pitched_data=track)
        self.assertEqual(_golden(result), [1])

    def test_octaves_are_ignored(self):
        track = _track((0.0, 1.0, D_SHARP4, 0.9), (3.0, 3.8, C5, 0.9))
        result = mark_golden_notes(self._two_notes(), bpm=120.0, count=1, max_fraction=1.0,
                                   pitched_data=track)
        self.assertEqual(_golden(result), [1])

    def test_unconfident_frames_do_not_count(self):
        track = _track((0.0, 1.0, D_SHARP4, 0.9), (3.0, 3.8, C4, 0.3))
        result = mark_golden_notes(self._two_notes(), bpm=120.0, count=1, max_fraction=1.0,
                                   pitched_data=track)
        self.assertEqual(_golden(result), [0])

    def test_unparsable_note_name_never_wins_with_a_pitch_track(self):
        segments = [_seg("odd ", 0.0, 1.0, note="?"), _seg("held ", 3.0, 3.8)]
        track = _track((0.0, 1.0, C4, 0.9), (3.0, 3.8, C4, 0.9))
        result = mark_golden_notes(segments, bpm=120.0, count=1, max_fraction=1.0, pitched_data=track)
        self.assertEqual(_golden(result), [1])


class TestPhrasePeak(unittest.TestCase):
    def test_highest_note_of_a_phrase_gets_a_bonus(self):
        # One phrase (no pause): the highest note wins over a slightly longer lower one.
        segments = [_seg("low ", 0.0, 0.55, note="C4"), _seg("high ", 0.55, 1.05, note="G4")]
        result = mark_golden_notes(segments, bpm=120.0, count=1, max_fraction=1.0)
        self.assertEqual(_golden(result), [1])

    def test_bonus_does_not_beat_a_much_longer_note(self):
        segments = [_seg("low ", 0.0, 1.0, note="C4"), _seg("high ", 1.0, 1.5, note="G4")]
        result = mark_golden_notes(segments, bpm=120.0, count=1, max_fraction=1.0)
        self.assertEqual(_golden(result), [0])


if __name__ == "__main__":
    unittest.main()
