"""Tests for the steady_pitch.py module."""

import unittest

import numpy as np

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Pitcher.steady_pitch import follow_steady_sung_pitch, steady_sung_pitch

C5 = 523.25


def _hz(semitones_from_c5):
    return C5 * 2 ** (semitones_from_c5 / 12)


def _track(semitones, confidence=0.95, end=1.0):
    """Pitch track with 10 ms frames up to *end*: one pitch, or one per frame."""
    times = np.round(np.arange(0.0, end, 0.01), 3)
    semitones = np.broadcast_to(np.asarray(semitones, dtype=float), times.shape)
    confidence = np.broadcast_to(np.asarray(confidence, dtype=float), times.shape)
    return PitchedData(list(times), [_hz(x) for x in semitones], list(confidence))


def _note(note="C5", start=0.0, end=1.0, note_type=":"):
    return MidiSegment(note=note, start=start, end=end, word="word ", note_type=note_type)


class TestFollowSteadySungPitch(unittest.TestCase):
    def _moved(self, note, track):
        segment = _note(note)
        result = follow_steady_sung_pitch([segment], track)
        self.assertIs(result[0], segment)
        return segment.note

    def test_a_note_beside_the_sung_pitch_moves_to_it(self):
        self.assertEqual(self._moved("G4", _track(0.0)), "C5")
        self.assertEqual(self._moved("D5", _track(0.0)), "C5")
        self.assertEqual(self._moved("B4", _track(0.1)), "C5")

    def test_a_note_on_the_sung_pitch_stays(self):
        self.assertEqual(self._moved("C5", _track(0.0)), "C5")
        # sung a little beside it: within the tolerance the games score with
        self.assertEqual(self._moved("C5", _track(0.7)), "C5")
        self.assertEqual(self._moved("C5", _track(-0.7)), "C5")

    def test_the_note_goes_to_the_nearest_octave_of_the_sung_pitch(self):
        self.assertEqual(self._moved("G5", _track(0.0)), "C6")
        self.assertEqual(self._moved("C4", _track(2.0)), "D4")
        # the same pitch class an octave away is the sung pitch
        self.assertEqual(self._moved("C3", _track(0.0)), "C3")

    def test_singing_that_is_not_steady_changes_nothing(self):
        times = np.arange(0.0, 1.0, 0.01)
        run = np.repeat([0.0, 2.0, 4.0, 5.0], 25)
        wide_vibrato = 1.6 * np.sin(2 * np.pi * 5.5 * times)
        for name, semitones in (("run", run), ("wide vibrato", wide_vibrato)):
            with self.subTest(name):
                self.assertEqual(self._moved("G4", _track(semitones)), "G4")

    def test_vibrato_around_one_pitch_is_steady(self):
        times = np.arange(0.0, 1.0, 0.01)
        self.assertEqual(self._moved("G4", _track(0.6 * np.sin(2 * np.pi * 5.5 * times))), "C5")

    def test_octave_errors_of_the_pitch_track_do_not_make_it_unsteady(self):
        semitones = np.zeros(100)
        semitones[30:50] = 12.0
        self.assertEqual(self._moved("G4", _track(semitones)), "C5")

    def test_a_note_with_too_few_voiced_frames_stays(self):
        confidence = np.full(100, 0.95)
        confidence[:50] = 0.3
        self.assertEqual(self._moved("G4", _track(0.0, confidence)), "G4")
        confidence[:35] = 0.95  # 85 of 100 frames voiced
        self.assertEqual(self._moved("G4", _track(0.0, confidence)), "C5")

    def test_only_the_frames_of_the_note_count(self):
        semitones = np.zeros(200)
        semitones[100:] = 4.0
        segments = [_note("G4", 0.0, 1.0), _note("G4", 1.0, 2.0)]
        follow_steady_sung_pitch(segments, _track(semitones, end=2.0))
        # each to the nearest C or E from where it is charted
        self.assertEqual([segment.note for segment in segments], ["C5", "E4"])

    def test_a_very_short_note_stays(self):
        segment = _note("G4", 0.0, 0.03)
        follow_steady_sung_pitch([segment], _track(0.0))
        self.assertEqual(segment.note, "G4")

    def test_freestyle_and_rap_notes_stay(self):
        for note_type in ("F", "R", "G"):
            with self.subTest(note_type=note_type):
                segment = _note("G4", note_type=note_type)
                follow_steady_sung_pitch([segment], _track(0.0))
                self.assertEqual(segment.note, "G4")

    def test_golden_notes_move_too(self):
        segment = _note("G4", note_type="*")
        follow_steady_sung_pitch([segment], _track(0.0))
        self.assertEqual((segment.note, segment.note_type), ("C5", "*"))

    def test_without_notes_or_pitch_track(self):
        segments = [_note("G4")]
        self.assertEqual(follow_steady_sung_pitch([], _track(0.0)), [])
        self.assertIs(follow_steady_sung_pitch(segments, None), segments)
        self.assertIs(follow_steady_sung_pitch(segments, PitchedData([], [], [])), segments)
        self.assertEqual(segments[0].note, "G4")

    def test_frames_without_a_pitch_are_not_voiced(self):
        track = _track(0.0)
        track.frequencies = [0.0] * 60 + track.frequencies[60:]
        self.assertEqual(self._moved("G4", track), "G4")


class TestSteadySungPitch(unittest.TestCase):
    def test_the_middle_of_the_voiced_frames(self):
        midi = np.array([72.0, 72.2, 71.9, 72.1, 60.0, 72.0])
        voiced = np.array([True, True, True, True, False, True])
        self.assertAlmostEqual(steady_sung_pitch(midi, voiced), 72.0)

    def test_none_when_spread_apart(self):
        midi = np.array([70.0, 71.0, 72.0, 73.0, 74.0, 75.0])
        self.assertIsNone(steady_sung_pitch(midi, np.ones(6, dtype=bool)))

    def test_none_with_too_few_frames(self):
        self.assertIsNone(steady_sung_pitch(np.full(4, 72.0), np.ones(4, dtype=bool)))


if __name__ == "__main__":
    unittest.main()
