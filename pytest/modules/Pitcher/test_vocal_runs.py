"""Tests for the vocal_runs.py module."""

import unittest

import librosa

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Pitcher.vocal_runs import (
    flip_interval,
    scale_pitch_classes,
    split_notes_at_vocal_runs,
    written_beat_seconds,
)

C5 = 523.25


def _hz(semitones_from_c5):
    return C5 * 2 ** (semitones_from_c5 / 12)


def _track(*spans, confidence=0.95):
    """Pitch track with 10 ms frames: spans of (start s, end s, semitones from C5)."""
    times, freqs, conf = [], [], []
    for start, end, semitones in spans:
        t = start
        while t < end - 1e-9:
            times.append(round(t, 3)); freqs.append(_hz(semitones)); conf.append(confidence)
            t += 0.01
    return PitchedData(times, freqs, conf)


def _held(word="leave ", start=0.0, end=1.0, note_type=":"):
    return MidiSegment(note="C5", start=start, end=end, word=word, note_type=note_type)


def _flip(semitones, start=0.4, end=0.5, note_end=1.0):
    """A note held on C5 that flips *semitones* away between *start* and *end*."""
    return _track((0.0, start, 0.0), (start, end, semitones), (end, note_end, 0.0))


def _shape(segments):
    return [(seg.note, seg.word) for seg in segments]


class TestSplitNotesAtVocalRuns(unittest.TestCase):
    def test_flip_down_a_tone_goes_to_the_key(self):
        # A quick flip sung short of its pitch: 1.3 semitones down measures
        # as a semitone (B), the key of D minor has B flat
        segment = _held()
        segment.line_break_after = True
        result = split_notes_at_vocal_runs([segment], _flip(-1.3), bpm=120.0, music_key="D minor")
        self.assertEqual(_shape(result), [("C5", "leave"), ("A♯4", "~"), ("C5", "~ ")])
        self.assertEqual([seg.line_break_after for seg in result], [False, False, True])

    def test_without_key_the_rounded_deviation(self):
        result = split_notes_at_vocal_runs([_held()], _flip(-1.3), bpm=120.0)
        self.assertEqual(_shape(result), [("C5", "leave"), ("B4", "~"), ("C5", "~ ")])

    def test_flip_up(self):
        result = split_notes_at_vocal_runs([_held()], _flip(2.0), bpm=120.0, music_key="C major")
        self.assertEqual(result[1].note, "D5")

    def test_pieces_are_on_the_beat_grid_and_cover_the_note(self):
        beat = written_beat_seconds(120.0)
        result = split_notes_at_vocal_runs([_held()], _flip(-2.0), bpm=120.0)
        self.assertEqual(result[0].start, 0.0)
        self.assertEqual(result[-1].end, 1.0)
        for before, after in zip(result, result[1:]):
            self.assertEqual(before.end, after.start)
        for seg in result[1:]:
            self.assertAlmostEqual(seg.start / beat, round(seg.start / beat))
        self.assertGreaterEqual(result[1].end - result[1].start, beat - 1e-9)

    def test_octave_error_is_not_a_flip(self):
        result = split_notes_at_vocal_runs([_held()], _flip(12.0), bpm=120.0)
        self.assertEqual(_shape(result), [("C5", "leave ")])

    def test_too_short_or_too_long_is_not_a_flip(self):
        for start, end in ((0.4, 0.43), (0.2, 0.7)):
            with self.subTest(ms=round((end - start) * 1000)):
                result = split_notes_at_vocal_runs([_held()], _flip(-2.0, start, end), bpm=120.0)
                self.assertEqual(len(result), 1)

    def test_a_change_that_does_not_come_back_is_not_a_flip(self):
        track = _track((0.0, 0.5, 0.0), (0.5, 1.0, -2.0))
        result = split_notes_at_vocal_runs([_held()], track, bpm=120.0)
        self.assertEqual(len(result), 1)

    def test_vibrato_is_not_a_flip(self):
        spans = [(i * 0.05, i * 0.05 + 0.05, 0.4 if i % 2 else -0.4) for i in range(20)]
        result = split_notes_at_vocal_runs([_held()], _track(*spans), bpm=120.0)
        self.assertEqual(len(result), 1)

    def test_freestyle_rap_and_short_notes_stay(self):
        segments = [_held(note_type="F"), _held(note_type="R"), _held(end=0.3)]
        for segment in segments:
            with self.subTest(note_type=segment.note_type, end=segment.end):
                result = split_notes_at_vocal_runs([segment], _flip(-2.0, 0.1, 0.2), bpm=120.0)
                self.assertEqual(result, [segment])

    def test_golden_note_pieces_stay_golden(self):
        result = split_notes_at_vocal_runs([_held(note_type="*")], _flip(-2.0), bpm=120.0)
        self.assertEqual([seg.note_type for seg in result], ["*", "*", "*"])

    def test_unconfident_frames_are_not_used(self):
        track = _flip(-2.0)
        track.confidence = [0.5] * len(track.confidence)
        self.assertEqual(len(split_notes_at_vocal_runs([_held()], track, bpm=120.0)), 1)

    def test_continuation_note_keeps_its_tilde(self):
        result = split_notes_at_vocal_runs([_held(word="~")], _flip(-2.0), bpm=120.0)
        self.assertEqual([seg.word for seg in result], ["~", "~", "~"])

    def test_input_is_not_changed(self):
        segment = _held()
        segments = [segment]
        split_notes_at_vocal_runs(segments, _flip(-2.0), bpm=120.0)
        self.assertEqual(segments, [segment])
        self.assertEqual((segment.word, segment.end), ("leave ", 1.0))

    def test_no_pitch_track(self):
        segments = [_held()]
        self.assertIs(split_notes_at_vocal_runs(segments, None, bpm=120.0), segments)
        self.assertIs(split_notes_at_vocal_runs(segments, PitchedData([], [], []), bpm=120.0), segments)

    def test_later_notes_are_kept_in_order(self):
        segments = [_held(), _held(word="me ", start=1.0, end=1.2)]
        track = _track((0.0, 0.4, 0.0), (0.4, 0.5, -2.0), (0.5, 1.2, 0.0))
        result = split_notes_at_vocal_runs(segments, track, bpm=120.0)
        self.assertEqual([seg.word for seg in result], ["leave", "~", "~ ", "me "])


class TestFlipInterval(unittest.TestCase):
    def test_rounded_at_least_a_semitone(self):
        self.assertEqual(flip_interval(72, -0.8, None), -1)
        self.assertEqual(flip_interval(72, 2.4, None), 2)

    def test_larger_than_four_semitones_is_an_error(self):
        self.assertIsNone(flip_interval(72, 5.0, None))
        self.assertIsNone(flip_interval(72, -12.0, None))
        self.assertEqual(flip_interval(72, 4.4, None), 4)

    def test_snaps_to_the_scale_on_the_same_side(self):
        d_minor = scale_pitch_classes("D minor")
        self.assertEqual(flip_interval(72, -1.3, d_minor), -2)  # C5 -> B flat 4
        self.assertEqual(flip_interval(72, 1.2, d_minor), 2)  # C5 -> D5 (C sharp is not in the key)

    def test_keeps_a_pitch_of_the_scale(self):
        self.assertEqual(flip_interval(72, -1.0, scale_pitch_classes("C major")), -1)  # C5 -> B4


class TestScalePitchClasses(unittest.TestCase):
    def test_keys(self):
        self.assertEqual(scale_pitch_classes("C major"), {0, 2, 4, 5, 7, 9, 11})
        self.assertEqual(scale_pitch_classes("D minor"), {2, 4, 5, 7, 9, 10, 0})
        self.assertEqual(scale_pitch_classes("F# major"), {6, 8, 10, 11, 1, 3, 5})

    def test_unknown(self):
        for key in (None, "", "H minor", "C dorian", "C"):
            with self.subTest(key=key):
                self.assertIsNone(scale_pitch_classes(key))


class TestFlipPitchTrack(unittest.TestCase):
    """The pipeline searches flips in the track of the audio the note pitches come from."""

    def _process_data(self):
        from types import SimpleNamespace
        paths = SimpleNamespace(vocals_audio_file_path="vocals.wav", whisper_audio_path="mix.wav")
        return SimpleNamespace(process_data_paths=paths, pitched_data=_flip(-2.0))

    def test_vocal_stem_uses_the_pitch_data(self):
        from src.UltraSinger import _flip_pitch_track
        data = self._process_data()
        self.assertIs(_flip_pitch_track(data, "vocals.wav"), data.pitched_data)

    def test_lead_stem_is_pitched(self):
        from unittest import mock
        from src import UltraSinger
        lead = _flip(-1.0)
        with mock.patch.object(UltraSinger, "get_pitch_with_file", return_value=lead) as pitch:
            self.assertIs(UltraSinger._flip_pitch_track(self._process_data(), "lead.wav"), lead)
        pitch.assert_called_once_with("lead.wav")

    def test_unreadable_lead_stem_gives_no_track(self):
        from unittest import mock
        from src import UltraSinger
        with mock.patch.object(UltraSinger, "get_pitch_with_file", side_effect=OSError("gone")):
            self.assertIsNone(UltraSinger._flip_pitch_track(self._process_data(), "lead.wav"))


class TestWrittenBeat(unittest.TestCase):
    def test_matches_the_writer_grid(self):
        # 120 BPM is written as 30 x 14 = 420 BPM: a beat of 60 / (120 * 14) s
        self.assertAlmostEqual(written_beat_seconds(120.0), 60.0 / (120.0 * 14))


if __name__ == "__main__":
    unittest.main()
