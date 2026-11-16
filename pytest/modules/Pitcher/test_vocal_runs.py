"""Tests for the vocal_runs.py module."""

import os
import unittest

import librosa

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Pitcher.vocal_runs import (
    CLEAR_STEP_PRICE,
    change_price,
    find_plateaus,
    flip_interval,
    scale_pitch_classes,
    split_notes_at_clear_steps,
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


def _held(word="hold ", start=0.0, end=1.0, note_type=":"):
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
        self.assertEqual(_shape(result), [("C5", "hold"), ("A♯4", "~"), ("C5", "~ ")])
        self.assertEqual([seg.line_break_after for seg in result], [False, False, True])

    def test_without_key_the_rounded_deviation(self):
        result = split_notes_at_vocal_runs([_held()], _flip(-1.3), bpm=120.0)
        self.assertEqual(_shape(result), [("C5", "hold"), ("B4", "~"), ("C5", "~ ")])

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
        self.assertEqual(_shape(result), [("C5", "hold ")])

    def test_too_short_is_not_a_flip(self):
        result = split_notes_at_vocal_runs([_held()], _flip(-2.0, 0.4, 0.43), bpm=120.0)
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
        self.assertEqual((segment.word, segment.end), ("hold ", 1.0))

    def test_no_pitch_track(self):
        segments = [_held()]
        self.assertIs(split_notes_at_vocal_runs(segments, None, bpm=120.0), segments)
        self.assertIs(split_notes_at_vocal_runs(segments, PitchedData([], [], []), bpm=120.0), segments)

    def test_later_notes_are_kept_in_order(self):
        segments = [_held(), _held(word="on ", start=1.0, end=1.2)]
        track = _track((0.0, 0.4, 0.0), (0.4, 0.5, -2.0), (0.5, 1.2, 0.0))
        result = split_notes_at_vocal_runs(segments, track, bpm=120.0)
        self.assertEqual([seg.word for seg in result], ["hold", "~", "~ ", "on "])


class TestSteps(unittest.TestCase):
    """Pitches the singing settles on within one note are notes of their own."""

    def _split(self, track, segment=None, **kwargs):
        return split_notes_at_vocal_runs([segment or _held()], track, bpm=120.0, **kwargs)

    def test_a_held_step_is_a_note_of_its_own(self):
        result = self._split(_track((0.0, 0.5, 0.0), (0.5, 1.0, -2.0)))
        self.assertEqual(_shape(result), [("C5", "hold"), ("A♯4", "~ ")])
        self.assertAlmostEqual(result[0].end, 0.5, delta=0.02)

    def test_a_long_departure_is_a_step_there_and_back(self):
        # 500 ms away: too long for a flip
        result = self._split(_flip(-2.0, 0.2, 0.7))
        self.assertEqual(_shape(result), [("C5", "hold"), ("A♯4", "~"), ("C5", "~ ")])

    def test_a_run_over_several_pitches(self):
        track = _track((0.0, 0.3, 0.0), (0.3, 0.5, 2.0), (0.5, 0.7, 4.0), (0.7, 1.0, 5.0))
        self.assertEqual([seg.note for seg in self._split(track)], ["C5", "D5", "E5", "F5"])

    def test_the_step_is_not_moved_to_the_key(self):
        # held long enough to measure: C sharp stays, although D minor has none
        result = self._split(_track((0.0, 0.5, 0.0), (0.5, 1.0, 1.0)), music_key="D minor")
        self.assertEqual([seg.note for seg in result], ["C5", "C♯5"])

    def test_the_scoop_into_a_note_is_no_step(self):
        self.assertEqual(len(self._split(_track((0.0, 0.1, -3.0), (0.1, 1.0, 0.0)))), 1)

    def test_the_voice_falling_off_at_the_end_is_no_step(self):
        self.assertEqual(len(self._split(_track((0.0, 0.9, 0.0), (0.9, 1.0, -3.0)))), 1)

    def test_a_glide_between_two_pitches_is_no_note_of_its_own(self):
        import numpy as np
        times = np.round(np.arange(0.0, 1.0, 0.01), 3)
        semitones = np.clip((times - 0.35) / 0.3, 0.0, 1.0) * 4.0  # C5, 300 ms up to E5, E5
        track = PitchedData(list(times), [_hz(x) for x in semitones], [0.95] * len(times))
        self.assertEqual([seg.note for seg in self._split(track)], ["C5", "E5"])

    def test_an_octave_error_of_the_pitch_track_is_no_step(self):
        for semitones in (12.0, -12.0):
            with self.subTest(semitones=semitones):
                self.assertEqual(len(self._split(_track((0.0, 0.5, 0.0), (0.5, 1.0, semitones)))), 1)

    def test_a_leap_of_more_than_four_semitones_is_no_step(self):
        self.assertEqual(len(self._split(_track((0.0, 0.5, 0.0), (0.5, 1.0, 6.0)))), 1)

    def test_the_note_keeps_its_pitch_on_the_plateau_it_has_it_from(self):
        # The note is charted on C5, sung for the shorter part: the longer part is the step
        result = self._split(_track((0.0, 0.3, 0.0), (0.3, 1.0, 2.0)))
        self.assertEqual([seg.note for seg in result], ["C5", "D5"])

    def test_a_semitone_step_keeps_the_note_on_its_own_pitch(self):
        # Both plateaus are within a semitone of the note: it is the one on C5
        result = self._split(_track((0.0, 0.35, 0.0), (0.35, 1.0, 1.0)))
        self.assertEqual([seg.note for seg in result], ["C5", "C♯5"])

    def test_a_note_charted_a_semitone_beside_the_singing(self):
        # Charted on C5, sung on B4 and then on D5: the nearer plateau is the note's
        result = self._split(_track((0.0, 0.5, -1.0), (0.5, 1.0, 2.0)))
        self.assertEqual([seg.note for seg in result], ["C5", "D♯5"])

    def test_the_octave_of_the_note_is_kept(self):
        # Charted an octave above the singing: the step is charted from there
        segment = _held()
        segment.note = "C6"
        result = self._split(_track((0.0, 0.5, 0.0), (0.5, 1.0, -2.0)), segment)
        self.assertEqual([seg.note for seg in result], ["C6", "A♯5"])

    def test_a_note_sung_on_other_pitches_than_its_own_stays(self):
        segment = _held()
        segment.note = "F5"
        self.assertEqual(self._split(_track((0.0, 0.5, 0.0), (0.5, 1.0, -2.0)), segment), [segment])

    def test_wide_vibrato_is_no_step(self):
        import numpy as np
        times = np.round(np.arange(0.0, 1.0, 0.01), 3)
        semitones = 1.2 * np.sin(2 * np.pi * 5.5 * times)
        track = PitchedData(list(times), [_hz(x) for x in semitones], [0.95] * len(times))
        for threshold in (1.0, 0.6):
            with self.subTest(threshold=threshold):
                self.assertEqual(len(self._split(track, threshold=threshold)), 1)

    def test_a_lower_threshold_finds_a_shorter_step(self):
        # a semitone up for 150 ms, then back
        track = _flip(1.0, 0.4, 0.55)
        track.confidence = [0.8] * len(track.confidence)  # voiced, but not sure enough for a flip
        self.assertEqual(len(self._split(track)), 1)
        self.assertEqual([seg.note for seg in self._split(track, threshold=0.5)], ["C5", "C♯5", "C5"])

    def test_pieces_are_on_the_beat_grid_and_cover_the_note(self):
        beat = written_beat_seconds(120.0)
        segment = _held()
        segment.line_break_after = True
        result = self._split(_track((0.0, 0.33, 0.0), (0.33, 0.61, 2.0), (0.61, 1.0, 4.0)), segment)
        self.assertEqual((result[0].start, result[-1].end), (0.0, 1.0))
        for before, after in zip(result, result[1:]):
            self.assertEqual(before.end, after.start)
            self.assertAlmostEqual(after.start / beat, round(after.start / beat))
        self.assertEqual([seg.line_break_after for seg in result], [False, False, True])
        self.assertEqual([seg.word for seg in result], ["hold", "~", "~ "])

    def test_unvoiced_frames_are_not_used(self):
        track = _track((0.0, 0.5, 0.0), (0.5, 1.0, -2.0), confidence=0.5)
        self.assertEqual(len(self._split(track)), 1)

    def test_golden_freestyle_rap_and_short_notes(self):
        track = _track((0.0, 0.5, 0.0), (0.5, 1.0, -2.0))
        self.assertEqual([seg.note_type for seg in self._split(track, _held(note_type="*"))], ["*", "*"])
        for segment in (_held(note_type="F"), _held(note_type="R")):
            with self.subTest(note_type=segment.note_type):
                self.assertEqual(self._split(track, segment), [segment])
        short = _held(end=0.18)
        self.assertEqual(self._split(_track((0.0, 0.09, 0.0), (0.09, 0.18, -2.0)), short), [short])

    def test_a_flip_within_a_step(self):
        # C5, a step to D5, and on D5 a quick flip down to C5
        track = _track((0.0, 0.3, 0.0), (0.3, 0.6, 2.0), (0.6, 0.66, 0.0), (0.66, 1.0, 2.0))
        result = self._split(track)
        self.assertEqual([seg.note for seg in result], ["C5", "D5", "C5", "D5"])
        self.assertEqual([seg.word for seg in result], ["hold", "~", "~", "~ "])


class TestHeldNoteInSeveralNotes(unittest.TestCase):
    """A held note that is charted in several notes on one pitch is also looked at as a whole."""

    def _pieces(self, cut=0.4, second_word="~ ", second_note="C5", pause=0.0):
        first = _held(word="hold", end=cut)
        second = _held(word=second_word, start=cut + pause, end=1.0)
        second.note = second_note
        return [first, second]

    def test_a_flip_right_at_the_cut_is_found(self):
        # The flip starts where the second note does: no steady voice before it there
        track = _flip(-2.0, 0.4, 0.5)
        result = split_notes_at_vocal_runs(self._pieces(), track, bpm=120.0)
        self.assertEqual(_shape(result), [("C5", "hold"), ("A♯4", "~"), ("C5", "~ ")])
        self.assertEqual((result[0].start, result[-1].end), (0.0, 1.0))

    def test_the_line_break_stays_at_the_end(self):
        pieces = self._pieces()
        pieces[1].line_break_after = True
        result = split_notes_at_vocal_runs(pieces, _flip(-2.0, 0.4, 0.5), bpm=120.0)
        self.assertEqual([seg.line_break_after for seg in result], [False, False, True])

    def test_without_anything_to_find_the_notes_stay_as_they_are(self):
        pieces = self._pieces()
        self.assertEqual(split_notes_at_vocal_runs(pieces, _track((0.0, 1.0, 0.0)), bpm=120.0), pieces)

    def test_what_the_pieces_show_as_well_is_charted_in_them(self):
        # The flip lies within the second note: its cut from the first one stays
        pieces = self._pieces(cut=0.2)
        result = split_notes_at_vocal_runs(pieces, _flip(-2.0, 0.6, 0.7), bpm=120.0)
        self.assertEqual(_shape(result), [("C5", "hold"), ("C5", "~"), ("A♯4", "~"), ("C5", "~ ")])
        self.assertEqual(result[0].end, 0.2)

    def test_only_notes_of_one_syllable_on_one_pitch_without_a_pause_are_one_held_note(self):
        track = _flip(-2.0, 0.4, 0.5)
        for name, pieces in (("another word", self._pieces(second_word="on ")),
                             ("another pitch", self._pieces(second_note="D5")),
                             ("a pause", self._pieces(pause=0.1))):
            with self.subTest(name):
                self.assertEqual(len(split_notes_at_vocal_runs(pieces, track, bpm=120.0)), 2)
        pieces = self._pieces()
        pieces[0].line_break_after = True
        self.assertEqual(len(split_notes_at_vocal_runs(pieces, track, bpm=120.0)), 2)

    def test_the_clear_steps_keep_to_each_note(self):
        pieces = self._pieces()
        self.assertEqual(split_notes_at_clear_steps(pieces, _flip(-2.0, 0.4, 0.5), bpm=120.0), pieces)


class TestClearSteps(unittest.TestCase):
    """Without the option, only the clearest steps are charted, and no flips."""

    def test_a_step_held_for_long_is_charted(self):
        result = split_notes_at_clear_steps([_held()], _track((0.0, 0.5, 0.0), (0.5, 1.0, -2.0)), bpm=120.0)
        self.assertEqual(_shape(result), [("C5", "hold"), ("A♯4", "~ ")])

    def test_a_short_step_is_left_to_the_option(self):
        # two semitones up for 180 ms and back: a step for the option, not a clear one
        track = _flip(2.0, 0.4, 0.58)
        track.confidence = [0.8] * len(track.confidence)
        self.assertEqual(len(split_notes_at_vocal_runs([_held()], track, bpm=120.0)), 3)
        self.assertEqual(len(split_notes_at_clear_steps([_held()], track, bpm=120.0)), 1)

    def test_no_flips(self):
        track = _flip(-2.0)
        self.assertEqual(len(split_notes_at_vocal_runs([_held()], track, bpm=120.0)), 3)
        self.assertEqual(len(split_notes_at_clear_steps([_held()], track, bpm=120.0)), 1)

    def test_the_price_is_the_highest_of_the_option(self):
        self.assertEqual(CLEAR_STEP_PRICE, change_price(100.0))

    def test_no_pitch_track(self):
        segments = [_held()]
        self.assertIs(split_notes_at_clear_steps(segments, None, bpm=120.0), segments)


class TestFindPlateaus(unittest.TestCase):
    def _frames(self, *spans):
        import numpy as np
        track = _track(*spans)
        return np.array(track.times), librosa.hz_to_midi(np.array(track.frequencies))

    def test_one_pitch_is_no_plateau_list(self):
        self.assertEqual(find_plateaus(*self._frames((0.0, 1.0, 0.0))), [])

    def test_plateaus_start_where_the_pitch_changes(self):
        plateaus = find_plateaus(*self._frames((0.0, 0.4, 0.0), (0.4, 1.0, 3.0)))
        self.assertEqual([pitch for _, _, pitch in plateaus], [0, 3])
        self.assertAlmostEqual(plateaus[1][0], 0.4, places=6)
        self.assertEqual(plateaus[0][1], plateaus[1][0])

    def test_too_few_frames(self):
        self.assertEqual(find_plateaus(*self._frames((0.0, 0.03, 0.0), (0.03, 0.06, 3.0))), [])

    def test_the_price_of_a_change_follows_the_threshold(self):
        self.assertLess(change_price(0.5), change_price(1.0))
        self.assertEqual(change_price(0.01), change_price(0.02))  # limited below
        self.assertEqual(change_price(50.0), change_price(100.0))  # and above


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


class TestStemPitchTrack(unittest.TestCase):
    """The pipeline compares the notes with the pitch of the stem as it was separated."""

    def setUp(self):
        import tempfile
        self._folder = tempfile.TemporaryDirectory()
        self.addCleanup(self._folder.cleanup)
        self.stem = os.path.join(self._folder.name, "vocals.wav")
        with open(self.stem, "wb") as f:
            f.write(b"stem")

    def _process_data(self):
        from types import SimpleNamespace
        paths = SimpleNamespace(cache_folder_path=self._folder.name)
        # The pitch data of the denoised vocal, which smears quick flips
        return SimpleNamespace(process_data_paths=paths, pitched_data=_flip(-0.5), stem_pitch_tracks={})

    def test_the_stem_is_pitched_once_per_conversion(self):
        from unittest import mock
        from src import UltraSinger
        stem_track = _flip(-2.0)
        process_data = self._process_data()
        with mock.patch.object(UltraSinger, "get_pitch_with_file", return_value=stem_track) as pitch:
            first = UltraSinger._stem_pitch_track(process_data, self.stem)
            with mock.patch.object(UltraSinger.settings, "skip_cache_pitch_detection", True):
                again = UltraSinger._stem_pitch_track(process_data, self.stem)
        pitch.assert_called_once_with(self.stem)
        self.assertIs(first, stem_track)
        self.assertIs(again, stem_track)

    def test_a_later_conversion_reads_the_cached_track(self):
        from unittest import mock
        from src import UltraSinger
        stem_track = _flip(-2.0)
        with mock.patch.object(UltraSinger, "get_pitch_with_file", return_value=stem_track) as pitch:
            UltraSinger._stem_pitch_track(self._process_data(), self.stem)
            again = UltraSinger._stem_pitch_track(self._process_data(), self.stem)
        pitch.assert_called_once_with(self.stem)
        self.assertEqual((again.times, again.frequencies), (stem_track.times, stem_track.frequencies))

    def test_skipping_the_pitch_cache_pitches_the_stem_again(self):
        from unittest import mock
        from src import UltraSinger
        with mock.patch.object(UltraSinger, "get_pitch_with_file", return_value=_flip(-2.0)) as pitch:
            UltraSinger._stem_pitch_track(self._process_data(), self.stem)
            with mock.patch.object(UltraSinger.settings, "skip_cache_pitch_detection", True):
                UltraSinger._stem_pitch_track(self._process_data(), self.stem)
        self.assertEqual(pitch.call_count, 2)

    def test_each_stem_has_its_own_track(self):
        from unittest import mock
        from src import UltraSinger
        lead = os.path.join(self._folder.name, "lead.wav")
        with open(lead, "wb") as f:
            f.write(b"lead")
        tracks = {self.stem: _flip(-2.0), lead: _flip(1.0)}
        process_data = self._process_data()
        with mock.patch.object(UltraSinger, "get_pitch_with_file", side_effect=tracks.get) as pitch:
            self.assertIs(UltraSinger._stem_pitch_track(process_data, self.stem), tracks[self.stem])
            self.assertIs(UltraSinger._stem_pitch_track(process_data, lead), tracks[lead])
        self.assertEqual(pitch.call_count, 2)

    def test_unreadable_stem_gives_no_track_and_is_tried_once(self):
        from unittest import mock
        from src import UltraSinger
        process_data = self._process_data()
        with mock.patch.object(UltraSinger, "get_pitch_with_file", side_effect=OSError("gone")) as pitch:
            self.assertIsNone(UltraSinger._stem_pitch_track(process_data, self.stem))
            self.assertIsNone(UltraSinger._stem_pitch_track(process_data, self.stem))
        pitch.assert_called_once_with(self.stem)


class TestThreshold(unittest.TestCase):
    def test_a_lower_threshold_finds_smaller_flips(self):
        track = _flip(-0.7)
        self.assertEqual(len(split_notes_at_vocal_runs([_held()], track, bpm=120.0)), 1)
        result = split_notes_at_vocal_runs([_held()], track, bpm=120.0, threshold=0.5)
        self.assertEqual(_shape(result), [("C5", "hold"), ("B4", "~"), ("C5", "~ ")])

    def test_a_lower_threshold_keeps_the_flips_of_a_note_with_vibrato(self):
        # +-0.6 semitone vibrato at 6 Hz around C5, a 2-semitone flip of 100 ms
        import numpy as np
        times = np.round(np.arange(0.0, 1.0, 0.01), 3)
        semitones = 0.6 * np.sin(2 * np.pi * 6 * times)
        semitones[(times >= 0.4) & (times < 0.5)] = -2.0
        track = PitchedData(list(times), [_hz(x) for x in semitones], [0.95] * len(times))
        for threshold in (1.0, 0.7, 0.5):
            with self.subTest(threshold=threshold):
                result = split_notes_at_vocal_runs([_held()], track, bpm=120.0, threshold=threshold)
                self.assertIn("A♯4", [seg.note for seg in result])

    def test_a_higher_threshold_keeps_smaller_flips_held(self):
        self.assertEqual(len(split_notes_at_vocal_runs([_held()], _flip(-1.3), bpm=120.0, threshold=1.5)), 1)
        self.assertEqual(len(split_notes_at_vocal_runs([_held()], _flip(-2.0), bpm=120.0, threshold=1.5)), 3)


class TestWrittenBeat(unittest.TestCase):
    def test_matches_the_writer_grid(self):
        # 120 BPM is written as 30 x 14 = 420 BPM: a beat of 60 / (120 * 14) s
        self.assertAlmostEqual(written_beat_seconds(120.0), 60.0 / (120.0 * 14))


if __name__ == "__main__":
    unittest.main()
