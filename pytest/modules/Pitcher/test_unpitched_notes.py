"""Tests for the unpitched_notes.py module."""

import os
import tempfile
import unittest

import numpy as np
import soundfile

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Pitcher.unpitched_notes import (
    MIN_PROMINENCE_DB,
    SAMPLE_RATE,
    as_freestyle,
    find_unpitched_notes,
    judge_notes,
    loud_frames,
    loudness_db,
    mark_unpitched_notes,
    note_evidence,
    partial_prominence,
)

HOP = 0.016


def _voice(hz, seconds, vibrato=0.0, partials=12):
    """A sung voice: *partials* partials of *hz*, with a vibrato of *vibrato* semitones."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    phase = 2 * np.pi * np.cumsum(hz * 2 ** (vibrato * np.sin(2 * np.pi * 5.5 * t) / 12)) / SAMPLE_RATE
    return sum(np.sin(k * phase) / k for k in range(1, partials + 1)) * 0.2


def _noise(seconds, seed=1):
    return np.random.default_rng(seed).normal(0.0, 0.15, int(seconds * SAMPLE_RATE))


def _times(seconds):
    return np.arange(0.008, seconds, HOP)


def _note(start, end, word="word ", note="C4", note_type=":"):
    return MidiSegment(note=note, start=start, end=end, word=word, note_type=note_type)


def _frames(*spans):
    """Frame times with has_pitch and loud from spans of (start, end, has_pitch, loud)."""
    end = max(span[1] for span in spans)
    times = _times(end)
    has_pitch = np.zeros(len(times), dtype=bool)
    loud = np.zeros(len(times), dtype=bool)
    for start, stop, pitch, is_loud in spans:
        inside = (times >= start) & (times < stop)
        has_pitch[inside], loud[inside] = pitch, is_loud
    return times, has_pitch, loud


class TestPartialProminence(unittest.TestCase):
    def _prominence(self, audio):
        return partial_prominence(audio.astype(np.float32), SAMPLE_RATE, _times(len(audio) / SAMPLE_RATE)[10:-10])

    def test_a_sung_voice_has_prominent_partials(self):
        self.assertGreater(np.median(self._prominence(_voice(220.0, 1.0))), MIN_PROMINENCE_DB + 5)

    def test_a_voice_with_vibrato_too(self):
        self.assertGreater(np.median(self._prominence(_voice(330.0, 1.0, vibrato=0.5))), MIN_PROMINENCE_DB)

    def test_several_voices_at_once_too(self):
        chord = _voice(220.0, 1.0, 0.3) + _voice(277.2, 1.0, 0.4) + _voice(329.6, 1.0, 0.2) + _voice(440.0, 1.0, 0.3)
        self.assertGreater(np.median(self._prominence(chord)), MIN_PROMINENCE_DB)

    def test_noise_has_none(self):
        self.assertLess(np.median(self._prominence(_noise(1.0))), MIN_PROMINENCE_DB - 2)

    def test_noise_shaped_like_a_vowel_has_none(self):
        # Noise through two resonances: the shape of the spectrum is no partial
        noise = _noise(1.0)
        spectrum = np.fft.rfft(noise)
        hz = np.fft.rfftfreq(len(noise), 1 / SAMPLE_RATE)
        shape = 1.0 / (1.0 + ((hz - 700.0) / 150.0) ** 2) + 0.5 / (1.0 + ((hz - 1800.0) / 250.0) ** 2)
        shaped = np.fft.irfft(spectrum * shape, len(noise))
        self.assertLess(np.median(self._prominence(shaped / np.abs(shaped).max() * 0.5)), MIN_PROMINENCE_DB - 2)

    def test_silence_and_times_outside_the_audio(self):
        result = partial_prominence(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE, np.array([-1.0, 0.5, 5.0]))
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all(result < MIN_PROMINENCE_DB))


class TestLoudness(unittest.TestCase):
    def test_level_of_a_full_scale_sine(self):
        t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        level = loudness_db(np.sin(2 * np.pi * 440 * t).astype(np.float32), SAMPLE_RATE, np.array([0.5]))
        self.assertAlmostEqual(level[0], -3.0, delta=0.2)

    def test_loud_frames_are_measured_from_the_loud_singing(self):
        level = np.array([-80.0, -20.0, -22.0, -45.0, -55.0, -21.0])
        self.assertEqual(list(loud_frames(level)), [False, True, True, True, False, True])
        # a quiet recording as a whole is judged the same way
        self.assertEqual(list(loud_frames(level - 15.0)), [False, True, True, True, False, True])

    def test_no_sound_at_all(self):
        self.assertFalse(loud_frames(np.full(5, -90.0)).any())


class TestLongSongs(unittest.TestCase):
    """The frames of a song are analysed in blocks, with the same result."""

    def test_blocks_give_the_same_loudness_and_prominence(self):
        from unittest import mock
        from src.modules.Pitcher import unpitched_notes
        audio = np.concatenate([_voice(220.0, 0.5, 0.3), _noise(0.5)]).astype(np.float32)
        times = _times(1.0)
        whole = loudness_db(audio, SAMPLE_RATE, times), partial_prominence(audio, SAMPLE_RATE, times)
        with mock.patch.object(unpitched_notes, "BLOCK_FRAMES", 7):
            blocks = loudness_db(audio, SAMPLE_RATE, times), partial_prominence(audio, SAMPLE_RATE, times)
        np.testing.assert_allclose(blocks[0], whole[0])
        np.testing.assert_allclose(blocks[1], whole[1])


class TestNoteEvidence(unittest.TestCase):
    def test_frames_with_a_pitch_speak_for_a_sung_note(self):
        self.assertLess(note_evidence(20, 18), 0)
        self.assertGreater(note_evidence(20, 1), 0)

    def test_more_frames_are_more_evidence_up_to_a_limit(self):
        self.assertGreater(note_evidence(20, 0), note_evidence(5, 0))
        self.assertEqual(note_evidence(400, 0), note_evidence(200, 0))

    def test_no_loud_frame_says_nothing(self):
        self.assertEqual(note_evidence(0, 0), 0.0)


class TestJudgeNotes(unittest.TestCase):
    def _judge(self, counts, gap=0.0):
        spans = [(k * (0.3 + gap), k * (0.3 + gap) + 0.3) for k in range(len(counts))]
        return judge_notes(spans, [note_evidence(*count) for count in counts])

    def test_notes_without_a_pitch(self):
        self.assertEqual(self._judge([(18, 17), (18, 0), (18, 1), (18, 0), (18, 18)]),
                         [False, True, True, True, False])

    def test_sung_notes(self):
        self.assertEqual(self._judge([(18, 17), (18, 12), (18, 18)]), [False, False, False])

    def test_one_unsure_short_note_within_singing_stays_sung(self):
        self.assertEqual(self._judge([(18, 18), (18, 17), (4, 1), (18, 18), (18, 16)]), [False] * 5)

    def test_a_note_with_nothing_to_count_follows_its_neighbours(self):
        self.assertEqual(self._judge([(18, 0), (18, 0), (0, 0), (18, 1), (18, 0)]), [True] * 5)
        self.assertEqual(self._judge([(18, 18), (0, 0), (18, 17)]), [False] * 3)

    def test_a_short_sung_note_within_harsh_vocals_follows_them(self):
        self.assertEqual(self._judge([(18, 0), (18, 0), (4, 3), (18, 0), (18, 1)]), [True] * 5)

    def test_a_long_sung_note_within_harsh_vocals_stays_sung(self):
        self.assertEqual(self._judge([(18, 0), (18, 0), (40, 38), (18, 0), (18, 1)]),
                         [True, True, False, True, True])

    def test_after_a_long_pause_the_notes_are_judged_anew(self):
        # the unsure note follows harsh vocals right before it, but not across a pause
        counts = [(18, 0), (18, 0), (6, 3)]
        self.assertEqual(self._judge(counts), [True, True, True])
        spans = [(0.0, 0.3), (0.3, 0.6), (10.0, 10.3)]
        self.assertEqual(judge_notes(spans, [note_evidence(*count) for count in counts]), [True, True, False])

    def test_no_notes(self):
        self.assertEqual(judge_notes([], []), [])


class TestFindUnpitchedNotes(unittest.TestCase):
    def test_a_passage_without_a_pitch(self):
        segments = [_note(0.0, 0.4), _note(0.4, 0.8), _note(0.8, 1.2), _note(1.2, 1.6)]
        frames = _frames((0.0, 0.4, True, True), (0.4, 1.2, False, True), (1.2, 1.6, True, True))
        self.assertEqual(find_unpitched_notes(segments, *frames), [False, True, True, False])

    def test_quiet_frames_do_not_count(self):
        # no pitch, but nothing sung either: a note over a pause stays what its neighbours are
        segments = [_note(0.0, 0.4), _note(0.4, 0.8), _note(0.8, 1.2)]
        frames = _frames((0.0, 0.4, True, True), (0.4, 0.8, False, False), (0.8, 1.2, True, True))
        self.assertEqual(find_unpitched_notes(segments, *frames), [False, False, False])

    def test_the_notes_of_a_syllable_are_of_one_kind(self):
        # a long scream that ends on a short sung piece, and a sung syllable with a short unsure piece
        segments = [_note(0.0, 0.3), _note(0.3, 1.3, "scream"), _note(1.3, 1.5, "~"), _note(1.5, 1.7, "~ "),
                    _note(1.7, 2.7, "sung"), _note(2.7, 2.8, "~ "), _note(2.8, 3.2)]
        frames = _frames((0.0, 0.3, True, True), (0.3, 1.5, False, True), (1.5, 2.7, True, True),
                         (2.7, 2.8, False, True), (2.8, 3.2, True, True))
        self.assertEqual(find_unpitched_notes(segments, *frames),
                         [False, True, True, True, False, False, False])

    def test_freestyle_and_rap_notes_stay_what_they_are(self):
        segments = [_note(0.0, 0.4, note_type="F"), _note(0.4, 0.8, note_type="R"), _note(0.8, 1.2)]
        frames = _frames((0.0, 1.2, False, True))
        self.assertEqual(find_unpitched_notes(segments, *frames), [False, False, True])


class TestAsFreestyle(unittest.TestCase):
    def test_a_run_goes_to_the_pitch_of_the_last_sung_note(self):
        segments = [_note(0.0, 0.3, note="D4"), _note(0.3, 0.6, note="A5"), _note(0.6, 0.9, note="F3"),
                    _note(0.9, 1.2, note="G4"), _note(1.2, 1.5, note="C6")]
        result = as_freestyle(segments, [False, True, True, False, True])
        self.assertEqual([(s.note_type, s.note) for s in result],
                         [(":", "D4"), ("F", "D4"), ("F", "D4"), (":", "G4"), ("F", "G4")])

    def test_a_run_at_the_start_goes_to_the_first_sung_note(self):
        segments = [_note(0.0, 0.3, note="A5"), _note(0.3, 0.6, note="D4")]
        result = as_freestyle(segments, [True, False])
        self.assertEqual([(s.note_type, s.note) for s in result], [("F", "D4"), (":", "D4")])

    def test_without_any_sung_note_the_pitches_stay(self):
        segments = [_note(0.0, 0.3, note="A5"), _note(0.3, 0.6, note="D4")]
        result = as_freestyle(segments, [True, True])
        self.assertEqual([(s.note_type, s.note) for s in result], [("F", "A5"), ("F", "D4")])

    def test_freestyle_notes_of_the_lyrics_keep_their_pitch_and_give_none(self):
        segments = [_note(0.0, 0.3, note="D4"), _note(0.3, 0.6, note="B4", note_type="F"), _note(0.6, 0.9, note="A5")]
        result = as_freestyle(segments, [False, False, True])
        self.assertEqual([(s.note_type, s.note) for s in result], [(":", "D4"), ("F", "B4"), ("F", "D4")])

    def test_the_given_notes_are_not_changed(self):
        segment = _note(0.0, 0.3, note="A5")
        segment.line_break_after = True
        result = as_freestyle([_note(0.0, 0.0, note="D4"), segment], [False, True])
        self.assertEqual((segment.note_type, segment.note), (":", "A5"))
        self.assertEqual((result[1].word, result[1].start, result[1].end, result[1].line_break_after),
                         ("word ", 0.0, 0.3, True))


class TestMarkUnpitchedNotes(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.stem = os.path.join(folder.name, "vocals.wav")

    def _song(self):
        """Two seconds sung, two seconds of harsh vocals, one second sung by several voices."""
        choir = _voice(220.0, 1.0, 0.3) + _voice(277.2, 1.0, 0.4) + _voice(329.6, 1.0, 0.2)
        audio = np.concatenate([_voice(220.0, 2.0, 0.3), _noise(2.0), choir * 0.6])
        soundfile.write(self.stem, audio.astype(np.float32), SAMPLE_RATE)
        times = _times(5.0)
        # the pitch detection is sure in the solo singing only
        confidence = np.where(times < 2.0, 0.95, 0.3)
        segments = [_note(k * 0.5, k * 0.5 + 0.5, note="D4" if k < 4 else "A5") for k in range(10)]
        return segments, PitchedData(list(times), [220.0] * len(times), list(confidence))

    def test_harsh_vocals_become_freestyle_and_singing_stays(self):
        segments, pitched_data = self._song()
        result = mark_unpitched_notes(segments, pitched_data, self.stem)
        self.assertEqual("".join(s.note_type for s in result), "::::FFFF::")
        self.assertEqual({s.note for s in result[4:8]}, {"D4"})
        self.assertEqual([s.note for s in result[8:]], ["A5", "A5"])
        self.assertEqual("".join(s.note_type for s in segments), ":" * 10)

    def test_a_song_without_them_gives_the_notes_back(self):
        soundfile.write(self.stem, _voice(220.0, 2.0, 0.3).astype(np.float32), SAMPLE_RATE)
        times = _times(2.0)
        segments = [_note(0.0, 1.0), _note(1.0, 2.0)]
        pitched_data = PitchedData(list(times), [220.0] * len(times), [0.95] * len(times))
        self.assertIs(mark_unpitched_notes(segments, pitched_data, self.stem), segments)

    def test_nothing_to_judge(self):
        segments = [_note(0.0, 1.0)]
        self.assertIs(mark_unpitched_notes(segments, None, self.stem), segments)
        self.assertIs(mark_unpitched_notes(segments, PitchedData([], [], []), self.stem), segments)
        self.assertEqual(mark_unpitched_notes([], PitchedData([0.0], [220.0], [0.9]), self.stem), [])

    def test_a_failing_analysis_gives_the_notes_back(self):
        # also when the memory runs out: the chart is never lost over this pass
        import contextlib
        import io
        from unittest import mock
        from src.modules.Pitcher import unpitched_notes
        segments, pitched_data = self._song()
        for error in (MemoryError(), KeyError("frames"), Exception("no decoder")):
            with self.subTest(error=type(error).__name__):
                out = io.StringIO()
                with mock.patch.object(unpitched_notes, "partial_prominence", side_effect=error), \
                        contextlib.redirect_stdout(out):
                    self.assertIs(mark_unpitched_notes(segments, pitched_data, self.stem), segments)
                self.assertIn("the freestyle detection failed", out.getvalue())

    def test_a_stem_that_cannot_be_read_gives_the_notes_back(self):
        segments = [_note(0.0, 1.0)]
        pitched_data = PitchedData([0.5], [220.0], [0.2])
        missing = os.path.join(os.path.dirname(self.stem), "missing.wav")
        self.assertIs(mark_unpitched_notes(segments, pitched_data, missing), segments)


if __name__ == "__main__":
    unittest.main()
