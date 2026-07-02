"""Tests for the ptAKF chart refit module."""

from __future__ import annotations

from modules.Midi.MidiSegment import MidiSegment
from modules.Refinement.ptakf_refit import (
    _best_class,
    _continuation_word,
    _fold,
    _hits_for_class,
    _segment_beat_tones,
    _segment_midi,
    _smooth_segments,
    refit_notes_ptakf,
)


# ---------------------------------------------------------------------------
# _fold / _hits_for_class / _best_class
# ---------------------------------------------------------------------------

class TestFold:
    def test_zero(self):
        assert _fold(0) == 0

    def test_within_half_octave(self):
        assert _fold(5) == 5
        assert _fold(-5) == -5

    def test_folds_above_six(self):
        assert _fold(7) == -5
        assert _fold(12) == 0
        assert _fold(17) == 5

    def test_folds_negative(self):
        assert _fold(-7) == 5
        assert _fold(-12) == 0


class TestHitsForClass:
    def test_exact_and_tolerance(self):
        # tones 10, 11, 12 vs class 10 with tol 1: 10 (0), 11 (1), 12 (2)
        assert _hits_for_class([10, 11, 12], 10, tol=1.0) == 2

    def test_octave_folding(self):
        # tone 22 = class 10 one octave up -> hits class 10
        assert _hits_for_class([22], 10, tol=1.0) == 1

    def test_unvoiced_never_hits(self):
        assert _hits_for_class([-1, -2], 0, tol=2.0) == 0


class TestBestClass:
    def test_majority_wins(self):
        cls, hits = _best_class([10, 10, 10, 3])
        assert cls in (9, 10, 11)  # class 10 or a +-1 neighbour ties at 3
        assert hits == 3

    def test_all_unvoiced(self):
        _, hits = _best_class([-1, -1])
        assert hits == 0


class TestSegmentMidi:
    def test_keeps_octave(self):
        # tones around 22 (D#3-ish region): median matching = 22 -> MIDI 58
        assert _segment_midi([22, 22, 23], 22 % 12) == 22 + 36

    def test_falls_back_to_all_voiced(self):
        # no tone matches class 0; falls back to median of voiced tones
        assert _segment_midi([5, 5, 5], 0) == 5 + 36


# ---------------------------------------------------------------------------
# _segment_beat_tones (DP segmentation)
# ---------------------------------------------------------------------------

class TestSegmentBeatTones:
    def test_all_unvoiced_returns_empty(self):
        assert _segment_beat_tones([-1, -1, -2]) == []

    def test_single_stable_run(self):
        segments = _segment_beat_tones([10] * 6)
        assert segments == [(0, 6, 46)]

    def test_trims_leading_and_trailing_unvoiced(self):
        segments = _segment_beat_tones([-1, -1, 10, 10, 10, -1])
        assert segments == [(2, 3, 46)]

    def test_splits_at_pitch_change(self):
        segments = _segment_beat_tones([10] * 4 + [17] * 4)
        assert len(segments) == 2
        assert segments[0] == (0, 4, 46)
        assert segments[1] == (4, 4, 53)

    def test_unvoiced_gap_stays_uncharted(self):
        segments = _segment_beat_tones([10, 10, -1, -1, 10, 10])
        assert segments == [(0, 2, 46), (4, 2, 46)]

    def test_single_beat_blip_does_not_split(self):
        segments = _segment_beat_tones([10] * 5 + [17] + [10] * 5)
        assert len(segments) == 1
        assert segments[0][0] == 0
        assert segments[0][1] == 11

    def test_short_run_kept_as_single_note(self):
        segments = _segment_beat_tones([10])
        assert segments == [(0, 1, 46)]


# ---------------------------------------------------------------------------
# _smooth_segments
# ---------------------------------------------------------------------------

class TestSmoothSegments:
    def test_lossless_merge_of_short_segment(self):
        bts = [10] * 6
        segs = [(0, 2, 46), (2, 4, 46)]
        result = _smooth_segments(segs, bts, min_note_beats=3.0)
        assert result == [(0, 6, 46)]

    def test_lossy_merge_rejected(self):
        bts = [0, 0, 5, 5, 5]
        segs = [(0, 2, 36), (2, 3, 41)]
        result = _smooth_segments(segs, bts, min_note_beats=3.0)
        assert result == segs

    def test_no_merge_across_unvoiced_gap(self):
        bts = [10, 10, -1, -1, 10, 10]
        segs = [(0, 2, 46), (4, 2, 46)]
        result = _smooth_segments(segs, bts, min_note_beats=3.0)
        assert result == segs

    def test_disabled_when_min_zero(self):
        bts = [10] * 6
        segs = [(0, 2, 46), (2, 4, 46)]
        assert _smooth_segments(segs, bts, min_note_beats=0.0) == segs


# ---------------------------------------------------------------------------
# _continuation_word
# ---------------------------------------------------------------------------

class TestContinuationWord:
    def test_trailing_space_style(self):
        assert _continuation_word("hello ") == "~ "

    def test_no_space_style(self):
        assert _continuation_word("hello") == "~"


# ---------------------------------------------------------------------------
# refit_notes_ptakf (fail-open behaviour)
# ---------------------------------------------------------------------------

class TestRefitNotesPtakf:
    def test_empty_segments_returned_unchanged(self):
        assert refit_notes_ptakf([], "does_not_exist.wav", 120.0) == []

    def test_fails_open_on_missing_audio(self):
        segments = [
            MidiSegment(note="C4", start=1.0, end=1.5, word="test "),
            MidiSegment(note="D4", start=1.5, end=2.0, word="case "),
        ]
        result = refit_notes_ptakf(segments, "does_not_exist.wav", 120.0)
        assert result is segments
        assert result[0].note == "C4"
