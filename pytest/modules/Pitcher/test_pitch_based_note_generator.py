"""Tests for pitch-based note generation."""

import pytest

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Speech_Recognition.TranscribedData import TranscribedData
from src.modules.Pitcher.pitch_based_note_generator import (
    _find_voiced_regions,
    _median_filter_midi,
    _segment_voiced_region,
    _merge_short_notes,
    _merge_same_pitch_neighbors,
    _overlay_lyrics,
    create_midi_segments_from_pitch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pitched_data(
    n_frames: int = 100,
    freq: float = 440.0,
    confidence: float = 0.9,
    dt: float = 0.016,
) -> PitchedData:
    """Create simple pitched data with constant pitch."""
    return PitchedData(
        times=[i * dt for i in range(n_frames)],
        frequencies=[freq] * n_frames,
        confidence=[confidence] * n_frames,
    )


def _make_two_pitch_data(
    n1: int = 50, n2: int = 50,
    freq1: float = 440.0, freq2: float = 660.0,
    confidence: float = 0.9, dt: float = 0.016,
) -> PitchedData:
    """Create pitched data with two distinct pitch regions."""
    n = n1 + n2
    return PitchedData(
        times=[i * dt for i in range(n)],
        frequencies=[freq1] * n1 + [freq2] * n2,
        confidence=[confidence] * n,
    )


# ---------------------------------------------------------------------------
# _find_voiced_regions
# ---------------------------------------------------------------------------

class TestFindVoicedRegions:

    def test_all_voiced(self):
        pd = _make_pitched_data(n_frames=20, confidence=0.9)
        regions = _find_voiced_regions(pd, confidence_threshold=0.3)
        assert len(regions) == 1
        assert regions[0] == (0, 20)

    def test_all_unvoiced(self):
        pd = _make_pitched_data(n_frames=20, confidence=0.1)
        regions = _find_voiced_regions(pd, confidence_threshold=0.3)
        assert regions == []

    def test_empty_data(self):
        pd = PitchedData(times=[], frequencies=[], confidence=[])
        assert _find_voiced_regions(pd) == []

    def test_gap_bridging(self):
        """Small unvoiced gap (<=3 frames) between voiced should be bridged."""
        n = 30
        conf = [0.9] * 10 + [0.1] * 2 + [0.9] * 18
        pd = PitchedData(
            times=[i * 0.016 for i in range(n)],
            frequencies=[440.0] * n,
            confidence=conf,
        )
        regions = _find_voiced_regions(pd, bridge_gap_frames=3)
        assert len(regions) == 1  # gap of 2 is bridged

    def test_large_gap_not_bridged(self):
        """Large unvoiced gap (>3 frames) should not be bridged."""
        n = 30
        conf = [0.9] * 10 + [0.1] * 10 + [0.9] * 10
        pd = PitchedData(
            times=[i * 0.016 for i in range(n)],
            frequencies=[440.0] * n,
            confidence=conf,
        )
        regions = _find_voiced_regions(pd, bridge_gap_frames=3, min_region_frames=5)
        assert len(regions) == 2

    def test_short_region_filtered(self):
        """Regions shorter than min_region_frames are dropped."""
        n = 20
        conf = [0.9] * 3 + [0.1] * 7 + [0.9] * 10
        pd = PitchedData(
            times=[i * 0.016 for i in range(n)],
            frequencies=[440.0] * n,
            confidence=conf,
        )
        regions = _find_voiced_regions(pd, min_region_frames=5)
        # First region is only 3 frames → filtered out
        assert len(regions) == 1
        assert regions[0] == (10, 20)

    def test_zero_frequency_not_voiced(self):
        """Frames with frequency=0 should not be counted as voiced."""
        n = 20
        pd = PitchedData(
            times=[i * 0.016 for i in range(n)],
            frequencies=[0.0] * n,
            confidence=[0.9] * n,
        )
        assert _find_voiced_regions(pd) == []


# ---------------------------------------------------------------------------
# _median_filter_midi
# ---------------------------------------------------------------------------

class TestMedianFilterMidi:

    def test_constant_pitch_unchanged(self):
        freqs = [440.0] * 10
        confs = [0.9] * 10
        result = _median_filter_midi(freqs, confs, window=5)
        assert len(result) == 10
        # All values should be the same (constant pitch)
        valid = [v for v in result if v is not None]
        assert len(valid) == 10
        assert all(abs(v - valid[0]) < 0.01 for v in valid)

    def test_low_confidence_becomes_none(self):
        freqs = [440.0, 440.0, 440.0]
        confs = [0.9, 0.1, 0.9]
        result = _median_filter_midi(freqs, confs, confidence_threshold=0.3)
        assert result[1] is None

    def test_empty_input(self):
        assert _median_filter_midi([], [], window=5) == []

    def test_single_frame(self):
        result = _median_filter_midi([440.0], [0.9], window=5)
        assert len(result) == 1
        assert result[0] is not None


# ---------------------------------------------------------------------------
# _merge_short_notes
# ---------------------------------------------------------------------------

class TestMergeShortNotes:

    def test_no_short_notes(self):
        segs = [
            MidiSegment("C4", 0.0, 0.5, "hello "),
            MidiSegment("E4", 0.5, 1.0, "world "),
        ]
        result = _merge_short_notes(segs, min_note_duration_ms=100)
        assert len(result) == 2

    def test_short_note_absorbed(self):
        segs = [
            MidiSegment("C4", 0.0, 0.5, "hello "),
            MidiSegment("C#4", 0.5, 0.52, "~ "),  # 20ms = too short
            MidiSegment("D4", 0.52, 1.0, "world "),
        ]
        result = _merge_short_notes(segs, min_note_duration_ms=100)
        assert len(result) == 2

    def test_single_note_unchanged(self):
        segs = [MidiSegment("C4", 0.0, 0.01, "x ")]
        result = _merge_short_notes(segs, min_note_duration_ms=100)
        assert len(result) == 1

    def test_empty_list(self):
        assert _merge_short_notes([], min_note_duration_ms=100) == []


# ---------------------------------------------------------------------------
# _merge_same_pitch_neighbors
# ---------------------------------------------------------------------------

class TestMergeSamePitchNeighbors:

    def test_same_pitch_merged(self):
        segs = [
            MidiSegment("C4", 0.0, 0.5, "hello "),
            MidiSegment("C4", 0.5, 1.0, "~ "),
        ]
        result = _merge_same_pitch_neighbors(segs)
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 1.0
        assert result[0].word == "hello "

    def test_different_pitches_not_merged(self):
        segs = [
            MidiSegment("C4", 0.0, 0.5, "hello "),
            MidiSegment("E4", 0.5, 1.0, "world "),
        ]
        result = _merge_same_pitch_neighbors(segs)
        assert len(result) == 2

    def test_keeps_word_over_continuation(self):
        """When merging, keep the word from the note that has text."""
        segs = [
            MidiSegment("C4", 0.0, 0.5, "~ "),
            MidiSegment("C4", 0.5, 1.0, "hello "),
        ]
        result = _merge_same_pitch_neighbors(segs)
        assert result[0].word == "hello "

    def test_empty_list(self):
        assert _merge_same_pitch_neighbors([]) == []


# ---------------------------------------------------------------------------
# _overlay_lyrics
# ---------------------------------------------------------------------------

class TestOverlayLyrics:

    def test_basic_overlay(self):
        segs = [
            MidiSegment("C4", 0.0, 0.5, "~ "),
            MidiSegment("E4", 0.5, 1.0, "~ "),
            MidiSegment("G4", 1.0, 1.5, "~ "),
        ]
        words = [
            TranscribedData(word="Hello ", start=0.1, end=0.4),
            TranscribedData(word="world ", start=0.6, end=0.9),
        ]
        result = _overlay_lyrics(segs, words)
        assert result[0].word == "Hello "
        assert result[1].word == "world "
        assert result[2].word == "~ "  # no word assigned

    def test_no_words(self):
        segs = [MidiSegment("C4", 0.0, 0.5, "~ ")]
        result = _overlay_lyrics(segs, [])
        assert result[0].word == "~ "

    def test_empty_segments(self):
        words = [TranscribedData(word="Hello ", start=0.0, end=0.5)]
        assert _overlay_lyrics([], words) == []

    def test_word_assigned_to_max_overlap(self):
        """Word should be assigned to the note with maximum time overlap."""
        segs = [
            MidiSegment("C4", 0.0, 0.3, "~ "),
            MidiSegment("E4", 0.3, 1.0, "~ "),
        ]
        words = [TranscribedData(word="hello ", start=0.2, end=0.8)]
        result = _overlay_lyrics(segs, words)
        # Overlap with seg[0]: 0.3-0.2=0.1, seg[1]: 0.8-0.3=0.5
        assert result[1].word == "hello "
        assert result[0].word == "~ "


# ---------------------------------------------------------------------------
# create_midi_segments_from_pitch (integration)
# ---------------------------------------------------------------------------

class TestCreateMidiSegmentsFromPitch:

    def test_constant_pitch_single_note(self):
        pd = _make_pitched_data(n_frames=100, freq=440.0)
        words = [TranscribedData(word="hello ", start=0.0, end=1.0)]
        result = create_midi_segments_from_pitch(pd, words)
        assert len(result) >= 1
        # At least one note should have the word
        word_notes = [s for s in result if "hello" in s.word]
        assert len(word_notes) >= 1

    def test_two_pitches_create_multiple_notes(self):
        pd = _make_two_pitch_data(n1=80, n2=80, freq1=261.63, freq2=523.25)
        words = [
            TranscribedData(word="low ", start=0.0, end=0.8),
            TranscribedData(word="high ", start=0.8, end=1.6),
        ]
        result = create_midi_segments_from_pitch(pd, words)
        # Should produce at least 2 distinct notes for the two pitches
        assert len(result) >= 2

    def test_empty_pitched_data(self):
        pd = PitchedData(times=[], frequencies=[], confidence=[])
        assert create_midi_segments_from_pitch(pd, []) == []

    def test_no_transcribed_data(self):
        """Should still produce notes, just with ~ placeholders."""
        pd = _make_pitched_data(n_frames=100)
        result = create_midi_segments_from_pitch(pd, [])
        assert len(result) >= 1
        assert all(s.word in ("~", "~ ") for s in result)

    def test_all_low_confidence(self):
        pd = _make_pitched_data(n_frames=100, confidence=0.05)
        result = create_midi_segments_from_pitch(pd, [])
        assert result == []

    def test_segments_have_valid_timing(self):
        pd = _make_pitched_data(n_frames=200, freq=440.0)
        result = create_midi_segments_from_pitch(pd, [])
        for seg in result:
            assert seg.end > seg.start
            assert seg.start >= 0.0

    def test_segments_are_sorted(self):
        pd = _make_two_pitch_data(n1=100, n2=100)
        result = create_midi_segments_from_pitch(pd, [])
        for i in range(len(result) - 1):
            assert result[i].start <= result[i + 1].start

    def test_none_pitched_data_returns_empty(self):
        assert create_midi_segments_from_pitch(None, []) == []
