"""Tests for the hybrid fusion engine."""

import pytest

from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher.hybrid_fusion import (
    WordTiming,
    _find_overlapping_words,
    _split_note_at_word_boundaries,
    fuse_pitch_notes_with_lyrics,
    transcribed_data_to_word_timings,
)


class TestFindOverlappingWords:
    """Test word-to-note overlap detection."""

    def test_single_word_full_overlap(self):
        note = MidiSegment("C4", 1.0, 2.0, "~ ")
        words = [WordTiming("hello", 1.0, 2.0)]
        result = _find_overlapping_words(note, words, min_overlap_ratio=0.3)
        assert len(result) == 1
        assert result[0].word == "hello"

    def test_no_overlap(self):
        note = MidiSegment("C4", 1.0, 2.0, "~ ")
        words = [WordTiming("hello", 3.0, 4.0)]
        result = _find_overlapping_words(note, words, min_overlap_ratio=0.3)
        assert len(result) == 0

    def test_partial_overlap_below_threshold(self):
        note = MidiSegment("C4", 1.0, 2.0, "~ ")
        # Word is 1s long, overlap is 0.2s = 20% < 30% threshold
        words = [WordTiming("hello", 1.8, 2.8)]
        result = _find_overlapping_words(note, words, min_overlap_ratio=0.3)
        assert len(result) == 0

    def test_partial_overlap_above_threshold(self):
        note = MidiSegment("C4", 1.0, 2.0, "~ ")
        # Word is 1s long, overlap is 0.5s = 50% > 30% threshold
        words = [WordTiming("hello", 1.5, 2.5)]
        result = _find_overlapping_words(note, words, min_overlap_ratio=0.3)
        assert len(result) == 1

    def test_multiple_words_overlap(self):
        note = MidiSegment("C4", 1.0, 3.0, "~ ")
        words = [
            WordTiming("hello", 1.0, 1.8),
            WordTiming("world", 1.9, 2.8),
        ]
        result = _find_overlapping_words(note, words, min_overlap_ratio=0.3)
        assert len(result) == 2


class TestSplitNoteAtWordBoundaries:
    """Test splitting a pitch note at word boundaries."""

    def test_single_word_no_split(self):
        note = MidiSegment("C4", 1.0, 2.0, "~ ")
        words = [WordTiming("hello", 1.0, 2.0)]
        result = _split_note_at_word_boundaries(note, words)
        assert len(result) == 1
        assert result[0].word == "hello "

    def test_two_words_split(self):
        note = MidiSegment("E4", 1.0, 3.0, "~ ")
        words = [
            WordTiming("hello", 1.0, 1.8),
            WordTiming("world", 2.0, 2.8),
        ]
        result = _split_note_at_word_boundaries(note, words)
        assert len(result) == 2
        assert result[0].word == "hello "
        assert result[1].word == "world "
        assert result[0].note == "E4"
        assert result[1].note == "E4"

    def test_empty_words(self):
        note = MidiSegment("C4", 1.0, 2.0, "~ ")
        result = _split_note_at_word_boundaries(note, [])
        assert len(result) == 1
        assert result[0].word == "~ "


class TestFusePitchNotesWithLyrics:
    """Test the main fusion function."""

    def test_empty_pitch_notes(self):
        result = fuse_pitch_notes_with_lyrics([], [])
        assert result == []

    def test_single_note_single_word(self):
        notes = [MidiSegment("C4", 1.0, 2.0, "~ ")]
        words = [WordTiming("hello", 1.0, 2.0)]
        result = fuse_pitch_notes_with_lyrics(notes, words)
        assert len(result) == 1
        assert result[0].note == "C4"
        assert "hello" in result[0].word

    def test_melisma_one_word_three_notes(self):
        """One word spanning three pitch notes — first gets word, rest get ~."""
        notes = [
            MidiSegment("C4", 1.0, 1.5, "~ "),
            MidiSegment("D4", 1.5, 2.0, "~ "),
            MidiSegment("E4", 2.0, 2.5, "~ "),
        ]
        words = [WordTiming("ah", 1.0, 2.5)]
        result = fuse_pitch_notes_with_lyrics(notes, words)
        # All three notes should exist, first gets "ah"
        assert len(result) == 3
        assert "ah" in result[0].word
        assert result[0].note == "C4"
        assert result[1].note == "D4"
        assert result[2].note == "E4"

    def test_multiple_words_on_one_note(self):
        """Two words overlapping a single pitch note — note gets split."""
        notes = [MidiSegment("C4", 1.0, 3.0, "~ ")]
        words = [
            WordTiming("hello", 1.0, 1.8),
            WordTiming("world", 2.0, 2.8),
        ]
        result = fuse_pitch_notes_with_lyrics(notes, words)
        assert len(result) == 2
        assert "hello" in result[0].word
        assert "world" in result[1].word

    def test_note_without_word_keeps_placeholder(self):
        """Pitch note without overlapping word keeps placeholder."""
        notes = [
            MidiSegment("C4", 1.0, 2.0, "~ "),
            MidiSegment("D4", 5.0, 6.0, "~ "),  # no word here
        ]
        words = [WordTiming("hello", 1.0, 2.0)]
        result = fuse_pitch_notes_with_lyrics(notes, words)
        assert len(result) == 2
        assert "hello" in result[0].word
        assert result[1].word.strip() in ("~", "~ ")

    def test_word_without_note_creates_gap(self):
        """Word without pitch note gets a gap note when pitched_data provided."""
        from modules.Pitcher.pitched_data import PitchedData

        notes = [MidiSegment("C4", 1.0, 2.0, "~ ")]
        words = [
            WordTiming("hello", 1.0, 2.0),
            WordTiming("world", 3.0, 4.0),  # no pitch note here
        ]
        # Create minimal pitched_data covering the gap
        times = [i * 0.016 for i in range(300)]  # ~4.8 seconds
        freqs = [261.63] * 300  # C4
        confs = [0.8] * 300
        pitched = PitchedData(times, freqs, confs)

        result = fuse_pitch_notes_with_lyrics(
            notes, words, pitched_data=pitched,
        )
        # Should have the original note + a gap note for "world"
        assert len(result) >= 2
        words_found = [s.word.strip() for s in result if s.word.strip() not in ("~", "")]
        assert "hello" in words_found
        assert "world" in words_found


class TestTranscribedDataToWordTimings:
    """Test conversion from TranscribedData to WordTiming."""

    def test_basic_conversion(self):
        from modules.Speech_Recognition.TranscribedData import TranscribedData

        td = [
            TranscribedData(word="hello", start=1.0, end=1.5, confidence=0.9, is_word_end=True),
            TranscribedData(word="world", start=1.6, end=2.0, confidence=0.8, is_word_end=True),
        ]
        result = transcribed_data_to_word_timings(td)
        assert len(result) == 2
        assert result[0].word == "hello"
        assert result[0].start == 1.0
        assert result[1].word == "world"
