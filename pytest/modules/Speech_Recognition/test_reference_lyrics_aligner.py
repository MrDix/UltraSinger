"""Tests for reference_lyrics_aligner module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from modules.Speech_Recognition.reference_lyrics_aligner import (
    parse_lrc_synced_lyrics,
    _normalize_segment_timestamps,
    _compute_note_for_word,
    _split_word_at_pitch_changes,
    create_midi_segments_from_reference_lyrics,
)
from modules.Pitcher.pitched_data import PitchedData
from modules.Midi.MidiSegment import MidiSegment


# ---------------------------------------------------------------------------
# LRC parsing
# ---------------------------------------------------------------------------


class TestParseLrcSyncedLyrics:
    """Tests for parse_lrc_synced_lyrics."""

    def test_basic_parsing(self):
        lrc = "[00:17.21] Hello world\n[00:23.25] Second line"
        segs = parse_lrc_synced_lyrics(lrc)
        assert len(segs) == 2
        assert segs[0]["text"] == "Hello world"
        assert segs[0]["start"] == pytest.approx(17.21)
        assert segs[0]["end"] == pytest.approx(23.25)
        assert segs[1]["text"] == "Second line"
        assert segs[1]["start"] == pytest.approx(23.25)

    def test_last_segment_end(self):
        lrc = "[01:00.00] Only line"
        segs = parse_lrc_synced_lyrics(lrc)
        assert len(segs) == 1
        assert segs[0]["end"] == pytest.approx(70.0)  # start + 10s

    def test_empty_lines_skipped(self):
        lrc = "[00:10.00] First\n[00:15.00] \n[00:20.00] Third"
        segs = parse_lrc_synced_lyrics(lrc)
        assert len(segs) == 2
        assert segs[0]["text"] == "First"
        assert segs[1]["text"] == "Third"
        # First segment ends at third's start (empty line skipped)
        assert segs[0]["end"] == pytest.approx(20.0)

    def test_empty_input(self):
        assert parse_lrc_synced_lyrics("") == []
        assert parse_lrc_synced_lyrics("   ") == []

    def test_invalid_lines_ignored(self):
        lrc = "Not a valid line\n[00:10.00] Valid line\nAlso invalid"
        segs = parse_lrc_synced_lyrics(lrc)
        assert len(segs) == 1
        assert segs[0]["text"] == "Valid line"

    def test_minute_conversion(self):
        lrc = "[02:30.50] Two minutes thirty"
        segs = parse_lrc_synced_lyrics(lrc)
        assert segs[0]["start"] == pytest.approx(150.50)

    def test_windows_line_endings(self):
        lrc = "[00:10.00] First\r\n[00:20.00] Second"
        segs = parse_lrc_synced_lyrics(lrc)
        assert len(segs) == 2

    def test_sorted_output(self):
        # Even if input is out of order
        lrc = "[00:20.00] Second\n[00:10.00] First"
        segs = parse_lrc_synced_lyrics(lrc)
        assert segs[0]["start"] < segs[1]["start"]
        assert segs[0]["text"] == "First"

    def test_melismatic_text(self):
        lrc = "[01:55.98] Oh-oh-oh-oh-oh\n[02:00.03] Ooh, ooh, ooh-ooh-ooh"
        segs = parse_lrc_synced_lyrics(lrc)
        assert len(segs) == 2
        assert segs[0]["text"] == "Oh-oh-oh-oh-oh"
        assert segs[1]["text"] == "Ooh, ooh, ooh-ooh-ooh"


# ---------------------------------------------------------------------------
# Timestamp normalization
# ---------------------------------------------------------------------------


class TestNormalizeSegmentTimestamps:
    """Tests for _normalize_segment_timestamps."""

    def test_evenly_spreads_two_segments(self):
        segs = [{"start": 30.0, "end": 60.0}, {"start": 60.0, "end": 90.0}]
        result = _normalize_segment_timestamps(segs, audio_duration=120.0)
        assert result[0]["start"] == pytest.approx(0.0)
        assert result[0]["end"] == pytest.approx(60.0)
        assert result[1]["start"] == pytest.approx(60.0)
        assert result[1]["end"] == pytest.approx(120.0)

    def test_covers_full_audio_duration(self):
        segs = [
            {"start": 10.0, "end": 160.0, "text": "First"},
            {"start": 160.0, "end": 310.0, "text": "Second"},
        ]
        result = _normalize_segment_timestamps(segs, audio_duration=200.0)
        assert result[0]["start"] == pytest.approx(0.0)
        assert result[-1]["end"] == pytest.approx(200.0)

    def test_empty_segments(self):
        assert _normalize_segment_timestamps([], 60.0) == []

    def test_zero_duration(self):
        segs = [{"start": 5.0, "end": 10.0}]
        result = _normalize_segment_timestamps(segs, audio_duration=0.0)
        # Should remain unchanged when audio_duration <= 0
        assert result[0]["start"] == pytest.approx(5.0)

    def test_equal_spacing(self):
        segs = [
            {"start": 0.0, "end": 10.0, "text": "A"},
            {"start": 10.0, "end": 20.0, "text": "B"},
            {"start": 20.0, "end": 30.0, "text": "C"},
        ]
        result = _normalize_segment_timestamps(segs, audio_duration=90.0)
        assert result[0]["start"] == pytest.approx(0.0)
        assert result[0]["end"] == pytest.approx(30.0)
        assert result[1]["start"] == pytest.approx(30.0)
        assert result[1]["end"] == pytest.approx(60.0)
        assert result[2]["start"] == pytest.approx(60.0)
        assert result[2]["end"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Pitch assignment
# ---------------------------------------------------------------------------


def _make_pitched_data(duration_s=10.0, base_freq=440.0, confidence=0.9):
    """Create synthetic PitchedData for testing."""
    # SwiftF0: 16kHz, hop=256 → ~62.5 fps → 16ms per frame
    fps = 62.5
    n_frames = int(duration_s * fps)
    times = [i / fps for i in range(n_frames)]
    frequencies = [base_freq] * n_frames
    confidences = [confidence] * n_frames
    return PitchedData(times=times, frequencies=frequencies, confidence=confidences)


class TestComputeNoteForWord:
    """Tests for _compute_note_for_word."""

    def test_basic_pitch(self):
        pd = _make_pitched_data(base_freq=440.0)  # A4
        note = _compute_note_for_word(1.0, 2.0, pd)
        assert note == "A4"

    def test_c4_pitch(self):
        pd = _make_pitched_data(base_freq=261.63)  # C4
        note = _compute_note_for_word(1.0, 2.0, pd)
        assert note in {"C4", "B3"}  # Exact match, rounding may yield B3

    def test_silent_region_fallback(self):
        pd = _make_pitched_data(base_freq=0.0, confidence=0.0)
        note = _compute_note_for_word(1.0, 2.0, pd)
        assert note == "C4"  # fallback

    def test_key_quantization(self):
        pd = _make_pitched_data(base_freq=440.0)  # A4
        # Allow only C major notes (no A)
        allowed = {"C", "D", "E", "F", "G", "B"}
        note = _compute_note_for_word(1.0, 2.0, pd, allowed_notes=allowed)
        # Should be quantized to nearest allowed note (no accidentals)
        pitch_class = note.rstrip("0123456789")
        assert pitch_class in allowed


# ---------------------------------------------------------------------------
# Melisma splitting
# ---------------------------------------------------------------------------


class TestSplitWordAtPitchChanges:
    """Tests for _split_word_at_pitch_changes."""

    def test_stable_pitch_single_note(self):
        pd = _make_pitched_data(base_freq=440.0)
        result = _split_word_at_pitch_changes("hello", 1.0, 2.0, pd)
        assert len(result) == 1
        assert result[0].word == "hello"

    def test_pitch_change_creates_split(self):
        """A large pitch change mid-word should split into 2+ segments."""
        fps = 62.5
        n_frames = int(5.0 * fps)  # 5 seconds
        times = [i / fps for i in range(n_frames)]
        frequencies = []
        confidences = [0.9] * n_frames
        mid = n_frames // 2
        for i in range(n_frames):
            if i < mid:
                frequencies.append(440.0)  # A4
            else:
                frequencies.append(880.0)  # A5 (12 semitones up)
        pd = PitchedData(times=times, frequencies=frequencies, confidence=confidences)

        result = _split_word_at_pitch_changes("oooh", 0.5, 4.5, pd, threshold_st=2.0)
        assert len(result) >= 2
        assert result[0].word == "oooh"
        assert result[1].word == "~ "

    def test_short_word_no_split(self):
        pd = _make_pitched_data(base_freq=440.0)
        # Very short word - not enough frames to split
        result = _split_word_at_pitch_changes("a", 1.0, 1.05, pd)
        assert len(result) == 1

    def test_silent_word(self):
        pd = _make_pitched_data(base_freq=0.0, confidence=0.0)
        result = _split_word_at_pitch_changes("hmm", 1.0, 2.0, pd)
        assert len(result) == 1

    def test_boundaries_preserved(self):
        """First segment should start at word start, last should end at word end."""
        fps = 62.5
        n_frames = int(5.0 * fps)
        times = [i / fps for i in range(n_frames)]
        frequencies = []
        for i in range(n_frames):
            if i < n_frames // 2:
                frequencies.append(440.0)
            else:
                frequencies.append(880.0)
        pd = PitchedData(times=times, frequencies=frequencies, confidence=[0.9] * n_frames)

        result = _split_word_at_pitch_changes("test", 0.5, 4.5, pd)
        assert result[0].start == pytest.approx(0.5)
        assert result[-1].end == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# Integration (with mocked WhisperX)
# ---------------------------------------------------------------------------


class TestCreateMidiSegmentsFromReferenceLyrics:
    """Tests for create_midi_segments_from_reference_lyrics (mocked alignment)."""

    @patch("modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio")
    def test_basic_pipeline(self, mock_align):
        """Test full pipeline with mocked alignment."""
        mock_align.return_value = [
            {"word": "Hello", "start": 1.0, "end": 1.5},
            {"word": "world", "start": 1.6, "end": 2.0},
        ]

        pd = _make_pitched_data(base_freq=440.0, duration_s=5.0)
        lrc = "[00:01.00] Hello world"

        result = create_midi_segments_from_reference_lyrics(
            synced_lyrics=lrc,
            audio_path="/fake/audio.wav",
            language="en",
            pitched_data=pd,
            melisma_split=False,
        )

        assert len(result) == 2
        assert result[0].word == "Hello"
        assert result[0].start == pytest.approx(1.0)
        assert result[1].word == "world"

    @patch("modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio")
    def test_empty_alignment_returns_empty(self, mock_align):
        mock_align.return_value = []
        pd = _make_pitched_data()
        result = create_midi_segments_from_reference_lyrics(
            "[00:01.00] Test", "/fake/audio.wav", "en", pd
        )
        assert result == []

    def test_empty_lrc_returns_empty(self):
        pd = _make_pitched_data()
        result = create_midi_segments_from_reference_lyrics(
            "", "/fake/audio.wav", "en", pd
        )
        assert result == []

    @patch("modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio")
    def test_melisma_split_enabled(self, mock_align):
        """With melisma split and a pitch change, should produce extra notes."""
        mock_align.return_value = [
            {"word": "oooh", "start": 0.5, "end": 4.5},
        ]

        # Create pitched data with a clear pitch change
        fps = 62.5
        n = int(5.0 * fps)
        times = [i / fps for i in range(n)]
        freqs = [440.0 if i < n // 2 else 880.0 for i in range(n)]
        pd = PitchedData(times=times, frequencies=freqs, confidence=[0.9] * n)

        result = create_midi_segments_from_reference_lyrics(
            "[00:00.50] oooh",
            "/fake/audio.wav",
            "en",
            pd,
            melisma_split=True,
        )

        assert len(result) >= 2
        assert result[0].word == "oooh"
        # Additional segments should be continuations
        for seg in result[1:]:
            assert seg.word == "~ "

    @patch("modules.Speech_Recognition.reference_lyrics_aligner.align_lyrics_to_audio")
    def test_no_melisma_split(self, mock_align):
        """Without melisma split, even pitch changes produce single note."""
        mock_align.return_value = [
            {"word": "oooh", "start": 0.5, "end": 4.5},
        ]

        fps = 62.5
        n = int(5.0 * fps)
        times = [i / fps for i in range(n)]
        freqs = [440.0 if i < n // 2 else 880.0 for i in range(n)]
        pd = PitchedData(times=times, frequencies=freqs, confidence=[0.9] * n)

        result = create_midi_segments_from_reference_lyrics(
            "[00:00.50] oooh",
            "/fake/audio.wav",
            "en",
            pd,
            melisma_split=False,
        )

        assert len(result) == 1
        assert result[0].word == "oooh"
