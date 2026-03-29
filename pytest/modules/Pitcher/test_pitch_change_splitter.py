"""Tests for pitch_change_splitter.py"""

import pytest

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Pitcher.pitch_change_splitter import (
    split_notes_at_pitch_changes,
    _detect_pitch_change_points,
    _merge_short_fragments,
    _split_single_segment,
    _get_frames_for_segment,
)


def _make_pitched_data(
    times: list[float],
    frequencies: list[float],
    confidence: list[float] | None = None,
) -> PitchedData:
    """Helper to create PitchedData with optional default confidence."""
    if confidence is None:
        confidence = [0.9] * len(times)
    return PitchedData(times, frequencies, confidence)


def _hz(note: str) -> float:
    """Convert note name to Hz for readability."""
    import librosa
    return float(librosa.note_to_hz(note))


class TestConstantPitch:
    """No split should occur when pitch is constant."""

    def test_single_note_constant_pitch(self):
        """A segment with constant pitch should not be split."""
        # 10 frames at C4 (~262 Hz), 16ms apart
        times = [i * 0.016 for i in range(10)]
        freqs = [_hz("C4")] * 10
        pitched = _make_pitched_data(times, freqs)

        segment = MidiSegment("C4", 0.0, 0.16, "hello ")
        result = split_notes_at_pitch_changes([segment], pitched)

        assert len(result) == 1
        assert result[0].word == "hello "
        assert result[0].note == "C4"

    def test_empty_segments_list(self):
        """Empty input should return empty output."""
        pitched = _make_pitched_data([], [], [])
        result = split_notes_at_pitch_changes([], pitched)
        assert result == []

    def test_single_frame_segment(self):
        """A segment spanning only one frame should not be split."""
        times = [0.0, 0.016]
        freqs = [_hz("C4"), _hz("E4")]
        pitched = _make_pitched_data(times, freqs)

        segment = MidiSegment("C4", 0.0, 0.008, "x ")
        result = split_notes_at_pitch_changes([segment], pitched)

        assert len(result) == 1
        assert result[0].word == "x "


class TestClearPitchChange:
    """Segments with clear pitch changes should be split."""

    def test_two_distinct_pitches(self):
        """C4 for 5 frames then E4 for 5 frames should produce 2 segments."""
        times = [i * 0.016 for i in range(10)]
        freqs = [_hz("C4")] * 5 + [_hz("E4")] * 5
        pitched = _make_pitched_data(times, freqs)

        segment = MidiSegment("C4", 0.0, 0.16, "la ")
        result = split_notes_at_pitch_changes(
            [segment], pitched, min_semitone_change=2.0, min_note_duration_ms=60.0
        )

        assert len(result) >= 2
        # First sub-note should have the original text
        assert result[0].word == "la"
        # Continuation notes should be "~"
        for sub in result[1:]:
            assert sub.word in ("~", "~ ")
        # Last sub-note should have trailing space
        assert result[-1].word.endswith(" ")

    def test_three_distinct_pitches(self):
        """C4 -> E4 -> G4 should produce 3 segments."""
        times = [i * 0.016 for i in range(15)]
        freqs = [_hz("C4")] * 5 + [_hz("E4")] * 5 + [_hz("G4")] * 5
        pitched = _make_pitched_data(times, freqs)

        segment = MidiSegment("C4", 0.0, 0.24, "run ")
        result = split_notes_at_pitch_changes(
            [segment], pitched, min_semitone_change=2.0, min_note_duration_ms=60.0
        )

        assert len(result) >= 3
        assert result[0].word == "run"
        assert all(r.word.rstrip() == "~" for r in result[1:])


class TestBriefFluctuation:
    """Brief pitch fluctuations below min_duration should not cause splits."""

    def test_brief_glitch_no_split(self):
        """A single-frame pitch glitch should not trigger a split."""
        times = [i * 0.016 for i in range(10)]
        # C4 with one frame of E4 glitch
        freqs = [_hz("C4")] * 4 + [_hz("E4")] + [_hz("C4")] * 5
        pitched = _make_pitched_data(times, freqs)

        segment = MidiSegment("C4", 0.0, 0.16, "steady ")
        result = split_notes_at_pitch_changes(
            [segment], pitched, min_semitone_change=2.0, min_note_duration_ms=60.0
        )

        # The E4 glitch is only 16ms, below 60ms threshold
        assert len(result) == 1
        assert result[0].word == "steady "

    def test_small_pitch_variation_no_split(self):
        """Pitch changes below min_semitone_change should not trigger a split."""
        times = [i * 0.016 for i in range(10)]
        # Alternating between C4 and C#4 (1 semitone)
        c4 = _hz("C4")
        cs4 = _hz("C#4")
        freqs = [c4, cs4, c4, cs4, c4, cs4, c4, cs4, c4, cs4]
        pitched = _make_pitched_data(times, freqs)

        segment = MidiSegment("C4", 0.0, 0.16, "vibrato ")
        result = split_notes_at_pitch_changes(
            [segment], pitched, min_semitone_change=2.0, min_note_duration_ms=60.0
        )

        assert len(result) == 1


class TestVibratoResistance:
    """Vibrato (periodic oscillation around a centre pitch) should not cause splits."""

    def test_wide_vibrato_no_split(self):
        """Vibrato spanning ±1.5 semitones (3 ST peak-to-peak) around C4 should
        not trigger splits because the region median stays near C4."""
        import librosa
        import numpy as np

        # Simulate 5 Hz vibrato over 40 frames (640ms at 16ms/frame)
        # ±1.5 ST around C4 (MIDI 60) means oscillating between ~58.5 and ~61.5
        times = [i * 0.016 for i in range(40)]
        c4_midi = 60.0
        midi_vibrato = [c4_midi + 1.5 * np.sin(2 * np.pi * 5.0 * t) for t in times]
        freqs = [float(librosa.midi_to_hz(m)) for m in midi_vibrato]
        pitched = _make_pitched_data(times, freqs)

        segment = MidiSegment("C4", 0.0, times[-1], "vibrato ")
        result = split_notes_at_pitch_changes(
            [segment], pitched, min_semitone_change=2.0, min_note_duration_ms=80.0
        )

        # Vibrato should be filtered by region-median comparison — no split
        assert len(result) == 1
        assert result[0].word == "vibrato "

    def test_vibrato_then_real_change(self):
        """Vibrato followed by a genuine pitch change should produce exactly 2 notes."""
        import librosa
        import numpy as np

        # 20 frames of vibrato around C4, then 20 frames of stable E4
        times = [i * 0.016 for i in range(40)]
        c4_midi = 60.0
        e4_midi = 64.0
        midi_vals = (
            [c4_midi + 1.0 * np.sin(2 * np.pi * 6.0 * t) for t in times[:20]]
            + [e4_midi] * 20
        )
        freqs = [float(librosa.midi_to_hz(m)) for m in midi_vals]
        pitched = _make_pitched_data(times, freqs)

        segment = MidiSegment("C4", 0.0, times[-1], "slide ")
        result = split_notes_at_pitch_changes(
            [segment], pitched, min_semitone_change=2.0, min_note_duration_ms=80.0
        )

        # Should split into vibrato region + stable E4 region
        assert len(result) == 2
        assert result[0].word == "slide"
        assert result[1].word in ("~", "~ ")


class TestShortFragmentMerging:
    """Short fragments should be merged with nearest neighbor."""

    def test_merge_short_fragments(self):
        """Fragments below min_duration_ms should be merged."""
        # Create segments where one is very short
        segments = [
            MidiSegment("C4", 0.0, 0.1, "hello"),
            MidiSegment("E4", 0.1, 0.12, "~"),  # 20ms - too short
            MidiSegment("G4", 0.12, 0.3, "~ "),
        ]

        result = _merge_short_fragments(segments, min_note_duration_ms=80.0)

        # The 20ms fragment should be absorbed
        assert len(result) < 3

    def test_no_merge_when_long_enough(self):
        """Fragments that meet min_duration should not be merged."""
        segments = [
            MidiSegment("C4", 0.0, 0.1, "hello"),
            MidiSegment("E4", 0.1, 0.2, "~"),
            MidiSegment("G4", 0.2, 0.3, "~ "),
        ]

        result = _merge_short_fragments(segments, min_note_duration_ms=80.0)

        assert len(result) == 3

    def test_single_segment_no_merge(self):
        """A single segment should pass through unchanged."""
        segments = [MidiSegment("C4", 0.0, 0.01, "x ")]
        result = _merge_short_fragments(segments, min_note_duration_ms=80.0)
        assert len(result) == 1
        assert result[0].word == "x "


class TestTextPreservation:
    """Text should be preserved correctly on split notes."""

    def test_first_note_keeps_text(self):
        """First sub-note should keep the original word text."""
        times = [i * 0.016 for i in range(10)]
        freqs = [_hz("C4")] * 5 + [_hz("E4")] * 5
        pitched = _make_pitched_data(times, freqs)

        segment = MidiSegment("C4", 0.0, 0.16, "word ")
        result = split_notes_at_pitch_changes(
            [segment], pitched, min_semitone_change=2.0, min_note_duration_ms=60.0
        )

        if len(result) > 1:
            # First note has original text without trailing space
            assert result[0].word == "word"
            # Continuation notes use "~"
            for r in result[1:-1]:
                assert r.word == "~"
            # Last note has trailing space
            assert result[-1].word.endswith(" ")

    def test_no_trailing_space_preserved(self):
        """Words without trailing space should not gain one."""
        times = [i * 0.016 for i in range(10)]
        freqs = [_hz("C4")] * 5 + [_hz("E4")] * 5
        pitched = _make_pitched_data(times, freqs)

        segment = MidiSegment("C4", 0.0, 0.16, "syl")
        result = split_notes_at_pitch_changes(
            [segment], pitched, min_semitone_change=2.0, min_note_duration_ms=60.0
        )

        # No sub-note should end with space since original didn't
        for r in result:
            assert not r.word.endswith(" ")

    def test_tilde_word_not_split_text(self):
        """A segment with '~' as word should keep '~' on all sub-notes."""
        times = [i * 0.016 for i in range(10)]
        freqs = [_hz("C4")] * 5 + [_hz("E4")] * 5
        pitched = _make_pitched_data(times, freqs)

        segment = MidiSegment("C4", 0.0, 0.16, "~ ")
        result = split_notes_at_pitch_changes(
            [segment], pitched, min_semitone_change=2.0, min_note_duration_ms=60.0
        )

        if len(result) > 1:
            assert result[0].word == "~"
            assert result[-1].word.endswith(" ")


class TestMultipleSegments:
    """Multiple segments should be processed independently."""

    def test_mixed_constant_and_changing(self):
        """Only segments with pitch changes should be split."""
        # 20 frames total: first 10 constant, last 10 with change
        times = [i * 0.016 for i in range(20)]
        freqs = [_hz("C4")] * 10 + [_hz("C4")] * 5 + [_hz("E4")] * 5
        pitched = _make_pitched_data(times, freqs)

        segments = [
            MidiSegment("C4", 0.0, 0.16, "one "),
            MidiSegment("C4", 0.16, 0.32, "two "),
        ]

        result = split_notes_at_pitch_changes(
            segments, pitched, min_semitone_change=2.0, min_note_duration_ms=60.0
        )

        # First segment (constant) should not be split
        # Second segment (changing) may be split
        assert len(result) >= 2

    def test_preserves_segment_order(self):
        """Output segments should maintain chronological order."""
        times = [i * 0.016 for i in range(20)]
        freqs = [_hz("C4")] * 5 + [_hz("E4")] * 5 + [_hz("G4")] * 5 + [_hz("C5")] * 5
        pitched = _make_pitched_data(times, freqs)

        segments = [
            MidiSegment("C4", 0.0, 0.16, "a "),
            MidiSegment("G4", 0.16, 0.32, "b "),
        ]

        result = split_notes_at_pitch_changes(
            segments, pitched, min_semitone_change=2.0, min_note_duration_ms=60.0
        )

        # Verify chronological order
        for i in range(len(result) - 1):
            assert result[i].start <= result[i + 1].start


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_low_confidence_frames(self):
        """Low confidence frames should not trigger false splits."""
        times = [i * 0.016 for i in range(10)]
        freqs = [_hz("C4")] * 5 + [_hz("E4")] * 5
        # Low confidence on the E4 frames
        confs = [0.9] * 5 + [0.1] * 5
        pitched = _make_pitched_data(times, freqs, confs)

        segment = MidiSegment("C4", 0.0, 0.16, "test ")
        result = split_notes_at_pitch_changes(
            [segment], pitched, min_semitone_change=2.0, min_note_duration_ms=60.0
        )

        # Low confidence frames should not form a valid split
        assert len(result) == 1

    def test_zero_frequency_frames(self):
        """Zero frequency (silence) frames should be handled gracefully."""
        times = [i * 0.016 for i in range(10)]
        freqs = [_hz("C4")] * 3 + [0.0] * 4 + [_hz("C4")] * 3
        confs = [0.9] * 3 + [0.0] * 4 + [0.9] * 3
        pitched = _make_pitched_data(times, freqs, confs)

        segment = MidiSegment("C4", 0.0, 0.16, "gap ")
        result = split_notes_at_pitch_changes([segment], pitched)

        # Should handle gracefully without error
        assert len(result) >= 1
        assert result[0].word.rstrip() in ("gap", "~")

    def test_custom_thresholds(self):
        """Custom min_semitone_change and min_note_duration_ms should work."""
        times = [i * 0.016 for i in range(20)]
        # C4 -> D4 (2 semitones)
        freqs = [_hz("C4")] * 10 + [_hz("D4")] * 10
        pitched = _make_pitched_data(times, freqs)

        segment = MidiSegment("C4", 0.0, 0.32, "test ")

        # With high threshold (3 semitones), 2-semitone change should not split
        result_high = split_notes_at_pitch_changes(
            [segment], pitched, min_semitone_change=3.0, min_note_duration_ms=60.0
        )
        assert len(result_high) == 1

        # With low threshold (1 semitone), 2-semitone change should split
        result_low = split_notes_at_pitch_changes(
            [segment], pitched, min_semitone_change=1.0, min_note_duration_ms=60.0
        )
        assert len(result_low) >= 2

    def test_time_boundaries_preserved(self):
        """Start of first and end of last sub-note should match original."""
        times = [i * 0.016 for i in range(10)]
        freqs = [_hz("C4")] * 5 + [_hz("E4")] * 5
        pitched = _make_pitched_data(times, freqs)

        segment = MidiSegment("C4", 0.0, 0.16, "bound ")
        result = split_notes_at_pitch_changes(
            [segment], pitched, min_semitone_change=2.0, min_note_duration_ms=60.0
        )

        if len(result) > 1:
            assert result[0].start == segment.start
            assert result[-1].end == segment.end

    def test_large_octave_jump(self):
        """An octave jump (12 semitones) should trigger a split."""
        times = [i * 0.016 for i in range(10)]
        freqs = [_hz("C4")] * 5 + [_hz("C5")] * 5
        pitched = _make_pitched_data(times, freqs)

        segment = MidiSegment("C4", 0.0, 0.16, "jump ")
        result = split_notes_at_pitch_changes(
            [segment], pitched, min_semitone_change=2.0, min_note_duration_ms=60.0
        )

        assert len(result) >= 2
