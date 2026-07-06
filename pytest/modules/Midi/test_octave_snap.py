"""Tests for snap_isolated_octave_spikes (isolated octave-jump removal)."""
import librosa

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Midi.midi_creator import snap_isolated_octave_spikes


def _segs(midis):
    return [MidiSegment(note=librosa.midi_to_note(m), start=i, end=i + 0.5, word="la")
            for i, m in enumerate(midis)]


def _midis(segs):
    return [int(librosa.note_to_midi(s.note)) for s in segs]


def test_isolated_up_spike_folded_down():
    # one note an octave above a stable line
    out = snap_isolated_octave_spikes(_segs([60, 60, 72, 60, 60]))
    assert _midis(out) == [60, 60, 60, 60, 60]


def test_isolated_down_spike_folded_up():
    out = snap_isolated_octave_spikes(_segs([60, 60, 48, 60, 60]))
    assert _midis(out) == [60, 60, 60, 60, 60]


def test_genuine_leap_untouched():
    # a sustained move up (not an isolated spike) must stay
    out = snap_isolated_octave_spikes(_segs([60, 60, 67, 67, 67]))
    assert _midis(out) == [60, 60, 67, 67, 67]


def test_small_interval_untouched():
    # a fifth spike (7 semitones) is a legit interval, below min_gap
    out = snap_isolated_octave_spikes(_segs([60, 60, 67, 60, 60]))
    assert _midis(out) == [60, 60, 67, 60, 60]


def test_unstable_context_untouched():
    # neighbours disagree (moving passage) -> do not fold
    out = snap_isolated_octave_spikes(_segs([55, 55, 72, 66, 66]))
    assert _midis(out) == [55, 55, 72, 66, 66]


def test_pitch_class_preserved():
    # a C5 spike over an A4 line folds to C4 (same class C), not to A
    out = snap_isolated_octave_spikes(_segs([69, 69, 84, 69, 69]))
    assert librosa.note_to_midi(out[2].note) % 12 == 0  # C


def test_short_input_noop():
    # fewer than 3 notes: returned unchanged
    out = snap_isolated_octave_spikes(_segs([60, 72]))
    assert _midis(out) == [60, 72]
