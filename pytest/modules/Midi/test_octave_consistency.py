"""Tests for enforce_octave_consistency (Viterbi per-note octave assignment)."""
import librosa

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Midi.midi_creator import enforce_octave_consistency


def _segs(midis):
    return [MidiSegment(note=librosa.midi_to_note(m), start=i, end=i + 0.5, word="la")
            for i, m in enumerate(midis)]


def _midis(segs):
    return [int(librosa.note_to_midi(s.note)) for s in segs]


def test_single_spike_folded():
    out = enforce_octave_consistency(_segs([60, 60, 72, 60, 60]))
    assert _midis(out) == [60, 60, 60, 60, 60]


def test_short_run_folded():
    # a 2-note wrong-octave run costs 2*fidelity=4 to fold vs 8 boundary cost
    out = enforce_octave_consistency(_segs([60, 60, 48, 48, 60, 60]))
    assert _midis(out) == [60, 60, 60, 60, 60, 60]


def test_alternating_scatter_unified():
    # rapid octave alternation (the "extremely confusing" case) collapses
    out = enforce_octave_consistency(_segs([60, 48, 60, 48, 60, 48, 60]))
    vals = _midis(out)
    assert max(vals) - min(vals) == 0


def test_long_genuine_octave_passage_kept():
    # >= 5-note octave passage: keeping (2 boundary jumps) is cheaper than
    # folding (fidelity per note) -> genuine chorus jumps survive
    line = [60, 60, 60, 72, 72, 72, 72, 72, 72, 60, 60, 60]
    out = enforce_octave_consistency(_segs(line))
    assert _midis(out) == line


def test_gradual_wide_range_untouched():
    # stepwise movement over a wide range has no jump beyond the hinge
    line = [55, 58, 62, 66, 69, 72, 69, 66, 62, 58, 55]
    out = enforce_octave_consistency(_segs(line))
    assert _midis(out) == line


def test_pitch_class_preserved():
    out = enforce_octave_consistency(_segs([69, 69, 84, 69, 69]))
    assert all(librosa.note_to_midi(s.note) % 12 in (9, 0) for s in out)
    # the spike (class C) may move octaves but never change class
    assert librosa.note_to_midi(out[2].note) % 12 == 0


def test_short_input_noop():
    out = enforce_octave_consistency(_segs([60, 72]))
    assert _midis(out) == [60, 72]


def test_malformed_note_skipped_not_crashing():
    # librosa raises ParameterError (NOT a ValueError subclass) for bad
    # note strings — the pass must treat such segments as unvoiced.
    segs = _segs([60, 60, 72, 60, 60])
    segs[1].note = "not-a-note"
    out = enforce_octave_consistency(segs)
    assert out[1].note == "not-a-note"  # untouched
    assert librosa.note_to_midi(out[2].note) == 60  # spike still folded
