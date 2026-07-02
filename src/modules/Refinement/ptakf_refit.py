"""ptAKF chart refit — rebuild note boundaries and pitches from the game's
own pitch detector.

The USDX-style scorer samples one ptAKF frame per beat and compares its
octave-folded tone against the charted pitch.  Upper-bound analysis showed
that charts derived from lyrics timing + SwiftF0 leave ~20 percentage
points on the table: SwiftF0 and ptAKF disagree on ~15-25% of beats, and
notes span unvoiced beats (guaranteed misses).

This pass keeps lyrics, line structure, BPM and GAP, but refits every
note onto the ptAKF beat-tone sequence:

    1. Sample the ptAKF tone at every beat of every note (exactly like
       the scorer does).
    2. Chart only the voiced beat runs inside each note; unvoiced gaps
       (breaths, consonants, reverb tails) stay note-free.
    3. Split runs at pitch changes via dynamic programming (a split must
       gain more hits than ``_SPLIT_PENALTY``).
    4. Merge adjacent short segments back when no hits are lost
       (playability smoothing, ``min_note_ms``).

The first sub-note keeps the syllable text, the rest become "~"
continuations.  Freestyle and rap notes pass through unchanged.

Benchmark (10 songs): Medium score 72.8% -> 90.0%, Easy 81.1% -> 94.2%.
"""

from __future__ import annotations

import os
import statistics

import librosa

from modules.Midi.MidiSegment import MidiSegment
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted

_HOP_SEC = 1024 / 44100.0  # ptAKF frame hop used by ultrastar-score
_TOL = 1.0  # optimise for Medium difficulty (octave-folded +-1 semitone)
_MIN_SEG_BEATS = 2
_SPLIT_PENALTY = 0.25
_PTAKF_TONE_OFFSET = 36  # ptAKF tone 0 = C2 = MIDI 36


def _fold(diff: int) -> int:
    """Fold a tone difference into [-6, 6] (octave-independent)."""
    d = diff % 12
    return d - 12 if d > 6 else d


def _hits_for_class(beat_tones: list[int], cls: int, tol: float = _TOL) -> int:
    """Beats of ``beat_tones`` a note of pitch class ``cls`` would hit."""
    return sum(1 for bt in beat_tones if bt >= 0 and abs(_fold(bt - cls)) <= tol)


def _best_class(beat_tones: list[int], tol: float = _TOL) -> tuple[int, int]:
    """Pitch class (0-11) with the most hits, and its hit count."""
    best_c, best_h = 0, -1
    for c in range(12):
        h = _hits_for_class(beat_tones, c, tol)
        if h > best_h:
            best_c, best_h = c, h
    return best_c, best_h


def _segment_midi(beat_tones: list[int], cls: int) -> int:
    """Absolute MIDI note for a segment: median of the tones matching the
    chosen pitch class (keeps the real octave for readability)."""
    matching = [bt for bt in beat_tones
                if bt >= 0 and abs(_fold(bt - cls)) <= _TOL]
    if not matching:
        matching = [bt for bt in beat_tones if bt >= 0]
    return int(statistics.median(matching)) + _PTAKF_TONE_OFFSET


def _segment_beat_tones(beat_tones: list[int]) -> list[tuple[int, int, int]]:
    """Partition a note's beat-tone list into refit segments.

    Returns a list of (beat_offset, length, midi).  Empty list means the
    note has no voiced beats and should be kept unchanged.
    """
    # voiced runs; unvoiced gaps between them stay uncharted
    runs = []
    cur_start = None
    for i, bt in enumerate(beat_tones + [-1]):
        if bt >= 0 and cur_start is None:
            cur_start = i
        elif bt < 0 and cur_start is not None:
            runs.append((cur_start, i))
            cur_start = None
    if not runs:
        return []

    segments: list[tuple[int, int, int]] = []
    for run_start, run_end in runs:
        run = beat_tones[run_start:run_end]
        n = len(run)
        # DP: best partition of run[:i] into segments of >= _MIN_SEG_BEATS
        # (a shorter final tail is allowed); each split costs _SPLIT_PENALTY
        neg = float("-inf")
        dp = [neg] * (n + 1)
        back: list[int | None] = [None] * (n + 1)
        dp[0] = 0.0
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] == neg:
                    continue
                if i - j < _MIN_SEG_BEATS and i != n:
                    continue
                _, best_h = _best_class(run[j:i])
                val = dp[j] + best_h - _SPLIT_PENALTY
                if val > dp[i]:
                    dp[i] = val
                    back[i] = j
        cuts = []
        i = n
        while i > 0:
            j = back[i]
            cuts.append((j, i))
            i = j
        cuts.reverse()
        for j, i in cuts:
            seg = run[j:i]
            cls, _ = _best_class(seg)
            segments.append((run_start + j, i - j, _segment_midi(seg, cls)))
    return segments


def _smooth_segments(
    segments: list[tuple[int, int, int]],
    beat_tones: list[int],
    min_note_beats: float,
) -> list[tuple[int, int, int]]:
    """Merge segments shorter than ``min_note_beats`` into a contiguous
    neighbour when the merge loses no hits (score-neutral smoothing)."""
    if min_note_beats <= 0 or len(segments) <= 1:
        return segments
    segs = list(segments)
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for k, (off, length, midi) in enumerate(segs):
            if length >= min_note_beats:
                continue
            best = None
            for nb in (k - 1, k + 1):
                if not 0 <= nb < len(segs):
                    continue
                noff, nlen, nmidi = segs[nb]
                # only merge contiguous segments; bridging an unvoiced gap
                # would add guaranteed-miss beats to the note
                if noff != off + length and off != noff + nlen:
                    continue
                lo = min(off, noff)
                hi = max(off + length, noff + nlen)
                merged = beat_tones[lo:hi]
                cur_hits = (
                    _hits_for_class(beat_tones[off:off + length], midi - _PTAKF_TONE_OFFSET)
                    + _hits_for_class(beat_tones[noff:noff + nlen], nmidi - _PTAKF_TONE_OFFSET)
                )
                cls, merged_hits = _best_class(merged)
                loss = cur_hits - merged_hits
                if loss <= 0 and (best is None or loss < best[0]):
                    best = (loss, nb, lo, hi - lo, _segment_midi(merged, cls))
            if best is not None:
                _, nb, lo, ln, midi_new = best
                a, b = sorted((k, nb))
                segs[a] = (lo, ln, midi_new)
                del segs[b]
                changed = True
                break
    return segs


def _continuation_word(word: str) -> str:
    """Continuation syllable marker matching the word's spacing style."""
    return "~ " if word.endswith(" ") else "~"


def refit_notes_ptakf(
    midi_segments: list[MidiSegment],
    vocal_audio_path: str,
    bpm: float,
    *,
    min_note_ms: float = 100.0,
) -> list[MidiSegment]:
    """Refit all notes onto the ptAKF beat-tone sequence.

    Fails open: on any error the original segments are returned unchanged.

    Args:
        midi_segments: Notes to refit (not mutated; a new list is returned).
        vocal_audio_path: Path to the vocal-only audio file.
        bpm: Real BPM for beat/time conversion.
        min_note_ms: Segments shorter than this are merged back into a
            neighbour when the merge is score-neutral.

    Returns:
        New list of MidiSegments (or the original list on failure).
    """
    if not midi_segments:
        return midi_segments

    try:
        return _refit(midi_segments, vocal_audio_path, bpm, min_note_ms)
    except (ImportError, OSError, ValueError, RuntimeError,
            AttributeError, KeyError, TypeError, IndexError) as e:
        print(
            f"{ULTRASINGER_HEAD} Warning: ptAKF refit failed: {e}. "
            f"Keeping original notes."
        )
        return midi_segments


def _refit(
    midi_segments: list[MidiSegment],
    vocal_audio_path: str,
    bpm: float,
    min_note_ms: float,
) -> list[MidiSegment]:
    from ultrastar_score.audio import load_audio
    from ultrastar_score.parser import parse_ultrastar
    from ultrastar_score.pitch import PitchDetector

    from modules.Refinement.refine_from_vocal import _write_temp_ultrastar_txt

    # Reproduce the exact beat grid the final TXT will have
    tmp_txt = _write_temp_ultrastar_txt(midi_segments, bpm)
    try:
        song = parse_ultrastar(tmp_txt)
    finally:
        try:
            os.unlink(tmp_txt)
        except OSError:
            pass

    parsed_notes = [n for line in song.lines for n in line.notes]
    if len(parsed_notes) != len(midi_segments):
        print(
            f"{ULTRASINGER_HEAD} Warning: note count mismatch "
            f"(segments={len(midi_segments)}, parsed={len(parsed_notes)}), "
            f"skipping ptAKF refit"
        )
        return midi_segments

    # ptAKF tones for the whole track (same detector settings as the scorer)
    audio = load_audio(vocal_audio_path)
    detector = PitchDetector(sample_rate=44100, volume_threshold=0.01)
    frames = detector.detect_all(audio)
    tones = [f["tone"] for f in frames]

    def beat_tone(beat: int) -> int:
        idx = int(round(song.beat_to_seconds(beat) / _HOP_SEC))
        return tones[idx] if 0 <= idx < len(tones) else -2

    spb = song.beat_to_seconds(1) - song.beat_to_seconds(0)
    gap_s = song.gap / 1000.0
    min_note_beats = min_note_ms / 1000.0 / spb if spb > 0 else 0.0

    # Beat -> seconds conversion chosen so the final writer reproduces the
    # intended integer beats exactly: the first segment defines the GAP, so
    # its epsilon must stay below the others to keep all relative beat
    # offsets strictly positive before floor().
    first_eps, rest_eps, end_backoff = 0.02, 0.30, 0.35

    def seg_times(start_beat: int, duration: int, is_first: bool) -> tuple[float, float]:
        eps = first_eps if is_first else rest_eps
        start = gap_s + (start_beat + eps) * spb
        end = start + (duration - end_backoff) * spb
        return start, end

    new_segments: list[MidiSegment] = []
    refit_count = 0

    def emit_passthrough(orig: MidiSegment, note) -> None:
        """Re-emit an unchanged note on the parsed beat grid.

        Only the seconds representation is normalised — the note keeps the
        exact beats (``note.start_beat``/``note.duration``) the writer would
        have produced from the original timings.  Keeping the original
        seconds instead is NOT safe: the first emitted segment redefines
        the GAP, and mixing time conventions can flip a passthrough note
        by one beat when its fractional beat position falls below the new
        GAP epsilon (verified end-to-end: max beat-time deviation dropped
        from ~34 ms, i.e. a full beat, to the intended epsilon of ~9 ms).
        """
        start, end = seg_times(note.start_beat, note.duration,
                               is_first=not new_segments)
        new_segments.append(MidiSegment(
            note=orig.note,
            start=start,
            end=end,
            word=orig.word,
            note_type=orig.note_type,
            line_break_after=orig.line_break_after,
        ))

    for orig, note in zip(midi_segments, parsed_notes, strict=True):
        if orig.note_type in ("F", "R"):
            emit_passthrough(orig, note)
            continue

        beat_tones = [beat_tone(note.start_beat + off) for off in range(note.duration)]
        parts = _segment_beat_tones(beat_tones)
        if not parts:
            emit_passthrough(orig, note)
            continue
        parts = _smooth_segments(parts, beat_tones, min_note_beats)

        refit_count += 1
        for k, (off, length, midi) in enumerate(parts):
            start, end = seg_times(note.start_beat + off, length,
                                   is_first=not new_segments)
            new_segments.append(MidiSegment(
                note=librosa.midi_to_note(midi),
                start=start,
                end=end,
                word=orig.word if k == 0 else _continuation_word(orig.word),
                note_type=orig.note_type,
                line_break_after=orig.line_break_after and k == len(parts) - 1,
            ))

    print(
        f"{ULTRASINGER_HEAD} ptAKF refit: "
        f"{blue_highlighted(str(refit_count))} notes refitted, "
        f"{blue_highlighted(str(len(midi_segments)))} -> "
        f"{blue_highlighted(str(len(new_segments)))} notes"
    )
    return new_segments
