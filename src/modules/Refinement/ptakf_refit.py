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
from functools import lru_cache

import librosa

from modules.Midi.MidiSegment import MidiSegment
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted

_HOP_SEC = 1024 / 44100.0  # ptAKF frame hop used by ultrastar-score
# Segmentation/pitch tolerance in octave-folded semitones. 0.0 = exact-tone
# resolution: with the former 1.0 (Medium tolerance) a run of +-1-semitone
# steps was coverable by a single note, so the DP never split it - runs and
# ornaments stayed visually flat and Hard scores suffered. Exact segmentation
# makes staircases explicit; measured on real material it lifted Medium from
# ~89% to ~98% and Hard from ~70% to ~94% at unchanged Easy. It also makes
# the score-neutral smoothing staircase-preserving (only identical tones
# merge losslessly).
_TOL = 0.0
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


@lru_cache(maxsize=None)
def _get_hyphenator(language: str):
    """Resolve and cache a Hyphenator for ``language``.

    Reuses the same language-to-dictionary resolution and error handling as
    the transcript-level hyphenation pass (``modules.Speech_Recognition.
    hyphenation.language_check``). Returns ``None`` (and prints the same
    warning) when the language has no installed/downloadable dictionary.
    """
    from hyphen import Hyphenator

    from modules.Speech_Recognition.hyphenation import language_check

    lang_region = language_check(language)
    if lang_region is None:
        return None
    try:
        return Hyphenator(lang_region)
    except Exception:
        return None


def _distribute_syllables(word: str, n_parts: int, language: str | None) -> list[str]:
    """Distribute a word's syllables across ``n_parts`` split notes.

    Uses the project's existing hyphenation infrastructure (same dictionary
    lookup as transcript-level hyphenation) to spread a multi-syllable word
    over its split sub-notes for nicer karaoke display. Purely cosmetic: the
    returned strings only affect displayed lyrics, never the note timing or
    pitch used for scoring.

    Falls back to today's behaviour (``[word, "~", "~", ...]`` with spacing
    matching ``_continuation_word``) when there is no language, no usable
    hyphenator, only one syllable, or hyphenation fails for any reason.
    """
    trailing_space = word.endswith(" ")
    continuation = _continuation_word(word)

    if n_parts <= 1:
        return [word]

    fallback = [word] + [continuation] * (n_parts - 1)

    if not language:
        return fallback

    hyphenator = _get_hyphenator(language)
    if hyphenator is None:
        return fallback

    stripped = word[:-1] if trailing_space else word
    try:
        from modules.Speech_Recognition.hyphenation import hyphenation as hyphenate_word

        syllables = hyphenate_word(stripped, hyphenator)
    except Exception:
        return fallback

    if not syllables or len(syllables) <= 1:
        return fallback

    m = len(syllables)
    if n_parts >= m:
        parts = list(syllables) + ["~"] * (n_parts - m)
    else:
        parts = list(syllables[: n_parts - 1]) + ["".join(syllables[n_parts - 1:])]

    if trailing_space:
        parts[-1] = parts[-1] + " "
    return parts


def _plan_fill_segments(
    grid_tones: list[int],
    min_fill_beats: int,
    min_note_beats: float,
) -> list[tuple[int, int, int]]:
    """Plan extra notes for sung-but-uncharted regions (ad-libs, vocalises).

    ``grid_tones`` holds one tone per beat starting at beat 0; beats already
    covered by chart notes must be pre-masked (any value < 0).  Only maximal
    voiced runs of at least ``min_fill_beats`` beats are charted, so short
    blips and separation bleed stay note-free.

    Returns a list of (start_beat, length, midi).
    """
    fills: list[tuple[int, int, int]] = []
    run_start = None
    for i, bt in enumerate(grid_tones + [-1]):
        if bt >= 0 and run_start is None:
            run_start = i
        elif bt < 0 and run_start is not None:
            if i - run_start >= min_fill_beats:
                run = grid_tones[run_start:i]
                parts = _segment_beat_tones(run)
                parts = _smooth_segments(parts, run, min_note_beats)
                fills.extend((run_start + off, length, midi)
                             for off, length, midi in parts)
            run_start = None
    return fills


def refit_notes_ptakf(
    midi_segments: list[MidiSegment],
    vocal_audio_path: str,
    bpm: float,
    *,
    min_note_ms: float = 100.0,
    fill: bool = False,
    fill_min_ms: float = 300.0,
    language: str | None = None,
) -> list[MidiSegment]:
    """Refit all notes onto the ptAKF beat-tone sequence.

    Fails open: on any error the original segments are returned unchanged.

    Args:
        midi_segments: Notes to refit (not mutated; a new list is returned).
        vocal_audio_path: Path to the vocal-only audio file.
        bpm: Real BPM for beat/time conversion.
        min_note_ms: Segments shorter than this are merged back into a
            neighbour when the merge is score-neutral.
        fill: Also chart sung regions outside all existing notes (ad-libs,
            vocalises, melisma tails) as "~" notes.
        fill_min_ms: Minimum length of an uncharted voiced run before it
            is filled (guards against separation bleed and noise).
        language: Optional language code used to hyphenate multi-syllable
            words so their split sub-notes get individual syllables instead
            of "~" filler. Purely cosmetic (score-neutral); falls back to
            the previous behaviour when omitted or unsupported.

    Returns:
        New list of MidiSegments (or the original list on failure).
    """
    if not midi_segments:
        return midi_segments

    try:
        return _refit(midi_segments, vocal_audio_path, bpm, min_note_ms,
                      fill, fill_min_ms, language)
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
    fill: bool = False,
    fill_min_ms: float = 300.0,
    language: str | None = None,
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

    # --- plan phase: collect (start_beat, duration, note_name, word, type,
    # line_break_after) for every output note, all on the parsed beat grid.
    #
    # Passthrough notes (freestyle/rap/no voiced beats) are re-emitted on
    # the parsed beat grid too — only the seconds representation is
    # normalised, the note keeps the exact beats the writer would have
    # produced from the original timings.  Keeping the original seconds
    # instead is NOT safe: the first emitted segment redefines the GAP, and
    # mixing time conventions can flip a passthrough note by one beat when
    # its fractional beat position falls below the new GAP epsilon
    # (verified end-to-end: max beat-time deviation dropped from ~34 ms,
    # i.e. a full beat, to the intended epsilon of ~9 ms).
    planned: list[tuple[int, int, str, str, str, bool]] = []
    covered: set[int] = set()
    refit_count = 0

    for orig, note in zip(midi_segments, parsed_notes, strict=True):
        covered.update(range(note.start_beat, note.start_beat + note.duration))

        if orig.note_type in ("F", "R"):
            planned.append((note.start_beat, note.duration, orig.note,
                            orig.word, orig.note_type, orig.line_break_after))
            continue

        beat_tones = [beat_tone(note.start_beat + off) for off in range(note.duration)]
        parts = _segment_beat_tones(beat_tones)
        if not parts:
            planned.append((note.start_beat, note.duration, orig.note,
                            orig.word, orig.note_type, orig.line_break_after))
            continue
        parts = _smooth_segments(parts, beat_tones, min_note_beats)

        refit_count += 1
        words = _distribute_syllables(orig.word, len(parts), language)
        for k, (off, length, midi) in enumerate(parts):
            planned.append((
                note.start_beat + off,
                length,
                librosa.midi_to_note(midi),
                words[k],
                orig.note_type,
                orig.line_break_after and k == len(parts) - 1,
            ))

    # --- fill phase: chart sung regions outside all existing notes
    fill_count = 0
    if fill and spb > 0:
        audio_dur = len(tones) * _HOP_SEC
        max_beat = max(0, int((audio_dur - gap_s) / spb))
        grid = [beat_tone(b) if b not in covered else -3
                for b in range(max_beat + 1)]
        min_fill_beats = max(1, int(round(fill_min_ms / 1000.0 / spb)))
        for start_beat, length, midi in _plan_fill_segments(
                grid, min_fill_beats, min_note_beats):
            planned.append((start_beat, length, librosa.midi_to_note(midi),
                            "~ ", ":", False))
            fill_count += 1
        planned.sort(key=lambda p: p[0])

    # --- emit phase
    new_segments: list[MidiSegment] = []
    for idx, (start_beat, duration, note_name, word, note_type, brk) in enumerate(planned):
        start, end = seg_times(start_beat, duration, is_first=idx == 0)
        new_segments.append(MidiSegment(
            note=note_name,
            start=start,
            end=end,
            word=word,
            note_type=note_type,
            line_break_after=brk,
        ))

    fill_info = f", {blue_highlighted(str(fill_count))} fill notes" if fill else ""
    print(
        f"{ULTRASINGER_HEAD} ptAKF refit: "
        f"{blue_highlighted(str(refit_count))} notes refitted, "
        f"{blue_highlighted(str(len(midi_segments)))} -> "
        f"{blue_highlighted(str(len(new_segments)))} notes{fill_info}"
    )
    return new_segments
