"""Reference-Lyrics-First pipeline: align LRCLIB synced lyrics to audio.

Instead of relying on Whisper ASR to transcribe words (which misses ~50%
of sung text), this module uses verified synced lyrics from LRCLIB as the
text source and wav2vec2 forced alignment (via WhisperX) to obtain precise
word-level timing.

Pipeline:
    1. Parse LRC synced lyrics -> line segments with timestamps
    2. Feed segments to WhisperX forced alignment -> word-level timing
    2a. (reserved -- boundary rebalancing disabled)
    2b. Trim word boundaries to actual voiced regions (confidence clipping)
    2c. Split long notes at silence gaps (gap splitting)
    3. For each word, compute median pitch from SwiftF0 pitched data
    4. Optionally split notes at pitch changes within word boundaries
       (melisma-aware splitting)

This produces MidiSegments where every word from the reference lyrics has
a note with accurate timing and pitch -- solving the lyrics coverage and
timing accuracy problems of the Whisper-only pipeline.
"""

from __future__ import annotations

import re
from typing import Optional

import librosa
import numpy as np

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitched_data_helper import get_frequencies_with_high_confidence
from modules.Midi.midi_creator import (
    confidence_weighted_median_note,
    find_nearest_index,
)
from modules.Audio.key_detector import quantize_note_to_key


# ---------------------------------------------------------------------------
# LRC parsing
# ---------------------------------------------------------------------------

_LRC_LINE_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)")


_BRACKET_RE = re.compile(r"[\(\[\{]([^)\]\}]*)[\)\]\}]")


def parse_lrc_synced_lyrics(
    synced_lyrics: str,
    audio_duration: Optional[float] = None,
) -> list[dict]:
    """Parse LRC format synced lyrics into line segments.

    Args:
        synced_lyrics: Raw LRC text, e.g. ``"[01:23.45] Hello world\\n..."``.
        audio_duration: Total audio duration in seconds.  Used to set the
            last segment's ``end`` time.  Falls back to ``start + 10 s``
            when *None*.

    Returns:
        List of ``{"text": str, "start": float, "end": float}`` dicts,
        sorted by start time.  ``end`` is set to the next line's start
        (or *audio_duration* / ``start + 10 s`` for the last line).

        Each word in the text that originated from a bracketed region
        (backing vocals) is prefixed with ``\\x00`` so downstream code
        can detect and mark it as freestyle.
    """
    lines: list[dict] = []
    for raw_line in synced_lyrics.strip().splitlines():
        m = _LRC_LINE_RE.match(raw_line)
        if not m:
            continue
        minutes, seconds, text = m.groups()
        time_s = int(minutes) * 60 + float(seconds)
        text = text.strip()
        if text:
            lines.append({"time": time_s, "text": text})

    if not lines:
        return []

    # Sort by time (should already be sorted, but be safe)
    lines.sort(key=lambda x: x["time"])

    # Build segments with start/end
    segments = []
    for i, line in enumerate(lines):
        if i + 1 < len(lines):
            end = lines[i + 1]["time"]
        elif audio_duration is not None:
            end = audio_duration
        else:
            end = line["time"] + 10.0

        # Mark bracketed words (backing vocals) with \x00 prefix
        # e.g. "(oh yeah)" -> "\x00oh \x00yeah"
        raw_text = line["text"]
        marked_text = _mark_bracketed_words(raw_text)

        segments.append({
            "text": marked_text,
            "start": line["time"],
            "end": end,
        })

    return segments


def _mark_bracketed_words(text: str) -> str:
    """Mark words inside brackets with \\x00 prefix for freestyle detection.

    Brackets ``()``, ``[]``, ``{}`` indicate backing vocals in LRC lyrics.
    The bracket characters are removed, but the words are preserved and
    prefixed with a NUL byte so the alignment pipeline can detect them.

    Examples:
        >>> _mark_bracketed_words("Hello (oh yeah) world")
        'Hello \\x00oh \\x00yeah world'
        >>> _mark_bracketed_words("No brackets here")
        'No brackets here'
    """
    result_parts: list[str] = []
    last_end = 0
    for m in _BRACKET_RE.finditer(text):
        # Add text before the bracket as-is
        before = text[last_end:m.start()].strip()
        if before:
            result_parts.append(before)
        # Mark each word inside brackets with \x00
        inner = m.group(1).strip()
        if inner:
            for w in inner.split():
                result_parts.append(f"\x00{w}")
        last_end = m.end()
    # Add remaining text after last bracket
    after = text[last_end:].strip()
    if after:
        result_parts.append(after)
    return " ".join(result_parts) if result_parts else text


# ---------------------------------------------------------------------------
# Forced alignment
# ---------------------------------------------------------------------------

def align_lyrics_to_audio(
    segments: list[dict],
    audio_path: str,
    language: str,
    device: str = "cpu",
    align_model_name: Optional[str] = None,
) -> list[dict]:
    """Align LRC line segments to audio using wav2vec2 forced alignment.

    Uses WhisperX's alignment infrastructure (which is already a project
    dependency) to get word-level timing for each line segment.

    Args:
        segments: LRC segments from :func:`parse_lrc_synced_lyrics`.
        audio_path: Path to the vocals audio file.
        language: ISO language code (e.g. ``"en"``, ``"de"``).
        device: ``"cpu"`` or ``"cuda"``.
        align_model_name: Optional custom alignment model (e.g. from
            ``--whisper_align_model``).  Passed to
            ``whisperx.load_align_model(model_name=...)``.

    Returns:
        List of ``{"word": str, "start": float, "end": float}`` dicts,
        one per word, sorted by start time.
    """
    import whisperx

    print(f"{ULTRASINGER_HEAD} Loading alignment model for reference lyrics")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language, device=device, model_name=align_model_name,
    )

    audio = whisperx.load_audio(audio_path)
    audio_duration = librosa.get_duration(path=audio_path)

    # Merge all lyrics into a single segment spanning the full audio.
    # This gives CTC alignment maximum flexibility to find words wherever
    # they actually are, regardless of how the LRC timestamps relate to
    # the audio version.  Per-line segments with narrow windows failed
    # catastrophically when song structure differed between releases.
    all_text = " ".join(s["text"] for s in segments)

    # Build ordered list of original words to track which are backing vocals
    # (\x00-prefixed) and which line each word belongs to.
    original_words: list[str] = []
    line_end_indices: list[int] = []  # index of last word of each LRC line
    for seg in segments:
        seg_words = seg["text"].split()
        original_words.extend(seg_words)
        if seg_words:
            line_end_indices.append(len(original_words) - 1)

    # Strip \x00 markers before feeding to alignment (CTC can't handle them)
    clean_text = all_text.replace("\x00", "")
    merged_segments = [{"text": clean_text, "start": 0.0, "end": audio_duration}]

    total_lrc_words = len(clean_text.split())
    print(
        f"{ULTRASINGER_HEAD} Aligning {blue_highlighted(str(total_lrc_words))} "
        f"reference lyric words to audio"
    )
    aligned = whisperx.align(
        merged_segments, align_model, align_metadata, audio, device,
        return_char_alignments=False,
    )

    # Map aligned words back to original word list to restore backing-vocal
    # markers and line boundary information.
    # Build a clean->original index mapping: the clean word list matches
    # original_words with \x00 stripped.
    clean_original = [w.lstrip("\x00") for w in original_words]
    backing_set = {i for i, w in enumerate(original_words) if w.startswith("\x00")}
    line_end_set = set(line_end_indices)

    # Collect all words with valid timing
    words: list[dict] = []
    aligned_idx = 0
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            if "start" in w and "end" in w and w.get("word", "").strip():
                aligned_word = w["word"].strip()

                # Find matching original word by sequential scan with fuzzy match
                is_backing = False
                is_line_end = False
                if aligned_idx < len(clean_original):
                    # Try exact match at current position
                    if aligned_word.lower() == clean_original[aligned_idx].lower():
                        is_backing = aligned_idx in backing_set
                        is_line_end = aligned_idx in line_end_set
                        aligned_idx += 1
                    else:
                        # Skip forward to find match (alignment may have dropped words)
                        for scan in range(aligned_idx + 1, min(aligned_idx + 5, len(clean_original))):
                            if aligned_word.lower() == clean_original[scan].lower():
                                is_backing = scan in backing_set
                                is_line_end = scan in line_end_set
                                aligned_idx = scan + 1
                                break
                        else:
                            # No match found -- advance anyway
                            aligned_idx += 1

                words.append({
                    "word": aligned_word,
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "backing": is_backing,
                    "line_end": is_line_end,
                })

    # Sort by start time
    words.sort(key=lambda x: x["start"])

    print(
        f"{ULTRASINGER_HEAD} Reference alignment: "
        f"{blue_highlighted(str(len(words)))} of "
        f"{blue_highlighted(str(total_lrc_words))} words aligned"
    )

    return words


# ---------------------------------------------------------------------------
# Pitch assignment
# ---------------------------------------------------------------------------

def _compute_note_for_word(
    start: float,
    end: float,
    pitched_data: PitchedData,
    allowed_notes: Optional[set[str]] = None,
) -> str:
    """Compute the note name for a word's time range from pitched data.

    Uses the same confidence-weighted median approach as the standard
    pipeline (``create_midi_note_from_pitched_data``).
    """
    start_idx = find_nearest_index(pitched_data.times, start)
    end_idx = find_nearest_index(pitched_data.times, end)

    # Clamp indices to valid range (find_nearest_index may return len() for edge cases)
    n = len(pitched_data.frequencies)
    if n == 0:
        return "C4"
    start_idx = max(0, min(start_idx, n - 1))
    end_idx = max(0, min(end_idx, n))  # end_idx used for slicing, so n is valid

    if start_idx == end_idx:
        freqs = [pitched_data.frequencies[start_idx]]
        confs = [pitched_data.confidence[start_idx]]
    else:
        freqs = pitched_data.frequencies[start_idx:end_idx]
        confs = pitched_data.confidence[start_idx:end_idx]

    conf_f, conf_weights = get_frequencies_with_high_confidence(freqs, confs)

    if not conf_f:
        note = "C4"
    else:
        try:
            note = confidence_weighted_median_note(conf_f, conf_weights)
        except ValueError:
            note = "C4"

    if allowed_notes is not None:
        note = quantize_note_to_key(note, allowed_notes)

    return note


# ---------------------------------------------------------------------------
# Confidence-based trimming and gap splitting
# ---------------------------------------------------------------------------

# Minimum confidence for a frame to be considered voiced.
# SwiftF0 reports a frequency for EVERY frame (never returns 0 Hz), even in
# silence -- only the confidence value distinguishes voiced from unvoiced.
# The SwiftF0 detector uses confidence_threshold=0.9 internally, so frames
# below 0.9 are effectively unvoiced.
_VOICED_CONFIDENCE = 0.9
# Minimum frequency (Hz) for a frame to be considered voiced.
_VOICED_MIN_FREQ = 50.0
# Buffer (seconds) added before/after trimmed boundaries to avoid cutting
# off consonant onsets or release tails.
_TRIM_BUFFER_S = 0.05
# Minimum silence gap (seconds) to split a note.
_GAP_MIN_S = 0.15
# Only notes longer than this (seconds) are candidates for gap splitting.
_GAP_SPLIT_MIN_NOTE_S = 1.5


def _rebalance_word_boundaries(
    words: list[dict],
    pitched_data: PitchedData,
    threshold_st: float = 2.0,
) -> list[dict]:
    """No-op placeholder -- rebalancing is disabled.

    Boundary rebalancing was found to cause more problems than it solves:
    it shifts word boundaries based on pitch heuristics, but even with
    safety checks it can destroy lyrics-to-note alignment (e.g. moving
    the first word of line 2 into line 1 when pitches happen to match).

    The function is kept as a no-op to avoid breaking the pipeline call
    site and tests.  The ``"Mmm" 3-tone`` problem is instead addressed
    by the pitch-change-split detecting mismatched prefixes within words.
    """
    return list(words)


def _trim_word_to_voiced(
    word: dict,
    pitched_data: PitchedData,
) -> dict:
    """Trim a word's start/end to the actual voiced region.

    CTC alignment often stretches the last word in a line until the next
    line starts, assigning several seconds of silence to a single word.
    This function clips the word boundaries to the first/last voiced
    frame (with a small buffer), so that the subsequent pitch computation
    only considers frames where the singer is actually producing sound.

    Args:
        word: ``{"word": str, "start": float, "end": float}``.
        pitched_data: SwiftF0 pitch data.

    Returns:
        A new dict with (potentially) tighter ``start`` and ``end``.
    """
    times = np.array(pitched_data.times)
    freqs = np.array(pitched_data.frequencies)
    confs = np.array(pitched_data.confidence)

    mask = (times >= word["start"]) & (times <= word["end"])
    w_times = times[mask]
    w_freqs = freqs[mask]
    w_confs = confs[mask]

    if len(w_times) < 2:
        return word

    voiced = (w_freqs > _VOICED_MIN_FREQ) & (w_confs > _VOICED_CONFIDENCE)
    voiced_indices = np.where(voiced)[0]

    if len(voiced_indices) == 0:
        # No voiced frames at all -- keep original boundaries so that
        # the note still exists (pitch will fallback to C4).
        return word

    first_voiced = w_times[voiced_indices[0]]
    last_voiced = w_times[voiced_indices[-1]]

    new_start = max(word["start"], first_voiced - _TRIM_BUFFER_S)
    new_end = min(word["end"], last_voiced + _TRIM_BUFFER_S)

    # Safety: never shrink below 50ms
    if new_end - new_start < 0.05:
        return word

    return {"word": word["word"], "start": new_start, "end": new_end}


def _split_word_at_silence_gaps(
    word: dict,
    pitched_data: PitchedData,
) -> list[dict]:
    """Split a long word into sub-words at silence gaps.

    When a singer pauses mid-melisma (e.g. breath marks within a long
    sustained syllable), the resulting note should be split at those
    pauses.  The first sub-word keeps the original text; continuation
    sub-words get ``"~ "``.

    Only words longer than ``_GAP_SPLIT_MIN_NOTE_S`` are candidates.

    Args:
        word: ``{"word": str, "start": float, "end": float}``.
        pitched_data: SwiftF0 pitch data.

    Returns:
        List of word dicts (1 if no gaps found, >1 if split).
    """
    duration = word["end"] - word["start"]
    if duration < _GAP_SPLIT_MIN_NOTE_S:
        return [word]

    times = np.array(pitched_data.times)
    freqs = np.array(pitched_data.frequencies)
    confs = np.array(pitched_data.confidence)

    mask = (times >= word["start"]) & (times <= word["end"])
    w_times = times[mask]
    w_freqs = freqs[mask]
    w_confs = confs[mask]

    if len(w_times) < 5:
        return [word]

    voiced = (w_freqs > _VOICED_MIN_FREQ) & (w_confs > _VOICED_CONFIDENCE)

    # Derive frame duration from data
    frame_dt = float(w_times[1] - w_times[0]) if len(w_times) >= 2 else 0.016
    gap_min_frames = max(1, int(_GAP_MIN_S / frame_dt))

    # Find silence gaps: runs of unvoiced frames >= gap_min_frames
    gaps: list[tuple[int, int]] = []  # (start_idx, end_idx) of each gap
    run_start = None
    for i, v in enumerate(voiced):
        if not v:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len >= gap_min_frames:
                    gaps.append((run_start, i))
                run_start = None

    if not gaps:
        return [word]

    # Build sub-words by splitting at gap midpoints
    result: list[dict] = []
    seg_start = word["start"]
    text = word["word"]

    for gap_s_idx, gap_e_idx in gaps:
        gap_start_t = float(w_times[gap_s_idx])
        gap_end_t = float(w_times[min(gap_e_idx, len(w_times) - 1)])

        # End current sub-word at gap start
        if gap_start_t - seg_start >= 0.05:  # minimum 50ms
            result.append({"word": text, "start": seg_start, "end": gap_start_t})
            text = "~ "  # continuation for subsequent sub-words

        # Next sub-word starts after the gap
        seg_start = gap_end_t

    # Final sub-word
    if word["end"] - seg_start >= 0.05:
        result.append({"word": text, "start": seg_start, "end": word["end"]})

    if not result:
        return [word]

    return result


# ---------------------------------------------------------------------------
# Melisma-aware splitting
# ---------------------------------------------------------------------------

def _split_word_at_pitch_changes(
    word: str,
    start: float,
    end: float,
    pitched_data: PitchedData,
    allowed_notes: Optional[set[str]] = None,
    threshold_st: float = 2.0,
    min_note_ms: float = 80.0,
) -> list[MidiSegment]:
    """Split a single word into multiple notes at pitch changes or voice gaps.

    Detects two types of note boundaries within a word:

    1. **Pitch changes**: When the smoothed pitch shifts by more than
       *threshold_st* semitones from the last change point.
    2. **Voice gaps**: When consecutive voiced frames are separated by
       more than 100 ms of unvoiced audio (breath marks, plosives,
       micro-pauses).  These always force a split regardless of pitch
       difference, because a silence break within a sustained note
       is a strong indicator of a new melodic segment.

    If neither condition is met, returns a single MidiSegment.
    The first segment keeps the word text; subsequent ones get ``"~ "``.

    Args:
        word: The word text.
        start: Word start time in seconds.
        end: Word end time in seconds.
        pitched_data: Raw pitch data from SwiftF0.
        allowed_notes: Optional key quantization set.
        threshold_st: Semitone threshold for detecting pitch changes.
        min_note_ms: Minimum note duration in milliseconds.

    Returns:
        List of MidiSegments (1 or more).
    """
    # Minimum time gap (seconds) between consecutive voiced frames
    # to force a split -- breath marks, plosives, micro-pauses.
    _VOICE_GAP_SPLIT_S = 0.1

    times = np.array(pitched_data.times)
    freqs = np.array(pitched_data.frequencies)
    confs = np.array(pitched_data.confidence)

    # Get frames within word boundary
    mask = (times >= start) & (times <= end)
    word_times = times[mask]
    word_freqs = freqs[mask]
    word_confs = confs[mask]

    # Filter for voiced frames (must match the detector's confidence threshold)
    voiced_mask = (word_freqs > _VOICED_MIN_FREQ) & (word_confs > _VOICED_CONFIDENCE)

    if voiced_mask.sum() < 2:
        # Too few voiced frames -- single note
        note = _compute_note_for_word(start, end, pitched_data, allowed_notes)
        return [MidiSegment(note, start, end, word)]

    # Convert frequencies to MIDI for pitch change detection
    voiced_freqs = word_freqs[voiced_mask]
    voiced_times = word_times[voiced_mask]
    midi_values = 69 + 12 * np.log2(voiced_freqs / 440.0)

    # Apply median filter to smooth vibrato while preserving melodic arcs.
    # Size 3 (3 frames x 16ms = 48ms window) removes single-frame spikes
    # from vibrato but keeps genuine pitch transitions intact.  Size 5
    # over-smooths and masks gradual chromatic passages (e.g. rise-then-
    # fall arcs of ~2 semitones).
    if len(midi_values) >= 3:
        from scipy.ndimage import median_filter
        midi_smooth = median_filter(midi_values, size=3)
    else:
        midi_smooth = midi_values

    # Find change points from three sources:
    # (a) pitch jumps exceeding threshold_st semitones from last change
    # (b) voiced-frame time gaps exceeding _VOICE_GAP_SPLIT_S
    # (c) range-based: the pitch rose then fell (or vice versa) by
    #     threshold_st within the current segment -- split at the
    #     turning point (peak or valley).  This catches gradual
    #     melodic arcs where no single frame jumps by the threshold
    #     relative to the segment start.
    #
    # SwiftF0 uses 16kHz sample rate with STFT hop=256 -> ~16ms per frame.
    # Derive frame duration from actual data when possible.
    if len(voiced_times) >= 2:
        frame_ms = (voiced_times[1] - voiced_times[0]) * 1000.0
    else:
        frame_ms = 16.0  # SwiftF0 default: 16kHz sample rate, hop=256
    min_frames = max(1, int(min_note_ms / frame_ms))

    change_points = [0]
    last_change = 0

    # Range tracking for the current segment
    seg_min = seg_max = float(midi_smooth[0])
    seg_min_idx = seg_max_idx = 0

    for i in range(1, len(midi_smooth)):
        # Update range tracking
        if midi_smooth[i] < seg_min:
            seg_min = float(midi_smooth[i])
            seg_min_idx = i
        if midi_smooth[i] > seg_max:
            seg_max = float(midi_smooth[i])
            seg_max_idx = i

        if i - last_change < min_frames:
            continue

        # (a) Direct pitch jump from last change
        pitch_jump = abs(midi_smooth[i] - midi_smooth[last_change]) >= threshold_st
        # (b) Voice gap -- silence break between consecutive voiced frames
        voice_gap = (voiced_times[i] - voiced_times[i - 1]) > _VOICE_GAP_SPLIT_S

        if pitch_jump or voice_gap:
            change_points.append(i)
            last_change = i
            seg_min = seg_max = float(midi_smooth[i])
            seg_min_idx = seg_max_idx = i
            continue

        # (c) Range-based: pitch arc exceeded threshold
        if (seg_max - seg_min) >= threshold_st:
            # Determine the turning point -- the extremum reached before
            # the pitch reversed.  If current pitch is near the minimum,
            # the peak was the turning point (rise-then-fall), and vice
            # versa.
            if abs(midi_smooth[i] - seg_min) < abs(midi_smooth[i] - seg_max):
                turning_idx = seg_max_idx   # rose then fell -> split at peak
            else:
                turning_idx = seg_min_idx   # fell then rose -> split at valley

            if (turning_idx > last_change
                    and turning_idx - last_change >= min_frames):
                change_points.append(turning_idx)
                last_change = turning_idx
                # Re-compute range from the new reference onward
                seg_min = seg_max = float(midi_smooth[last_change])
                seg_min_idx = seg_max_idx = last_change
                for j in range(last_change + 1, i + 1):
                    if midi_smooth[j] < seg_min:
                        seg_min = float(midi_smooth[j])
                        seg_min_idx = j
                    if midi_smooth[j] > seg_max:
                        seg_max = float(midi_smooth[j])
                        seg_max_idx = j

    if len(change_points) <= 1:
        # No significant pitch changes or gaps -- single note
        note = _compute_note_for_word(start, end, pitched_data, allowed_notes)
        return [MidiSegment(note, start, end, word)]

    # Create segments at change points
    segments: list[MidiSegment] = []
    for j, cp in enumerate(change_points):
        seg_start = voiced_times[cp]
        seg_end = voiced_times[change_points[j + 1] - 1] if j + 1 < len(change_points) else end

        # Ensure minimum duration
        if (seg_end - seg_start) < min_note_ms / 1000.0:
            continue

        note = _compute_note_for_word(seg_start, seg_end, pitched_data, allowed_notes)
        text = word if j == 0 else "~ "
        segments.append(MidiSegment(note, seg_start, seg_end, text))

    if not segments:
        note = _compute_note_for_word(start, end, pitched_data, allowed_notes)
        return [MidiSegment(note, start, end, word)]

    # Merge adjacent segments with the same pitch.  Range-based detection
    # may split a note at a pitch arc peak/valley, but if both halves
    # resolve to the same median note the split is useless and only
    # hurts scoring through unnecessary note boundaries.
    merged: list[MidiSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.note == prev.note:
            # Same pitch -- merge by extending the previous segment
            merged[-1] = MidiSegment(prev.note, prev.start, seg.end, prev.word)
        else:
            merged.append(seg)
    segments = merged

    if len(segments) <= 1:
        # All segments had the same pitch after merging -- single note
        note = segments[0].note if segments else _compute_note_for_word(
            start, end, pitched_data, allowed_notes,
        )
        return [MidiSegment(note, start, end, word)]

    # Ensure first segment starts at word start and last ends at word end
    segments[0] = MidiSegment(segments[0].note, start, segments[0].end, segments[0].word)
    segments[-1] = MidiSegment(segments[-1].note, segments[-1].start, end, segments[-1].word)

    return segments


# ---------------------------------------------------------------------------
# Note name -> MIDI helper
# ---------------------------------------------------------------------------

_NOTE_SEMITONES = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
}
# Match note names with ASCII '#' or Unicode sharp (U+266F, used by librosa)
_NOTE_RE = re.compile(r"^([A-G][#\u266f]?)(-?\d+)$")


def _note_name_to_midi(note: str) -> float | None:
    """Convert a note name like ``"A4"``, ``"D#3"``, or ``"D\u266f3"`` to MIDI.

    Accepts both ASCII ``#`` and Unicode sharp (U+266F) for sharps, as
    librosa's ``hz_to_note`` returns the Unicode variant.

    Returns ``None`` if the note name cannot be parsed.
    """
    m = _NOTE_RE.match(note)
    if not m:
        return None
    name, octave = m.groups()
    # Normalize Unicode sharp to ASCII for lookup
    name = name.replace("\u266f", "#")
    semitone = _NOTE_SEMITONES.get(name)
    if semitone is None:
        return None
    return float((int(octave) + 1) * 12 + semitone)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_midi_segments_from_reference_lyrics(
    synced_lyrics: str,
    audio_path: str,
    language: str,
    pitched_data: PitchedData,
    device: str = "cpu",
    allowed_notes: Optional[set[str]] = None,
    melisma_split: bool = True,
    threshold_st: float = 2.0,
    min_note_ms: float = 80.0,
    align_model_name: Optional[str] = None,
) -> list[MidiSegment]:
    """Create MidiSegments from LRCLIB synced lyrics + forced alignment + pitch.

    This is the main entry point for the Reference-Lyrics-First pipeline.

    Args:
        synced_lyrics: Raw LRC synced lyrics from LRCLIB.
        audio_path: Path to the separated vocals audio file.
        language: Language code for the alignment model.
        pitched_data: SwiftF0 pitch data.
        device: ``"cpu"`` or ``"cuda"``.
        allowed_notes: Optional key quantization set.
        melisma_split: Whether to split notes at pitch changes within words.
        threshold_st: Semitone threshold for melisma detection.
        min_note_ms: Minimum note duration for melisma splits.
        align_model_name: Optional custom alignment model name.

    Returns:
        List of MidiSegments with lyrics and pitch.
    """
    # Step 1: Parse LRC (use actual audio duration for the last segment)
    audio_duration = librosa.get_duration(path=audio_path)
    segments = parse_lrc_synced_lyrics(synced_lyrics, audio_duration=audio_duration)
    if not segments:
        print(f"{ULTRASINGER_HEAD} No valid LRC segments found -- "
              f"falling back to standard pipeline")
        return []

    # Step 2: Forced alignment -> word-level timing
    words = align_lyrics_to_audio(
        segments, audio_path, language, device, align_model_name,
    )
    if not words:
        print(f"{ULTRASINGER_HEAD} Forced alignment returned no words -- "
              f"falling back to standard pipeline")
        return []

    # Step 2b: Trim word boundaries to actual voiced regions.
    # CTC alignment stretches the last word in each line to fill the gap
    # until the next line -- this assigns seconds of silence to a single
    # word.  Trimming removes leading/trailing silence so that the pitch
    # computation only considers frames where the singer is singing.
    trimmed_count = 0
    for i, w in enumerate(words):
        trimmed = _trim_word_to_voiced(w, pitched_data)
        if trimmed["start"] != w["start"] or trimmed["end"] != w["end"]:
            trimmed_count += 1
        words[i] = trimmed

    if trimmed_count:
        print(
            f"{ULTRASINGER_HEAD} Trimmed {blue_highlighted(str(trimmed_count))} "
            f"words to voiced regions"
        )

    # Step 2c: Split long notes at silence gaps.
    # Within long words (>1.5s), the singer may pause (breath marks,
    # phrase boundaries).  Splitting at these gaps gives each sub-note
    # its own median pitch -- dramatically improving accuracy for
    # melismatic passages.
    expanded_words: list[dict] = []
    gap_splits = 0
    for w in words:
        sub_words = _split_word_at_silence_gaps(w, pitched_data)
        if len(sub_words) > 1:
            gap_splits += 1
            # Propagate backing and line_end flags to sub-words:
            # - backing applies to all sub-words of a backing-vocal word
            # - line_end only applies to the LAST sub-word
            for j, sw in enumerate(sub_words):
                sw["backing"] = w.get("backing", False)
                sw["line_end"] = w.get("line_end", False) and j == len(sub_words) - 1
        expanded_words.extend(sub_words)

    if gap_splits:
        print(
            f"{ULTRASINGER_HEAD} Split {blue_highlighted(str(gap_splits))} "
            f"long notes at silence gaps "
            f"({len(expanded_words) - len(words)} new sub-notes)"
        )
    words = expanded_words

    # Step 3: Assign pitch to each word (with optional melisma splitting)
    midi_segments: list[MidiSegment] = []
    words_with_pitch = 0
    melisma_splits = 0
    freestyle_count = 0

    for i, w in enumerate(words):
        # Append trailing space to mark word boundaries -- the UltraStar
        # writer uses this to detect where linebreaks may be inserted.
        # Melisma continuation notes (~ prefix) must NOT get a space.
        word_text = w["word"]
        is_last = (i == len(words) - 1)
        if not word_text.startswith("~") and not is_last:
            word_text = word_text + " "

        is_backing = w.get("backing", False)
        is_line_end = w.get("line_end", False)

        if melisma_split:
            segs = _split_word_at_pitch_changes(
                word_text, w["start"], w["end"],
                pitched_data, allowed_notes,
                threshold_st, min_note_ms,
            )
            if len(segs) > 1:
                melisma_splits += 1
            # Apply backing vocal -> freestyle and line break flags
            for j, seg in enumerate(segs):
                if is_backing:
                    seg.note_type = "F"
                    freestyle_count += 1
                if is_line_end and j == len(segs) - 1:
                    seg.line_break_after = True
            midi_segments.extend(segs)
        else:
            note = _compute_note_for_word(
                w["start"], w["end"], pitched_data, allowed_notes,
            )
            note_type = "F" if is_backing else ":"
            if is_backing:
                freestyle_count += 1
            seg = MidiSegment(note, w["start"], w["end"], word_text,
                              note_type=note_type, line_break_after=is_line_end)
            midi_segments.append(seg)

        words_with_pitch += 1

    total_notes = len(midi_segments)
    info_parts = []
    if melisma_splits > 0:
        info_parts.append(f"{melisma_splits} melisma splits")
    if freestyle_count > 0:
        info_parts.append(f"{freestyle_count} freestyle/backing")
    line_break_count = sum(1 for s in midi_segments if s.line_break_after)
    if line_break_count > 0:
        info_parts.append(f"{line_break_count} LRC linebreaks")
    info_suffix = f" ({', '.join(info_parts)})" if info_parts else ""
    print(
        f"{ULTRASINGER_HEAD} Reference-first pipeline: "
        f"{blue_highlighted(str(total_notes))} notes from "
        f"{blue_highlighted(str(words_with_pitch))} words"
        + info_suffix
    )

    return midi_segments


def create_midi_segments_from_plain_lyrics(
    plain_lyrics: Optional[str],
    audio_path: str,
    language: str,
    pitched_data: PitchedData,
    device: str = "cpu",
    allowed_notes: Optional[set[str]] = None,
    melisma_split: bool = True,
    threshold_st: float = 2.0,
    min_note_ms: float = 80.0,
    align_model_name: Optional[str] = None,
) -> list[MidiSegment]:
    """Create MidiSegments from plain lyrics (no timestamps) + forced alignment.

    This is a fallback when LRCLIB provides plain_lyrics but no synced_lyrics.
    The plain text is fed directly to WhisperX forced alignment which uses
    CTC to find word positions in the audio.

    Args:
        plain_lyrics: Raw plain text lyrics (line breaks separate lines).
        audio_path: Path to the separated vocals audio file.
        language: Language code for the alignment model.
        pitched_data: SwiftF0 pitch data.
        device: ``"cpu"`` or ``"cuda"``.
        allowed_notes: Optional key quantization set.
        melisma_split: Whether to split notes at pitch changes within words.
        threshold_st: Semitone threshold for melisma detection.
        min_note_ms: Minimum note duration for melisma splits.
        align_model_name: Optional custom alignment model name.

    Returns:
        List of MidiSegments with lyrics and pitch.
    """
    if not plain_lyrics or not plain_lyrics.strip():
        return []

    # Clean up the lyrics: normalize line breaks, strip empty lines
    lines = [line.strip() for line in plain_lyrics.replace("\r\n", "\n").split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        return []

    all_text = " ".join(lines)
    print(
        f"{ULTRASINGER_HEAD} Plain lyrics alignment: "
        f"{blue_highlighted(str(len(all_text.split())))} words to align"
    )

    # Create a single pseudo-segment spanning the full audio
    audio_duration = librosa.get_duration(path=audio_path)
    segments = [{"text": all_text, "start": 0.0, "end": audio_duration}]

    # Use the same alignment function as the synced pipeline
    words = align_lyrics_to_audio(
        segments, audio_path, language, device, align_model_name,
    )
    if not words:
        print(f"{ULTRASINGER_HEAD} Plain lyrics alignment returned no words -- "
              f"falling back to standard pipeline")
        return []

    # Step 3: Assign pitch to each word (same as synced pipeline)
    midi_segments: list[MidiSegment] = []
    words_with_pitch = 0
    melisma_splits_count = 0

    for i, w in enumerate(words):
        word_text = w["word"]
        is_last = (i == len(words) - 1)
        if not word_text.startswith("~") and not is_last:
            word_text = word_text + " "

        if melisma_split:
            segs = _split_word_at_pitch_changes(
                word_text, w["start"], w["end"],
                pitched_data, allowed_notes,
                threshold_st, min_note_ms,
            )
            if len(segs) > 1:
                melisma_splits_count += 1
            midi_segments.extend(segs)
        else:
            note = _compute_note_for_word(
                w["start"], w["end"], pitched_data, allowed_notes,
            )
            midi_segments.append(MidiSegment(note, w["start"], w["end"], word_text))

        words_with_pitch += 1

    total_notes = len(midi_segments)
    print(
        f"{ULTRASINGER_HEAD} Plain lyrics pipeline: "
        f"{blue_highlighted(str(total_notes))} notes from "
        f"{blue_highlighted(str(words_with_pitch))} words"
        + (f" ({melisma_splits_count} melisma splits)" if melisma_splits_count > 0 else "")
    )

    return midi_segments
