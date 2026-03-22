"""Reference-Lyrics-First pipeline: align LRCLIB synced lyrics to audio.

Instead of relying on Whisper ASR to transcribe words (which misses ~50%
of sung text), this module uses verified synced lyrics from LRCLIB as the
text source and wav2vec2 forced alignment (via WhisperX) to obtain precise
word-level timing.

Pipeline:
    1. Parse LRC synced lyrics → line segments with timestamps
    2. Feed segments to WhisperX forced alignment → word-level timing
    3. For each word, compute median pitch from SwiftF0 pitched data
    4. Optionally split notes at pitch changes within word boundaries
       (melisma-aware splitting)

This produces MidiSegments where every word from the reference lyrics has
a note with accurate timing and pitch — solving the lyrics coverage and
timing accuracy problems of the Whisper-only pipeline.
"""

from __future__ import annotations

import re
from typing import Optional

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


def parse_lrc_synced_lyrics(synced_lyrics: str) -> list[dict]:
    """Parse LRC format synced lyrics into line segments.

    Args:
        synced_lyrics: Raw LRC text, e.g. ``"[01:23.45] Hello world\\n..."``.

    Returns:
        List of ``{"text": str, "start": float, "end": float}`` dicts,
        sorted by start time.  ``end`` is set to the next line's start
        (or start + 10 s for the last line).
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
        end = lines[i + 1]["time"] if i + 1 < len(lines) else line["time"] + 10.0
        segments.append({
            "text": line["text"],
            "start": line["time"],
            "end": end,
        })

    return segments


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
    import librosa

    print(f"{ULTRASINGER_HEAD} Loading alignment model for reference lyrics")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language, device=device, model_name=align_model_name,
    )

    audio = whisperx.load_audio(audio_path)
    audio_duration = librosa.get_duration(filename=audio_path)

    # Merge all lyrics into a single segment spanning the full audio.
    # This gives CTC alignment maximum flexibility to find words wherever
    # they actually are, regardless of how the LRC timestamps relate to
    # the audio version.  Per-line segments with narrow windows failed
    # catastrophically when song structure differed between releases.
    all_text = " ".join(s["text"] for s in segments)
    merged_segments = [{"text": all_text, "start": 0.0, "end": audio_duration}]

    total_lrc_words = len(all_text.split())
    print(
        f"{ULTRASINGER_HEAD} Aligning {blue_highlighted(str(total_lrc_words))} "
        f"reference lyric words to audio"
    )
    aligned = whisperx.align(
        merged_segments, align_model, align_metadata, audio, device,
        return_char_alignments=False,
    )

    # Collect all words with valid timing
    words: list[dict] = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            if "start" in w and "end" in w and w.get("word", "").strip():
                words.append({
                    "word": w["word"].strip(),
                    "start": float(w["start"]),
                    "end": float(w["end"]),
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
    """Split a single word into multiple notes if pitch changes within it.

    If the pitch stays stable throughout the word, returns a single
    MidiSegment.  If the pitch changes by more than ``threshold_st``
    semitones, creates multiple segments — the first gets the word text,
    subsequent ones get ``"~ "``.

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
    times = np.array(pitched_data.times)
    freqs = np.array(pitched_data.frequencies)
    confs = np.array(pitched_data.confidence)

    # Get frames within word boundary
    mask = (times >= start) & (times <= end)
    word_times = times[mask]
    word_freqs = freqs[mask]
    word_confs = confs[mask]

    # Filter for voiced frames
    voiced_mask = (word_freqs > 50) & (word_confs > 0.3)

    if voiced_mask.sum() < 2:
        # Too few voiced frames — single note
        note = _compute_note_for_word(start, end, pitched_data, allowed_notes)
        return [MidiSegment(note, start, end, word)]

    # Convert frequencies to MIDI for pitch change detection
    voiced_freqs = word_freqs[voiced_mask]
    voiced_times = word_times[voiced_mask]
    midi_values = 69 + 12 * np.log2(voiced_freqs / 440.0)

    # Apply median filter to smooth vibrato
    if len(midi_values) >= 5:
        from scipy.ndimage import median_filter
        midi_smooth = median_filter(midi_values, size=5)
    else:
        midi_smooth = midi_values

    # Find pitch change points
    # SwiftF0 uses 16kHz sample rate with STFT hop=256 → ~16ms per frame (62.5 fps).
    # Derive frame duration from actual data when possible, otherwise use 16ms default.
    if len(voiced_times) >= 2:
        frame_ms = (voiced_times[1] - voiced_times[0]) * 1000.0
    else:
        frame_ms = 16.0
    min_frames = max(1, int(min_note_ms / frame_ms))
    change_points = [0]
    last_change = 0

    for i in range(1, len(midi_smooth)):
        if abs(midi_smooth[i] - midi_smooth[last_change]) >= threshold_st:
            if i - last_change >= min_frames:
                change_points.append(i)
                last_change = i

    if len(change_points) <= 1:
        # No significant pitch changes — single note
        note = _compute_note_for_word(start, end, pitched_data, allowed_notes)
        return [MidiSegment(note, start, end, word)]

    # Create segments at pitch change points
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

    # Ensure first segment starts at word start and last ends at word end
    segments[0] = MidiSegment(segments[0].note, start, segments[0].end, segments[0].word)
    segments[-1] = MidiSegment(segments[-1].note, segments[-1].start, end, segments[-1].word)

    return segments


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
    # Step 1: Parse LRC
    segments = parse_lrc_synced_lyrics(synced_lyrics)
    if not segments:
        print(f"{ULTRASINGER_HEAD} No valid LRC segments found — "
              f"falling back to standard pipeline")
        return []

    # Step 2: Forced alignment → word-level timing
    words = align_lyrics_to_audio(
        segments, audio_path, language, device, align_model_name,
    )
    if not words:
        print(f"{ULTRASINGER_HEAD} Forced alignment returned no words — "
              f"falling back to standard pipeline")
        return []

    # Step 3: Assign pitch to each word (with optional melisma splitting)
    midi_segments: list[MidiSegment] = []
    words_with_pitch = 0
    melisma_splits = 0

    for w in words:
        if melisma_split:
            segs = _split_word_at_pitch_changes(
                w["word"], w["start"], w["end"],
                pitched_data, allowed_notes,
                threshold_st, min_note_ms,
            )
            if len(segs) > 1:
                melisma_splits += 1
            midi_segments.extend(segs)
        else:
            note = _compute_note_for_word(
                w["start"], w["end"], pitched_data, allowed_notes,
            )
            midi_segments.append(MidiSegment(note, w["start"], w["end"], w["word"]))

        words_with_pitch += 1

    total_notes = len(midi_segments)
    print(
        f"{ULTRASINGER_HEAD} Reference-first pipeline: "
        f"{blue_highlighted(str(total_notes))} notes from "
        f"{blue_highlighted(str(words_with_pitch))} words"
        + (f" ({melisma_splits} melisma splits)" if melisma_splits > 0 else "")
    )

    return midi_segments
