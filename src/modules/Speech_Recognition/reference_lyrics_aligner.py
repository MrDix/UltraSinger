"""Reference-Lyrics-First pipeline: align LRCLIB synced lyrics to audio.

Instead of relying on Whisper ASR to transcribe words (which misses ~50%
of sung text), this module uses verified synced lyrics from LRCLIB as the
text source and wav2vec2 forced alignment (via WhisperX) to obtain precise
word-level timing.

Pipeline:
    1. Parse LRC synced lyrics -> line segments with timestamps
    2. Align the lines with wav2vec2 forced alignment (WhisperX's models and
       path search) and check them line by line against their LRC times
       (see lyrics_alignment) -> word-level timing
    2a. (reserved -- boundary rebalancing disabled)
    2b. Trim word boundaries to actual voiced regions (confidence clipping)
    2c. Split long notes at silence gaps (gap splitting)
    3. For each word, compute median pitch from SwiftF0 pitched data
    4. Optionally split notes at pitch changes within word boundaries
       (melisma-aware splitting)

Plain lyrics (without timestamps) get their line times from the words of a
transcription and then go the same way; without enough of these times, their
lines are aligned on one path over the whole song.

This produces MidiSegments where every word from the reference lyrics has
a note with accurate timing and pitch -- solving the lyrics coverage and
timing accuracy problems of the Whisper-only pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, gold_highlighted
from modules.DeviceDetection.device_detection import release_gpu_memory
from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitched_data_helper import get_frequencies_with_high_confidence
from modules.Midi.midi_creator import (
    confidence_weighted_median_note,
    find_nearest_index,
)
from modules.Audio.key_detector import quantize_note_to_key
from modules.Speech_Recognition.lyrics_alignment import (
    ANCHORED_SHARE,
    RELIABLE_LINES,
    SAMPLE_RATE,
    align_lyrics,
    blank_index,
    emission_in_pieces,
    line_times_from_words,
    lyrics_tokens,
)


# ---------------------------------------------------------------------------
# LRC parsing
# ---------------------------------------------------------------------------

# A bracket at the start of an LRC line: a time ("[01:23.45]"), an ID tag
# ("[ar:Artist]"), a section label ("[Chorus]"), a repeat mark ("(x2)") or a
# backing vocal ("[oh yeah]", "(la)")
_LRC_TAG_RE = re.compile(r"\s*(\[[^\]]*\]|\([^)]*\)|\{[^}]*\})")
_LRC_TIME_RE = re.compile(r"(\d+):(\d+(?:\.\d+)?)")
_LRC_ID_TAG_RE = re.compile(r"[A-Za-z#]+:")
# A repeat mark: "x2", "2x", "3 times", "repeat", "repeat chorus x2" (not
# "twice": "(twice)" may be a sung echo)
_REPEAT = r"(?:[x×]\s*\d+|\d+\s*[x×]|\d+\s*times)"
_REPEAT_RE = re.compile(rf"{_REPEAT}|repeat(?:\s*chorus)?(?:\s*{_REPEAT})?", re.IGNORECASE)
# A repeat mark in brackets anywhere in a line: "Hello world (x2)"
_REPEAT_MARK_RE = re.compile(rf"[\(\[\{{]\s*(?:{_REPEAT_RE.pattern})\s*[\)\]\}}]", re.IGNORECASE)
# The name of a section: "Chorus", "Pre-Chorus", "Pre Chorus", "1. Strophe",
# "2x Refrain", "Guitar Solo" (also the words for verse and chorus in German,
# Spanish and Italian lyrics)
_LABEL = (rf"(?:{_REPEAT}\s*:?\s*)?(?:\d+\.?\s*)?(?:(?:pre|post)[-\s]?)?"
          r"(?:chorus|verse|bridge|intro|outro|hook|refrain|interlude|instrumental|coda|strophe|strofa|coro"
          r"|estribillo|ritornello|(?:(?:guitar|drum|drums|bass|piano|keyboard|sax|synth)\s+)?solo)")
# A section label: "Chorus", "Verse 2", "Chorus x2", "Verse 1: Artist"
_LRC_SECTION_RE = re.compile(
    rf"(?:{_LABEL}|(?:(?:pre|post)[-\s]?)?break(?:down)?)(?:\s*\d+)?(?:\s*{_REPEAT})?(?:\s*:.*)?",
    re.IGNORECASE,
)
# A section label without brackets, alone on its line: "Chorus", "Verse 2",
# "Pre-Chorus x2" (not "Break" or "Breakdown": these may be shouted)
_BARE_LABEL_RE = re.compile(rf"{_LABEL}(?:\s*\d+)?(?:\s*{_REPEAT})?", re.IGNORECASE)
# A repeat mark without brackets at the end of a line, before closing
# punctuation or backing vocals: "Hello world x2", "Come on x2!", "Go x2,"
_TRAILING_REPEAT_RE = re.compile(
    r"\s+(?:[x×]\s*\d+|\d+\s*[x×])(?=[.,;:!?…\"'”’]*(?:\s*[\(\[\{][^)\]\}]*[\)\]\}])*$)", re.IGNORECASE)
_BRACKET_CHAR_RE = re.compile(r"[\[\]\(\)\{\}]")
_WORD_CHAR_RE = re.compile(r"\w")
_REPEAT_IN_TEXT_RE = re.compile(_REPEAT, re.IGNORECASE)
# A word time of enhanced LRC: "<01:23.45>"
_LRC_WORD_TIME_RE = re.compile(r"<\d+:\d+(?:\.\d+)?>")


_BRACKET_RE = re.compile(r"[\(\[\{]([^)\]\}]*)[\)\]\}]")


def _lyrics_line(raw_line: str) -> tuple[list[float], str]:
    """The times (s) of the time tags of a line of lyrics and its sung text.

    Brackets at the start of the line are time tags (``"[01:23.45]"``), ID
    tags (``"[ar:Artist]"``), section labels (``"[Chorus]"``,
    ``"[Verse 2]"``, also nested: ``"[[Refrain (2x)]]"``), repeat marks
    (``"(x2)"``) or backing vocals. Repeat marks are not sung wherever they
    are, and neither is a line that is only a section label in other
    brackets (``"(Chorus)"``), with a colon (``"Chorus:"``) or alone
    (``"Chorus"``, ``"Verse 2"``, ``"2x Refrain"``, ``"1. Strophe"``; not
    ``"Break"``, which may be shouted), or only a repeat mark without
    brackets (``"x2"``, ``"Repeat chorus"``), nor ``"x2"`` at the end of a
    line. Word times of enhanced LRC (``"<00:12.50>"``) are left out. The
    other brackets are backing vocals: their words are prefixed with
    ``\\x00`` (see ``_mark_bracketed_words``).
    """
    times: list[float] = []
    sung: list[str] = []  # backing vocals among the brackets at the start
    # Repeat marks in brackets go first, also inside another bracket ("[Refrain (2x)]")
    raw_line = _REPEAT_MARK_RE.sub(" ", raw_line)
    pos = 0
    while m := _LRC_TAG_RE.match(raw_line, pos):
        bracket = m.group(1)
        content = bracket[1:-1].strip()
        time_tag = _LRC_TIME_RE.fullmatch(content) if bracket.startswith("[") else None
        if time_tag:
            times.append(int(time_tag.group(1)) * 60 + float(time_tag.group(2)))
        elif not ((bracket.startswith("[")
                   and (_LRC_ID_TAG_RE.match(content) or _LRC_SECTION_RE.fullmatch(_label_text(content))))
                  or _REPEAT_RE.fullmatch(content)):
            sung.append(bracket)
        pos = m.end()
    text = " ".join(_LRC_WORD_TIME_RE.sub("", " ".join(sung + [raw_line[pos:]])).split())
    # A label with a repeat mark before closing punctuation: "Chorus x2,", "2x Refrain!"
    # (a label alone with punctuation, "Chorus.", may be sung and stays)
    bare = text.rstrip(".,;:!?…")
    if bare != text and _REPEAT_IN_TEXT_RE.search(bare) and _BARE_LABEL_RE.fullmatch(bare):
        text = ""
    # A repeat mark without brackets at the end: "Hello world x2", "Chorus 2x"
    text = _TRAILING_REPEAT_RE.sub("", text)
    # Only a label: "(Chorus)", "{Verse 2}", "((Chorus))", "Chorus:" (but "(Break) it down" is sung)
    inner = text.strip(" ({})")
    in_brackets = text[:1] in ("(", "{") and text[-1:] in (")", "}") and not _BRACKET_CHAR_RE.search(inner)
    if ((in_brackets and _LRC_SECTION_RE.fullmatch(inner))
            or (text.endswith(":") and ":" not in text[:-1] and _LRC_SECTION_RE.fullmatch(text[:-1].rstrip()))):
        text = ""
    # Labels and repeat marks without brackets: a line that is only "Chorus",
    # "Verse 2", "x2" or "Repeat chorus" (also "x2,"); and a line that is
    # only punctuation (the "," left of "(x2),")
    if (_BARE_LABEL_RE.fullmatch(text) or _REPEAT_RE.fullmatch(text.rstrip(".,;:!?…"))
            or not _WORD_CHAR_RE.search(text)):
        text = ""
    return times, _mark_bracketed_words(text)


def _label_text(content: str) -> str:
    """The content of a bracket as it is compared with the section labels:
    without stray brackets of a nested one ("[[Refrain]]" is "Refrain")."""
    return content.strip(" []{}()")


def parse_lrc_synced_lyrics(
    synced_lyrics: str,
    audio_duration: Optional[float] = None,
) -> list[dict]:
    """Parse LRC format synced lyrics into line segments.

    A line with several time tags (``"[00:12.00][01:30.00] Chorus line"``)
    is sung at each of these times. Word times of enhanced LRC
    (``"<00:12.50>"``), ID tags (``"[ar:Artist]"``), section labels among
    the brackets at the start of a line (``"[Chorus]"``, ``"[Verse 2]"``) and
    repeat marks (``"(x2)"``) are not sung and are left out; other brackets
    are backing vocals, wherever they are (see ``_lyrics_line``).

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
        times, text = _lyrics_line(raw_line)
        if text:
            lines.extend({"time": time_s, "text": text} for time_s in times)

    if not lines:
        return []

    # Sort by time: a line with several times comes again later
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

        segments.append({
            "text": line["text"],
            "start": line["time"],
            "end": end,
        })

    return segments


def parse_plain_lyrics(plain_lyrics: str) -> list[str]:
    """The sung lines of plain lyrics (no timestamps), read as the lines of
    LRC lyrics are (see ``parse_lrc_synced_lyrics``): ID tags, section labels
    and repeat marks are left out, and the words in other brackets are
    marked as backing vocals with a ``\\x00`` prefix. Empty lines are skipped.
    """
    texts = []
    for raw_line in plain_lyrics.strip().splitlines():
        _, text = _lyrics_line(raw_line)
        if text:
            texts.append(text)
    return texts


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
    marked = " ".join(result_parts) if result_parts else text
    # Bracket characters left over by a nested pair ("[[oh yeah]]") are not sung
    words = []
    for w in marked.split():
        word = w.replace("\x00", "").strip("[](){}")
        if word:
            words.append(("\x00" if w.startswith("\x00") else "") + word)
    return " ".join(words)


# ---------------------------------------------------------------------------
# Forced alignment
# ---------------------------------------------------------------------------

_WHOLE_SONG = "whole song"
_PIECES = "30-s pieces"


def align_lyrics_to_audio(
    segments: list[dict],
    audio_path: str,
    language: str,
    device: str = "cpu",
    align_model_name: Optional[str] = None,
    line_time: Optional[str] = "LRC time",
) -> list[dict]:
    """Align LRC line segments to audio using wav2vec2 forced alignment.

    Uses WhisperX's alignment models and path search (see
    ``lyrics_alignment``). With several segments that have times (the lines
    of synced lyrics), these times choose between alignments and check the
    lines one by one. Segments without times (plain lyrics), or a single
    segment, are aligned on one path over the whole audio, as WhisperX
    aligns one segment.

    Args:
        segments: LRC segments from :func:`parse_lrc_synced_lyrics`.
        audio_path: Path to the vocals audio file.
        language: ISO language code (e.g. ``"en"``, ``"de"``).
        device: ``"cpu"`` or ``"cuda"``.
        align_model_name: Optional custom alignment model (e.g. from
            ``--whisper_align_model``).  Passed to
            ``whisperx.load_align_model(model_name=...)``.
        line_time: What the ``start`` of the segments is, as the console
            names it: ``"LRC time"`` for synced lyrics, ``"time from the
            transcription"`` for plain lyrics with line times from its words.
            None: the segments have no times.

    Returns:
        List of ``{"word", "start", "end", "backing", "line_end"}`` dicts,
        one per aligned word, in the order of the lyrics: a line moved to its
        LRC time may start up to ``OVERLAP_S`` (0.3 s) before the previous
        line ends. Words of lines at the end that the recording does not have
        are left out.
    """
    import whisperx

    from modules.Speech_Recognition.Whisper import whisperx_model_loading

    # The words of each line without the backing-vocal marker (\x00), which
    # CTC cannot handle, and per word whether it had the marker
    lines: list[list[str]] = []
    line_times: list[float] = []
    backing: list[bool] = []
    for seg in segments:
        seg_words = [w for w in seg["text"].split() if w.replace("\x00", "")]
        if seg_words:
            lines.append([w.replace("\x00", "") for w in seg_words])
            line_times.append(seg["start"])
            backing.extend(w.startswith("\x00") for w in seg_words)
    total_lrc_words = len(backing)
    timed = line_time is not None and len(lines) > 1

    print(f"{ULTRASINGER_HEAD} Loading alignment model for reference lyrics")
    align_model = audio = None
    try:
        try:
            with whisperx_model_loading():
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=language, device=device, model_name=align_model_name,
                )
        except ValueError:
            # No built-in wav2vec2 model exists for this language and no custom
            # model was given -- degrade gracefully to English instead of
            # aborting (the caller falls back to the standard pipeline on any
            # exception, which would otherwise hit the same crash again).
            if align_model_name is None and language != "en":
                print(
                    f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} "
                    f"No wav2vec2 alignment model available for language "
                    f"{blue_highlighted(language)} -- falling back to the "
                    f"{blue_highlighted('en')} alignment model. For better "
                    f"results, provide a matching model with "
                    f"--whisper_align_model [huggingface]."
                )
                with whisperx_model_loading():
                    align_model, align_metadata = whisperx.load_align_model(
                        language_code="en", device=device, model_name=None,
                    )
            else:
                raise

        audio = whisperx.load_audio(audio_path)
        n_samples = len(audio)
        print(
            f"{ULTRASINGER_HEAD} Aligning {blue_highlighted(str(total_lrc_words))} "
            f"reference lyric words to audio"
        )
        dictionary = align_metadata["dictionary"]
        tokens = lyrics_tokens(lines, dictionary, align_metadata["language"])
        emissions = []
        if len(tokens.ids) and n_samples:
            emissions = _alignment_emissions(align_model, align_metadata["type"], audio, device, timed)
    finally:
        # Drop the model (and the audio) before going on: the first call imports
        # libraries, one of which keeps the traceback of a failed optional
        # import - and with it this call's frame and all its variables (see
        # transcribe_with_whisper). The model would stay on the GPU for the
        # rest of the run.
        align_model = audio = None
        release_gpu_memory()

    result = None
    if emissions:
        frame_s = n_samples / SAMPLE_RATE / len(emissions[0][1])
        result = align_lyrics(
            tokens, emissions, blank_index(dictionary), frame_s, line_times if timed else None,
        )

    words: list[dict] = []
    if result is not None:
        if timed:
            _print_line_check(result, line_time)
        last_of_line = {}  # line -> index of its last word
        for w, start, end in result.words:
            last_of_line[int(tokens.word_line[w])] = len(words)
            words.append({
                "word": tokens.words[w],
                "start": start,
                "end": end,
                "backing": backing[tokens.word_source[w]],
                "line_end": False,
            })
        for k in last_of_line.values():
            words[k]["line_end"] = True

    print(
        f"{ULTRASINGER_HEAD} Reference alignment: "
        f"{blue_highlighted(str(len(words)))} of "
        f"{blue_highlighted(str(total_lrc_words))} words aligned"
    )

    return words


def _alignment_emissions(model, model_type: str, audio: np.ndarray, device: str,
                         timed: bool) -> list[tuple[str, np.ndarray]]:
    """The emissions to align with, as (label, log-probabilities): of the whole
    song and, for lines with times, of 30-s pieces; of the pieces alone when
    the whole song does not fit in the GPU memory."""

    def run(samples: np.ndarray) -> np.ndarray:
        return _align_model_log_probs(model, model_type, samples, device)

    pieces = emission_in_pieces(run, audio) if timed else None
    whole = None
    try:
        whole = run(audio)
    except RuntimeError as error:
        if not _out_of_memory(error):
            raise
    if whole is None:
        print(f"{ULTRASINGER_HEAD} Not enough memory to align the whole song at once "
              f"-- aligning it in 30-s pieces")
        if pieces is None:
            release_gpu_memory()
            pieces = emission_in_pieces(run, audio)
        return [(_PIECES, pieces)]
    # The pieces match the whole song frame by frame (both have the model's
    # 320-sample hop); a model with another hop gets the whole song only
    if pieces is None or len(pieces) != len(whole):
        return [(_WHOLE_SONG, whole)]
    return [(_WHOLE_SONG, whole), (_PIECES, pieces)]


def _align_model_log_probs(model, model_type: str, samples: np.ndarray, device: str) -> np.ndarray:
    """Log-probabilities (frames x vocabulary) of the alignment model for
    16-kHz samples, computed as whisperx.align does it (inputs shorter than
    400 samples are padded)."""
    import torch

    waveform = torch.from_numpy(samples).unsqueeze(0)
    lengths = None
    if waveform.shape[-1] < 400:
        lengths = torch.as_tensor([waveform.shape[-1]]).to(device)
        waveform = torch.nn.functional.pad(waveform, (0, 400 - waveform.shape[-1]))
    with torch.inference_mode():
        if model_type == "torchaudio":
            emissions, _ = model(waveform.to(device), lengths=lengths)
        elif model_type == "huggingface":
            emissions = model(waveform.to(device)).logits
        else:
            raise NotImplementedError(f"Align model of type {model_type} not supported.")
        emissions = torch.log_softmax(emissions, dim=-1)
    return emissions[0].float().cpu().numpy()


def _out_of_memory(error: RuntimeError) -> bool:
    """Whether the model ran out of memory: on the GPU, or on the CPU, whose
    allocator reports "not enough memory"."""
    import torch

    message = str(error)
    return (isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in message
            or "not enough memory" in message)


def _print_line_check(result, line_time: str) -> None:
    """Which alignment of the lines with times was used, and what the line check changed."""
    end = "capped end" if result.capped else "free end"
    print(
        f"{ULTRASINGER_HEAD} Lyrics alignment: {blue_highlighted(str(result.fit))} of "
        f"{blue_highlighted(str(result.lines))} lines at their {line_time} ({result.emission}, {end})"
    )
    changes = []
    if result.moved:
        changes.append(f"moved {_count_lines(result.moved)} to their {line_time}")
    if result.dropped:
        changes.append(
            f"left out {_count_lines(len(result.dropped))} at the end that the recording does not have")
    if changes:
        text = "; ".join(changes)
        print(f"{ULTRASINGER_HEAD} {text[0].upper()}{text[1:]}")


def _count_lines(n: int) -> str:
    return f"{blue_highlighted(str(n))} {'line' if n == 1 else 'lines'}"


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

    # Preserve ALL other keys (backing, line_end, ...): CTC stretches
    # precisely the LAST word of each LRC line into the following pause,
    # so almost every line-end word lands here - rebuilding the dict from
    # scratch used to silently drop the line_end flag, which killed nearly
    # all LRCLIB linebreaks in the output file.
    return {**word, "start": new_start, "end": new_end}


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
        # The first segment that is kept carries the word: a first part that is
        # too short was skipped above and must not take the word with it.
        text = word if not segments else "~ "
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

    return _notes_from_words(words, pitched_data, allowed_notes, melisma_split, threshold_st, min_note_ms,
                             "Reference-first pipeline", "LRC linebreaks")


def _notes_from_words(
    words: list[dict],
    pitched_data: PitchedData,
    allowed_notes: Optional[set[str]],
    melisma_split: bool,
    threshold_st: float,
    min_note_ms: float,
    pipeline: str,
    line_breaks: str,
) -> list[MidiSegment]:
    """MidiSegments of the words from :func:`align_lyrics_to_audio` (steps 2b
    to 4): words in brackets become freestyle notes, a line break follows the
    last word of each line. ``pipeline`` names the result in the console, and
    ``line_breaks`` where the line breaks come from."""
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
        info_parts.append(f"{line_break_count} {line_breaks}")
    info_suffix = f" ({', '.join(info_parts)})" if info_parts else ""
    print(
        f"{ULTRASINGER_HEAD} {pipeline}: "
        f"{blue_highlighted(str(total_notes))} notes from "
        f"{blue_highlighted(str(words_with_pitch))} words"
        + info_suffix
    )

    return midi_segments


@dataclass
class PlainLyricsPlacement:
    """How the lines of plain lyrics were placed in time."""
    lines: int = 0              # lines with words
    anchored: int = 0           # of these, the lines a transcribed word gave a time
    with_words: bool = False    # whether there were words of a transcription
    line_by_line: bool = False  # aligned with these line times, as synced lyrics are
    words_from: str = "transcription"  # what transcribed the words

    def __str__(self) -> str:
        if self.line_by_line:
            return (f"line by line ({self.anchored} of {self.lines} lines anchored by the "
                    f"{self.words_from}'s words)")
        if not self.with_words:
            return f"as one block (no {self.words_from} words)"
        return (f"as one block ({self.anchored} of {self.lines} lines anchored by the {self.words_from}'s words, "
                f"too few)")


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
    transcription_words: Optional[list[tuple[str, float, float]]] = None,
    words_from: str = "transcription",
) -> tuple[list[MidiSegment], PlainLyricsPlacement]:
    """Create MidiSegments from plain lyrics (no timestamps) + forced alignment.

    This is a fallback when LRCLIB provides plain_lyrics but no synced_lyrics.
    Each line of the lyrics, read as by :func:`parse_plain_lyrics`, becomes a
    line of the chart. The words of a transcription give the lines times
    (``line_times_from_words``). When at least ``ANCHORED_SHARE`` of the
    lines, and at least ``RELIABLE_LINES``, got one, the lines are aligned
    with these times as synced lyrics are; otherwise all of them on one path
    over the whole song, as WhisperX aligns one segment.

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
        transcription_words: The words of a transcription of the song as
            ``(text, start s, end s)``, where the transcriber placed them.
        words_from: What transcribed them, as the report names it
            (``"remote transcription"`` for remote speech-to-text).

    Returns:
        The MidiSegments with lyrics and pitch (none when the lyrics could
        not be aligned), and how the lines were placed.
    """
    placement = PlainLyricsPlacement(with_words=bool(transcription_words), words_from=words_from)
    texts, lines = [], []
    for text in parse_plain_lyrics(plain_lyrics or ""):
        # the words as align_lyrics_to_audio takes them from the text
        line = [w.replace("\x00", "") for w in text.split() if w.replace("\x00", "")]
        if line:
            texts.append(text)
            lines.append(line)
    if not lines:
        return [], placement

    times, placement.anchored = line_times_from_words(lines, transcription_words or [])
    placement.lines = len(lines)
    placement.line_by_line = (times is not None and placement.anchored >= RELIABLE_LINES
                              and placement.anchored >= ANCHORED_SHARE * len(lines))
    print(
        f"{ULTRASINGER_HEAD} Plain lyrics alignment: "
        f"{blue_highlighted(str(sum(len(line) for line in lines)))} words in "
        f"{blue_highlighted(str(len(lines)))} lines to align"
    )
    print(f"{ULTRASINGER_HEAD} Plain lyrics are placed {placement}")

    # Without line times, the start of the segments is not used
    segments = [{"text": text, "start": times[i] if placement.line_by_line else 0.0}
                for i, text in enumerate(texts)]
    words = align_lyrics_to_audio(
        segments, audio_path, language, device, align_model_name,
        line_time="time from the transcription" if placement.line_by_line else None,
    )
    if not words:
        print(f"{ULTRASINGER_HEAD} Plain lyrics alignment returned no words -- "
              f"falling back to standard pipeline")
        return [], placement

    # The end of the lyrics is no line break: lyrics in one line (such as the
    # text of remote speech-to-text) leave the line breaks to the pauses
    words[-1]["line_end"] = False
    return _notes_from_words(words, pitched_data, allowed_notes, melisma_split, threshold_st, min_note_ms,
                             "Plain lyrics pipeline", "lyrics linebreaks"), placement
