"""Whisper Speech Recognition Module"""
import os
import inspect
import textwrap
from contextlib import contextmanager

# Set environment variable to handle cuDNN loading issues
# CUDA_MODULE_LOADING=LAZY allows PyTorch to continue even if some cuDNN modules are not found
#os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import torch
import whisperx
from enum import Enum
from torch.cuda import OutOfMemoryError

from modules.DeviceDetection.device_detection import release_gpu_memory
from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, gold_highlighted, red_highlighted
from modules.Speech_Recognition.TranscribedData import TranscribedData, from_whisper

#Addition for numbers to words
import re
from decimal import Decimal
from num2words import CONVERTER_CLASSES, num2words

#Addition for numbers to words
# A word (numbers and words joined by "." or ",", like "1,000", count as one), or other characters
_NUMBER_TOKEN_RE = re.compile(r"\w+(?:[.,]\w+)*|\W+")
_DIGITS_RE = re.compile(r"\d+")
# A number with thousands separators, maybe with decimals after the other
# separator: "1,000", "1.000.000", "1,000.50", "1.000,50"
_THOUSANDS_RE = re.compile(r"\d{1,3}([.,])\d{3}(?:\1\d{3})*(?:(?!\1)[.,]\d+)?")
# An English ordinal number: "1st", "22nd", "3rd", "4th"
_EN_ORDINAL_RE = re.compile(r"(\d+)(?:st|nd|rd|th)", re.IGNORECASE)


MEMORY_ERROR_MESSAGE = f"{ULTRASINGER_HEAD} {blue_highlighted('whisper')} ran out of GPU memory; reduce --whisper_batch_size or force usage of cpu with --force_cpu"


from modules.Speech_Recognition.language_constants import (
    CORE_LANGUAGES,
    LANG_CONFIDENCE_THRESHOLD as _LANG_CONFIDENCE_THRESHOLD,
)


@contextmanager
def whisperx_model_loading():
    """Load WhisperX's own models (VAD, alignment) as before PyTorch 2.6.

    PyTorch 2.6 changed the default of torch.load to weights_only=True, and
    older WhisperX/pyannote versions load their pickled model files without
    passing weights_only. Inside this block a torch.load without it unpickles
    as before; a caller that passes weights_only keeps its choice. torch.load
    is only replaced for the duration of the block, so other model files (a
    downloaded segmentation model, for example) are never unpickled.
    """
    original_load = torch.load

    def load(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original_load


def detect_language_from_audio(audio_path: str, device: str = "cpu", compute_type: str = None) -> str:
    """Detect the language from audio (see :func:`detect_language_with_confidence`)."""
    return detect_language_with_confidence(audio_path, device, compute_type)[0]


def detect_language_with_confidence(audio_path: str, device: str = "cpu",
                                    compute_type: str = None) -> tuple[str, float]:
    """Detect language from audio using Whisper tiny model; returns ``(language, probability)``.

    Much faster than full transcription (~2-3s vs ~2min) because it only
    loads the tiny model and processes a ~30s window of audio.

    Songs with a long instrumental intro (e.g. "unplugged"/acoustic
    versions) can have no singing at all in the first 30s. Without VAD
    filtering, Whisper tiny would guess the language from near-silence /
    instrumental-only audio, producing a low-confidence, effectively
    random result -- observed in practice as e.g. a German song being
    misdetected as English at ~30% confidence. ``vad_filter=True`` makes
    faster-whisper skip non-speech regions first, so detection runs on
    the first actual vocal segment instead of whatever happens to be at
    the very start of the file.

    When confidence is below the threshold and the detected language is not
    in the well-supported set, falls back to English — the most common
    language in music — to avoid wrong transcription/alignment.

    Args:
        audio_path: Path to the audio file.
        device: ``"cpu"`` or ``"cuda"``.
        compute_type: As for the transcription (``--whisper_compute_type``);
            None picks the same default (float16 on CUDA, int8 on the CPU).
            When the device does not support it (float16 on older GPUs), the
            fastest type it supports is used instead.

    Returns:
        ISO language code (e.g. ``"en"``, ``"de"``) and Whisper's probability
        for it (the probability of the detected language, also when English
        is substituted as fallback).
    """
    from faster_whisper import WhisperModel as FasterWhisperModel

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"
    try:
        model = FasterWhisperModel("tiny", device=device, compute_type=compute_type)
    except ValueError as error:
        print(
            f"{ULTRASINGER_HEAD} {gold_highlighted('Note:')} compute type {compute_type} is not "
            f"supported here ({error}) -- detecting the language with the fastest supported type"
        )
        model = FasterWhisperModel("tiny", device=device, compute_type="auto")
    audio = whisperx.load_audio(audio_path)

    language, probability, _ = model.detect_language(audio, vad_filter=True)

    print(
        f"{ULTRASINGER_HEAD} Language detected: "
        f"{blue_highlighted(language)} ({probability:.0%} confidence)"
    )

    if probability < _LANG_CONFIDENCE_THRESHOLD:
        if language not in CORE_LANGUAGES:
            print(
                f"{ULTRASINGER_HEAD} {gold_highlighted('Low confidence for non-core language')} "
                f"— falling back to {blue_highlighted('en')}"
            )
            language = "en"
        else:
            print(
                f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} Low confidence "
                f"({probability:.0%}) for detected language {blue_highlighted(language)}. "
                f"If results are poor, set the language explicitly with --language."
            )

    return language, float(probability)


class WhisperModel(Enum):
    """Whisper model"""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    # English-only models (whisperx then transcribes in English)
    TINY_EN = "tiny.en"
    BASE_EN = "base.en"
    SMALL_EN = "small.en"
    MEDIUM_EN = "medium.en"

#Addition for numbers to words (Using previous code from louispan in PR#135)
def number_to_words(line,language='en'):
    """``line`` with its numbers written as words of ``language``.

    Only numbers are converted: digits, also in groups of a thousand ("1,000",
    with decimals "1,000.50") and, in English, ordinals ("21st"). Other words with digits ("80s", "mp3")
    stay as they are, and so do all numbers when num2words does not know the
    language.
    """
    # https://github.com/m-bain/whisperX
    # Transcript words which do not contain characters in the alignment models dictionary e.g. "2014." or "£13.60" cannot be aligned and therefore are not given a timing.
    # Therefore, convert numbers to words
    if not _DIGITS_RE.search(line):
        return line
    if not language or (language not in CONVERTER_CLASSES and language[:2] not in CONVERTER_CLASSES):
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} Unknown language for number transcription. Keeping numbers as numeric characters for line: {line}"
        )
        return line
    out_tokens = []
    for token in _NUMBER_TOKEN_RE.findall(line):
        if not _DIGITS_RE.search(token):
            out_tokens.append(token)
        elif thousands := _THOUSANDS_RE.fullmatch(token):
            # without the thousands separators; the other one, if any, marks the decimals
            number = token.replace(thousands.group(1), "").replace(",", ".")
            out_tokens.append(_number_as_words(token, number, language))
        else:
            # Numbers joined by "." or "," otherwise ("1,2,3", "9.30") one by one
            out_tokens.extend(_word_to_words(part, language) for part in re.split(r"([.,])", token))
    return ''.join(out_tokens)


def _word_to_words(word: str, language: str) -> str:
    """A number as words, an English ordinal number as ordinal; other words unchanged."""
    if _DIGITS_RE.fullmatch(word) and (len(word) == 1 or not word.startswith("0")):
        return _number_as_words(word, word, language)
    ordinal = _EN_ORDINAL_RE.fullmatch(word) if language[:2] == "en" else None
    if ordinal:
        return _number_as_words(word, ordinal.group(1), language, "ordinal")
    return word


def _number_as_words(token: str, number: str, language: str, to: str = "cardinal") -> str:
    """``number`` (digits, maybe with a "." before decimals) in words, else ``token``."""
    try:
        return num2words(Decimal(number) if "." in number else int(number), lang=language, to=to)
    except Exception:
        return token

def replace_code_lines(source, start_token, end_token,
                       replacement, escape_tokens=True):
    """Replace the source code between `start_token` and `end_token`
    in `source` with `replacement`. The `start_token` portion is included
    in the replaced code. If `escape_tokens` is True (default),
    escape the tokens to avoid them being treated as a regular expression."""

    if escape_tokens:
        start_token = re.escape(start_token)
        end_token = re.escape(end_token)

    def replace_with_indent(match):
        indent = match.group(1)
        return textwrap.indent(replacement, indent)

    return re.sub(r"^(\s+)({}[\s\S]+?)(?=^\1{})".format(start_token, end_token),
                  replace_with_indent, source, flags=re.MULTILINE)

def transcribe_with_whisper(
    audio_path: str,
    model: WhisperModel,
    device="cpu",
    alignment_model: str = None,
    batch_size: int = 16,
    compute_type: str = None,
    language: str = None,
    keep_numbers: bool = False,
    vad_onset: float = 0.35,
    vad_offset: float = 0.20,
    no_speech_threshold: float = 0.4,
) -> TranscriptionResult:
    """Transcribe with whisper"""
    # Info: Regardless of the audio sampling rate used in the original audio file, whisper resample the audio signal to 16kHz (via ffmpeg). So the standard input from (44.1 or 48 kHz) should work.
    print(
        f"{ULTRASINGER_HEAD} Loading {blue_highlighted('whisper')} with model {blue_highlighted(model.value)} and {red_highlighted(device)} as worker"
    )
    if alignment_model is not None:
        print(f"{ULTRASINGER_HEAD} using alignment model {blue_highlighted(alignment_model)}")

    # Fixme: Why it is not none for cpu?
    #compute_type = "int8"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    loaded_whisper_model = audio = None
    try:
        torch.cuda.empty_cache()

        # VAD options tuned for singing (lower thresholds capture more vocal segments)
        vad_options = {
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
        }
        # ASR options tuned for singing (lower no_speech_threshold prevents
        # Whisper from classifying singing as silence)
        asr_options = {
            "no_speech_threshold": no_speech_threshold,
        }

        print(
            f"{ULTRASINGER_HEAD} VAD thresholds: onset={vad_onset}, offset={vad_offset}; "
            f"no_speech_threshold={no_speech_threshold}"
        )

        with whisperx_model_loading():
            loaded_whisper_model = whisperx.load_model(
                model.value,
                language=language,
                device=device,
                compute_type=compute_type,
                vad_options=vad_options,
                asr_options=asr_options,
            )

        audio = whisperx.load_audio(audio_path)

        # The song's language, unless it is given (or the model knows only
        # English), is found before the transcription runs in it
        transcription_language = language
        if transcription_language is None and not model.value.endswith(".en"):
            transcription_language = sung_language(loaded_whisper_model, audio)

        print(f"{ULTRASINGER_HEAD} Transcribing {audio_path}")

        result = loaded_whisper_model.transcribe(
            audio, batch_size=batch_size, language=transcription_language
        )

        detected_language = result["language"]
        if language is None:
            language = detected_language
            if language not in CORE_LANGUAGES and alignment_model is None:
                print(
                    f"{ULTRASINGER_HEAD} Detected language: "
                    f"{blue_highlighted(language)} "
                    f"— {gold_highlighted('not a core language')}. "
                    f"If results are poor, try --language en"
                )
            else:
                print(
                    f"{ULTRASINGER_HEAD} Detected language: "
                    f"{blue_highlighted(language)}"
                )
        elif language != detected_language:
            print(
                f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} "
                f"Detected language {blue_highlighted(detected_language)} "
                f"differs from specified {blue_highlighted(language)}. "
                f"Using {blue_highlighted(language)} for alignment. "
                f"Consider omitting --language for auto-detection."
            )

        # The alignment does not need the Whisper model: its memory goes to
        # the alignment model
        loaded_whisper_model = None
        release_gpu_memory()
        transcribed_data = align_segments(result["segments"], audio, language, device, alignment_model,
                                          keep_numbers)

        return TranscriptionResult(transcribed_data, detected_language)
    except ValueError as value_error:
        if (
            "Requested float16 compute type, but the target device or backend do not support efficient float16 computation."
            in str(value_error)
        ):
            print(value_error)
            print(
                f"{ULTRASINGER_HEAD} Your GPU does not support efficient float16 computation; run UltraSinger with '--whisper_compute_type int8'"
            )

        raise value_error
    except OutOfMemoryError as oom_exception:
        print(oom_exception)
        print(MEMORY_ERROR_MESSAGE)
        raise oom_exception
    except Exception as exception:
        if "CUDA failed with error out of memory" in str(exception):
            print(exception)
            print(MEMORY_ERROR_MESSAGE)
        raise exception
    finally:
        # Drop the model (and the audio) before leaving. The first call
        # imports libraries, one of which keeps the traceback of a failed
        # optional import - and with it this call's frame and all its
        # variables. The model would then stay on the GPU for the rest of
        # the run (about 4 GB with large-v2), and later GPU steps, like the
        # lead-vocal separation, spilled into shared system memory and became
        # many times slower.
        loaded_whisper_model = audio = None
        release_gpu_memory()


def sung_language(whisper_pipeline, audio) -> str:
    """The language that a loaded WhisperX pipeline hears in ``audio`` (16 kHz).

    WhisperX detects it on the first 30 s of the audio. On a separated vocal
    track an instrumental intro is near silence there, and for that Whisper
    names a rare language: on a benchmark of 100 vocal tracks, Norwegian
    Nynorsk ("nn") for three songs in English. A language outside
    CORE_LANGUAGES is therefore detected again on the first 30 s of singing
    (voice activity detection, as detect_language_with_confidence does), which
    found English for all three and changed no other song. The first
    detection stands when no singing is found.
    """
    language = whisper_pipeline.detect_language(audio)
    if language in CORE_LANGUAGES:
        return language
    try:
        sung, probability, _ = whisper_pipeline.model.detect_language(audio, vad_filter=True)
    except Exception:  # noqa: BLE001 - no singing found: a second opinion only
        return language
    if sung != language:
        print(
            f"{ULTRASINGER_HEAD} Detected language {blue_highlighted(language)} in the first 30 s, "
            f"{blue_highlighted(sung)} ({probability:.0%}) in the first 30 s of singing: "
            f"transcribing in {blue_highlighted(sung)}"
        )
    return sung


def align_segments(segments: list[dict], audio, language: str, device: str,
                   alignment_model: str = None, keep_numbers: bool = False) -> list[TranscribedData]:
    """Word timings for transcribed text segments, by forced alignment.

    ``segments`` are ``{"text", "start", "end"}`` dicts, times in seconds of
    ``audio`` (16 kHz mono); each text is aligned within its time span with
    the wav2vec2 model of ``language``, or ``alignment_model`` when given.
    """
    model_a = None
    try:
        # load alignment model and metadata
        try:
            with whisperx_model_loading():
                model_a, metadata = whisperx.load_align_model(
                    language_code=language, device=device, model_name=alignment_model
                )
        except ValueError as ve:
            # No built-in wav2vec2 model exists for this language and no
            # custom model was given -- degrade gracefully to the English
            # alignment model instead of aborting the whole run. Alignment
            # quality will suffer for this language, but that is strictly
            # better than crashing.
            if alignment_model is None and language != "en":
                print(
                    f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} "
                    f"No wav2vec2 alignment model available for language "
                    f"{blue_highlighted(language)} -- falling back to the "
                    f"{blue_highlighted('en')} alignment model. For better "
                    f"results, provide a matching model with "
                    f"--whisper_align_model [huggingface]."
                )
                with whisperx_model_loading():
                    model_a, metadata = whisperx.load_align_model(
                        language_code="en", device=device, model_name=None
                    )
            else:
                print(
                    f"{red_highlighted(f'{ve}')}"
                    f"\n"
                    f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} Unknown language. "
                    f"Try add it with --align_model [huggingface]."
                )
                raise ve

        #Addition for numbers to words (Using previous code from louispan in PR#135)
        if keep_numbers == False:
            for obj in segments:
                obj["text"] = number_to_words(obj["text"],language)

        # align the transcribed text
        result_aligned = whisperx.align(
            segments,
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        return convert_to_transcribed_data(result_aligned)
    finally:
        # Drop the model before leaving (see transcribe_with_whisper)
        model_a = None
        release_gpu_memory()


def convert_to_transcribed_data(result_aligned):
    transcribed_data = []
    for segment in result_aligned["segments"]:
        for obj in segment["words"]:
            vtd = from_whisper(obj)  # create custom Word object
            vtd.word = vtd.word + " "  # add space to end of word
            if len(obj) < 4:
                #Addition for numbers to words (Using previous code from louispan in PR#135)
                if len(transcribed_data) == 0: # if the first word doesn't have any timing data
                    vtd.start = 0.0
                    vtd.end = 0.1
                    msg = f'Error: There is no timestamp for word: "{obj["word"]}". ' \
                        f'Fixing it by placing it at beginning. At start: {vtd.start} end: {vtd.end}. Fix it manually!'
                else:
                    previous = transcribed_data[-1] if len(transcribed_data) != 0 else TranscribedData()
                    vtd.start = previous.end + 0.1
                    vtd.end = previous.end + 0.2
                    msg = f'Error: There is no timestamp for word: "{obj["word"]}". ' \
                          f'Fixing it by placing it after the previous word: "{previous.word}". At start: {vtd.start} end: {vtd.end}. Fix it manually!'
                print(f"{red_highlighted(msg)}")
            transcribed_data.append(vtd)  # and add it to list
    return transcribed_data
