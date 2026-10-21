"""Qwen3-ASR as the transcriber of the sung lyrics.

Qwen3-ASR, an audio language model trained on singing as well, recognizes
sung words better than Whisper: on a benchmark of 100 separated vocal tracks
its 1.7B model made 26.9 % word errors with the song language given (27.7 %
when it found the language itself) against 30.0 % for Whisper large-v2,
mostly because it invents fewer words. It needs a GPU with about 5 GB of
free memory; the smaller variants that fit less memory were all worse than
Whisper there (the 0.6B model 31.9 %, the 1.7B model in 4 bit 33.4 %), so on
smaller GPUs, without a GPU and for the languages Qwen3-ASR does not know,
Whisper transcribes.

The vocal track is transcribed in pieces of about half a minute, cut at quiet
points: given a whole song at once, the model leaves out repeated parts. The
word timings come from the same forced alignment as Whisper's text.
"""

from __future__ import annotations

import importlib.util
import logging
import math
from collections import Counter
from dataclasses import dataclass

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.DeviceDetection.device_detection import free_gpu_memory, release_gpu_memory
from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult

QWEN3_SIZE = "1.7B"
QWEN3_MODEL = f"Qwen/Qwen3-ASR-{QWEN3_SIZE}"
SAMPLE_RATE = 16000
# Length of the pieces the vocal track is transcribed in
PIECE_SECONDS = 30
# Half a minute of singing holds far fewer words; the cap ends a decoding
# that got stuck repeating itself
MAX_PIECE_TOKENS = 384
# Shorter pieces are left out, the model does not take them
MIN_PIECE_SECONDS = 0.5
# Free GPU memory needed, in GiB: the model peaked at 4.4 with two pieces at a
# time and at 5.3 with eight, which it takes from BATCH_FREE_GIB on
MIN_FREE_GIB = 5.0
BATCH_FREE_GIB = 7.0
# The model sometimes gives up on a piece after its first words (a vocalise
# at its start) and leaves the rest of it out. A span of at least
# RETRY_GAP_S without words, where the voice is audible on RETRY_VOICED_SHARE
# of its frames of RETRY_FRAME_S (no more than RETRY_LEVEL_DB below the loud
# parts of the song), is transcribed once more on its own.
RETRY_GAP_S = 8.0
RETRY_VOICED_SHARE = 0.5
RETRY_FRAME_S = 0.05
RETRY_LEVEL_DB = 25.0

TRANSCRIBERS = ("auto", "qwen3", "whisper")

# The languages Qwen3-ASR transcribes, with its names, under the codes the
# rest of the pipeline uses (Whisper's: "tl" for Filipino, "yue" for Cantonese)
QWEN_LANGUAGES = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German", "fr": "French",
    "es": "Spanish", "pt": "Portuguese", "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish", "hi": "Hindi", "ms": "Malay",
    "nl": "Dutch", "sv": "Swedish", "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "tl": "Filipino", "fa": "Persian", "el": "Greek", "hu": "Hungarian", "mk": "Macedonian", "ro": "Romanian",
}
_ISO_OF = {name.lower(): code for code, name in QWEN_LANGUAGES.items()}


class QwenUnavailable(Exception):
    """Qwen3-ASR cannot transcribe this song; the message tells why."""


class QwenFailed(QwenUnavailable):
    """Qwen3-ASR broke off: out of GPU memory, the model could not be downloaded, ..."""


@dataclass
class TranscriberChoice:
    """The transcriber a run uses, and why."""
    name: str           # "qwen3" or "whisper"
    batch_size: int = 1  # pieces at a time for qwen3
    reason: str = ""


def choose_transcriber(requested: str, device: str, language: str | None = None) -> TranscriberChoice:
    """The transcriber for ``requested`` ("auto", "qwen3" or "whisper").

    Qwen3-ASR runs on a CUDA GPU with at least ``MIN_FREE_GIB`` of free
    memory, needs its package, and transcribes only the languages of
    ``QWEN_LANGUAGES`` (``language`` is the one given for the song, if any);
    otherwise Whisper transcribes, also for "qwen3" (the reason tells why).
    """
    if requested == "whisper":
        return TranscriberChoice("whisper", reason="selected")
    if language and language not in QWEN_LANGUAGES:
        return TranscriberChoice("whisper", reason=f"Qwen3-ASR does not transcribe the language {language}")
    if device != "cuda":
        return TranscriberChoice("whisper", reason="Qwen3-ASR needs a GPU")
    if not _qwen_asr_installed():
        return TranscriberChoice("whisper", reason="Qwen3-ASR is not installed")
    try:
        free = free_gpu_memory()
    except RuntimeError as e:
        return TranscriberChoice("whisper", reason=f"the free GPU memory could not be read ({e})")
    if free is None:
        return TranscriberChoice("whisper", reason="Qwen3-ASR needs a GPU")
    free_gib = free / 2 ** 30
    if free_gib < MIN_FREE_GIB:
        # Rounded down, so that 4.96 GB does not read as the 5.0 GB needed
        return TranscriberChoice("whisper", reason=f"only {math.floor(free_gib * 10) / 10:.1f} GB of GPU memory "
                                                   f"free, Qwen3-ASR needs {MIN_FREE_GIB:.1f} GB")
    return TranscriberChoice("qwen3", 8 if free_gib >= BATCH_FREE_GIB else 2,
                             f"{free_gib:.1f} GB of GPU memory free")


def song_language(audio_path: str, language: str | None) -> str | None:
    """The language to give Qwen3-ASR: ``language`` when set, else a confident
    quick detection; None lets the model decide for each piece.

    The quick detection runs on the CPU, so that the GPU memory found free for
    the model stays free.
    """
    if language:
        return language
    from modules.Speech_Recognition.Whisper import detect_language_with_confidence
    from modules.Speech_Recognition.text_language import CONFIDENT_AUDIO_LANGUAGE

    detected, probability = detect_language_with_confidence(audio_path, device="cpu")
    return detected if probability >= CONFIDENT_AUDIO_LANGUAGE else None


def transcribe_with_qwen3(audio_path: str, choice: TranscriberChoice, device: str = "cuda",
                          language: str | None = None, alignment_model: str | None = None,
                          keep_numbers: bool = False) -> TranscriptionResult:
    """Transcribe the vocal track with Qwen3-ASR and align the words to it.

    Raises ``QwenUnavailable`` when the song is in a language Qwen3-ASR does
    not transcribe, and ``QwenFailed`` when the model cannot be loaded or
    breaks off; errors of the alignment are passed on as they are.
    """
    import whisperx

    from modules.Speech_Recognition.Whisper import align_segments

    language = song_language(audio_path, language)
    if language is not None and language not in QWEN_LANGUAGES:
        raise QwenUnavailable(f"Qwen3-ASR does not transcribe the language {language}")
    audio = whisperx.load_audio(audio_path)
    model = None
    try:
        pieces = audio_pieces(audio)
        if not pieces:
            return TranscriptionResult([], language or "en")
        print(f"{ULTRASINGER_HEAD} Loading {blue_highlighted('Qwen3-ASR')} ({choice.reason})")
        model = _load_model(choice)
        print(f"{ULTRASINGER_HEAD} Transcribing {audio_path} in {len(pieces)} pieces")
        results = model.transcribe(audio=[(wav, SAMPLE_RATE) for wav, _ in pieces],
                                   language=QWEN_LANGUAGES[language] if language else None)
    except (RuntimeError, OSError, ImportError) as e:
        # Out of GPU memory, the first download of the model failed, a broken install, ...
        raise QwenFailed(f"Qwen3-ASR failed ({_error_summary(e)})") from None
    finally:
        # Drop the model before going on (see transcribe_with_whisper); after
        # an error, the caller releases what the error's frames still held
        model = None
        release_gpu_memory()

    detected = language or transcribed_language(results)
    print(f"{ULTRASINGER_HEAD} Detected language: {blue_highlighted(detected)}")
    segments = [{"text": r.text.strip(), "start": offset, "end": offset + len(wav) / SAMPLE_RATE}
                for (wav, offset), r in zip(pieces, results) if r.text and r.text.strip()]
    if not segments:
        # nothing heard at all: the same pieces again would give the same
        return TranscriptionResult([], detected)
    transcribed_data = align_segments(segments, audio, detected, device, alignment_model, keep_numbers)
    # the model is loaded once more for these (it was released for the alignment)
    spans = unheard_spans(transcribed_data, audio)
    if spans:
        transcribed_data = _with_spans_heard_again(transcribed_data, spans, audio, choice, detected, device,
                                                   alignment_model, keep_numbers)
    return TranscriptionResult(transcribed_data, detected)


def unheard_spans(words: list, audio) -> list[tuple[float, float]]:
    """The spans (start s, end s) of ``audio`` (16 kHz) of at least
    RETRY_GAP_S without ``words`` (TranscribedData, in time order) where the
    voice is audible (see RETRY_GAP_S)."""
    import numpy as np

    frame = int(RETRY_FRAME_S * SAMPLE_RATE)
    n_frames = len(audio) // frame
    if n_frames == 0:
        return []
    samples = np.asarray(audio[:n_frames * frame], dtype=np.float64).reshape(n_frames, frame)
    rms = np.sqrt((samples ** 2).mean(axis=1))
    loud = float(np.percentile(rms, 95))
    if loud <= 0:
        return []
    voiced = rms >= loud * 10 ** (-RETRY_LEVEL_DB / 20)
    duration = n_frames * RETRY_FRAME_S
    times = [(0.0, 0.0)] + [(w.start, w.end) for w in words if w.start is not None and w.end is not None] \
        + [(duration, duration)]
    spans = []
    for (_, start), (end, _) in zip(times, times[1:]):
        if end - start >= RETRY_GAP_S and voiced[int(start / RETRY_FRAME_S):int(end / RETRY_FRAME_S)].mean() \
                >= RETRY_VOICED_SHARE:
            spans.append((start, end))
    return spans


def _with_spans_heard_again(words: list, spans: list[tuple[float, float]], audio, choice: TranscriberChoice,
                            language: str, device: str, alignment_model: str | None,
                            keep_numbers: bool) -> list:
    """The ``words`` with the words of the ``spans`` transcribed once more on
    their own (see RETRY_GAP_S); the words alone when that fails."""
    from modules.Speech_Recognition.Whisper import align_segments

    pieces = []  # (samples, offset s) of the spans, in pieces as long as the first ones at most
    for start, end in spans:
        first = int(start * SAMPLE_RATE)
        pieces += [(wav, start + offset) for wav, offset in audio_pieces(audio[first:int(end * SAMPLE_RATE)])]
    if not pieces:
        return words
    model = None
    try:
        model = _load_model(choice)
        results = model.transcribe(audio=[(wav, SAMPLE_RATE) for wav, _ in pieces],
                                   language=QWEN_LANGUAGES.get(language))
    except (RuntimeError, OSError, ImportError) as e:
        print(f"{ULTRASINGER_HEAD} Could not transcribe the parts without words once more ({_error_summary(e)})")
        return words
    finally:
        model = None
        release_gpu_memory()
    segments = [{"text": r.text.strip(), "start": offset, "end": offset + len(wav) / SAMPLE_RATE}
                for (wav, offset), r in zip(pieces, results) if r.text and r.text.strip()]
    try:
        found = align_segments(segments, audio, language, device, alignment_model, keep_numbers) if segments else []
    except (RuntimeError, OSError, ImportError, ValueError) as e:
        print(f"{ULTRASINGER_HEAD} Could not align the words of the parts without words ({_error_summary(e)})")
        return words
    # only words within the spans: the alignment may stretch a word beyond its piece
    found = [w for w in found if w.start is not None and any(start <= w.start < end for start, end in spans)]
    print(f"{ULTRASINGER_HEAD} Transcribed {len(spans)} sung {'part' if len(spans) == 1 else 'parts'} without "
          f"words once more: {blue_highlighted(str(len(found)))} words")
    # each found word before the first word that starts after it; the order of the others stays
    out = list(words)
    for w in sorted(found, key=lambda w: w.start):
        at = next((k for k, x in enumerate(out) if x.start is not None and x.start > w.start), len(out))
        out.insert(at, w)
    return out


def audio_pieces(audio) -> list:
    """``(samples, offset in seconds)`` pieces of about ``PIECE_SECONDS``, cut at
    quiet points; pieces shorter than ``MIN_PIECE_SECONDS`` are left out."""
    from qwen_asr.inference.utils import split_audio_into_chunks

    return [(wav, offset) for wav, offset in split_audio_into_chunks(audio, SAMPLE_RATE, max_chunk_sec=PIECE_SECONDS)
            if len(wav) >= MIN_PIECE_SECONDS * SAMPLE_RATE]


def transcribed_language(results) -> str:
    """The ISO code of the language the pieces were transcribed in.

    The model's language tag of a piece is unreliable: it tagged most pieces
    of a clearly Spanish transcription "English". So the words decide where
    they can (see ``words_override_tag``), else the tags, else "en".
    """
    from modules.Speech_Recognition.text_language import detect_text_language, words_override_tag

    text = " ".join(r.text.strip() for r in results if r.text and r.text.strip())
    from_tags = majority_language([r.language for r in results], default=None)
    from_words = detect_text_language(text)
    if not words_override_tag(from_words, from_tags):
        return from_tags or "en"
    tags = f"say {blue_highlighted(from_tags)}" if from_tags else "name no known language"
    print(f"{ULTRASINGER_HEAD} The transcription's language tags {tags}, its words say "
          f"{blue_highlighted(from_words)} — taking {from_words}")
    return from_words


def majority_language(names: list[str | None], default: str | None = "en") -> str | None:
    """The ISO code of the language most pieces were in (``default`` when none is known).

    A piece the model decoded in several chunks has the languages of the
    chunks joined by commas ("Spanish,English"); each of them counts.
    """
    codes = [_ISO_OF[part.strip().lower()] for name in names if name
             for part in name.split(",") if part.strip().lower() in _ISO_OF]
    return Counter(codes).most_common(1)[0][0] if codes else default


def _error_summary(error: BaseException) -> str:
    """``Type: first line of the message`` of an error."""
    lines = str(error).strip().splitlines()
    return f"{type(error).__name__}: {lines[0]}" if lines else type(error).__name__


def _qwen_asr_installed() -> bool:
    """Whether the qwen_asr package is installed; importing it takes seconds,
    so it is only looked up."""
    return importlib.util.find_spec("qwen_asr") is not None


def _load_model(choice: TranscriberChoice):
    import torch
    from qwen_asr import Qwen3ASRModel

    # GPUs before compute capability 8 only emulate bfloat16, much slower
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    # The generation config of the model sets a temperature without sampling,
    # which transformers reports as an invalid flag when it loads the model;
    # the model never samples, so the notice says nothing
    generation_logger = logging.getLogger(_GENERATION_CONFIG_LOGGER)
    generation_logger.addFilter(_without_temperature_notice)
    try:
        return Qwen3ASRModel.from_pretrained(QWEN3_MODEL, dtype=dtype, device_map="cuda:0",
                                             max_inference_batch_size=choice.batch_size,
                                             max_new_tokens=MAX_PIECE_TOKENS)
    finally:
        generation_logger.removeFilter(_without_temperature_notice)


_GENERATION_CONFIG_LOGGER = "transformers.generation.configuration_utils"


def _without_temperature_notice(record: logging.LogRecord) -> bool:
    """False for the notice of transformers that the temperature is no valid generation flag."""
    message = record.getMessage()
    return not ("generation flags are not valid" in message and "['temperature']" in message)
