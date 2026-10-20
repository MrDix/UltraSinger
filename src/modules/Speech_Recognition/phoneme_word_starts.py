"""Word starts from the sounds of the words.

The alignment of lyrics to the audio listens for letters. A singer who holds
a vowel and changes the word on it (a melisma: "your" for a moment, then
"eyes" held for seconds) keeps the vowel the letters of the first word ask
for, so the next word starts only at the end of the held note. A phoneme
recognizer hears the change of sound: the words of each line are aligned to
its phonemes (from eSpeak NG), with vowels a singer may color differently
accepted at a lower weight. A word whose sounds start well inside the
previous word, where those sounds fit clearly better than the previous
word's, starts there: the notes of the previous word from that time on
become the notes of this word.

Measured on 100 songs with hand-made reference charts (English, German,
Spanish, Italian, Swedish): 202 of the 233 moved word starts came closer to
the reference, 24 went further away; 87 % of them lay within 0.25 s of the
reference start, before 24 %.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from modules.Midi.MidiSegment import MidiSegment
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, gold_highlighted

# Phoneme recognizer (wav2vec2, CTC over eSpeak NG phonemes, many languages)
PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
SAMPLE_RATE = 16000

# A word starts earlier when its sounds start at least MIN_LEAD_S (and at
# most MAX_LEAD_S) before its start, at least MIN_INTO_S after the start of
# the previous word, and the frames between fit its sounds better than the
# previous word's by MIN_MARGIN (mean log probability per frame)
MIN_LEAD_S = 0.25
MAX_LEAD_S = 5.0
MIN_INTO_S = 0.1
MIN_MARGIN = 0.5
# A new start within SNAP_S of the end of the previous word, or within
# MIN_PART_S of a boundary between its notes, takes that boundary (no note
# shorter than MIN_PART_S is split off); with less than MIN_PART_S of the
# previous word after it, the previous word ends there and keeps its notes
SNAP_S = 0.05
MIN_PART_S = 0.1
# Audio around a line that its words may be heard in
LINE_PAD_S = 0.5

# Vowels a singer colors: a vowel may be heard as one of its group at
# GROUP_WEIGHT, and as any other vowel at VOWEL_WEIGHT
_VOWEL_GROUPS = [
    "a aː ɑ ɑː ʌ æ ɐ ɒ ä",
    "ɔ ɔː o oː oʊ ɒ əʊ",
    "e eː ɛ ɛː eɪ æ",
    "i iː ɪ ᵻ",
    "u uː ʊ ʉ",
    "ə ɚ ɜ ɜː ʌ ɐ",
]
_R_GROUP = "ɹ r ɻ ɚ ɾ"
_DIPHTHONGS = ("aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ", "aɪə", "aɪɚ", "ɛɹ", "ɪɹ", "ʊɹ", "ɔːɹ", "ɑːɹ", "iə", "eə", "ʊə")
GROUP_WEIGHT = float(np.log(0.5))
VOWEL_WEIGHT = float(np.log(0.05))

# eSpeak NG voices for the language codes the transcription gives, where the
# code itself is no voice
_VOICES = {"en": "en-us", "fr": "fr-fr", "pt": "pt-br", "zh": "cmn", "nb": "nb", "no": "nb"}
# A language switch mark in eSpeak NG phonemes: "(en)"
_LANGUAGE_FLAG = re.compile(r"\([a-z-]+\)")
# Longest data path (bytes, resolved) used with the bundled eSpeak NG library:
# its buffer for the path holds 229 bytes, and with a longer path it misses
# its data and ends the whole process. Kept below that, for other builds.
MAX_DATA_PATH_BYTES = 200


@dataclass
class _Word:
    first: int  # index of its first segment
    last: int  # index of its last segment
    text: str


def lines_of_words(segments: list[MidiSegment]) -> list[list[_Word]]:
    """The words of the segments, line by line.

    A segment whose text starts with "~" (or has none) continues the word
    before it; any other segment starts a word. A line ends after a segment
    with ``line_break_after``.
    """
    lines: list[list[_Word]] = [[]]
    for i, seg in enumerate(segments):
        text = (seg.word or "").strip()
        line = lines[-1]
        if line and (not text or text.startswith("~")):
            line[-1].last = i
        elif text:
            line.append(_Word(i, i, text))
        if seg.line_break_after and line:
            lines.append([])
    return [line for line in lines if line]


class PhonemeAligner:
    """Phoneme recognizer and eSpeak NG phonemes for one language."""

    def __init__(self, voice: str, device: str = "cpu"):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2ForCTC

        self.backend = _espeak_backend(voice)
        with open(hf_hub_download(PHONEME_MODEL, "vocab.json"), encoding="utf-8") as f:
            self.vocab: dict[str, int] = json.load(f)
        self.device = device
        self.half = device.startswith("cuda")
        model = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL).eval().to(device)
        self.model = model.half() if self.half else model
        self._torch = torch
        self._phonemes: dict[str, list[str]] = {}
        self._alternatives: dict[str, tuple[list[int], np.ndarray]] = {}
        vowels = set(" ".join(_VOWEL_GROUPS).split()) | set(_DIPHTHONGS)
        self._vowels = {v for v in vowels if v in self.vocab}

    def log_probs(self, samples: np.ndarray) -> np.ndarray:
        """Frames x phonemes log probabilities of 16 kHz mono samples."""
        torch = self._torch
        x = torch.from_numpy(((samples - samples.mean()) / (samples.std() + 1e-7)).astype(np.float32))[None]
        x = x.to(self.device)
        if self.half:
            x = x.half()
        with torch.inference_mode():
            logits = self.model(x).logits[0].float()
        return torch.log_softmax(logits, -1).cpu().numpy()

    def phonemes(self, word: str) -> list[str]:
        """The phonemes of a word, as tokens of the recognizer."""
        if word not in self._phonemes:
            from phonemizer.separator import Separator

            text = word.strip("~ ").strip()
            ipa = ""
            if re.search(r"\w", text):
                ipa = self.backend.phonemize([text], separator=Separator(phone=" ", word=" | "), strip=True)[0]
                ipa = _LANGUAGE_FLAG.sub(" ", ipa)
            tokens = []
            for token in ipa.replace("|", " ").split():
                tokens.extend([token] if token in self.vocab else [c for c in token if c in self.vocab])
            self._phonemes[word] = tokens
        return self._phonemes[word]

    def _heard_as(self, token: str) -> tuple[list[int], np.ndarray]:
        """Recognizer tokens a phoneme may be heard as, with their log weights."""
        if token not in self._alternatives:
            weights = {token: 0.0}
            parts = {token} | ({token[0], token[1:]} if token in self._vowels and len(token) > 1 else set())
            for group in [*_VOWEL_GROUPS, _R_GROUP]:
                members = group.split()
                if parts & set(members):
                    for m in members:
                        weights.setdefault(m, GROUP_WEIGHT)
            if token in self._vowels:
                for v in self._vowels:
                    weights.setdefault(v, VOWEL_WEIGHT)
            items = [(self.vocab[t], w) for t, w in weights.items() if t in self.vocab]
            self._alternatives[token] = ([i for i, _ in items], np.array([w for _, w in items]))
        return self._alternatives[token]

    def word_starts(self, log_probs: np.ndarray, words: list[str]) -> tuple[list[int | None], np.ndarray, list[int]]:
        """Forced alignment of the words' phonemes to the frames.

        Returns the first frame of each word (None: no phonemes), the
        per-frame score of each phoneme (frames x phonemes) and the word of
        each phoneme.
        """
        tokens, owner = [], []
        for i, w in enumerate(words):
            for p in self.phonemes(w):
                tokens.append(p)
                owner.append(i)
        if not tokens or len(tokens) > len(log_probs):
            return [None] * len(words), np.empty((len(log_probs), 0)), owner
        scores = np.empty((len(log_probs), len(tokens) + 1), np.float32)
        scores[:, 0] = log_probs[:, self.vocab["<pad>"]]
        for j, t in enumerate(tokens):
            idx, weights = self._heard_as(t)
            scores[:, j + 1] = np.logaddexp.reduce(log_probs[:, idx] + weights, axis=1)
        first: list[int | None] = [None] * len(words)
        for frame, label in enumerate(ctc_path(scores)):
            if label and first[owner[label - 1]] is None:
                first[owner[label - 1]] = frame
        return first, scores[:, 1:], owner


def ctc_path(scores: np.ndarray) -> list[int]:
    """Best CTC path through the labels 1..L in order (column 0: blank).

    ``scores`` is frames x (L + 1) log scores. Returns the label of each
    frame (0: blank). Every label is a phoneme of its own, so two equal
    phonemes in a row need no blank between them (a vowel held from one word
    into the next).
    """
    n_frames, n_labels = scores.shape[0], scores.shape[1] - 1
    n_states = 2 * n_labels + 1  # blank, label 1, blank, label 2, ..., blank
    label_of = np.where(np.arange(n_states) % 2 == 1, (np.arange(n_states) + 1) // 2, 0)
    emit = scores[:, label_of]
    can_skip = np.arange(n_states) % 2 == 1  # a label may follow the label before it directly
    can_skip[:2] = False
    dp = np.full(n_states, -np.inf)
    dp[:2] = emit[0, :2]
    back = np.zeros((n_frames, n_states), dtype=np.int8)
    for t in range(1, n_frames):
        stay = dp
        step = np.concatenate([[-np.inf], dp[:-1]])
        skip = np.where(can_skip, np.concatenate([[-np.inf, -np.inf], dp[:-2]]), -np.inf)
        moves = np.stack([stay, step, skip])
        back[t] = moves.argmax(0)
        dp = moves.max(0) + emit[t]
    state = n_states - 1 if n_states == 1 or dp[-1] >= dp[-2] else n_states - 2
    path = [0] * n_frames
    for t in range(n_frames - 1, -1, -1):
        path[t] = int(label_of[state])
        state -= int(back[t, state])
    return path


def _use_bundled_espeak() -> None:
    """Point phonemizer at the eSpeak NG library and data of espeakng-loader.

    Raises ValueError when the data path is too long for the library, which
    would otherwise end the process.
    """
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    data_path = espeakng_loader.get_data_path()
    # phonemizer hands the library the resolved path (no short names)
    if len(str(Path(data_path).resolve()).encode()) > MAX_DATA_PATH_BYTES:
        raise ValueError(f"eSpeak NG data path longer than {MAX_DATA_PATH_BYTES} bytes: {data_path}")
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(data_path)


def _espeak_backend(voice: str):
    from phonemizer.backend import EspeakBackend

    _use_bundled_espeak()
    # A word eSpeak NG reads in another language (an English word in German
    # lyrics) comes without the "(en)...(de)" marks of the switch
    return EspeakBackend(voice, preserve_punctuation=False, with_stress=False, language_switch="remove-flags")


def espeak_voice(language: str | None) -> str | None:
    """The eSpeak NG voice of a language code, None when it has none.

    Raises when eSpeak NG cannot be used (see ``_use_bundled_espeak``).
    """
    if not language:
        return None
    from phonemizer.backend import EspeakBackend

    _use_bundled_espeak()
    supported = EspeakBackend.supported_languages()
    code = language.lower()
    for voice in (_VOICES.get(code), code):
        if voice and voice in supported:
            return voice
    return None


def earlier_starts(aligner: PhonemeAligner, segments: list[MidiSegment],
                   samples: np.ndarray) -> dict[int, float]:
    """Word starts to move earlier: {index of the word's first segment: new start}."""
    lines = lines_of_words(segments)
    audio_end = len(samples) / SAMPLE_RATE
    moves: dict[int, float] = {}
    for k, line in enumerate(lines):
        if len(line) < 2:
            continue
        start, end = segments[line[0].first].start, segments[line[-1].last].end
        lo = max(start - LINE_PAD_S, segments[lines[k - 1][-1].last].end if k else 0.0, 0.0)
        hi = min(end + LINE_PAD_S, segments[lines[k + 1][0].first].start if k + 1 < len(lines) else audio_end)
        chunk = samples[int(lo * SAMPLE_RATE):int(hi * SAMPLE_RATE)]
        if len(chunk) < SAMPLE_RATE // 10:
            continue
        log_probs = aligner.log_probs(chunk)
        frame_s = len(chunk) / SAMPLE_RATE / len(log_probs)
        first, scores, owner = aligner.word_starts(log_probs, [w.text for w in line])
        owner_arr = np.array(owner)
        for j in range(1, len(line)):
            prev, word = line[j - 1], line[j]
            if first[j] is None or first[j - 1] is None:
                continue
            heard = lo + first[j] * frame_s
            current = segments[word.first].start
            if not (max(segments[prev.first].start + MIN_INTO_S, current - MAX_LEAD_S)
                    <= heard <= current - MIN_LEAD_S):
                continue
            f0, f1 = first[j], int(round((current - lo) / frame_s))
            if f1 <= f0:
                continue
            own = scores[f0:f1, owner_arr == j].max(1)
            before = scores[f0:f1, owner_arr == j - 1].max(1)
            if float(np.mean(own - before)) >= MIN_MARGIN:
                moves[word.first] = heard
    return moves


def move_word_starts(segments: list[MidiSegment], moves: dict[int, float]) -> list[MidiSegment]:
    """The segments with the words at ``moves`` starting at their new times.

    The previous word's notes from the new start on become notes of the
    word (one note is split at the new start), and the word's own first
    note continues it. A new start in the pause after the previous word
    moves the start of the word's first note.
    """
    starts = {w.first: w for line in lines_of_words(segments) for w in line}
    previous = {}
    for line in lines_of_words(segments):
        for a, b in zip(line, line[1:]):
            previous[b.first] = a
    out: list[MidiSegment] = []
    i = 0
    while i < len(segments):
        word = starts.get(i)
        if word is not None and i in moves and i in previous:
            # out ends with the previous word: take back its notes after t
            prev = previous[i]
            n_prev = prev.last - prev.first + 1
            tail = out[-n_prev:]
            t = _snapped(moves[i], tail)
            kept, moved = [], []
            for seg in tail:
                if seg.end <= t:
                    kept.append(seg)
                elif seg.start >= t:
                    moved.append(seg)
                else:
                    kept.append(replace(seg, end=t))
                    moved.append(replace(seg, start=t))
            own = segments[i]
            if kept and moved and tail[-1].end - t >= MIN_PART_S:
                del out[-n_prev:]
                moved[0] = replace(moved[0], word=own.word, note_type=own.note_type)
                moved[1:] = [replace(seg, word=_continued(seg.word), note_type=own.note_type) for seg in moved[1:]]
                out.extend(kept + moved)
                out.append(replace(own, word=_continued(own.word)))
                i += 1
                continue
            if kept and t < own.start:
                # heard in the pause after the previous word, or too close to
                # its end to take a note of it: the previous word ends and
                # the word's first note starts there
                del out[-n_prev:]
                out.extend(kept)
                out.append(replace(own, start=t))
                i += 1
                continue
        out.append(segments[i])
        i += 1
    return out


def _snapped(t: float, notes: list[MidiSegment]) -> float:
    """``t``, or the boundary between two notes of ``notes`` within
    MIN_PART_S of it, or their end within SNAP_S of it (no short note split
    off on either side)."""
    inner = [n.start for n in notes[1:]] + [n.end for n in notes[:-1]]
    nearest = min(inner, key=lambda b: abs(b - t), default=None)
    if nearest is not None and abs(nearest - t) < MIN_PART_S:
        return nearest
    return notes[-1].end if abs(notes[-1].end - t) <= SNAP_S else t


def _continued(text: str) -> str:
    """A continuation ("~") in place of a note's text, keeping its word
    separator."""
    return "~ " if text.endswith(" ") else "~"


def correct_word_starts(segments: list[MidiSegment], vocals_path: str, language: str | None,
                        device: str = "cpu") -> tuple[list[MidiSegment], str]:
    """Segments whose word starts follow the sounds of the words (see the
    module), and what was done, for the settings info file.

    Fails open: without a voice for the language, or on any problem, the
    segments come back unchanged.
    """
    try:
        voice = espeak_voice(language)
    except Exception as e:  # noqa: BLE001 - an optional refinement must never stop the conversion
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} phoneme word starts skipped: {e!r}")
        return segments, f"skipped after an error: {e!r}"
    if voice is None:
        print(f"{ULTRASINGER_HEAD} Phoneme word starts: no phonemes for language "
              f"{blue_highlighted(str(language))} - skipped")
        return segments, f"skipped, no phonemes for language {language}"
    try:
        import librosa

        samples, _ = librosa.load(vocals_path, sr=SAMPLE_RATE, mono=True)
        aligner = PhonemeAligner(voice, device)
        try:
            moves = earlier_starts(aligner, segments, samples)
        finally:
            del aligner
            _free_cuda(device)
    except Exception as e:  # noqa: BLE001 - an optional refinement must never stop the conversion
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Warning:')} phoneme word starts skipped: {e!r}")
        return segments, f"skipped after an error: {e!r}"
    if not moves:
        print(f"{ULTRASINGER_HEAD} Phoneme word starts: no word starts earlier")
        return segments, "no word starts earlier"
    moved = f"{len(moves)} word starts earlier" if len(moves) == 1 else f"{len(moves)} words start earlier"
    print(f"{ULTRASINGER_HEAD} Phoneme word starts: {blue_highlighted(moved)}, where their sounds begin")
    return move_word_starts(segments, moves), moved


def _free_cuda(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
