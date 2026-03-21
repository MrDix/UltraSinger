"""Lyrics correction using reference lyrics from LRCLIB.

Aligns Whisper transcription output against known-good reference lyrics
using sequence matching. This is more reliable than LLM correction because
it uses verified ground-truth lyrics.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.Speech_Recognition.TranscribedData import TranscribedData


@dataclass
class LyricsLookupResult:
    """Summary of lyrics lookup correction."""
    words_corrected: int = 0
    words_kept: int = 0
    words_total: int = 0
    reference_words: int = 0


def correct_transcription_from_lyrics(
    transcribed_data: list[TranscribedData],
    plain_lyrics: str,
) -> tuple[list[TranscribedData], LyricsLookupResult]:
    """Correct Whisper transcription using reference lyrics.

    Uses ``difflib.SequenceMatcher`` to align transcribed words against
    reference lyrics and replaces misheard words while preserving all
    timing data from Whisper.

    Args:
        transcribed_data: Whisper transcription output (word-level).
        plain_lyrics: Plain-text reference lyrics from LRCLIB.

    Returns:
        Tuple of (possibly modified transcribed_data, result summary).
    """
    result = LyricsLookupResult()

    if not transcribed_data or not plain_lyrics or not plain_lyrics.strip():
        result.words_total = len(transcribed_data) if transcribed_data else 0
        result.words_kept = result.words_total
        return transcribed_data, result

    # Normalize reference lyrics into a flat word list
    ref_words = _normalize_lyrics(plain_lyrics)
    result.reference_words = len(ref_words)

    if not ref_words:
        result.words_total = len(transcribed_data)
        result.words_kept = result.words_total
        return transcribed_data, result

    # Extract words from transcription for matching
    whisper_words = [_normalize_word(td.word) for td in transcribed_data]
    result.words_total = len(whisper_words)

    # Use SequenceMatcher to find aligned blocks
    matcher = difflib.SequenceMatcher(
        None, whisper_words, ref_words, autojunk=False
    )

    corrected = 0
    kept = 0

    for op, w_start, w_end, r_start, r_end in matcher.get_opcodes():
        if op == "equal":
            # Words match — keep as-is
            kept += w_end - w_start
        elif op == "replace":
            # Words differ — replace Whisper words with reference
            w_len = w_end - w_start
            r_len = r_end - r_start

            if w_len == r_len:
                # 1:1 replacement — straightforward
                for i in range(w_len):
                    td = transcribed_data[w_start + i]
                    new_word = ref_words[r_start + i]
                    if _normalize_word(td.word) != new_word:
                        trailing = td.word[len(td.word.rstrip()):]
                        td.word = new_word + trailing
                        corrected += 1
                    else:
                        kept += 1
            elif w_len > r_len:
                # More Whisper words than reference — replace what we can,
                # keep the rest (likely ad-libs or split words)
                for i in range(r_len):
                    td = transcribed_data[w_start + i]
                    new_word = ref_words[r_start + i]
                    if _normalize_word(td.word) != new_word:
                        trailing = td.word[len(td.word.rstrip()):]
                        td.word = new_word + trailing
                        corrected += 1
                    else:
                        kept += 1
                kept += w_len - r_len  # Extra Whisper words kept as-is
            else:
                # Fewer Whisper words than reference — replace all Whisper words
                # using the first N reference words (we can't add new timed entries)
                for i in range(w_len):
                    td = transcribed_data[w_start + i]
                    new_word = ref_words[r_start + i]
                    if _normalize_word(td.word) != new_word:
                        trailing = td.word[len(td.word.rstrip()):]
                        td.word = new_word + trailing
                        corrected += 1
                    else:
                        kept += 1
        elif op == "insert":
            # Reference has words not in Whisper — we can't insert new
            # timed entries, so we skip these
            pass
        elif op == "delete":
            # Whisper has words not in reference — keep them (could be ad-libs)
            kept += w_end - w_start

    result.words_corrected = corrected
    result.words_kept = kept

    if corrected > 0:
        print(f"{ULTRASINGER_HEAD} Lyrics lookup corrected "
              f"{blue_highlighted(str(corrected))} word(s) "
              f"({kept} kept, {result.words_total} total)")
    else:
        print(f"{ULTRASINGER_HEAD} Lyrics lookup found no corrections needed "
              f"({result.words_total} words matched)")

    return transcribed_data, result


def _normalize_lyrics(plain_lyrics: str) -> list[str]:
    """Normalize reference lyrics into a flat list of lowercase words.

    Removes line breaks, extra whitespace, and common punctuation.
    """
    # Replace line breaks with spaces
    text = plain_lyrics.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    # Remove common punctuation but keep apostrophes in contractions
    text = re.sub(r"[^\w\s']", " ", text)
    # Split and normalize
    words = [w.lower().strip("'") for w in text.split() if w.strip()]
    return [w for w in words if w]


def _normalize_word(word: str) -> str:
    """Normalize a single word for comparison."""
    word = word.strip().lower()
    word = re.sub(r"[^\w']", "", word)
    word = word.strip("'")
    return word
