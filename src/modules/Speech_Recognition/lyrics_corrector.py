"""Lyrics correction using reference lyrics from LRCLIB.

Aligns Whisper transcription output against known-good reference lyrics
using sequence matching. This is more reliable than LLM correction because
it uses verified ground-truth lyrics.

Additionally transfers metadata from LRCLIB lyrics:
- Parenthesized text ``(backing vocals)`` or ``[ad-libs]`` → freestyle notes
- Line breaks from the plain lyrics → UltraStar linebreaks
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.Speech_Recognition.TranscribedData import TranscribedData


@dataclass
class _RefWord:
    """A single reference word with metadata from LRCLIB lyrics."""
    normalized: str  # Lowercase, stripped of punctuation (for matching)
    original: str  # Original word text (for replacement)
    is_freestyle: bool  # Inside parentheses/brackets
    line_break_after: bool  # A line break follows this word


@dataclass
class LyricsLookupResult:
    """Summary of lyrics lookup correction."""
    words_corrected: int = 0
    words_kept: int = 0
    words_total: int = 0
    reference_words: int = 0
    freestyle_words: int = 0
    linebreaks_applied: int = 0


def correct_transcription_from_lyrics(
    transcribed_data: list[TranscribedData],
    plain_lyrics: str,
) -> tuple[list[TranscribedData], LyricsLookupResult]:
    """Correct Whisper transcription using reference lyrics.

    Uses ``difflib.SequenceMatcher`` to align transcribed words against
    reference lyrics and replaces misheard words while preserving all
    timing data from Whisper.

    Also transfers freestyle (parenthesized) and line-break metadata
    from the reference lyrics to matched transcription entries.

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

    # Parse reference lyrics preserving bracket/linebreak metadata
    ref_words = _parse_reference_lyrics(plain_lyrics)
    result.reference_words = len(ref_words)

    if not ref_words:
        result.words_total = len(transcribed_data)
        result.words_kept = result.words_total
        return transcribed_data, result

    # Extract normalized words for matching
    ref_normalized = [rw.normalized for rw in ref_words]
    whisper_words = [_normalize_word(td.word) for td in transcribed_data]
    result.words_total = len(whisper_words)

    # Use SequenceMatcher to find aligned blocks
    matcher = difflib.SequenceMatcher(
        None, whisper_words, ref_normalized, autojunk=False
    )

    corrected = 0
    kept = 0

    for op, w_start, w_end, r_start, r_end in matcher.get_opcodes():
        if op == "equal":
            # Words match — keep text as-is, transfer metadata
            for i in range(w_end - w_start):
                _apply_ref_metadata(
                    transcribed_data[w_start + i], ref_words[r_start + i]
                )
            kept += w_end - w_start
        elif op == "replace":
            # Words differ — replace Whisper words with reference
            w_len = w_end - w_start
            r_len = r_end - r_start
            replace_count = min(w_len, r_len)

            for i in range(replace_count):
                td = transcribed_data[w_start + i]
                rw = ref_words[r_start + i]
                if _normalize_word(td.word) != rw.normalized:
                    trailing = td.word[len(td.word.rstrip()):]
                    td.word = rw.original + trailing
                    corrected += 1
                else:
                    kept += 1
                _apply_ref_metadata(td, rw)

            if w_len > r_len:
                # Extra Whisper words kept as-is
                kept += w_len - r_len
            # If r_len > w_len, extra reference words are dropped
            # (we can't insert new timed entries)
        elif op == "insert":
            # Reference has words not in Whisper — we can't insert new
            # timed entries, so we skip these
            pass
        elif op == "delete":
            # Whisper has words not in reference — keep them (could be ad-libs)
            kept += w_end - w_start

    result.words_corrected = corrected
    result.words_kept = kept
    result.freestyle_words = sum(
        1 for td in transcribed_data if td.is_freestyle
    )
    result.linebreaks_applied = sum(
        1 for td in transcribed_data if td.line_break_after
    )

    parts = []
    if corrected > 0:
        parts.append(f"corrected {blue_highlighted(str(corrected))} word(s)")
    if result.freestyle_words > 0:
        parts.append(
            f"{blue_highlighted(str(result.freestyle_words))} freestyle"
        )
    if result.linebreaks_applied > 0:
        parts.append(
            f"{blue_highlighted(str(result.linebreaks_applied))} linebreaks"
        )

    if parts:
        detail = ", ".join(parts)
        print(
            f"{ULTRASINGER_HEAD} Lyrics lookup: {detail} "
            f"({kept} kept, {result.words_total} total)"
        )
    else:
        print(
            f"{ULTRASINGER_HEAD} Lyrics lookup found no corrections needed "
            f"({result.words_total} words matched)"
        )

    return transcribed_data, result


def _apply_ref_metadata(td: TranscribedData, rw: _RefWord) -> None:
    """Transfer freestyle and line-break flags from a reference word."""
    if rw.is_freestyle:
        td.is_freestyle = True
    if rw.line_break_after:
        td.line_break_after = True


def _parse_reference_lyrics(plain_lyrics: str) -> list[_RefWord]:
    """Parse LRCLIB plain lyrics into _RefWord list.

    Preserves:
    - Parenthesized regions ``(...)`` and bracketed regions ``[...]`` as
      freestyle (backing vocals, ad-libs).
    - Line breaks between words for UltraStar linebreak placement.
    """
    ref_words: list[_RefWord] = []

    lines = plain_lyrics.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            # Empty line = paragraph break.  Mark the last word (if any)
            # so we get a linebreak in the output.
            if ref_words:
                ref_words[-1].line_break_after = True
            continue

        # Tokenize line into (word_text, is_parenthesized) pairs
        tokens = _tokenize_with_parens(line)

        for word_text, is_paren in tokens:
            normalized = _normalize_word(word_text)
            if not normalized:
                continue
            ref_words.append(
                _RefWord(
                    normalized=normalized,
                    original=word_text,
                    is_freestyle=is_paren,
                    line_break_after=False,
                )
            )

        # Mark end of non-empty line as a linebreak (except last line)
        if ref_words and line_idx < len(lines) - 1:
            ref_words[-1].line_break_after = True

    # Remove trailing linebreak on very last word
    if ref_words and ref_words[-1].line_break_after:
        ref_words[-1].line_break_after = False

    return ref_words


def _tokenize_with_parens(line: str) -> list[tuple[str, bool]]:
    """Split a line into ``(word, is_parenthesized)`` tuples.

    Handles ``(multi word)`` and ``[multi word]`` regions.  Words outside
    any bracket are marked ``is_parenthesized=False``.

    Examples::

        >>> _tokenize_with_parens("hello (oh yeah) world")
        [("hello", False), ("oh", True), ("yeah", True), ("world", False)]
        >>> _tokenize_with_parens("[ad-lib] test")
        [("ad-lib", True), ("test", False)]
    """
    tokens: list[tuple[str, bool]] = []

    # Split into segments: text outside brackets and text inside brackets
    # Pattern matches (content) or [content] as groups
    parts = re.split(r"(\([^)]*\)|\[[^\]]*\])", line)

    for part in parts:
        if not part:
            continue

        # Check if this part is a bracketed/parenthesized region
        if (part.startswith("(") and part.endswith(")")) or (
            part.startswith("[") and part.endswith("]")
        ):
            # Strip the brackets and split into words
            inner = part[1:-1].strip()
            for word in inner.split():
                if word.strip():
                    tokens.append((word.strip(), True))
        else:
            # Regular text — split into words
            for word in part.split():
                if word.strip():
                    tokens.append((word.strip(), False))

    return tokens


def _normalize_word(word: str) -> str:
    """Normalize a single word for comparison.

    Keeps letters, digits, and apostrophes. Removes hyphens and other
    punctuation so that e.g. "re-enter" becomes "reenter".
    """
    word = word.strip().lower()
    word = re.sub(r"[^\w']", "", word)
    word = word.strip("'")
    return word
