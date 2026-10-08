"""Lightweight language identification for lyrics text.

Used to reject lyrics that are clearly in another language than the one
being sung (e.g. a lyrics service returning the English version of a song
that is sung in German). Song lyrics are full of function words, so counting
them is reliable enough for this purpose and needs no extra dependency.
Non-Latin scripts are recognised by their characters.
"""

from __future__ import annotations

import re
from collections import Counter

# Frequent function words; words shared by several languages are fine, the
# decision needs a clear margin between the best and the second language.
_FUNCTION_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset("the and you i to a it me my of in is that on your we be for this all love don't i'm what so "
                    "with but can no just like when oh know it's are not will now they got never".split()),
    "de": frozenset("der die das und ich du nicht ist ein eine mit mich dich mir dir sie wir es zu auf in den "
                    "dem im noch nur so wie was mein dein auch kein keine wenn doch ja nein bin bist hab "
                    "habe sein immer alles uns".split()),
    "fr": frozenset("le la les et je tu il elle de des un une est pas que qui dans pour sur mon ma mes ton ta "
                    "tes ce cette avec moi toi nous vous on plus mais oui non c'est j'ai".split()),
    "es": frozenset("el la los las y que de en un una no es por con para mi tu te me se lo como pero yo más "
                    "mas si ya muy porque cuando todo esta este eres soy".split()),
    "pt": frozenset("o os as e que de do da em um uma não nao é eu você voce meu minha com para se mas por "
                    "seu sua nos isso muito quando tudo está sou".split()),
    "it": frozenset("il la le i e che di un una non è per con mi ti si lo gli del della sono come ma io tu "
                    "più piu anche ho hai questo quando tutto sei".split()),
    "nl": frozenset("de het een en ik je jij niet is van dat in op met mijn zijn wat maar voor ook als nog "
                    "wel ze we er dan naar geen".split()),
    "sv": frozenset("och jag du det att en ett är inte på för med mig dig som vi har till kan men så om nu "
                    "min din av vad".split()),
    "da": frozenset("og jeg du det at en et er ikke på for med mig dig som vi har til kan men så om nu min "
                    "din af hvad".split()),
    "no": frozenset("og jeg du det at en et er ikke på for med meg deg som vi har til kan men så om nå min "
                    "din av hva".split()),
    "pl": frozenset("i w nie na się to że z do jak ja ty mi mnie cię jest co ale tak po by za już moja "
                    "mój".split()),
    "tr": frozenset("ve bir bu ben sen ne de da mi için gibi çok var yok o beni seni ama daha her".split()),
}

# Languages that share too many function words to be told apart reliably;
# a mismatch between members of one group is never reported.
_GROUPS = [frozenset({"es", "pt"}), frozenset({"sv", "da", "no"})]

# Languages we can judge: a function-word profile or a recognisable script.
_KNOWN_LANGUAGES = frozenset(_FUNCTION_WORDS) | {"ja", "zh", "ko", "ru", "uk", "bg", "sr", "be"}

_MIN_WORDS = 30
_MIN_SHARE = 0.15     # share of all words that are function words of the winner
_MIN_MARGIN = 2.0     # winner must have at least this many times the hits of the runner-up

_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)
_LRC_TAG_RE = re.compile(r"\[[^\]]*\]")


def _script_language(text: str) -> str | None:
    """Language guess from the writing system for non-Latin scripts."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return None
    kana = sum(1 for ch in letters if "぀" <= ch <= "ヿ")
    han = sum(1 for ch in letters if "一" <= ch <= "鿿")
    cyr = sum(1 for ch in letters if "Ѐ" <= ch <= "ӿ")
    hangul = sum(1 for ch in letters if "가" <= ch <= "힯")
    n = len(letters)
    if (kana + han) / n > 0.5:
        return "ja" if kana / n > 0.05 else "zh"
    if hangul / n > 0.5:
        return "ko"
    if cyr / n > 0.5:
        return "ru"  # Cyrillic; see _same_language for uk/ru/bg
    return None


def detect_text_language(text: str) -> str | None:
    """Most likely language of ``text`` or ``None`` when it is not clear."""
    text = _LRC_TAG_RE.sub(" ", text or "")
    script = _script_language(text)
    if script:
        return script
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if len(words) < _MIN_WORDS:
        return None
    hits = Counter()
    for lang, vocab in _FUNCTION_WORDS.items():
        hits[lang] = sum(1 for w in words if w in vocab)
    (best, best_hits), *rest = hits.most_common()
    second_hits = rest[0][1] if rest else 0
    if best_hits / len(words) < _MIN_SHARE:
        return None
    if second_hits and best_hits < _MIN_MARGIN * second_hits:
        # ambiguous - unless the runner-up is a sibling language of the winner
        runner_up = rest[0][0]
        if not _same_language(best, runner_up):
            return None
    return best


def _same_language(a: str, b: str) -> bool:
    if a == b:
        return True
    cyrillic = {"ru", "uk", "bg", "sr", "be"}
    if a in cyrillic and b in cyrillic:
        return True
    return any(a in g and b in g for g in _GROUPS)


def lyrics_language_mismatch(lyrics: str, sung_language: str | None) -> str | None:
    """Language of ``lyrics`` if it is clearly NOT ``sung_language``, else ``None``.

    Only reports a mismatch when both languages are known and the lyrics
    language is unambiguous; anything uncertain is treated as matching.
    """
    if not sung_language:
        return None
    sung = sung_language.lower()[:2]
    if sung not in _KNOWN_LANGUAGES:
        return None  # no profile for the sung language: cannot judge
    detected = detect_text_language(lyrics)
    if detected is None or _same_language(detected, sung):
        return None
    return detected


# Whisper tiny language probability from which the sung language is trusted
# more than the language of found lyrics.
CONFIDENT_AUDIO_LANGUAGE = 0.8


def check_lyrics_language(lyrics: str, language: str | None,
                          language_is_confident: bool) -> tuple[bool, str | None]:
    """Decide whether found lyrics may be used for a song sung in ``language``.

    Returns ``(use_lyrics, other_language)``:

    * lyrics language unclear, or the same as ``language`` -> ``(True, None)``
    * it differs and ``language`` is trusted (set by the user, from a full
      transcription or a confident detection) -> ``(False, lyrics language)``:
      the lyrics belong to another language version of the song
    * it differs but ``language`` is only a weak guess -> ``(True, lyrics
      language)``: either side may be wrong, so the lyrics are kept; the
      caller can point out the disagreement
    """
    detected = lyrics_language_mismatch(lyrics, language)
    if detected is None:
        return True, None
    return (not language_is_confident), detected
