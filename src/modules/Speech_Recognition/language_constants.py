"""Language detection constants — side-effect-free module.

Extracted from Whisper.py so tests can import these constants without
triggering the torch.load patch or whisperx import side effects.
"""

# Core languages with high-quality Whisper transcription AND wav2vec2
# forced-alignment models.  These are the languages documented in the
# README and tested to produce good UltraStar results.
CORE_LANGUAGES = frozenset({
    "en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt",
})

# Minimum confidence threshold for language detection.  When Whisper
# tiny fast-detect reports confidence BELOW this threshold AND the
# language is not in CORE_LANGUAGES, we fall back to English — the
# most common language in music.
LANG_CONFIDENCE_THRESHOLD = 0.5
