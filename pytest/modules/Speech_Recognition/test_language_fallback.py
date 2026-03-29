"""Tests for language detection fallback logic."""

from modules.Speech_Recognition.Whisper import (
    CORE_LANGUAGES,
    _LANG_CONFIDENCE_THRESHOLD,
)


class TestCoreLanguages:
    """Verify the core language set matches README documentation."""

    def test_core_languages_contains_readme_list(self):
        """README lists: en, fr, de, es, it, ja, zh, nl, uk, pt."""
        readme_languages = {"en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt"}
        assert readme_languages == CORE_LANGUAGES

    def test_core_languages_is_frozenset(self):
        assert isinstance(CORE_LANGUAGES, frozenset)

    def test_confidence_threshold_is_reasonable(self):
        assert 0.0 < _LANG_CONFIDENCE_THRESHOLD < 1.0
