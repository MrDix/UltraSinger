"""Tests for YouTube manual subtitle extraction and LRC conversion."""

import pytest

from modules.Audio.youtube_subtitles import (
    _clean_subtitle_text,
    _convert_to_lrc,
    _is_non_lyric,
    _pick_language,
)


# ── _clean_subtitle_text ──────────────────────────────────────────


class TestCleanSubtitleText:
    """Tests for subtitle text cleaning."""

    def test_removes_musical_notes_prefix(self):
        assert _clean_subtitle_text("♪ Hello world") == "Hello world"

    def test_removes_musical_notes_suffix(self):
        assert _clean_subtitle_text("Hello world ♪") == "Hello world"

    def test_removes_musical_notes_both(self):
        assert _clean_subtitle_text("♪ Hello world ♪") == "Hello world"

    def test_removes_html_tags(self):
        assert _clean_subtitle_text("<b>Hello</b> world") == "Hello world"

    def test_normalizes_whitespace(self):
        assert _clean_subtitle_text("  Hello   world  ") == "Hello world"

    def test_empty_after_cleanup(self):
        assert _clean_subtitle_text("♪ ♪") == ""

    def test_keeps_regular_text(self):
        assert _clean_subtitle_text("I love you baby") == "I love you baby"

    def test_multiple_note_variants(self):
        assert _clean_subtitle_text("♫ Singing 🎵") == "Singing"


# ── _is_non_lyric ─────────────────────────────────────────────────


class TestIsNonLyric:
    """Tests for non-lyric line detection."""

    def test_music_marker(self):
        assert _is_non_lyric("[Music]") is True

    def test_instrumental(self):
        assert _is_non_lyric("(instrumental)") is True

    def test_applause(self):
        assert _is_non_lyric("[Applause]") is True

    def test_guitar_solo(self):
        assert _is_non_lyric("guitar solo") is True

    def test_actual_lyrics(self):
        assert _is_non_lyric("I love you baby") is False

    def test_case_insensitive(self):
        assert _is_non_lyric("[MUSIC]") is True


# ── _pick_language ─────────────────────────────────────────────────


class TestPickLanguage:
    """Tests for subtitle language selection."""

    def test_preferred_exact_match(self):
        subs = {"en": [], "de": [], "es": []}
        assert _pick_language(subs, "de") == "de"

    def test_preferred_prefix_match(self):
        subs = {"en-US": [], "de": []}
        assert _pick_language(subs, "en") == "en-US"

    def test_english_fallback(self):
        subs = {"en": [], "de": [], "fr": []}
        assert _pick_language(subs, "ja") == "en"

    def test_english_variant_fallback(self):
        subs = {"en-GB": [], "de": []}
        assert _pick_language(subs, "ja") == "en-GB"

    def test_any_available(self):
        subs = {"fr": [], "de": []}
        assert _pick_language(subs, "ja") in ("fr", "de")

    def test_empty_subs(self):
        assert _pick_language({}, "en") is None

    def test_no_preferred(self):
        subs = {"en": [], "de": []}
        assert _pick_language(subs, None) == "en"


# ── _convert_to_lrc ───────────────────────────────────────────────


class TestConvertToLrc:
    """Tests for VTT to LRC conversion."""

    def test_basic_vtt_conversion(self):
        vtt = """WEBVTT

00:00:05.000 --> 00:00:10.000
♪ Hello world ♪

00:00:10.000 --> 00:00:15.000
♪ I love you ♪
"""
        result = _convert_to_lrc(vtt)
        assert result is not None
        lines = result.split("\n")
        assert len(lines) == 2
        assert "[00:05.00] Hello world" in lines[0]
        assert "[00:10.00] I love you" in lines[1]

    def test_filters_music_markers(self):
        vtt = """WEBVTT

00:00:01.000 --> 00:00:03.000
[Music]

00:00:03.000 --> 00:00:08.000
♪ Actual lyrics here ♪
"""
        result = _convert_to_lrc(vtt)
        assert result is not None
        lines = result.split("\n")
        assert len(lines) == 1
        assert "Actual lyrics here" in lines[0]

    def test_deduplicates_repeated_cues(self):
        vtt = """WEBVTT

00:00:05.000 --> 00:00:10.000
Hello world

00:00:07.000 --> 00:00:12.000
Hello world
"""
        result = _convert_to_lrc(vtt)
        assert result is not None
        lines = result.split("\n")
        assert len(lines) == 1

    def test_empty_vtt_returns_none(self):
        vtt = """WEBVTT

"""
        assert _convert_to_lrc(vtt) is None

    def test_only_music_markers_returns_none(self):
        vtt = """WEBVTT

00:00:01.000 --> 00:00:05.000
[Music]

00:00:05.000 --> 00:00:10.000
[Applause]
"""
        assert _convert_to_lrc(vtt) is None

    def test_hours_in_timestamp(self):
        vtt = """WEBVTT

01:02:03.456 --> 01:02:10.000
Long song lyrics
"""
        result = _convert_to_lrc(vtt)
        assert result is not None
        # 1h 2m = 62 minutes
        assert "[62:03.46]" in result

    def test_multiline_cue(self):
        vtt = """WEBVTT

00:00:05.000 --> 00:00:10.000
Line one
Line two
"""
        result = _convert_to_lrc(vtt)
        assert result is not None
        assert "Line one Line two" in result
