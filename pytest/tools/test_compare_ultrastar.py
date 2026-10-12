"""Tests for tools/compare_ultrastar.py — format-aware UltraStar comparison tool."""

import json
import textwrap
from pathlib import Path

import pytest

# The tool is a standalone script in tools/ — import its internals directly
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from compare_ultrastar import (
    Note,
    WordGroup,
    ParsedFile,
    ComparisonResult,
    _beat_to_ms,
    _pitch_to_midi,
    _version_ge,
    _group_words,
    _normalise_word,
    _parse_bpm,
    _parse_gap,
    _pct,
    absolute_chart_text,
    align_words,
    compare,
    declared_encoding,
    parse_score_song,
    parse_ultrastar,
    read_chart_text,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_ultrastar(tmp_path: Path, content: str, name: str = "song.txt") -> Path:
    """Write an UltraStar TXT file and return its path."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ── Format detection ─────────────────────────────────────────────────────────

class TestFormatDetection:
    """Feature: auto-detect file format version and apply pitch conversion."""

    def test_v120_format_detected(self, tmp_path):
        """v1.2.0 header is parsed and pitch is offset by +48 to get MIDI."""
        txt = _write_ultrastar(tmp_path, """\
            #VERSION:1.2.0
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:5000
            : 0 5 12 Hello
            E
        """)
        parsed = parse_ultrastar(txt)
        assert parsed.version == "1.2.0"
        # pitch=12 in v1.2.0 → MIDI = 12 + 48 = 60
        assert parsed.notes[0].pitch == 12
        assert parsed.notes[0].midi == 60

    def test_legacy_format_no_version(self, tmp_path):
        """Legacy file without #VERSION treats pitch as raw MIDI."""
        txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:5000
            : 0 5 60 Hello
            E
        """)
        parsed = parse_ultrastar(txt)
        assert parsed.version is None
        # pitch=60 in legacy → MIDI = 60 (no offset)
        assert parsed.notes[0].pitch == 60
        assert parsed.notes[0].midi == 60

    def test_v110_format_no_offset(self, tmp_path):
        """v1.1.0 file still uses raw MIDI (offset only from v1.2.0+)."""
        txt = _write_ultrastar(tmp_path, """\
            #VERSION:1.1.0
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:5000
            : 0 5 60 Hello
            E
        """)
        parsed = parse_ultrastar(txt)
        assert parsed.version == "1.1.0"
        assert parsed.notes[0].midi == 60  # no offset for < 1.2.0

    def test_mixed_format_comparison_gives_zero_diff(self, tmp_path):
        """Comparing v1.2.0 (pitch=12) vs legacy (pitch=60) yields 0 diff.

        Both represent MIDI note 60 — the tool must normalise before comparing.
        """
        gen_txt = _write_ultrastar(tmp_path, """\
            #VERSION:1.2.0
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:5000
            : 0 5 12 Hello
            E
        """, name="gen.txt")

        ref_txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:5000
            : 0 5 60 Hello
            E
        """, name="ref.txt")

        result = compare(gen_txt, ref_txt)
        assert result.pitch_median == 0.0
        assert result.pitch_exact_pct == 100.0


# ── Pitch conversion unit tests ─────────────────────────────────────────────

class TestPitchToMidi:
    """Unit tests for the _pitch_to_midi helper."""

    def test_v120_adds_48(self):
        """v1.2.0 format adds +48 offset to convert to MIDI."""
        assert _pitch_to_midi(0, "1.2.0") == 48
        assert _pitch_to_midi(12, "1.2.0") == 60
        assert _pitch_to_midi(-12, "1.2.0") == 36

    def test_legacy_returns_raw(self):
        """Legacy format (no version) returns pitch as-is."""
        assert _pitch_to_midi(60, None) == 60
        assert _pitch_to_midi(48, None) == 48

    def test_v110_returns_raw(self):
        """v1.1.0 is below the 1.2.0 threshold, so no offset."""
        assert _pitch_to_midi(60, "1.1.0") == 60

    def test_v200_adds_48(self):
        """v2.0.0 is >= 1.2.0, so offset applies."""
        assert _pitch_to_midi(0, "2.0.0") == 48


class TestVersionGe:
    """Unit tests for the _version_ge semantic version comparison helper."""

    def test_equal(self):
        """Equal versions return True."""
        assert _version_ge("1.2.0", "1.2.0") is True

    def test_greater(self):
        """Greater version returns True."""
        assert _version_ge("2.0.0", "1.2.0") is True

    def test_less(self):
        """Lesser version returns False."""
        assert _version_ge("1.1.0", "1.2.0") is False

    def test_invalid_returns_false(self):
        """Malformed version string returns False gracefully."""
        assert _version_ge("abc", "1.2.0") is False

    def test_shorthand_version_padded(self):
        """Shorthand '1.2' is treated as '1.2.0' (padded with trailing zeros)."""
        assert _version_ge("1.2", "1.2.0") is True
        assert _version_ge("1.2.0", "1.2") is True

    def test_suffixed_version_parsed(self):
        """Suffixed versions like '1.2.0-beta' extract numeric part correctly."""
        assert _version_ge("1.2.0-beta", "1.2.0") is True
        assert _version_ge("1.3.0-rc1", "1.2.0") is True
        assert _version_ge("1.1.0-alpha", "1.2.0") is False

    def test_single_component_version(self):
        """Single-component version like '2' is treated as '2.0.0'."""
        assert _version_ge("2", "1.2.0") is True
        assert _version_ge("1", "1.2.0") is False


# ── Beat-to-ms conversion ───────────────────────────────────────────────────

class TestBeatToMs:
    """UltraStar BPM header represents quarter-notes at 4x the real BPM."""

    def test_zero_beat_returns_gap(self):
        """Beat 0 returns exactly the GAP value."""
        assert _beat_to_ms(0, 300.0, 5000.0) == 5000.0

    def test_formula(self):
        """Verify beat-to-ms formula: gap + (beat * 60000) / (bpm * 4)."""
        # bpm_header=300, real_bpm=1200
        # beat=120 → 5000 + (120 * 60000) / 1200 = 5000 + 6000 = 11000
        assert _beat_to_ms(120, 300.0, 5000.0) == 11000.0

    def test_zero_bpm_returns_gap(self):
        """Edge case: BPM=0 should not crash, just return GAP."""
        assert _beat_to_ms(100, 0, 5000.0) == 5000.0

    def test_comma_bpm_parsing(self):
        """BPM may use comma as decimal separator."""
        assert _parse_bpm("409,66") == 409.66

    def test_comma_gap_parsing(self):
        """GAP may use comma as decimal separator."""
        assert _parse_gap("5000,5") == 5000.5

    def test_empty_gap_is_zero(self):
        """An empty #GAP counts as 0, as the games read it."""
        assert _parse_gap("") == 0.0
        assert _parse_gap("  ") == 0.0


# ── Tilde note grouping ─────────────────────────────────────────────────────

class TestTildeGrouping:
    """Tests for grouping continuation notes (~) with their parent word."""

    def _make_note(self, word, midi=60, start_ms=0, end_ms=100):
        """Create a Note instance for testing."""
        return Note(
            note_type=":",
            start_beat=0,
            duration=5,
            pitch=midi,
            midi=midi,
            word=word,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    def test_tilde_groups_with_parent(self):
        """Tilde notes merge continuation text into the parent word."""
        notes = [
            self._make_note("Hel", start_ms=0, end_ms=100),
            self._make_note("~lo", start_ms=100, end_ms=200),
            self._make_note("World", start_ms=300, end_ms=400),
        ]
        groups = _group_words(notes)
        assert len(groups) == 2
        assert groups[0].word == "Hello"  # merged: "Hel" + "lo"
        assert len(groups[0].notes) == 2
        assert groups[0].end_ms == 200  # extended by tilde note
        assert groups[1].word == "World"
        assert len(groups[1].notes) == 1

    def test_multiple_tildes(self):
        """Multiple consecutive tildes all merge into the same parent."""
        notes = [
            self._make_note("Beau", start_ms=0, end_ms=50),
            self._make_note("~ti", start_ms=50, end_ms=100),
            self._make_note("~ful", start_ms=100, end_ms=150),
        ]
        groups = _group_words(notes)
        assert len(groups) == 1
        assert groups[0].word == "Beautiful"  # merged: "Beau" + "ti" + "ful"
        assert len(groups[0].notes) == 3
        assert groups[0].end_ms == 150

    def test_tilde_at_start_becomes_own_group(self):
        """A tilde as the very first note cannot attach to a parent."""
        notes = [
            self._make_note("~orphan", start_ms=0, end_ms=100),
            self._make_note("Word", start_ms=200, end_ms=300),
        ]
        groups = _group_words(notes)
        # ~orphan becomes its own group since there is no parent
        assert len(groups) == 2

    def test_primary_midi_is_first_note(self):
        """WordGroup.midi comes from the first note in the group."""
        notes = [
            self._make_note("Test", midi=60, start_ms=0, end_ms=100),
            self._make_note("~ing", midi=65, start_ms=100, end_ms=200),
        ]
        groups = _group_words(notes)
        assert groups[0].midi == 60  # first note's MIDI, not 65


# ── Word alignment ───────────────────────────────────────────────────────────

class TestWordAlignment:
    """Tests for fuzzy word alignment between generated and reference files."""

    def _make_wg(self, word, midi=60, start_ms=0):
        """Create a WordGroup instance for testing."""
        return WordGroup(word=word, midi=midi, start_ms=start_ms)

    def test_exact_match(self):
        """Identical word sequences yield 1:1 index pairs."""
        gen = [self._make_wg("Hello"), self._make_wg("World")]
        ref = [self._make_wg("Hello"), self._make_wg("World")]
        pairs = align_words(gen, ref)
        assert pairs == [(0, 0), (1, 1)]

    def test_partial_match_with_extra_words(self):
        """Shared words are matched even when surrounding words differ."""
        gen = [self._make_wg("the"), self._make_wg("quick"), self._make_wg("fox")]
        ref = [self._make_wg("quick"), self._make_wg("brown"), self._make_wg("fox")]
        pairs = align_words(gen, ref)
        # "quick" and "fox" should match
        matched_gen_words = [gen[gi].word for gi, ri in pairs]
        assert "quick" in matched_gen_words
        assert "fox" in matched_gen_words

    def test_case_insensitive(self):
        """Word matching is case-insensitive."""
        gen = [self._make_wg("HELLO")]
        ref = [self._make_wg("hello")]
        pairs = align_words(gen, ref)
        assert len(pairs) == 1

    def test_punctuation_ignored(self):
        """Punctuation is stripped before matching."""
        gen = [self._make_wg("don't")]
        ref = [self._make_wg("dont")]
        pairs = align_words(gen, ref)
        assert len(pairs) == 1

    def test_empty_words_skipped(self):
        """Empty word strings are excluded from alignment."""
        gen = [self._make_wg(""), self._make_wg("Hello")]
        ref = [self._make_wg("Hello"), self._make_wg("")]
        pairs = align_words(gen, ref)
        # Only "Hello" should match
        assert len(pairs) == 1


# ── Timing ───────────────────────────────────────────────────────────────────

class TestTiming:
    """Tests for timing difference calculations."""

    def test_first_word_excluded_from_timing(self, tmp_path):
        """Timing metrics exclude the first matched word (GAP offset effect)."""
        gen_txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:0
            : 0 5 60 First
            : 100 5 62 Second
            : 200 5 64 Third
            E
        """, name="gen.txt")

        ref_txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:0
            : 0 5 60 First
            : 100 5 62 Second
            : 200 5 64 Third
            E
        """, name="ref.txt")

        result = compare(gen_txt, ref_txt)
        # 3 matched words, but timing_diffs only has 2 (first excluded)
        assert result.matched_pairs == 3
        assert len(result.timing_diffs) == 2

    def test_timing_calculation_correct(self, tmp_path):
        """Verify timing differences are computed from beat→ms conversion."""
        # BPM=300, real_bpm=1200, ms_per_beat = 60000/1200 = 50ms
        gen_txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:1000
            : 0 5 60 Hello
            : 20 5 60 World
            E
        """, name="gen.txt")

        ref_txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:1000
            : 0 5 60 Hello
            : 22 5 60 World
            E
        """, name="ref.txt")

        result = compare(gen_txt, ref_txt)
        # "World": gen at beat 20 → 1000 + 20*50 = 2000ms
        #          ref at beat 22 → 1000 + 22*50 = 2100ms
        # diff = 2000 - 2100 = -100ms
        assert len(result.timing_diffs) == 1  # first word excluded
        assert result.timing_diffs[0] == pytest.approx(-100.0, abs=0.1)


# ── Intervals ────────────────────────────────────────────────────────────────

class TestIntervals:
    """Tests for pitch interval comparison between consecutive matched words."""

    def test_interval_format_independent(self, tmp_path):
        """Intervals cancel out any constant pitch offset between formats.

        v1.2.0 pitches 12, 17, 19 → MIDI 60, 65, 67
        Legacy pitches 60, 65, 67 → MIDI 60, 65, 67
        Intervals: +5, +2 vs +5, +2 → diffs all 0
        """
        gen_txt = _write_ultrastar(tmp_path, """\
            #VERSION:1.2.0
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:0
            : 0 5 12 One
            : 20 5 17 Two
            : 40 5 19 Three
            E
        """, name="gen.txt")

        ref_txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:0
            : 0 5 60 One
            : 20 5 65 Two
            : 40 5 67 Three
            E
        """, name="ref.txt")

        result = compare(gen_txt, ref_txt)
        assert len(result.interval_diffs) == 2
        assert all(d == 0 for d in result.interval_diffs)
        assert result.interval_exact_pct == 100.0

    def test_interval_detects_wrong_jump(self, tmp_path):
        """An incorrect interval is reported."""
        gen_txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:0
            : 0 5 60 One
            : 20 5 67 Two
            E
        """, name="gen.txt")

        ref_txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:0
            : 0 5 60 One
            : 20 5 65 Two
            E
        """, name="ref.txt")

        result = compare(gen_txt, ref_txt)
        # gen interval = 67 - 60 = 7
        # ref interval = 65 - 60 = 5
        # diff = 7 - 5 = 2
        assert result.interval_diffs == [2]


# ── Parsing ──────────────────────────────────────────────────────────────────

class TestParsing:
    """Tests for UltraStar TXT file parsing."""

    def test_note_types_parsed(self, tmp_path):
        """All standard note types are parsed correctly."""
        txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:0
            : 0 5 60 Normal
            * 10 5 62 Golden
            F 20 5 64 Freestyle
            R 30 5 66 Rap
            G 40 5 68 RapGolden
            E
        """)
        parsed = parse_ultrastar(txt)
        types = [n.note_type for n in parsed.notes]
        assert types == [":", "*", "F", "R", "G"]

    def test_linebreak_counting(self, tmp_path):
        """Linebreak markers are counted but not treated as notes."""
        txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:0
            : 0 5 60 Hello
            - 10
            : 20 5 62 World
            - 30
            : 40 5 64 End
            E
        """)
        parsed = parse_ultrastar(txt)
        assert parsed.linebreak_count == 2
        assert len(parsed.notes) == 3  # linebreaks not counted as notes

    def test_headers_collected(self, tmp_path):
        """All header tags are stored in the headers dict."""
        txt = _write_ultrastar(tmp_path, """\
            #TITLE:My Song
            #ARTIST:Some Artist
            #BPM:300
            #GAP:5000
            #LANGUAGE:English
            : 0 5 60 Hello
            E
        """)
        parsed = parse_ultrastar(txt)
        assert parsed.headers["TITLE"] == "My Song"
        assert parsed.headers["ARTIST"] == "Some Artist"
        assert parsed.headers["LANGUAGE"] == "English"

    def test_end_marker_stops_parsing(self, tmp_path):
        """Lines after 'E' are ignored."""
        txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:0
            : 0 5 60 Hello
            E
            : 100 5 62 ShouldBeIgnored
        """)
        parsed = parse_ultrastar(txt)
        assert len(parsed.notes) == 1

    def test_comma_bpm_gap(self, tmp_path):
        """European-style comma decimals are handled."""
        txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:409,66
            #GAP:12763,5
            : 0 5 60 Hello
            E
        """)
        parsed = parse_ultrastar(txt)
        assert parsed.bpm == pytest.approx(409.66, abs=0.01)
        assert parsed.gap == pytest.approx(12763.5, abs=0.1)


# ── JSON output ──────────────────────────────────────────────────────────────

class TestJsonOutput:
    """Tests for JSON serialisation of comparison results."""

    def test_json_has_expected_structure(self, tmp_path):
        """JSON output contains all expected top-level sections."""
        gen_txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:0
            : 0 5 60 Hello
            E
        """, name="gen.txt")

        ref_txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:0
            : 0 5 60 Hello
            E
        """, name="ref.txt")

        result = compare(gen_txt, ref_txt)
        data = json.loads(result.to_json())

        assert "pitch" in data
        assert "intervals" in data
        assert "timing" in data
        assert "lyrics" in data
        assert "structure" in data
        assert "meta" in data

        # Verify key fields exist
        assert "median_diff" in data["pitch"]
        assert "exact_pct" in data["pitch"]
        assert "within_2st_pct" in data["pitch"]
        assert "median_ms" in data["timing"]
        assert "within_100ms_pct" in data["timing"]
        assert "gen_notes" in data["structure"]
        assert "ref_notes" in data["structure"]
        assert "gen_version" in data["meta"]
        assert "ref_version" in data["meta"]


# ── Full comparison integration ──────────────────────────────────────────────

class TestFullComparison:
    """Integration tests running the full compare() pipeline."""

    def test_identical_files_perfect_score(self, tmp_path):
        """Comparing identical files should yield perfect metrics."""
        content = """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:5000
            : 0 5 60 Hello
            : 20 5 62 World
            : 40 5 64 Again
            - 50
            : 60 5 66 More
            E
        """
        gen_txt = _write_ultrastar(tmp_path, content, name="gen.txt")
        ref_txt = _write_ultrastar(tmp_path, content, name="ref.txt")

        result = compare(gen_txt, ref_txt)

        assert result.pitch_exact_pct == 100.0
        assert result.pitch_median == 0.0
        assert result.interval_exact_pct == 100.0
        assert result.timing_within_100 == 100.0
        assert result.lyrics_precision == 100.0
        assert result.lyrics_recall == 100.0
        assert result.matched_pairs == 4
        assert result.gen_linebreaks == 1
        assert result.ref_linebreaks == 1

    def test_no_matching_words(self, tmp_path):
        """No matching words → empty metrics, no crash."""
        gen_txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:0
            : 0 5 60 Alpha
            E
        """, name="gen.txt")

        ref_txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:0
            : 0 5 60 Beta
            E
        """, name="ref.txt")

        result = compare(gen_txt, ref_txt)
        assert result.matched_pairs == 0
        assert result.pitch_exact_pct == 0.0
        assert result.lyrics_recall == 0.0

    def test_summary_output_not_empty(self, tmp_path):
        """Human-readable summary produces non-empty output."""
        content = """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:0
            : 0 5 60 Hello
            E
        """
        gen_txt = _write_ultrastar(tmp_path, content, name="gen.txt")
        ref_txt = _write_ultrastar(tmp_path, content, name="ref.txt")

        result = compare(gen_txt, ref_txt)
        summary = result.to_summary()
        assert "UltraStar Comparison Report" in summary
        assert "Pitch" in summary
        assert "Timing" in summary


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and utility function tests."""

    def test_normalise_word(self):
        """Word normalisation strips punctuation and lowercases."""
        assert _normalise_word("Don't") == "dont"
        assert _normalise_word("HELLO!") == "hello"
        assert _normalise_word("  spaces  ") == "spaces"
        assert _normalise_word("café") == "cafe"  # accented → decomposed

    def test_pct_zero_total(self):
        """Percentage with zero denominator returns 0.0 instead of crashing."""
        assert _pct(5, 0) == 0.0
        assert _pct(0, 0) == 0.0

    def test_pct_normal(self):
        """Normal percentage calculation works correctly."""
        assert _pct(1, 4) == 25.0
        assert _pct(3, 3) == 100.0

    def test_lyrics_denominator_excludes_empty_words(self, tmp_path):
        """Lyrics precision/recall denominators exclude empty normalized words.

        align_words() filters out empty words, so denominators must use
        the same filter to avoid underreporting metrics.
        """
        gen_txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:0
            : 0 5 60 Hello
            : 10 5 60 -
            : 20 5 62 World
            E
        """, name="gen.txt")

        ref_txt = _write_ultrastar(tmp_path, """\
            #TITLE:Test
            #ARTIST:Test
            #BPM:300
            #GAP:0
            : 0 5 60 Hello
            : 20 5 62 World
            E
        """, name="ref.txt")

        result = compare(gen_txt, ref_txt)
        # "-" normalizes to "" → excluded from denominators
        # gen has 2 valid words (Hello, World), ref has 2
        assert result.gen_word_count == 2
        assert result.ref_word_count == 2
        assert result.matched_pairs == 2
        assert result.lyrics_precision == 100.0
        assert result.lyrics_recall == 100.0

    def test_tilde_bare_continuation_no_suffix(self):
        """Bare tilde '~' (no suffix text) does not alter parent word."""
        notes = [
            Note(note_type=":", start_beat=0, duration=5, pitch=60,
                 midi=60, word="Hold", start_ms=0, end_ms=100),
            Note(note_type=":", start_beat=5, duration=5, pitch=60,
                 midi=60, word="~", start_ms=100, end_ms=200),
        ]
        groups = _group_words(notes)
        assert len(groups) == 1
        assert groups[0].word == "Hold"  # unchanged — "~" has no suffix
        assert len(groups[0].notes) == 2

    def test_empty_file(self, tmp_path):
        """An empty file should parse without crashing."""
        txt = _write_ultrastar(tmp_path, "")
        parsed = parse_ultrastar(txt)
        assert parsed.notes == []
        assert parsed.word_groups == []
        assert parsed.bpm == 0.0


# ── Text encoding ───────────────────────────────────────────────────────────

class TestReadChartText:
    """Feature: charts are decoded the way the games read them."""

    def test_bom_does_not_hide_the_first_header(self, tmp_path):
        """A UTF-8 BOM before #VERSION must not drop the version (48 semitones off)."""
        p = tmp_path / "bom.txt"
        p.write_bytes(b"\xef\xbb\xbf#VERSION:1.2.0\n#BPM:300\n#GAP:0\n: 0 4 12 la\nE\n")
        parsed = parse_ultrastar(p)
        assert parsed.version == "1.2.0"
        assert parsed.notes[0].midi == 60

    def test_cp1252_lyrics(self, tmp_path):
        p = tmp_path / "cp1252.txt"
        p.write_bytes("#BPM:300\n#GAP:0\n: 0 4 60 sch\xf6n\nE\n".encode("cp1252"))
        assert parse_ultrastar(p).notes[0].word == "sch\xf6n"

    @pytest.mark.parametrize("newline", ["\n", "\r\n"])
    def test_encoding_tag_is_honoured(self, tmp_path, newline):
        """A CP1250 chart (Polish/Czech) is read in CP1250, not as Windows-1252."""
        p = tmp_path / "cp1250.txt"
        lines = ["#TITLE:Title", "#ENCODING:CP1250", "#BPM:300", "#GAP:0", ": 0 4 60 łąka", "E", ""]
        p.write_bytes(newline.join(lines).encode("cp1250"))
        assert parse_ultrastar(p).notes[0].word == "łąka"

    def test_utf8_wins_over_a_stale_encoding_tag(self, tmp_path):
        p = tmp_path / "utf8.txt"
        p.write_bytes("#ENCODING:CP1250\n#BPM:300\n#GAP:0\n: 0 4 60 łąka\nE\n".encode("utf-8"))
        assert read_chart_text(p).splitlines()[3].endswith("łąka")

    @pytest.mark.parametrize("value", ["Auto", "", "nonsense", "UTF-16", "base64"])
    def test_unusable_encoding_tags_are_ignored(self, value):
        assert declared_encoding(f"#TITLE:x\n#ENCODING:{value}\n".encode("ascii")) is None

    def test_known_encoding_tag(self):
        assert declared_encoding(b"#TITLE:x\r\n#encoding: cp1250 \r\n") == "cp1250"


# ── #RELATIVE charts ────────────────────────────────────────────────────────

_RELATIVE_CHART = [
    "#TITLE:Title", "#ARTIST:Artist", "#BPM:300", "#GAP:1000", "#RELATIVE:YES",
    ": 0 4 60 one", ": 4 4 62 two",
    "- 8 10",
    ": 0 4 64 three", ": 4 4 65 four",
    "- 6",
    ": 2 4 67 five",
    "E",
]


class TestRelativeCharts:
    """Feature: #RELATIVE:YES beats count from the start of their line."""

    @pytest.mark.parametrize("newline", ["\n", "\r\n"])
    def test_beats_are_made_absolute(self, tmp_path, newline):
        p = tmp_path / "rel.txt"
        p.write_bytes(newline.join(_RELATIVE_CHART + [""]).encode("utf-8"))
        parsed = parse_ultrastar(p)
        # "- 8 10" moves the line start to 10; "- 6" (no second number) by 6 more
        assert [n.start_beat for n in parsed.notes] == [0, 4, 10, 14, 18]
        assert parsed.notes[2].start_ms == pytest.approx(_beat_to_ms(10, 300, 1000))
        assert parsed.linebreak_count == 2

    def test_absolute_chart_unchanged(self):
        text = "\n".join(c for c in _RELATIVE_CHART if not c.startswith("#RELATIVE")) + "\n"
        assert absolute_chart_text(text) == text

    def test_conversion_drops_the_tag_and_keeps_other_lines(self):
        out = absolute_chart_text("\n".join(_RELATIVE_CHART) + "\n").splitlines()
        assert "#RELATIVE:YES" not in out
        assert out[:4] == _RELATIVE_CHART[:4]
        assert out[4:] == [": 0 4 60 one", ": 4 4 62 two", "- 8", ": 10 4 64 three", ": 14 4 65 four",
                           "- 16", ": 18 4 67 five", "E"]

    def test_duet_tracks_keep_their_own_line_start(self):
        text = "\n".join(["#BPM:300", "#RELATIVE:yes", "P1", ": 0 2 60 a", "- 4 8", ": 0 2 60 b",
                          "P2", ": 0 2 60 c", "- 2 3", ": 0 2 60 d", "E"])
        beats = [int(line.split()[1]) for line in absolute_chart_text(text).splitlines() if line.startswith(":")]
        assert beats == [0, 8, 0, 3]

    def test_score_library_gets_absolute_beats(self, tmp_path):
        pytest.importorskip("ultrastar_score")
        p = tmp_path / "rel.txt"
        p.write_bytes("\r\n".join(_RELATIVE_CHART + [""]).encode("utf-8"))
        song = parse_score_song(p)
        assert [n.start_beat for n in song.all_notes] == [0, 4, 10, 14, 18]
        assert len(song.lines) == 3
        assert p.read_bytes().count(b"#RELATIVE") == 1  # the chart itself is untouched


class TestParseScoreSong:
    """Feature: the score library reads charts the way the games do."""

    def test_empty_gap_reads_as_zero(self, tmp_path):
        pytest.importorskip("ultrastar_score")
        p = tmp_path / "gap.txt"
        p.write_bytes(b"#TITLE:Title\r\n#BPM:300\r\n#GAP:\r\n: 0 4 60 la\r\nE\r\n")
        song = parse_score_song(p)
        assert song.gap == 0.0 and len(song.all_notes) == 1

    def test_plain_chart_parsed_as_it_is(self, tmp_path):
        pytest.importorskip("ultrastar_score")
        p = tmp_path / "plain.txt"
        p.write_bytes(b"#TITLE:Title\n#MP3:song.mp3\n#BPM:300\n#GAP:500\n: 3 4 60 la\nE\n")
        song = parse_score_song(p)
        assert song.gap == 500.0 and song.audio == "song.mp3"
        assert [n.start_beat for n in song.all_notes] == [3]

    def test_non_numeric_bpm_still_fails(self, tmp_path):
        pytest.importorskip("ultrastar_score")
        p = tmp_path / "bad.txt"
        p.write_bytes(b"#TITLE:Title\n#BPM:fast\n#GAP:\n: 0 4 60 la\nE\n")
        with pytest.raises(ValueError):
            parse_score_song(p)
