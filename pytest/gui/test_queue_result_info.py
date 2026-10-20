"""Tests for queue result info parsing from UltraSinger stdout."""

import pytest
from unittest.mock import MagicMock, patch

# Skip the whole module (instead of failing collection) when PySide6 or
# its native Qt libraries are unavailable (e.g. a runner without the gui
# extra or missing system libs like libEGL).
pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

from src.gui.models import QueueItem
from src.gui.queue_manager import QueueManager


@pytest.fixture
def queue_manager():
    """Create a QueueManager with mocked Qt signals."""
    with patch("src.gui.queue_manager.QObject.__init__"):
        mgr = QueueManager.__new__(QueueManager)
        mgr._items = []
        mgr._runner = MagicMock()
        mgr._running = False
        mgr._current_item = None
        mgr._global_config = {}
        mgr._media_interceptor = None
        mgr._download_thread = None
        mgr._download_worker = None
        # Mock signals
        mgr.item_added = MagicMock()
        mgr.item_removed = MagicMock()
        mgr.item_status_changed = MagicMock()
        mgr.queue_started = MagicMock()
        mgr.queue_finished = MagicMock()
        mgr.line_output = MagicMock()
        mgr.stage_changed = MagicMock()
        mgr.item_result_info = MagicMock()
    return mgr


def _make_item():
    return QueueItem(
        input_source="https://example.com/video",
        input_type="url",
        title="Test Song",
        status="running",
    )


class TestParseOutputLine:
    """Tests for QueueManager._parse_output_line."""

    def test_game_score_parsed(self, queue_manager):
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Game score (ptAKF): Easy 96.7% | Medium 87.1% | Hard 66.6%"
        )
        assert item.result_info["uscore"] == "Easy 96.7% | Medium 87.1% | Hard 66.6%"

    def test_game_score_parsed_with_ansi(self, queue_manager):
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Game score (ptAKF): [94mEasy 96.7% | Medium 87.1% | Hard 66.6%[0m"
        )
        assert item.result_info["uscore"] == "Easy 96.7% | Medium 87.1% | Hard 66.6%"

    def test_language_detected_plain(self, queue_manager):
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Language detected: en (95% confidence)"
        )
        assert item.result_info["language"] == "en"

    def test_language_detected_with_ansi(self, queue_manager):
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Language detected: \x1b[34mde\x1b[0m (87% confidence)"
        )
        assert item.result_info["language"] == "de"

    def test_detected_language_whisper(self, queue_manager):
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Detected language: fr"
        )
        assert item.result_info["language"] == "fr"

    def test_lrclib_synced_found(self, queue_manager):
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Found lyrics on LRCLIB: Artist - Title [plain (120 words), synced]"
        )
        assert item.result_info["lrclib_result"] == "synced"

    def test_lrclib_lengths_parsed(self, queue_manager):
        """The lengths of the found version and of the audio; a later lookup without them drops them."""
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Found lyrics on LRCLIB: Artist - Title [plain (120 words), synced], "
            "245 s long (the audio: 391 s)"
        )
        assert item.result_info["lyrics_length_s"] == 245
        assert item.result_info["audio_length_s"] == 391
        queue_manager._parse_output_line(
            "[UltraSinger] Found lyrics on LRCLIB: Artist - Title [plain (120 words), synced]"
        )
        assert "lyrics_length_s" not in item.result_info

    def test_lrclib_lengths_parsed_from_the_conversions_line(self, queue_manager, capsys):
        """The line as the lookup prints it: colored, a title with brackets, lengths to a tenth of a second."""
        from modules import lrclib_client

        found = lrclib_client.LyricsInfo(track_name="Title [Live] (Extended)", artist_name="Artist [Band]",
                                         plain_lyrics="first line", synced_lyrics="[00:12.00] first line",
                                         duration=210.4)
        with patch.object(lrclib_client, "find_lyrics", return_value=found):
            lrclib_client.search_lyrics("Artist", "Title", duration=190.0)
        item = _make_item()
        queue_manager._current_item = item
        for line in capsys.readouterr().out.splitlines():
            queue_manager._parse_output_line(line)
        assert item.result_info["lrclib_result"] == "synced"
        assert item.result_info["lyrics_length_s"] == 210.4
        assert item.result_info["audio_length_s"] == 190.0
        # 20.4 s longer: another recording for the conversion and for the label
        item.result_info["pipeline"] = "reference"
        queue_manager._items = [item]
        queue_manager._running = True
        queue_manager._on_item_finished(0)
        assert item.result_info["lyrics_source"] == "other version"

    def test_lrclib_plain_only(self, queue_manager):
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Found lyrics on LRCLIB: Artist - Title [plain (80 words)]"
        )
        assert item.result_info["lrclib_result"] == "plain"

    def test_lrclib_no_lyrics(self, queue_manager):
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] No lyrics found on LRCLIB"
        )
        assert item.result_info["lrclib_result"] == "none"

    def test_synced_lyrics_skip_whisper(self, queue_manager):
        # the current wording and the one of older versions
        for line in ("[UltraSinger] Synced lyrics found — skipping the transcription",
                     "[UltraSinger] Synced lyrics found — skipping Whisper transcription"):
            item = _make_item()
            queue_manager._current_item = item
            queue_manager._parse_output_line(line)
            assert item.result_info["whisper_skipped"] is True

    def test_reference_pipeline_detected(self, queue_manager):
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Reference-first pipeline: 150 notes from 120 words"
        )
        assert item.result_info["pipeline"] == "reference"

    def test_reference_recovered_after_language_correction(self, queue_manager):
        item = _make_item()
        queue_manager._current_item = item
        # First: fallback happens
        queue_manager._parse_output_line(
            "[UltraSinger] Falling back to standard pipeline"
        )
        assert item.result_info["whisper_fallback"] is True
        # Then: recovery succeeds
        queue_manager._parse_output_line(
            "[UltraSinger] Reference pipeline recovered after language correction"
        )
        assert item.result_info["pipeline"] == "reference"
        assert item.result_info["reference_recovered"] is True
        assert "whisper_fallback" not in item.result_info

    def test_fallback_to_standard(self, queue_manager):
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Falling back to standard pipeline"
        )
        assert item.result_info["whisper_fallback"] is True
        assert item.result_info["pipeline"] == "whisper"

    def test_fallback_after_alignment_without_words(self, queue_manager):
        """The aligner's own fallback message is in lower case."""
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Alignment returned no words -- falling back to standard pipeline"
        )
        assert item.result_info["whisper_fallback"] is True
        assert item.result_info["pipeline"] == "whisper"

    def test_plain_lyrics_after_failed_synced_alignment(self, queue_manager):
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line("[UltraSinger] Falling back to standard pipeline")
        queue_manager._parse_output_line("[UltraSinger] Plain lyrics pipeline: 150 notes from 120 words")
        assert item.result_info["pipeline"] == "plain"
        assert "whisper_fallback" not in item.result_info

    def test_remote_text_aligned(self, queue_manager):
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line("[UltraSinger] Plain lyrics pipeline: 150 notes from 120 words")
        queue_manager._parse_output_line("[UltraSinger] Remote STT transcription aligned locally")
        assert item.result_info["pipeline"] == "remote"

    def test_remote_text_aligned_after_language_correction(self, queue_manager):
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line("[UltraSinger] Falling back to standard pipeline")
        queue_manager._parse_output_line("[UltraSinger] Remote STT transcription aligned locally")
        assert item.result_info["pipeline"] == "remote"
        assert "whisper_fallback" not in item.result_info

    def test_output_folder_parsed(self, queue_manager):
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Creating output folder. -> D:\\UltraStar\\Songs\\Test Song"
        )
        assert item.result_info["output_folder"] == "D:\\UltraStar\\Songs\\Test Song"

    def test_language_change_tracked(self, queue_manager):
        """When language changes between detections, track the change."""
        item = _make_item()
        queue_manager._current_item = item
        # First detection (fast Whisper tiny)
        queue_manager._parse_output_line(
            "[UltraSinger] Language detected: cy (85% confidence)"
        )
        assert item.result_info["language"] == "cy"
        # Second detection (full Whisper fallback)
        queue_manager._parse_output_line(
            "[UltraSinger] Detected language: en"
        )
        assert item.result_info["language"] == "en"
        assert item.result_info["language_changed"] is True
        assert item.result_info["initial_language"] == "cy"

    def test_language_no_change_no_flag(self, queue_manager):
        """Same language both times should not set language_changed."""
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Language detected: en (95% confidence)"
        )
        queue_manager._parse_output_line(
            "[UltraSinger] Detected language: en"
        )
        assert item.result_info["language"] == "en"
        assert "language_changed" not in item.result_info

    def test_output_folder_from_ultrastar_file(self, queue_manager):
        """Derive output folder from 'Creating UltraStar file' line."""
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Creating UltraStar file D:\\Songs\\Test Song\\Test Song.txt"
        )
        assert item.result_info["output_folder"] == "D:\\Songs\\Test Song"

    def test_language_from_video_platform_metadata(self, queue_manager):
        """'Using video platform language metadata: xx' should set language."""
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Using video platform language metadata: ja"
        )
        assert item.result_info["language"] == "ja"

    def test_language_from_cli_flag(self, queue_manager):
        """'Language set: xx (--language)' should set language."""
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Language set: de (--language)"
        )
        assert item.result_info["language"] == "de"

    def test_language_video_platform_then_whisper_correction(self, queue_manager):
        """Video platform metadata followed by Whisper correction should track change."""
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Using video platform language metadata: nn"
        )
        assert item.result_info["language"] == "nn"
        queue_manager._parse_output_line(
            "[UltraSinger] Detected language: en"
        )
        assert item.result_info["language"] == "en"
        assert item.result_info["language_changed"] is True
        assert item.result_info["initial_language"] == "nn"

    def test_language_low_confidence_fallback(self, queue_manager):
        """Low-confidence fallback line should update language to 'en'."""
        item = _make_item()
        queue_manager._current_item = item
        # First: fast-detect gives a non-core language
        queue_manager._parse_output_line(
            "[UltraSinger] Language detected: cy (42% confidence)"
        )
        assert item.result_info["language"] == "cy"
        # Then: low-confidence fallback
        queue_manager._parse_output_line(
            "[UltraSinger] Low confidence for non-core language "
            "— falling back to en"
        )
        assert item.result_info["language"] == "en"
        assert item.result_info["language_changed"] is True
        assert item.result_info["initial_language"] == "cy"

    def test_output_folder_skips_cache(self, queue_manager):
        """Cache folder paths should be ignored."""
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Creating output folder. -> D:\\UltraStar\\Songs\\cache"
        )
        assert "output_folder" not in item.result_info

    def test_output_folder_skips_cache_trailing_slash(self, queue_manager):
        """Cache folder paths with trailing slash should be ignored."""
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Creating output folder. -> /tmp/ultrasinger/cache/"
        )
        assert "output_folder" not in item.result_info

    def test_output_folder_prefers_ultrastar_file(self, queue_manager):
        """'Creating UltraStar file' should override any earlier folder."""
        item = _make_item()
        queue_manager._current_item = item
        # First: a non-cache output folder line
        queue_manager._parse_output_line(
            "[UltraSinger] Creating output folder. -> D:\\Songs\\Artist - Song"
        )
        assert item.result_info["output_folder"] == "D:\\Songs\\Artist - Song"
        # Then: the definitive UltraStar file line
        queue_manager._parse_output_line(
            "[UltraSinger] Creating UltraStar file D:\\Output\\Real Song\\Real Song.txt"
        )
        assert item.result_info["output_folder"] == "D:\\Output\\Real Song"

    def test_output_folder_non_cache_accepted(self, queue_manager):
        """Non-cache output folders should be accepted as fallback."""
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Creating output folder. -> D:\\Songs\\Artist - Title"
        )
        assert item.result_info["output_folder"] == "D:\\Songs\\Artist - Title"

    def test_no_current_item_safe(self, queue_manager):
        """Parsing with no current item should not crash."""
        queue_manager._current_item = None
        queue_manager._parse_output_line("Language detected: en")

    def test_unrelated_line_ignored(self, queue_manager):
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line("Some random log line")
        assert item.result_info == {}


class TestLyricsSourceDerivation:
    """Tests for lyrics_source derivation in _on_item_finished."""

    def test_reference_pipeline_synced(self, queue_manager):
        item = _make_item()
        item.result_info = {"pipeline": "reference", "language": "en"}
        queue_manager._items = [item]
        queue_manager._current_item = item
        queue_manager._running = True

        queue_manager._on_item_finished(0)

        assert item.result_info["lyrics_source"] == "synced"
        assert item.status == "done"

    def test_whisper_fallback_from_reference(self, queue_manager):
        item = _make_item()
        item.result_info = {
            "lrclib_result": "synced",
            "whisper_fallback": True,
            "pipeline": "whisper",
            "language": "en",
            "language_changed": True,
            "initial_language": "cy",
        }
        queue_manager._items = [item]
        queue_manager._current_item = item
        queue_manager._running = True

        queue_manager._on_item_finished(0)

        assert item.result_info["lyrics_source"] == "synced (fallback)"
        assert item.result_info["language_caused_fallback"] is True

    def test_whisper_fallback_no_language_change(self, queue_manager):
        """Fallback without language change should NOT flag language as cause."""
        item = _make_item()
        item.result_info = {
            "lrclib_result": "synced",
            "whisper_fallback": True,
            "pipeline": "whisper",
            "language": "en",
        }
        queue_manager._items = [item]
        queue_manager._current_item = item
        queue_manager._running = True

        queue_manager._on_item_finished(0)

        assert item.result_info["lyrics_source"] == "synced (fallback)"
        assert "language_caused_fallback" not in item.result_info

    def test_reference_recovered_is_synced(self, queue_manager):
        """When reference pipeline recovers after language correction, treat as synced."""
        item = _make_item()
        item.result_info = {
            "pipeline": "reference",
            "reference_recovered": True,
            "language": "en",
            "language_changed": True,
            "initial_language": "cy",
        }
        queue_manager._items = [item]
        queue_manager._current_item = item
        queue_manager._running = True

        queue_manager._on_item_finished(0)

        assert item.result_info["lyrics_source"] == "synced"

    def test_plain_lyrics_made_the_notes(self, queue_manager):
        """Synced lyrics were found but did not align; the plain lyrics made the notes."""
        for info, expected in (({"lrclib_result": "synced", "pipeline": "plain"}, "plain"),
                               ({"lrclib_result": "none", "pipeline": "remote"}, "transcribed"),
                               ({"lrclib_result": "synced", "pipeline": "remote"}, "synced (fallback)")):
            item = _make_item()
            item.result_info = dict(info, language="en")
            queue_manager._items = [item]
            queue_manager._current_item = item
            queue_manager._running = True

            queue_manager._on_item_finished(0)

            assert item.result_info["lyrics_source"] == expected

    def test_lyrics_of_another_version(self, queue_manager):
        """Found lyrics that made the notes, of a version much longer or shorter than the audio."""
        lengths = {"lyrics_length_s": 245, "audio_length_s": 391}
        for info, expected in (({"pipeline": "reference", **lengths}, "other version"),
                               ({"lrclib_result": "synced", "pipeline": "plain", **lengths}, "other version"),
                               ({"pipeline": "reference", "lyrics_length_s": 380, "audio_length_s": 391},
                                "synced"),
                               ({"lrclib_result": "synced", "whisper_fallback": True, **lengths},
                                "synced (fallback)"),
                               ({"lrclib_result": "none", "pipeline": "remote", **lengths}, "transcribed")):
            item = _make_item()
            item.result_info = dict(info, language="en")
            queue_manager._items = [item]
            queue_manager._current_item = item
            queue_manager._running = True

            queue_manager._on_item_finished(0)

            assert item.result_info["lyrics_source"] == expected

    def test_no_label_left_empty(self, queue_manager):
        """Synced lyrics found, but no line told what made the notes: the transcription did."""
        item = _make_item()
        item.result_info = {"lrclib_result": "synced", "language": "en"}
        queue_manager._items = [item]
        queue_manager._current_item = item
        queue_manager._running = True

        queue_manager._on_item_finished(0)

        assert item.result_info["lyrics_source"] == "transcribed"
        assert item.result_info["pipeline"] == "whisper"

    def test_transcribed_no_lrclib(self, queue_manager):
        item = _make_item()
        item.result_info = {"lrclib_result": "none", "language": "en"}
        queue_manager._items = [item]
        queue_manager._current_item = item
        queue_manager._running = True

        queue_manager._on_item_finished(0)

        assert item.result_info["lyrics_source"] == "transcribed"
        assert item.result_info["pipeline"] == "whisper"

    def test_no_info_defaults_to_transcribed(self, queue_manager):
        item = _make_item()
        item.result_info = {}
        queue_manager._items = [item]
        queue_manager._current_item = item
        queue_manager._running = True

        queue_manager._on_item_finished(0)

        assert item.result_info["lyrics_source"] == "transcribed"
        assert item.result_info["pipeline"] == "whisper"

    def test_result_info_signal_emitted(self, queue_manager):
        item = _make_item()
        item.result_info = {"pipeline": "reference", "language": "en"}
        queue_manager._items = [item]
        queue_manager._current_item = item
        queue_manager._running = True

        queue_manager._on_item_finished(0)

        queue_manager.item_result_info.emit.assert_called_once()
        call_args = queue_manager.item_result_info.emit.call_args
        assert call_args[0][0] == item.id
        assert call_args[0][1]["lyrics_source"] == "synced"
