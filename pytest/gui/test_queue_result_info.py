"""Tests for queue result info parsing from UltraSinger stdout."""

import pytest
from unittest.mock import MagicMock, patch

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
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Synced lyrics found — skipping Whisper transcription"
        )
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

    def test_language_from_youtube_metadata(self, queue_manager):
        """'Using YouTube language metadata: xx' should set language."""
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Using YouTube language metadata: ja"
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

    def test_language_youtube_then_whisper_correction(self, queue_manager):
        """YouTube metadata followed by Whisper correction should track change."""
        item = _make_item()
        queue_manager._current_item = item
        queue_manager._parse_output_line(
            "[UltraSinger] Using YouTube language metadata: nn"
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
