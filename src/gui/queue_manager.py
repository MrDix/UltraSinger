"""Queue manager for batch conversion of multiple items."""

from __future__ import annotations

import copy
import logging
import re
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QThread, Qt, Signal

from .models import QueueItem
from .ultrasinger_runner import UltraSingerRunner

if TYPE_CHECKING:
    from .media_interceptor import MediaInterceptor

logger = logging.getLogger(__name__)


class QueueManager(QObject):
    """Manages a queue of conversion items and runs them sequentially."""

    item_added = Signal(object)  # QueueItem
    item_removed = Signal(str)  # item_id
    item_status_changed = Signal(str, str)  # item_id, new_status
    queue_started = Signal()
    queue_finished = Signal(int, bool)  # failed_count, was_cancelled

    # Forwarded from the runner for the currently active item
    line_output = Signal(str)
    stage_changed = Signal(str)
    item_result_info = Signal(str, dict)  # item_id, result_info dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[QueueItem] = []
        self._runner = UltraSingerRunner(self)
        self._running = False
        self._current_item: QueueItem | None = None
        self._global_config: dict = {}
        self._media_interceptor: MediaInterceptor | None = None
        self._download_thread: QThread | None = None
        self._download_worker = None  # MediaDownloadWorker | None

        self._runner.line_output.connect(self.line_output.emit)
        self._runner.line_output.connect(self._parse_output_line)
        self._runner.stage_changed.connect(self.stage_changed.emit)
        # Use QueuedConnection so _on_item_finished runs on the main thread,
        # avoiding races with add_item/remove_item/cancel_all.
        self._runner.finished.connect(
            self._on_item_finished, Qt.ConnectionType.QueuedConnection
        )

    @property
    def runner(self) -> UltraSingerRunner:
        return self._runner

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def items(self) -> list[QueueItem]:
        return list(self._items)

    def set_global_config(self, config: dict):
        """Set the global config used as base for all conversions."""
        self._global_config = dict(config)

    def set_media_interceptor(self, interceptor: MediaInterceptor):
        """Set the media interceptor for capturing browser audio URLs."""
        self._media_interceptor = interceptor

    def add_item(
        self, source: str, input_type: str, title: str
    ) -> QueueItem:
        """Add an item to the queue. Returns the new item."""
        item = QueueItem(
            input_source=source,
            input_type=input_type,
            title=title,
        )
        self._items.append(item)
        self.item_added.emit(item)
        logger.info("Queue: added '%s' (%s)", title, input_type)
        return item

    def clone_item(self, item_id: str) -> QueueItem | None:
        """Clone a pending item — duplicate it with the same settings."""
        for item in self._items:
            if item.id == item_id and item.status == "pending":
                clone = QueueItem(
                    input_source=item.input_source,
                    input_type=item.input_type,
                    title=item.title,
                    video_id=item.video_id,
                    yt_language=item.yt_language,
                    settings_overrides=copy.deepcopy(item.settings_overrides),
                )
                self._items.append(clone)
                self.item_added.emit(clone)
                logger.info("Queue: cloned '%s'", item.title)
                return clone
        return None

    def remove_item(self, item_id: str):
        """Remove a pending item from the queue."""
        for i, item in enumerate(self._items):
            if item.id == item_id and item.status == "pending":
                self._items.pop(i)
                self.item_removed.emit(item_id)
                logger.info("Queue: removed '%s'", item.title)
                return

    def clear_pending(self):
        """Remove all pending items from the queue."""
        pending = [it for it in self._items if it.status == "pending"]
        for item in pending:
            self._items.remove(item)
            self.item_removed.emit(item.id)

    def clear_completed(self):
        """Remove all done/failed/cancelled items."""
        finished = [
            it for it in self._items
            if it.status in ("done", "failed", "cancelled")
        ]
        for item in finished:
            self._items.remove(item)
            self.item_removed.emit(item.id)

    def pending_count(self) -> int:
        return sum(1 for it in self._items if it.status == "pending")

    def start_all(self):
        """Start processing all pending items sequentially."""
        if self._running:
            logger.warning("Queue is already running")
            return
        if self.pending_count() == 0:
            return

        self._running = True
        self.queue_started.emit()
        self._run_next()

    def cancel_current(self):
        """Cancel the currently running item (runner or pre-download)."""
        if self._download_worker is not None:
            self._download_worker.cancel()
        if self._current_item and self._runner.is_running:
            self._runner.cancel()

    def cancel_all(self):
        """Cancel current and mark remaining pending items as cancelled."""
        self.cancel_current()
        for item in self._items:
            if item.status == "pending":
                item.status = "cancelled"
                self.item_status_changed.emit(item.id, "cancelled")

    def _emit_finished(self, cancelled: bool = False):
        """Emit queue_finished with summary counts."""
        failed = sum(1 for it in self._items if it.status == "failed")
        self.queue_finished.emit(failed, cancelled)

    def _run_next(self):
        """Pick the next pending item and start conversion."""
        next_item = next(
            (it for it in self._items if it.status == "pending"), None
        )
        if next_item is None:
            self._running = False
            self._current_item = None
            self._emit_finished()
            return

        self._current_item = next_item
        next_item.status = "running"
        self.item_status_changed.emit(next_item.id, "running")

        # Merge global config with per-song overrides
        merged = {**self._global_config, **next_item.settings_overrides}
        next_item.resolved_config = dict(merged)

        # Ensure per-provider API keys are available for build_args
        provider_id = merged.get("llm_provider_id", "")
        if provider_id and f"llm_api_key_{provider_id}" not in merged:
            try:
                from .secrets import get_secret
                key = get_secret(f"llm_api_key_{provider_id}", merged)
                if key:
                    merged[f"llm_api_key_{provider_id}"] = key
            except ImportError:
                pass

        self.line_output.emit(
            f"[Queue] Starting: {next_item.title}"
        )
        self.line_output.emit("")

        # Try interceptor path for URL items with a captured audio stream
        if (
            next_item.input_type == "url"
            and next_item.video_id
            and self._media_interceptor is not None
        ):
            stream = self._media_interceptor.get_stream(next_item.video_id)
            if stream and not stream.is_expired:
                self.line_output.emit(
                    f"[Queue] Browser audio stream available "
                    f"(expires in {stream.seconds_until_expiry:.0f}s)"
                )
                self._start_intercepted_download(next_item, merged, stream)
                return

        # Fallback: standard yt-dlp path
        args = self._runner.build_args(merged, next_item.input_source)

        # Auto-export cookies if available
        cookie_file = merged.get("cookie_file", "")
        if cookie_file and next_item.input_type == "url":
            self.line_output.emit(
                f"[Queue] Cookie file: {cookie_file}"
            )

        self._runner.start(args)

    def _start_intercepted_download(
        self, item: QueueItem, merged: dict, stream,
    ):
        """Pre-download audio via ffmpeg, then run UltraSinger with local file.

        This bypasses yt-dlp entirely — the URL was captured from the
        real Chromium browser, so YouTube sees it as normal playback.
        """
        from .media_downloader import start_media_download

        # Choose file extension from stream mime type
        _MIME_EXT = {"audio/webm": ".webm", "audio/mp4": ".m4a"}
        ext = _MIME_EXT.get(stream.mime_type, ".webm")

        # Always download into the configured output folder so UltraSinger
        # derives the correct output directory from the local file path.
        output_dir = merged.get("output_folder", "") or tempfile.gettempdir()
        audio_path = str(
            Path(output_dir) / f"_intercepted_{item.video_id}{ext}"
        )

        thread, worker = start_media_download(stream.url, audio_path, self)
        self._download_thread = thread
        self._download_worker = worker

        worker.progress.connect(self.line_output.emit)
        worker.finished.connect(
            lambda ok, path, err: self._on_predownload_finished(
                ok, path, err, item, merged
            )
        )

        thread.start()

    def _on_predownload_finished(
        self,
        success: bool,
        audio_path: str,
        error_msg: str,
        item: QueueItem,
        merged: dict,
    ):
        """Handle completion of the intercepted audio pre-download."""
        self._download_thread = None
        self._download_worker = None

        # Bail out if item was cancelled while downloading
        if item.status == "cancelled":
            try:
                Path(audio_path).unlink(missing_ok=True)
            except OSError:
                pass
            return

        if success and Path(audio_path).exists():
            self.line_output.emit(
                "[Queue] Using browser-intercepted audio (yt-dlp bypassed)"
            )
            # Run UltraSinger with the local audio file as input, but pass
            # the original YouTube URL so metadata (title, artist, thumbnail)
            # can still be fetched via yt-dlp extract_info(download=False).
            merged["youtube_url"] = item.input_source
            args = self._runner.build_args(merged, audio_path)
            self._runner.start(args)
        else:
            # Fallback to yt-dlp
            self.line_output.emit(
                f"[Queue] Browser download failed: {error_msg}"
            )
            self.line_output.emit(
                "[Queue] Falling back to yt-dlp..."
            )
            # Clean up failed download
            try:
                Path(audio_path).unlink(missing_ok=True)
            except OSError:
                pass

            args = self._runner.build_args(merged, item.input_source)
            cookie_file = merged.get("cookie_file", "")
            if cookie_file and item.input_type == "url":
                self.line_output.emit(f"[Queue] Cookie file: {cookie_file}")
            self._runner.start(args)

    def _parse_output_line(self, line: str):
        """Extract pipeline metadata from UltraSinger stdout lines."""
        item = self._current_item
        if item is None:
            return

        info = item.result_info

        # Strip ANSI escape codes for regex matching
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)

        # Language detection (fast Whisper tiny, full Whisper, YouTube metadata, or --language)
        lang_pat = r"([A-Za-z0-9_-]+)"
        m = (re.search(rf"Language detected:\s*{lang_pat}", clean)
             or re.search(rf"Detected language:\s*{lang_pat}", clean)
             or re.search(rf"Using YouTube language metadata:\s*{lang_pat}", clean)
             or re.search(rf"Language set:\s*{lang_pat}", clean))
        if m:
            new_lang = m.group(1)
            if "language" in info and info["language"] != new_lang:
                info["language_changed"] = True
                info.setdefault("initial_language", info["language"])
            info["language"] = new_lang
            return

        # Low-confidence fallback to English (non-core language detected)
        m = re.search(rf"falling back to\s+{lang_pat}", clean)
        if m and "confidence" in clean.lower():
            new_lang = m.group(1)
            if "language" in info and info["language"] != new_lang:
                info["language_changed"] = True
                info.setdefault("initial_language", info["language"])
            info["language"] = new_lang
            return

        # LRCLIB lyrics found — parse what types are available
        m = re.search(r"Found lyrics on LRCLIB:.*\[(.*?)\]", clean)
        if m:
            details = m.group(1).lower()
            if "synced" in details:
                info["lrclib_result"] = "synced"
            elif "plain" in details:
                info["lrclib_result"] = "plain"
            elif "instrumental" in details:
                info["lrclib_result"] = "instrumental"
            else:
                info["lrclib_result"] = "found"
            return

        # No lyrics on LRCLIB
        if "No lyrics found on LRCLIB" in clean:
            info["lrclib_result"] = "none"
            return

        # Synced lyrics found — skipping Whisper (reference pipeline)
        if "Synced lyrics found" in clean and "skipping Whisper" in clean:
            info["whisper_skipped"] = True
            return

        # Reference-first pipeline produced results
        if "Reference-first pipeline:" in clean:
            info["pipeline"] = "reference"
            return

        # Reference pipeline recovered after language correction
        if "Reference pipeline recovered" in clean:
            info["pipeline"] = "reference"
            info["reference_recovered"] = True
            info.pop("whisper_fallback", None)
            return

        # Reference pipeline fell back to Whisper
        if "Falling back to standard pipeline" in clean:
            info["whisper_fallback"] = True
            info["pipeline"] = "whisper"
            return

        # Output folder — derive from "Creating UltraStar file <path>".
        m = re.search(r"Creating UltraStar file\s+(.+)", clean)
        if m:
            try:
                raw_path = m.group(1).strip()
                path_obj = (
                    PureWindowsPath(raw_path)
                    if "\\" in raw_path or re.match(r"^[A-Za-z]:", raw_path)
                    else PurePosixPath(raw_path)
                )
                info["output_folder"] = str(path_obj.parent)
            except (ValueError, OSError):
                pass
            return

        # Output folder fallback — from create_folder() calls.
        m = re.search(r"Creating output folder\.\s*->\s*(.+)", clean)
        if m:
            folder = m.group(1).strip()
            if not folder.endswith(("cache", "cache/", "cache\\")):
                info["output_folder"] = folder
            return

    def _on_item_finished(self, exit_code: int):
        """Handle completion of a single item, then run next."""
        item = self._current_item
        if item is None:
            return

        item.exit_code = exit_code
        if exit_code == 0:
            item.status = "done"
        elif exit_code == -2:
            item.status = "cancelled"
        else:
            item.status = "failed"

        # Derive final lyrics_source from collected info
        info = item.result_info
        if exit_code == 0:
            if info.get("pipeline") == "reference":
                info["lyrics_source"] = "synced"
            elif info.get("lrclib_result") == "synced" and info.get("whisper_fallback"):
                info["lyrics_source"] = "synced (fallback)"
                if info.get("language_changed"):
                    info["language_caused_fallback"] = True
            elif info.get("lrclib_result") in ("plain", "found"):
                info["lyrics_source"] = "plain"
            elif info.get("lrclib_result") == "none":
                info["lyrics_source"] = "transcribed"
                info.setdefault("pipeline", "whisper")
            elif "pipeline" not in info and "lrclib_result" not in info:
                info["lyrics_source"] = "transcribed"
                info["pipeline"] = "whisper"

        if exit_code == 0 and info:
            self.item_result_info.emit(item.id, dict(info))

        self.item_status_changed.emit(item.id, item.status)
        self.line_output.emit("")

        status_text = {
            "done": "completed",
            "cancelled": "cancelled",
            "failed": f"failed (exit code {exit_code})",
        }.get(item.status, item.status)
        self.line_output.emit(
            f"[Queue] {item.title}: {status_text}"
        )
        self.line_output.emit("")

        # If cancelled, don't continue the queue
        if item.status == "cancelled":
            # Cancel remaining
            for it in self._items:
                if it.status == "pending":
                    it.status = "cancelled"
                    self.item_status_changed.emit(it.id, "cancelled")
            self._running = False
            self._current_item = None
            self._emit_finished(cancelled=True)
            return

        # Run next pending item
        self._run_next()
