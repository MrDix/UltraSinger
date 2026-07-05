"""Main application window with sidebar navigation and conversion queue."""

import importlib.metadata
import logging
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QWidget,
)

from .browser_tab import BrowserTab
from .config import load_config, save_config
from .models import LLMProvider
from .preferences_tab import PreferencesTab
from .queue_manager import QueueManager
from .queue_tab import QueueTab
from .settings_dialog import PerSongSettingsDialog, ReadOnlySettingsDialog
from .widgets.sidebar import Sidebar

logger = logging.getLogger(__name__)


class _PotokenWorker(QObject):
    """Ensures the bgutil PO-token provider is available (off the GUI thread).

    Starting the provider server can take several seconds, so this runs in
    a QThread and reports the result via ``finished``.  ``result_status``
    is also stored directly so the close handler can read it even if the
    queued ``finished`` signal has not been delivered yet.
    """

    finished = Signal(object)  # ProviderStatus | None

    def __init__(self, base_url: str, auto_start_node: bool,
                 auto_start_docker: bool, cancel):
        super().__init__()
        self._base_url = base_url
        self._auto_start_node = auto_start_node
        self._auto_start_docker = auto_start_docker
        self._cancel = cancel
        self.result_status = None

    def run(self):
        try:
            from .potoken_provider import ensure_provider

            status = ensure_provider(
                self._base_url,
                auto_start_node=self._auto_start_node,
                auto_start_docker=self._auto_start_docker,
                cancel=self._cancel,
            )
        except Exception as e:  # noqa: BLE001 — fail open, never crash startup
            logger.warning("PO-token provider check failed: %s", e)
            status = None
        self.result_status = status
        self.finished.emit(status)


class _YtdlpUpdateCheckWorker(QObject):
    """Checks PyPI for a newer yt-dlp release (off the GUI thread).

    Mirrors ``_PotokenWorker``: the network round-trip can take a couple
    of seconds, so this runs in a QThread and reports the result via
    ``finished``.  ``result`` is also stored directly so the close handler
    can read it even if the queued ``finished`` signal has not been
    delivered yet.  Never raises — fails open with empty version strings.
    """

    finished = Signal(str, str)  # installed, latest ("" on lookup failure)

    def __init__(self, cancel):
        super().__init__()
        self._cancel = cancel
        self.result = ("", "")

    def run(self):
        from .ytdlp_updater import get_installed_version, get_latest_version

        installed = get_installed_version()
        latest = ""
        if not (self._cancel is not None and self._cancel.is_set()):
            try:
                latest = get_latest_version()
            except Exception as e:  # noqa: BLE001 — fail open, never crash startup
                logger.debug("yt-dlp update check failed: %s", e)
                latest = ""
        self.result = (installed, latest)
        self.finished.emit(installed, latest)


class MainWindow(QMainWindow):
    """Main UltraSinger GUI window with sidebar navigation.

    Three-tab layout:
      - Video: embedded browser for video platform navigation
      - Console: log output and batch progress
      - Settings: unified conversion defaults, LLM providers, cookies
    """

    # Tab indices
    _TAB_VIDEO = 0
    _TAB_CONSOLE = 1
    _TAB_SETTINGS = 2

    def __init__(self, config: dict | None = None):
        super().__init__()
        try:
            _version = importlib.metadata.version("ultrasinger")
        except importlib.metadata.PackageNotFoundError:
            _version = "dev"
        self.setWindowTitle(f"UltraSinger v{_version}")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Load config. gui_main.py loads it earlier (to apply proxy settings
        # before any network activity) and passes it in here so it isn't read
        # from disk twice; standalone instantiation (e.g. tests) still works.
        self._config = config if config is not None else load_config()

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar (with drop zone and queue list)
        self._sidebar = Sidebar()
        self._sidebar.add_section("\U0001F310", "Video")
        self._sidebar.add_section("\U0001F4BB", "Console")
        self._sidebar.add_section("\u2699\uFE0F", "Settings")
        self._sidebar.finalize()
        main_layout.addWidget(self._sidebar)

        # Content stack
        self._stack = QStackedWidget()
        self._stack.setObjectName("contentArea")
        self._stack.setFrameShape(QFrame.Shape.NoFrame)
        main_layout.addWidget(self._stack, 1)

        # Create tabs (3 tabs: Video, Console, Settings)
        self._browser_tab = BrowserTab()
        self._browser_tab.probe_cookie_file = self._config.get("cookie_file", "")
        self._queue_tab = QueueTab()
        self._settings_tab = PreferencesTab(
            self._config, self._browser_tab.cookie_manager
        )
        self._settings_tab.log_message.connect(self._queue_tab.append_log)

        self._stack.addWidget(self._browser_tab)
        self._stack.addWidget(self._queue_tab)
        self._stack.addWidget(self._settings_tab)

        # Queue manager (owns the runner, drives batch execution)
        self._queue_mgr = QueueManager(self)
        self._queue_mgr.set_media_interceptor(
            self._browser_tab.media_interceptor
        )

        # Wire sidebar navigation
        self._sidebar.section_changed.connect(self._stack.setCurrentIndex)

        # Wire queue add flows
        self._browser_tab.convert_requested.connect(self._on_add_from_browser)
        self._sidebar.file_dropped.connect(self._on_add_from_file)

        # Wire queue list ↔ queue manager
        self._queue_mgr.item_added.connect(self._sidebar.queue_list.add_item)
        self._queue_mgr.item_removed.connect(self._sidebar.queue_list.remove_item)
        self._queue_mgr.item_status_changed.connect(
            self._sidebar.queue_list.update_status
        )
        self._sidebar.queue_list.remove_requested.connect(
            self._on_remove_from_queue
        )
        # Wire per-song settings gear icon
        self._sidebar.queue_list.settings_requested.connect(
            self._on_per_song_settings
        )
        # Wire re-queue button
        self._sidebar.queue_list.requeue_requested.connect(
            self._on_requeue_item
        )
        # Wire clone button
        self._sidebar.queue_list.clone_requested.connect(
            self._on_clone_item
        )

        # Wire Start All / Clear buttons
        self._sidebar.start_all_requested.connect(self._on_start_all)
        self._sidebar.clear_button.clicked.connect(self._on_clear_queue)

        # Wire queue manager → console tab
        self._queue_mgr.line_output.connect(self._queue_tab.append_log)
        self._queue_mgr.line_output.connect(self._check_for_bot_detection)
        self._queue_mgr.stage_changed.connect(self._queue_tab.on_stage_changed)
        self._queue_mgr.queue_started.connect(self._on_queue_started)
        self._queue_mgr.queue_finished.connect(self._on_queue_finished)
        self._queue_mgr.item_status_changed.connect(self._on_item_status_changed)
        self._queue_mgr.item_result_info.connect(
            self._sidebar.queue_list.set_result_info
        )

        # Wire console cancel → queue manager
        self._queue_tab.cancel_requested.connect(self._queue_mgr.cancel_all)

        # Initial button state
        self._update_queue_buttons()

        # Start / detect the PO-token provider in the background so video
        # downloads get full-quality formats (SABR delivery otherwise limits
        # us to 360p / HTTP 403).
        self._potoken_status = None
        self._potoken_thread = None
        self._potoken_worker = None
        self._potoken_cancel = None
        self._closing = False
        self._start_potoken_provider()

        # Check PyPI for a newer yt-dlp release in the background. This is
        # a check only — never an automatic update — the user updates via
        # the "Update yt-dlp" button in the Settings tab.
        self._ytdlp_check_thread = None
        self._ytdlp_check_worker = None
        self._ytdlp_check_cancel = None
        self._start_ytdlp_update_check()

    # ── PO-token provider ─────────────────────────────────────────────

    def _start_potoken_provider(self):
        """Ensure the bgutil PO-token provider is running (background)."""
        if not self._config.get("potoken_auto_start", True):
            return
        import threading

        from .potoken_provider import DEFAULT_BASE_URL

        base_url = self._config.get("potoken_base_url") or DEFAULT_BASE_URL
        self._potoken_cancel = threading.Event()
        self._potoken_thread = QThread(self)
        self._potoken_worker = _PotokenWorker(
            base_url,
            self._config.get("potoken_auto_start_node", True),
            self._config.get("potoken_auto_start_docker", True),
            self._potoken_cancel,
        )
        self._potoken_worker.moveToThread(self._potoken_thread)
        self._potoken_thread.started.connect(self._potoken_worker.run)
        self._potoken_worker.finished.connect(self._on_potoken_ready)
        self._potoken_worker.finished.connect(self._potoken_thread.quit)
        self._potoken_thread.start()

    def _on_potoken_ready(self, status):
        """Report PO-token provider status and unlock the Queue button."""
        self._potoken_status = status
        if status is None or self._closing:
            return
        prefix = "[PO-Token] " if status.running else "[PO-Token] WARNING: "
        self._queue_tab.append_log(prefix + status.detail)
        if status.running:
            # With the provider up, yt-dlp downloads any video in full
            # quality — enable the browser Queue button globally.
            self._browser_tab.set_provider_ready(True)
            self._queue_mgr.set_potoken_available(True)

    def _shutdown_potoken_provider(self):
        """Cancel a pending provider start and stop one we launched."""
        self._closing = True
        thread = self._potoken_thread
        if thread is not None and thread.isRunning():
            # Signal the worker to abort its wait/start loop, then join it
            # so the QThread is never destroyed while still running.
            if self._potoken_cancel is not None:
                self._potoken_cancel.set()
            thread.quit()
            if not thread.wait(8000):
                logger.warning("PO-token provider thread did not stop in time")

        # Prefer the status delivered via signal; fall back to the value the
        # worker stored directly (the queued signal may not have run yet).
        status = self._potoken_status
        if status is None and self._potoken_worker is not None:
            status = self._potoken_worker.result_status
        if status is not None:
            try:
                from .potoken_provider import stop_provider_if_started

                stop_provider_if_started(status)
            except Exception:  # noqa: BLE001 — never block shutdown
                logger.debug("PO-token provider cleanup failed", exc_info=True)

    # ── yt-dlp update check ───────────────────────────────────────────

    def _start_ytdlp_update_check(self):
        """Check PyPI for a newer yt-dlp release (background, non-blocking)."""
        import threading

        self._ytdlp_check_cancel = threading.Event()
        self._ytdlp_check_thread = QThread(self)
        self._ytdlp_check_worker = _YtdlpUpdateCheckWorker(self._ytdlp_check_cancel)
        self._ytdlp_check_worker.moveToThread(self._ytdlp_check_thread)
        self._ytdlp_check_thread.started.connect(self._ytdlp_check_worker.run)
        self._ytdlp_check_worker.finished.connect(self._on_ytdlp_update_checked)
        self._ytdlp_check_worker.finished.connect(self._ytdlp_check_thread.quit)
        self._ytdlp_check_thread.start()

    def _on_ytdlp_update_checked(self, installed: str, latest: str):
        """Report the yt-dlp update check result."""
        if self._closing:
            return
        from .ytdlp_updater import is_outdated

        if hasattr(self, "_settings_tab"):
            self._settings_tab.set_ytdlp_update_info(installed, latest)

        if is_outdated(installed, latest):
            self._queue_tab.append_log(
                f"[yt-dlp] Update verfügbar: {installed} -> {latest}. "
                "Klicke 'Update yt-dlp' im Settings-Tab."
            )
        else:
            logger.debug(
                "yt-dlp is up to date (installed=%r, latest=%r)", installed, latest
            )

    def _shutdown_ytdlp_update_check(self):
        """Cancel a pending update-check start and join its thread."""
        thread = self._ytdlp_check_thread
        if thread is not None and thread.isRunning():
            if self._ytdlp_check_cancel is not None:
                self._ytdlp_check_cancel.set()
            thread.quit()
            if not thread.wait(8000):
                logger.warning("yt-dlp update-check thread did not stop in time")

    # ── Queue operations ───────────────────────────────────────────────

    def _on_add_from_browser(self, url: str, title: str):
        """Add a video URL to the queue (from browser Convert button)."""
        logger.info("Add to queue from browser: %s", url)
        self._auto_export_cookies()
        item = self._queue_mgr.add_item(url, "url", title)

        # Extract video ID for interceptor lookup
        from urllib.parse import parse_qs, urlparse
        params = parse_qs(urlparse(url).query)
        video_id = params.get("v", [""])[0]
        # Store video platform language metadata as pipeline hint
        yt_lang = self._browser_tab.yt_language
        if yt_lang:
            item.yt_language = yt_lang

        if video_id:
            item.video_id = video_id
            # Check if we already have an intercepted stream and tell the
            # user right away — otherwise the queue will silently fall back
            # to yt-dlp (bot-detection risk) when the item starts.
            stream = self._browser_tab.media_interceptor.get_stream(video_id)
            if stream:
                logger.info(
                    "Intercepted audio available for %s (expires in %.0fs)",
                    video_id, stream.seconds_until_expiry,
                )
                self._queue_tab.append_log(
                    f"[Queue] Added: {title} - browser audio stream captured "
                    f"(expires in {stream.seconds_until_expiry:.0f}s)"
                )
            elif self._potoken_status is not None and self._potoken_status.running:
                # With the PO-token provider up, yt-dlp IS the intended
                # full-quality path — a missing browser capture is fine.
                self._queue_tab.append_log(
                    f"[Queue] Added: {title} - no browser stream captured; "
                    f"will download via yt-dlp with PO token (full quality)."
                )
            else:
                self._queue_tab.append_log(
                    f"[Queue] Added: {title} - WARNING: no browser audio "
                    f"stream captured yet and the PO-token provider is not "
                    f"running. Play the video for a few seconds in the "
                    f"browser tab, otherwise the download will use plain "
                    f"yt-dlp (bot-detection risk)."
                )
        else:
            self._queue_tab.append_log(
                f"[Queue] Added: {title} - no video ID found in URL, "
                f"download will use yt-dlp"
            )

        self._update_queue_buttons()

    def _on_add_from_file(self, path: str):
        """Add a local file to the queue (from drop zone).

        For UltraStar .txt files, validates that the referenced media
        file exists before queueing.
        """
        p = Path(path)
        if p.suffix.lower() == ".txt":
            title, error = _validate_ultrastar_txt(p)
            if error:
                self._queue_tab.append_log(f"[Error] {error}")
                self._sidebar.set_active(self._TAB_CONSOLE)
                return
        else:
            # Try to read metadata from audio/video file
            try:
                from modules.Audio.metadata_reader import read_media_metadata, format_display_title
                metadata = read_media_metadata(str(p))
                title = format_display_title(metadata, fallback=p.stem)
            except Exception as e:
                logger.debug("Failed to read metadata from %s: %s", p, e)
                title = p.stem

        logger.info("Add to queue from file: %s", title)
        self._queue_mgr.add_item(path, "file", title)
        self._update_queue_buttons()

    def _on_remove_from_queue(self, item_id: str):
        """Remove a pending item from the queue."""
        self._queue_mgr.remove_item(item_id)
        self._update_queue_buttons()

    def _on_clone_item(self, item_id: str):
        """Clone a pending item — duplicate it with same settings."""
        self._queue_mgr.clone_item(item_id)
        self._update_queue_buttons()

    def _on_clear_queue(self):
        """Clear completed and pending items from the queue."""
        self._queue_mgr.clear_completed()
        self._queue_mgr.clear_pending()
        self._update_queue_buttons()

    def _on_per_song_settings(self, item_id: str):
        """Open per-song settings (editable for pending, read-only otherwise)."""
        item = next(
            (it for it in self._queue_mgr.items if it.id == item_id), None
        )
        if item is None:
            return

        providers = self._settings_tab.get_llm_providers()

        # Non-pending items: read-only view of the resolved config
        if item.status != "pending":
            resolved = item.resolved_config
            if not resolved:
                # Fallback: show global config + overrides
                resolved = {**self._config, **item.settings_overrides}
            ReadOnlySettingsDialog(
                resolved_config=resolved,
                llm_providers=providers,
                title=item.title,
                parent=self,
            ).exec()
            return

        # Pending items: editable override dialog
        global_config = {**self._config, **self._settings_tab.collect_all()}

        dialog = PerSongSettingsDialog(
            global_config=global_config,
            overrides=item.settings_overrides,
            llm_providers=providers,
            title=item.title,
            parent=self,
        )

        if dialog.exec():
            item.settings_overrides = dialog.get_overrides()
            # Visual indicator: mark items with overrides
            has_overrides = bool(item.settings_overrides)
            self._sidebar.queue_list.set_has_overrides(item_id, has_overrides)
            logger.info(
                "Per-song overrides for '%s': %d keys",
                item.title, len(item.settings_overrides),
            )

    def _on_requeue_item(self, item_id: str):
        """Re-queue a completed item for re-conversion.

        Creates a new pending queue item with the same source, then opens
        the per-song settings dialog so the user can set overrides
        (e.g. manual language) before starting.
        """
        old_item = next(
            (it for it in self._queue_mgr.items if it.id == item_id), None
        )
        if old_item is None:
            return

        # Create a new pending item with the same source
        new_item = self._queue_mgr.add_item(
            old_item.input_source, old_item.input_type, old_item.title,
        )
        if old_item.video_id:
            new_item.video_id = old_item.video_id
        if old_item.yt_language:
            new_item.yt_language = old_item.yt_language

        self._update_queue_buttons()

        # Auto-open per-song settings so the user can adjust language etc.
        self._on_per_song_settings(new_item.id)

    def _on_start_all(self):
        """Start processing all pending queue items."""
        if self._queue_mgr.pending_count() == 0:
            self._queue_tab.append_log(
                "[Error] No items in queue. "
                "Drop a file or select a video first."
            )
            self._sidebar.set_active(self._TAB_CONSOLE)
            return

        # Collect and save current config from unified settings tab
        all_settings = self._settings_tab.collect_all()
        self._config.update(all_settings)
        save_config(self._config)

        # Auto-export cookies
        self._auto_export_cookies()

        # Set global config on queue manager and start
        self._queue_mgr.set_global_config(self._config)

        # Set output folder for the Open Folder button
        self._queue_tab.set_output_folder(self._config.get("output_folder", ""))

        # Switch to console tab
        self._sidebar.set_active(self._TAB_CONSOLE)
        self._queue_tab.on_queue_started()
        self._queue_mgr.start_all()

    def _on_queue_started(self):
        """Handle queue batch start."""
        self._bot_detection_handled = False
        self._update_queue_buttons()

    def _on_queue_finished(self, failed_count: int, cancelled: bool):
        """Handle queue batch completion."""
        self._queue_tab.on_queue_finished(failed_count, cancelled)
        self._update_queue_buttons()

    def _on_item_status_changed(self, item_id: str, status: str):
        """Update queue buttons when item status changes."""
        self._update_queue_buttons()

    def _update_queue_buttons(self):
        """Sync Start All / Clear button states with queue state."""
        has_pending = self._queue_mgr.pending_count() > 0
        is_running = self._queue_mgr.is_running
        self._sidebar.update_queue_buttons(has_pending, is_running)

    # ── Bot detection workaround ──────────────────────────────────────

    _bot_detection_handled = False

    def _check_for_bot_detection(self, line: str):
        """Detect yt-dlp bot-detection errors and trigger cookie reset.

        When the video platform returns a bot challenge, the exported
        cookies are stale.  We clear them, delete the cookie file, and
        ask the user to re-login in the embedded browser.

        We match on "not a bot" which is the stable, apostrophe-free
        tail of the platform's error message.
        """
        if "not a bot" not in line:
            return
        if self._bot_detection_handled:
            return
        self._bot_detection_handled = True

        logger.warning("Bot detection triggered — clearing cookies")

        # 1. Clear in-memory cookies (also wipes the QWebEngine cookie DB)
        self._browser_tab.cookie_manager.clear_all()

        # 2. Delete the exported Netscape cookie file
        cookie_path = self._config.get("cookie_file", "")
        if cookie_path:
            try:
                Path(cookie_path).unlink(missing_ok=True)
                logger.info("Deleted cookie file: %s", cookie_path)
            except OSError:
                logger.warning("Could not delete %s", cookie_path, exc_info=True)

        # 3. Reload the browser so the user sees a logged-out state
        self._browser_tab._view.reload()

        # 4. Switch to browser tab and inform the user
        self._sidebar.set_active(self._TAB_VIDEO)
        QMessageBox.information(
            self,
            "UltraSinger",
            "Die Video-Plattform hat eine Bot-Erkennung ausgelöst.\n\n"
            "Deine Cookies wurden gelöscht.\n"
            "Bitte melde dich im Browser erneut an\n"
            "und starte die Konvertierung danach neu.",
        )

    def _auto_export_cookies(self):
        """Export browser cookies to file if available."""
        cookie_path = self._config.get("cookie_file", "")
        if cookie_path and self._browser_tab.cookie_manager.has_video_cookies:
            try:
                self._browser_tab.cookie_manager.export_netscape(cookie_path)
                logger.info("Cookies exported to %s", cookie_path)
            except OSError:
                logger.warning(
                    "Failed to export cookies to %s", cookie_path,
                    exc_info=True,
                )

    def closeEvent(self, event):
        """Save configuration and shut down the browser cleanly on close.

        The browser must be shut down explicitly so Chromium's renderer
        subprocess can flush cookies and persistent storage to disk
        before the process exits.  Without this, the subprocess stays
        alive and locks the database files, preventing persistence.
        """
        try:
            all_settings = self._settings_tab.collect_all()
            self._config.update(all_settings)
            save_config(self._config)
        except (OSError, ValueError, TypeError):
            logger.warning("Failed to save config on close", exc_info=True)

        # Stop a pending provider start / the server we launched
        self._shutdown_potoken_provider()

        # Stop a pending yt-dlp update check
        self._shutdown_ytdlp_update_check()

        # Shut down the browser engine so Chromium can flush cookies
        self._browser_tab.shutdown()

        super().closeEvent(event)


def _validate_ultrastar_txt(txt_path: Path) -> tuple[str, str]:
    """Validate an UltraStar TXT file and extract its title.

    Returns (title, error_message).  If error_message is non-empty,
    the file should not be queued.
    """
    try:
        # Try UTF-8 first, fall back to latin-1
        try:
            lines = txt_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = txt_path.read_text(encoding="latin-1").splitlines()
    except OSError as e:
        return "", f"Cannot read {txt_path.name}: {e}"

    # Check it looks like an UltraStar file (has # tags)
    tags = {}
    for line in lines:
        if not line.startswith("#"):
            break
        if ":" in line:
            key, _, value = line[1:].partition(":")
            tags[key.strip().upper()] = value.strip()

    if not tags:
        return "", f"{txt_path.name} does not appear to be an UltraStar TXT file."

    # Extract title
    title = tags.get("TITLE", txt_path.stem)
    artist = tags.get("ARTIST", "")
    display_title = f"{artist} - {title}" if artist else title

    # Check for referenced media files relative to the TXT
    txt_dir = txt_path.parent
    media_keys = ["MP3", "AUDIO", "VIDEO"]
    found_media = False
    for key in media_keys:
        if key in tags:
            media_path = txt_dir / tags[key]
            if media_path.exists():
                found_media = True
            else:
                return "", (
                    f"{txt_path.name}: referenced {key} file "
                    f"'{tags[key]}' not found in {txt_dir}"
                )

    if not found_media:
        return "", (
            f"{txt_path.name}: no #MP3, #AUDIO, or #VIDEO tag found. "
            "UltraSinger needs a media file reference."
        )

    return display_title, ""
