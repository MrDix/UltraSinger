"""Main application window with sidebar navigation and conversion queue."""

import importlib.metadata
import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from .browser_tab import BrowserTab
from .config import load_config, save_config
from .models import LLMProvider
from .preferences_tab import PreferencesTab
from .queue_manager import QueueManager
from .queue_tab import QueueTab
from .settings_dialog import PerSongSettingsDialog
from .widgets.sidebar import Sidebar

logger = logging.getLogger(__name__)


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

    def __init__(self):
        super().__init__()
        try:
            _version = importlib.metadata.version("ultrasinger")
        except importlib.metadata.PackageNotFoundError:
            _version = "dev"
        self.setWindowTitle(f"UltraSinger v{_version}")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Load config
        self._config = load_config()

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
        main_layout.addWidget(self._stack, 1)

        # Create tabs (3 tabs: Video, Console, Settings)
        self._browser_tab = BrowserTab()
        self._queue_tab = QueueTab()
        self._settings_tab = PreferencesTab(
            self._config, self._browser_tab.cookie_manager
        )

        self._stack.addWidget(self._browser_tab)
        self._stack.addWidget(self._queue_tab)
        self._stack.addWidget(self._settings_tab)

        # Queue manager (owns the runner, drives batch execution)
        self._queue_mgr = QueueManager(self)

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

        # Wire Start All / Clear buttons
        self._sidebar.start_all_requested.connect(self._on_start_all)
        self._sidebar.clear_button.clicked.connect(self._on_clear_queue)

        # Wire queue manager → console tab
        self._queue_mgr.line_output.connect(self._queue_tab.append_log)
        self._queue_mgr.stage_changed.connect(self._queue_tab.on_stage_changed)
        self._queue_mgr.queue_started.connect(self._on_queue_started)
        self._queue_mgr.queue_finished.connect(self._on_queue_finished)
        self._queue_mgr.item_status_changed.connect(self._on_item_status_changed)

        # Wire console cancel → queue manager
        self._queue_tab.cancel_requested.connect(self._queue_mgr.cancel_all)

        # Initial button state
        self._update_queue_buttons()

    # ── Queue operations ───────────────────────────────────────────────

    def _on_add_from_browser(self, url: str, title: str):
        """Add a video URL to the queue (from browser Convert button)."""
        logger.info("Add to queue from browser: %s", url)
        self._auto_export_cookies()
        self._queue_mgr.add_item(url, "url", title)
        self._update_queue_buttons()

    def _on_add_from_file(self, path: str):
        """Add a local file to the queue (from drop zone)."""
        title = Path(path).stem
        logger.info("Add to queue from file: %s", title)
        self._queue_mgr.add_item(path, "file", title)
        self._update_queue_buttons()

    def _on_remove_from_queue(self, item_id: str):
        """Remove a pending item from the queue."""
        self._queue_mgr.remove_item(item_id)
        self._update_queue_buttons()

    def _on_clear_queue(self):
        """Clear completed and pending items from the queue."""
        self._queue_mgr.clear_completed()
        self._queue_mgr.clear_pending()
        self._update_queue_buttons()

    def _on_per_song_settings(self, item_id: str):
        """Open the per-song settings override dialog."""
        item = next(
            (it for it in self._queue_mgr.items if it.id == item_id), None
        )
        if item is None or item.status != "pending":
            return

        # Collect current global config from the settings tab
        global_config = {**self._config, **self._settings_tab.collect_all()}
        providers = self._settings_tab.get_llm_providers()

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

        # Switch to console tab
        self._sidebar.set_active(self._TAB_CONSOLE)
        self._queue_tab.on_queue_started()
        self._queue_mgr.start_all()

    def _on_queue_started(self):
        """Handle queue batch start."""
        self._update_queue_buttons()

    def _on_queue_finished(self):
        """Handle queue batch completion."""
        self._queue_tab.on_queue_finished()
        self._update_queue_buttons()

    def _on_item_status_changed(self, item_id: str, status: str):
        """Update queue buttons when item status changes."""
        self._update_queue_buttons()

    def _update_queue_buttons(self):
        """Sync Start All / Clear button states with queue state."""
        has_pending = self._queue_mgr.pending_count() > 0
        is_running = self._queue_mgr.is_running
        self._sidebar.update_queue_buttons(has_pending, is_running)

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
        """Save configuration on close."""
        try:
            all_settings = self._settings_tab.collect_all()
            self._config.update(all_settings)
            save_config(self._config)
        except (OSError, ValueError, TypeError):
            logger.warning("Failed to save config on close", exc_info=True)
        super().closeEvent(event)
