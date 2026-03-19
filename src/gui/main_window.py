"""Main application window with sidebar navigation."""

import importlib.metadata
import logging

from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from .browser_tab import BrowserTab
from .config import load_config, save_config
from .preferences_tab import PreferencesTab
from .queue_tab import QueueTab
from .settings_tab import SettingsTab
from .widgets.sidebar import Sidebar

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main UltraSinger GUI window with sidebar navigation.

    Input flow:
      - Drop a file on the sidebar drop zone -> Convert button appears -> click to start
      - Browse YouTube -> click Convert overlay -> Convert button appears -> click to start
      - Both flows: settings are read from the Settings tab, conversion runs in Queue tab
    """

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

        # Sidebar (with drop zone and convert button)
        self._sidebar = Sidebar()
        self._sidebar.add_section("\U0001F310", "YouTube")
        self._sidebar.add_section("\u2699\uFE0F", "Settings")
        self._sidebar.add_section("\U0001F3B5", "Queue")
        self._sidebar.add_section("\U0001F527", "Preferences")
        self._sidebar.finalize()
        main_layout.addWidget(self._sidebar)

        # Content stack
        self._stack = QStackedWidget()
        self._stack.setObjectName("contentArea")
        main_layout.addWidget(self._stack, 1)

        # Create tabs
        self._browser_tab = BrowserTab()
        self._settings_tab = SettingsTab(self._config)
        self._queue_tab = QueueTab()
        self._preferences_tab = PreferencesTab(
            self._config, self._browser_tab.cookie_manager
        )

        self._stack.addWidget(self._browser_tab)
        self._stack.addWidget(self._settings_tab)
        self._stack.addWidget(self._queue_tab)
        self._stack.addWidget(self._preferences_tab)

        # Wire sidebar navigation
        self._sidebar.section_changed.connect(self._stack.setCurrentIndex)

        # Wire Convert flows
        self._browser_tab.convert_requested.connect(self._on_convert_from_browser)
        self._sidebar.convert_requested.connect(self._on_start_conversion)

    def _on_convert_from_browser(self, url: str):
        """Handle Convert button click from YouTube browser.

        Sets the URL as input in the sidebar. The user can then click
        the sidebar Convert button to start, or adjust settings first.
        """
        logger.info("Convert requested for URL: %s", url)
        self._sidebar.set_youtube_input(url)

        # Auto-export cookies
        cookie_path = self._config.get("cookie_file", "")
        if cookie_path and self._browser_tab.cookie_manager.has_youtube_cookies:
            try:
                self._browser_tab.cookie_manager.export_netscape(cookie_path)
                logger.info("Cookies exported to %s", cookie_path)
            except OSError:
                logger.warning("Failed to export cookies to %s", cookie_path, exc_info=True)

    def _on_start_conversion(self):
        """Start the conversion with current input and settings."""
        input_source = self._sidebar.get_input_source()
        if not input_source:
            self._queue_tab.append_log(
                "[Error] No input source selected. "
                "Drop a file or select a YouTube video first."
            )
            self._sidebar.set_active(2)
            return

        # Collect all settings
        settings = self._settings_tab.collect_config()
        prefs = self._preferences_tab.collect_preferences()
        merged = {**self._config, **settings, **prefs}

        # Save config for next time
        self._config.update(merged)
        save_config(self._config)

        # Auto-export cookies if available
        cookie_file = merged.get("cookie_file", "")
        if cookie_file and self._browser_tab.cookie_manager.has_youtube_cookies:
            try:
                self._browser_tab.cookie_manager.export_netscape(cookie_file)
            except OSError:
                logger.warning("Failed to export cookies to %s", cookie_file, exc_info=True)

        # Build CLI args
        runner = self._queue_tab.runner
        args = runner.build_args(merged, input_source)

        # Switch to queue tab and start
        self._sidebar.set_active(2)
        self._queue_tab.start_conversion(args, merged.get("output_folder", ""))

    def closeEvent(self, event):
        """Save configuration on close."""
        try:
            settings = self._settings_tab.collect_config()
            prefs = self._preferences_tab.collect_preferences()
            self._config.update(settings)
            self._config.update(prefs)
            save_config(self._config)
        except (OSError, ValueError, TypeError):
            logger.warning("Failed to save config on close", exc_info=True)
        super().closeEvent(event)
