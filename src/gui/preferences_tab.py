"""Application preferences and configuration tab."""

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .widgets import AnimatedButton, SettingsCard, ToggleSwitch

logger = logging.getLogger(__name__)


class PreferencesTab(QWidget):
    """App preferences: output folder, LLM config, cookie management."""

    def __init__(self, config: dict, cookie_manager=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._cookie_manager = cookie_manager

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        header = QLabel("Preferences")
        header.setObjectName("sectionHeader")
        main_layout.addWidget(header)

        # ── Output Folder ────────────────────────────────────────────────
        card = SettingsCard("Default Output Folder")
        row = QHBoxLayout()
        self._output_folder = QLineEdit()
        self._output_folder.setText(config.get("output_folder", ""))
        self._output_folder.setPlaceholderText("Select default output folder...")
        row.addWidget(self._output_folder, 1)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_output)
        row.addWidget(browse)
        card.add_layout(row)
        main_layout.addWidget(card)

        # ── LLM Configuration ────────────────────────────────────────────
        llm_card = SettingsCard("LLM Lyric Correction (Groq)")

        llm_card.add_info(
            "UltraSinger can use an LLM API to post-correct Whisper transcription errors. "
            "This is optional and disabled by default.\n\n"
            "Recommended setup (free):\n"
            "1. Create a free account at https://console.groq.com\n"
            "2. Go to API Keys and create a new key\n"
            "3. Paste the key below\n"
            "4. Model: qwen/qwen3-32b (best quality, 0 degradations in testing)"
        )

        # Show keyring storage status
        try:
            from .secrets import get_keyring_backend_name, is_keyring_available

            if is_keyring_available():
                llm_card.add_info(
                    f"\U0001F512 API key stored securely in: {get_keyring_backend_name()}"
                )
            else:
                llm_card.add_info(
                    "\u26A0 No system keyring available. Set ULTRASINGER_LLM_API_KEY "
                    "environment variable as a secure alternative."
                )
        except ImportError:
            pass

        self._llm_url = QLineEdit()
        self._llm_url.setText(config.get("llm_api_base_url", "https://api.groq.com/openai/v1"))
        self._llm_url.setPlaceholderText("https://api.groq.com/openai/v1")
        llm_card.add_row("API Base URL", self._llm_url)

        # API key with visibility toggle
        key_row = QHBoxLayout()
        self._llm_key = QLineEdit()
        self._llm_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._llm_key.setText(config.get("llm_api_key", ""))
        self._llm_key.setPlaceholderText("gsk_...")
        key_row.addWidget(self._llm_key, 1)

        self._key_visible_btn = QPushButton("\U0001F441")
        self._key_visible_btn.setObjectName("ghostButton")
        self._key_visible_btn.setFixedWidth(36)
        self._key_visible_btn.setToolTip("Show/hide API key")
        self._key_visible_btn.clicked.connect(self._toggle_key_visibility)
        key_row.addWidget(self._key_visible_btn)

        # Add labeled key row with visibility toggle
        labeled_key_row = QHBoxLayout()
        labeled_key_row.setSpacing(12)
        key_label = QLabel("API Key")
        key_label.setMinimumWidth(180)
        labeled_key_row.addWidget(key_label)
        labeled_key_row.addLayout(key_row, 1)
        llm_card.add_layout(labeled_key_row)

        self._llm_model_pref = QLineEdit()
        self._llm_model_pref.setText(config.get("llm_model", "qwen/qwen3-32b"))
        llm_card.add_row("Default Model", self._llm_model_pref)

        llm_card.add_info(
            "Groq Free Plan limits (as of March 2026): "
            "1,000 requests/day, 500K tokens/day for qwen3-32b. "
            "This is typically enough for ~200 songs/day."
        )

        main_layout.addWidget(llm_card)

        # ── Cookie Management ─────────────────────────────────────────────
        cookie_card = SettingsCard("Cookie Management")

        if cookie_manager:
            self._cookie_status = QLabel("Checking...")
            cookie_card.add_widget(self._cookie_status)
            cookie_manager.cookies_changed.connect(self._update_cookie_status)
            self._update_cookie_status()

        cookie_card.add_info(
            "Cookies are used for YouTube Premium authentication. "
            "Log in via the YouTube browser tab to capture cookies automatically. "
            "They are exported in Netscape format for yt-dlp."
        )

        cookie_path_row = QHBoxLayout()
        self._cookie_path = QLineEdit()
        self._cookie_path.setText(config.get("cookie_file", ""))
        self._cookie_path.setPlaceholderText("Cookie file path...")
        self._cookie_path.setReadOnly(True)
        cookie_path_row.addWidget(self._cookie_path, 1)
        cookie_card.add_layout(cookie_path_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        export_btn = QPushButton("Export Cookies")
        export_btn.clicked.connect(self._export_cookies)
        export_btn.setEnabled(bool(cookie_manager))
        btn_row.addWidget(export_btn)

        clear_btn = QPushButton("Clear Cookies")
        clear_btn.setObjectName("dangerButton")
        clear_btn.clicked.connect(self._clear_cookies)
        clear_btn.setEnabled(bool(cookie_manager))
        btn_row.addWidget(clear_btn)

        btn_row.addStretch()
        cookie_card.add_layout(btn_row)

        main_layout.addWidget(cookie_card)

        # ── Save Button ───────────────────────────────────────────────────
        main_layout.addSpacing(8)
        save_btn = AnimatedButton("Save Preferences")
        save_btn.clicked.connect(self._save)
        main_layout.addWidget(save_btn)

        main_layout.addStretch(1)

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self._output_folder.setText(path)

    def _toggle_key_visibility(self):
        if self._llm_key.echoMode() == QLineEdit.EchoMode.Password:
            self._llm_key.setEchoMode(QLineEdit.EchoMode.Normal)
            self._key_visible_btn.setText("\U0001F441\u200D\U0001F5E8")
        else:
            self._llm_key.setEchoMode(QLineEdit.EchoMode.Password)
            self._key_visible_btn.setText("\U0001F441")

    def _update_cookie_status(self):
        if not hasattr(self, "_cookie_status"):
            return
        if self._cookie_manager and self._cookie_manager.has_youtube_cookies:
            count = self._cookie_manager.youtube_cookie_count
            self._cookie_status.setText(
                f"\u2705 {count} YouTube cookies captured"
            )
            self._cookie_status.setStyleSheet("color: #4caf50;")
        else:
            self._cookie_status.setText("\u26A0 No YouTube cookies (log in via browser tab)")
            self._cookie_status.setStyleSheet("color: #ffa726;")

    def _export_cookies(self):
        if self._cookie_manager:
            # Prefer UI path, fall back to config
            cookie_path = self._cookie_path.text() or self._config.get("cookie_file", "")
            if cookie_path:
                try:
                    exported = self._cookie_manager.export_netscape(cookie_path)
                    self._cookie_path.setText(str(exported))
                except OSError as e:
                    logger.error(
                        "Failed to export cookies to %s: %s", cookie_path, e,
                        exc_info=True,
                    )

    def _clear_cookies(self):
        if self._cookie_manager:
            self._cookie_manager.clear_all()

    def _save(self):
        """Save current preferences to config."""
        from .config import save_config

        self._config.update(self.collect_preferences())
        save_config(self._config)

    def collect_preferences(self) -> dict:
        """Return current preference values."""
        return {
            "output_folder": self._output_folder.text(),
            "llm_api_base_url": self._llm_url.text(),
            "llm_api_key": self._llm_key.text(),
            "llm_model": self._llm_model_pref.text(),
            "cookie_file": self._cookie_path.text(),
        }
