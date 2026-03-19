"""Unified Settings tab: output, conversion defaults, LLM providers, cookies.

This replaces the old separate Settings + Preferences tabs.
"""

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

from .models import LLMProvider
from .settings_tab import ConversionSettingsForm
from .widgets import AnimatedButton, LLMProviderListWidget, SettingsCard

logger = logging.getLogger(__name__)


class PreferencesTab(QWidget):
    """Unified settings: output folder, conversion defaults, LLM providers, cookies."""

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

        header = QLabel("Settings")
        header.setObjectName("sectionHeader")
        main_layout.addWidget(header)

        # ── Output Folder ────────────────────────────────────────────────
        card = SettingsCard("Output Folder")
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

        # ── Conversion Defaults (embedded form) ──────────────────────────
        defaults_header = QLabel("Default Conversion Settings")
        defaults_header.setObjectName("subsectionHeader")
        defaults_header.setStyleSheet("padding-top: 8px;")
        main_layout.addWidget(defaults_header)

        self._conversion_form = ConversionSettingsForm(config)
        main_layout.addWidget(self._conversion_form)

        # ── LLM Providers ────────────────────────────────────────────────
        llm_card = SettingsCard("LLM Providers")

        llm_card.add_info(
            "Configure one or more LLM API providers for lyric correction. "
            "Each provider has its own URL, API key, and default model.\n\n"
            "Recommended: Groq with qwen/qwen3-32b (free plan, best quality)."
        )

        # Show keyring storage status
        try:
            from .secrets import get_keyring_backend_name, is_keyring_available

            if is_keyring_available():
                llm_card.add_info(
                    f"\U0001F512 API keys stored securely in: {get_keyring_backend_name()}"
                )
            else:
                llm_card.add_info(
                    "\u26A0 No system keyring available. Set ULTRASINGER_LLM_API_KEY "
                    "environment variable as a secure alternative."
                )
        except ImportError:
            pass

        self._llm_provider_list = LLMProviderListWidget()
        llm_card.add_widget(self._llm_provider_list)

        # Load existing providers
        self._load_llm_providers()

        # Keep the conversion form's provider selector in sync
        self._llm_provider_list.providers_changed.connect(
            self._sync_provider_selector
        )
        self._sync_provider_selector()

        main_layout.addWidget(llm_card)

        # ── Cookie Management ────────────────────────────────────────────
        cookie_card = SettingsCard("Cookie Management")

        if cookie_manager:
            self._cookie_status = QLabel("Checking...")
            cookie_card.add_widget(self._cookie_status)
            cookie_manager.cookies_changed.connect(self._update_cookie_status)
            self._update_cookie_status()

        cookie_card.add_info(
            "Cookies are used for premium account authentication. "
            "Log in via the Video browser tab to capture cookies automatically. "
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

        # ── Save Button ──────────────────────────────────────────────────
        main_layout.addSpacing(8)
        save_btn = AnimatedButton("Save Settings")
        save_btn.clicked.connect(self._save)
        main_layout.addWidget(save_btn)

        main_layout.addStretch(1)

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── LLM Provider Management ──────────────────────────────────────────

    def _load_llm_providers(self):
        """Load providers from config and populate the list widget."""
        raw_providers = self._config.get("llm_providers", [])
        providers = []
        api_keys: dict[str, str] = {}
        for raw in raw_providers:
            if isinstance(raw, dict):
                p = LLMProvider.from_dict(raw)
                providers.append(p)
                # Load API key from config (was loaded from keyring by load_config)
                key = self._config.get(f"llm_api_key_{p.id}", "")
                if key:
                    api_keys[p.id] = key
        self._llm_provider_list.set_providers(providers, api_keys)

    def _sync_provider_selector(self):
        """Update the conversion form's LLM provider combobox."""
        providers = self._llm_provider_list.get_providers()
        self._conversion_form.set_llm_providers(providers)

    def get_llm_providers(self) -> list[LLMProvider]:
        """Return current LLM providers from the widget."""
        return self._llm_provider_list.get_providers()

    def get_llm_api_keys(self) -> dict[str, str]:
        """Return provider_id -> api_key mapping."""
        return self._llm_provider_list.get_api_keys()

    # ── Output / Cookie ──────────────────────────────────────────────────

    def _browse_output(self):
        start = self._output_folder.text() or str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            start,
            QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self._output_folder.setText(path)

    def _update_cookie_status(self):
        if not hasattr(self, "_cookie_status"):
            return
        if self._cookie_manager and self._cookie_manager.has_video_cookies:
            count = self._cookie_manager.video_cookie_count
            self._cookie_status.setText(
                f"\u2705 {count} cookies captured"
            )
            self._cookie_status.setStyleSheet("color: #4caf50;")
        else:
            self._cookie_status.setText("\u26A0 No cookies (log in via Video browser tab)")
            self._cookie_status.setStyleSheet("color: #ffa726;")

    def _export_cookies(self):
        if self._cookie_manager:
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

    # ── Save / Collect ───────────────────────────────────────────────────

    def _save(self):
        """Save current settings to config."""
        from .config import save_config

        self._config.update(self.collect_all())
        save_config(self._config)

    def collect_all(self) -> dict:
        """Return all settings: conversion + providers + output + cookies."""
        result = self._conversion_form.collect_config()
        result["output_folder"] = self._output_folder.text()
        result["cookie_file"] = self._cookie_path.text()

        # Serialize providers
        providers = self._llm_provider_list.get_providers()
        result["llm_providers"] = [p.to_dict() for p in providers]

        # Collect API keys (stored in keyring on save)
        api_keys = self._llm_provider_list.get_api_keys()
        for pid, key in api_keys.items():
            result[f"llm_api_key_{pid}"] = key

        return result

    # Backward-compatible alias used by main_window.py
    def collect_preferences(self) -> dict:
        """Alias for collect_all()."""
        return self.collect_all()
