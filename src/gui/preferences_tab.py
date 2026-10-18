"""Unified Settings tab: output, conversion defaults, LLM providers, cookies.

This replaces the old separate Settings + Preferences tabs.
"""

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .models import LLMProvider
from .settings_tab import ConversionSettingsForm, _NoScrollComboBox
from .widgets import AnimatedButton, LLMProviderListWidget, SettingsCard, ToggleSwitch

logger = logging.getLogger(__name__)

_SIGNIN_TOOLTIP = (
    "Off (default): the embedded browser does not open the video platform's sign-in "
    "pages, its account cookies are deleted, and downloads get no cookies. "
    "On: you can sign in, and the browser's cookies go to every download. The "
    "platform can restrict or lock accounts that are used for automated downloads: "
    "use at most a throwaway account, never your main account. Downloads work "
    "without signing in; only age-restricted or members-only videos need an account."
)

# The warning before signing in is allowed
SIGNIN_WARNING_TITLE = "Allow Signing In"
SIGNIN_WARNING_TEXT = "Signing in to the video platform puts the account at risk."
SIGNIN_WARNING_DETAILS = (
    "The platform can restrict or lock accounts that are used for automated "
    "downloads. While signing in is allowed, UltraSinger passes the browser's "
    "cookies, and with them the account, to every download.\n\n"
    "If you sign in anyway, use at most a throwaway account, never your main "
    "account.\n\n"
    "Downloads work without signing in. Only age-restricted or members-only "
    "videos need an account."
)
SIGNIN_CONFIRM_BUTTON = "I understand the risk, allow signing in"


class _YtdlpUpdateRunWorker(QObject):
    """Runs the yt-dlp update (``uv lock`` + ``uv sync``) off the GUI thread.

    Mirrors the ``_PotokenWorker`` / ``_YtdlpUpdateCheckWorker`` pattern in
    ``main_window.py``: the update can take a minute or two, so it must not
    block the GUI event loop.
    """

    finished = Signal(bool, str)  # ok, combined output

    def __init__(self, repo_root):
        super().__init__()
        self._repo_root = repo_root

    def run(self):
        from .ytdlp_updater import run_update

        try:
            ok, output = run_update(self._repo_root)
        except Exception as e:  # noqa: BLE001 — never crash the GUI thread
            logger.warning("yt-dlp update failed unexpectedly: %s", e, exc_info=True)
            ok, output = False, f"Unexpected error: {e}"
        self.finished.emit(ok, output)


class PreferencesTab(QWidget):
    """Unified settings: output folder, conversion defaults, LLM providers, cookies."""

    # Emitted so MainWindow can forward yt-dlp update messages to the
    # Console tab (this widget has no direct reference to it).
    log_message = Signal(str)
    # "Train..." next to the Segmentation Model; MainWindow opens the training window.
    training_requested = Signal()
    # "Allow signing in to the video platform" was switched (and saved); MainWindow applies it
    signin_allowed_changed = Signal(bool)

    def __init__(self, config: dict, cookie_manager=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._cookie_manager = cookie_manager
        self._signin_allowed = config.get("allow_platform_signin", False) is True

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
        output_header = QLabel("Output Folder")
        output_header.setObjectName("subsectionHeader")
        main_layout.addWidget(output_header)
        card = SettingsCard()
        row = QHBoxLayout()
        self._output_folder = QLineEdit()
        self._output_folder.setText(config.get("output_folder", ""))
        self._output_folder.setPlaceholderText("Select default output folder...")
        self._output_folder.setToolTip(
            "Where converted UltraStar files are saved. "
            "Each song gets its own subfolder with TXT, audio, and optional MIDI/plots."
        )
        row.addWidget(self._output_folder, 1)
        browse = QPushButton("Browse")
        browse.setToolTip("Open a folder picker to select the output directory.")
        browse.clicked.connect(self._browse_output)
        row.addWidget(browse)
        card.add_layout(row)
        main_layout.addWidget(card)

        # ── Conversion Defaults (embedded form) ──────────────────────────
        defaults_header = QLabel("Default Conversion Settings")
        defaults_header.setObjectName("subsectionHeader")
        main_layout.addWidget(defaults_header)

        self._conversion_form = ConversionSettingsForm(config, allow_training=True)
        self._conversion_form.training_requested.connect(self.training_requested.emit)
        main_layout.addWidget(self._conversion_form)

        # ── LLM Providers ────────────────────────────────────────────────
        llm_header = QLabel("LLM Providers")
        llm_header.setObjectName("subsectionHeader")
        main_layout.addWidget(llm_header)
        llm_card = SettingsCard()
        # Kept as attributes so the form's "Manage..." button can scroll here
        # (this section is easy to miss below the long conversion form).
        self._llm_section_header = llm_header
        self._llm_card = llm_card
        self._conversion_form.manage_providers_requested.connect(
            self._scroll_to_llm_providers
        )

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

        # ── Video Platform Sign-in (and the cookies) ─────────────────────
        cookie_header = QLabel("Video Platform Sign-in")
        cookie_header.setObjectName("subsectionHeader")
        main_layout.addWidget(cookie_header)
        self._signin_header = cookie_header
        cookie_card = SettingsCard()

        self._signin_switch = ToggleSwitch(checked=self._signin_allowed)
        cookie_card.add_toggle_row("Allow signing in to the video platform",
                                   self._signin_switch, _SIGNIN_TOOLTIP)
        self._signin_switch.toggled_signal.connect(self._on_signin_switch_toggled)

        cookie_card.add_info(
            "Downloads work without signing in; only age-restricted or members-only "
            "videos need an account. The video platform can restrict or lock accounts "
            "that are used for automated downloads, so signing in is off by default: "
            "the browser does not open the sign-in pages, account cookies are deleted, "
            "and no cookies go to downloads. If you allow it, use at most a throwaway "
            "account, never your main account."
        )

        if cookie_manager:
            self._cookie_status = QLabel("Checking...")
            cookie_card.add_widget(self._cookie_status)
            cookie_manager.cookies_changed.connect(self._update_cookie_status)
            self._update_cookie_status()

        cookie_card.add_info(
            "While signing in is allowed, the browser's cookies are exported in "
            "Netscape format to the file below and passed to yt-dlp. While it is "
            "off, nothing is exported and the file is deleted."
        )

        cookie_path_row = QHBoxLayout()
        self._cookie_path = QLineEdit()
        self._cookie_path.setText(config.get("cookie_file", ""))
        self._cookie_path.setPlaceholderText("Cookie file path...")
        self._cookie_path.setReadOnly(True)
        self._cookie_path.setToolTip(
            "Path to the exported Netscape cookie file. It is passed to yt-dlp "
            "only while signing in is allowed; while it is off, it is deleted."
        )
        cookie_path_row.addWidget(self._cookie_path, 1)
        cookie_card.add_layout(cookie_path_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        export_btn = QPushButton("Export Cookies")
        export_btn.setToolTip(
            "Save all browser cookies to the cookie file for yt-dlp. "
            "Only while signing in is allowed."
        )
        export_btn.clicked.connect(self._export_cookies)
        self._export_btn = export_btn
        self._update_export_enabled()
        btn_row.addWidget(export_btn)

        clear_btn = QPushButton("Clear Cookies")
        clear_btn.setObjectName("dangerButton")
        clear_btn.setToolTip(
            "Delete all cookies of the embedded browser, also those of a sign-in."
        )
        clear_btn.clicked.connect(self._clear_cookies)
        clear_btn.setEnabled(bool(cookie_manager))
        btn_row.addWidget(clear_btn)

        btn_row.addStretch()
        cookie_card.add_layout(btn_row)

        main_layout.addWidget(cookie_card)

        # ── Network Proxy ────────────────────────────────────────────────
        proxy_header = QLabel("Network Proxy")
        proxy_header.setObjectName("subsectionHeader")
        main_layout.addWidget(proxy_header)
        proxy_card = SettingsCard()

        self._proxy_mode = _NoScrollComboBox()
        self._proxy_mode.addItem("System / environment (default)", userData="system")
        self._proxy_mode.addItem("Manual", userData="manual")
        self._proxy_mode.addItem("No proxy", userData="none")
        configured_mode = config.get("proxy_mode", "system") or "system"
        mode_index = self._proxy_mode.findData(configured_mode)
        self._proxy_mode.setCurrentIndex(mode_index if mode_index >= 0 else 0)
        proxy_card.add_row(
            "Mode", self._proxy_mode,
            "How UltraSinger reaches the network. 'System / environment' uses "
            "your OS proxy settings or the HTTP_PROXY/HTTPS_PROXY environment "
            "variables. 'Manual' lets you enter a proxy host below. "
            "'No proxy' ignores any proxy configured elsewhere.",
        )

        self._proxy_url = QLineEdit()
        self._proxy_url.setText(config.get("proxy_url", ""))
        self._proxy_url.setPlaceholderText("http://proxy.company.com:3128")
        proxy_card.add_row(
            "Proxy URL", self._proxy_url,
            "HTTP(S) proxy address, used for both http:// and https:// "
            "traffic. Only applied in 'Manual' mode.",
        )

        self._proxy_no_proxy = QLineEdit()
        self._proxy_no_proxy.setText(config.get("proxy_no_proxy", ""))
        self._proxy_no_proxy.setPlaceholderText(".company.com,10.0.0.0/8")
        proxy_card.add_row(
            "No-proxy exceptions", self._proxy_no_proxy,
            "Comma-separated hosts/domains that should bypass the proxy. "
            "Only applied in 'Manual' mode. Localhost is always exempted "
            "automatically so the local PO-token provider keeps working.",
        )

        def _update_proxy_manual_enabled():
            is_manual = self._proxy_mode.currentData() == "manual"
            self._proxy_url.setEnabled(is_manual)
            self._proxy_no_proxy.setEnabled(is_manual)

        _update_proxy_manual_enabled()
        self._proxy_mode.currentIndexChanged.connect(_update_proxy_manual_enabled)

        proxy_card.add_info(
            "'System / environment' uses your Windows proxy settings or the "
            "HTTP_PROXY/HTTPS_PROXY environment variables. Automatic proxy "
            "configuration (PAC file) is supported by the embedded browser "
            "tab, but NOT by the conversion pipeline (downloads, LLM calls) "
            "- if your network uses a PAC file, switch to 'Manual' here and "
            "enter the proxy host directly. TLS-intercepting proxies are "
            "supported automatically via the operating system's certificate "
            "store, no extra configuration needed."
        )

        main_layout.addWidget(proxy_card)

        # ── yt-dlp Update ────────────────────────────────────────────────
        ytdlp_header = QLabel("yt-dlp")
        ytdlp_header.setObjectName("subsectionHeader")
        main_layout.addWidget(ytdlp_header)
        ytdlp_card = SettingsCard()

        from .ytdlp_updater import get_installed_version

        self._ytdlp_update_thread = None
        self._ytdlp_update_worker = None

        installed_version = get_installed_version()
        self._ytdlp_version_label = QLabel(
            f"Installed version: {installed_version}"
            if installed_version else "Installed version: unknown"
        )
        ytdlp_card.add_widget(self._ytdlp_version_label)

        self._ytdlp_update_available_label = QLabel("")
        self._ytdlp_update_available_label.setVisible(False)
        ytdlp_card.add_widget(self._ytdlp_update_available_label)

        ytdlp_card.add_info(
            "yt-dlp handles all video/audio downloads. Video platforms change "
            "their internals often, so an outdated yt-dlp can start failing downloads "
            "(HTTP 403). The install and update scripts fetch the newest release; "
            "use this button only if downloads fail in between — it "
            "re-runs 'uv lock' and 'uv sync' and takes 1-2 minutes."
        )

        ytdlp_btn_row = QHBoxLayout()
        self._ytdlp_update_btn = QPushButton("Update yt-dlp")
        self._ytdlp_update_btn.setToolTip(
            "Re-lock and re-sync the project's dependencies to pull in the "
            "latest yt-dlp release. Takes 1-2 minutes."
        )
        self._ytdlp_update_btn.clicked.connect(self._on_update_ytdlp_clicked)
        ytdlp_btn_row.addWidget(self._ytdlp_update_btn)
        ytdlp_btn_row.addStretch()
        ytdlp_card.add_layout(ytdlp_btn_row)

        main_layout.addWidget(ytdlp_card)

        # ── Save Button ──────────────────────────────────────────────────
        main_layout.addSpacing(8)
        save_btn = AnimatedButton("Save Settings")
        save_btn.setToolTip(
            "Save all settings to disk. "
            "Settings are also auto-saved when you start a conversion or close the app."
        )
        save_btn.clicked.connect(self._save)
        main_layout.addWidget(save_btn)

        main_layout.addStretch(1)

        self._scroll = scroll
        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── LLM Provider Management ──────────────────────────────────────────

    def _scroll_to_llm_providers(self):
        """Scroll the page to the 'LLM Providers' management section.

        Triggered by the "Manage..." button next to the provider selector in
        the conversion form, whose section sits far above this one.
        """
        self._scroll.ensureWidgetVisible(self._llm_section_header, ymargin=24)

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
        """Update the conversion form's LLM provider combobox.

        Keeps the selected provider (on start the saved one) while the
        provider list is edited.
        """
        providers = self._llm_provider_list.get_providers()
        selected = (self._conversion_form.get_selected_provider_id()
                    or self._config.get("llm_provider_id", ""))
        self._conversion_form.set_llm_providers(providers, selected_id=selected)

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

    # ── Signing in to the video platform ─────────────────────────────────

    @property
    def signin_allowed(self) -> bool:
        """Whether signing in to the video platform is allowed (the switch, as saved)."""
        return self._signin_allowed

    def _on_signin_switch_toggled(self, checked: bool):
        """Switching on asks first (Cancel leaves it off); either way the change
        is saved at once and applies at once."""
        if checked == self._signin_allowed:
            return  # set back after Cancel
        if checked and not self._confirm_signin():
            self._signin_switch.setChecked(False)
            return
        from .config import save_config

        self._signin_allowed = checked
        self._config["allow_platform_signin"] = checked
        try:
            save_config(self._config)
        except OSError:  # it applies all the same; the next save stores it
            logger.warning("Failed to save the sign-in setting", exc_info=True)
        logger.info("Signing in to the video platform %s", "allowed" if checked else "switched off")
        self._update_cookie_status()
        self._update_export_enabled()
        self.signin_allowed_changed.emit(checked)

    def _confirm_signin(self) -> bool:
        """The warning before signing in is allowed; True when the user confirms it."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(SIGNIN_WARNING_TITLE)
        box.setText(SIGNIN_WARNING_TEXT)
        box.setInformativeText(SIGNIN_WARNING_DETAILS)
        allow = box.addButton(SIGNIN_CONFIRM_BUTTON, QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        confirmed = box.clickedButton() == allow
        box.deleteLater()
        return confirmed

    def show_signin_switch(self):
        """Scroll to "Allow signing in to the video platform" (the browser's notice links here)."""
        self._scroll.ensureWidgetVisible(self._signin_header, ymargin=24)

    def _update_export_enabled(self):
        self._export_btn.setEnabled(bool(self._cookie_manager) and self._signin_allowed)

    def _update_cookie_status(self):
        if not hasattr(self, "_cookie_status"):
            return
        manager = self._cookie_manager
        if not self._signin_allowed:
            self._cookie_status.setText("Signing in is off: downloads get no cookies.")
            self._cookie_status.setStyleSheet("color: #a09888;")
        elif manager and manager.is_signed_in:
            self._cookie_status.setText(
                f"\u2705 Signed in ({manager.video_cookie_count} cookies captured)"
            )
            self._cookie_status.setStyleSheet("color: #4caf50;")
        else:
            self._cookie_status.setText("\u26A0 Not signed in (sign in via the Video browser tab)")
            self._cookie_status.setStyleSheet("color: #ffa726;")

    def _export_cookies(self):
        if self._cookie_manager and self._signin_allowed:
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

    # ── yt-dlp Update ──────────────────────────────────────────────────

    def set_ytdlp_update_info(self, installed: str, latest: str = ""):
        """Refresh the yt-dlp version label (called after the startup check)."""
        from .ytdlp_updater import is_outdated

        if installed:
            self._ytdlp_version_label.setText(f"Installed version: {installed}")
        if latest and is_outdated(installed, latest):
            self._ytdlp_update_available_label.setText(
                f"⚠ Update available: {latest}"
            )
            self._ytdlp_update_available_label.setStyleSheet("color: #ffa726;")
            self._ytdlp_update_available_label.setVisible(True)
        else:
            self._ytdlp_update_available_label.setVisible(False)

    @property
    def is_ytdlp_update_running(self) -> bool:
        """True while "Update yt-dlp" re-locks and re-syncs the environment."""
        thread = self._ytdlp_update_thread
        return thread is not None and thread.isRunning()

    def _on_update_ytdlp_clicked(self):
        reply = QMessageBox.question(
            self,
            "Update yt-dlp",
            "Update yt-dlp now?\n\n"
            "This is usually only needed when downloads are failing "
            "(e.g. HTTP 403 errors).\n"
            "It re-runs 'uv lock' and 'uv sync' and takes about 1-2 minutes.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._ytdlp_update_btn.setEnabled(False)
        self._ytdlp_update_btn.setText("Updating...")
        self.log_message.emit("[yt-dlp] Starting update ...")

        repo_root = Path(__file__).resolve().parent.parent.parent

        self._ytdlp_update_thread = QThread(self)
        self._ytdlp_update_worker = _YtdlpUpdateRunWorker(repo_root)
        self._ytdlp_update_worker.moveToThread(self._ytdlp_update_thread)
        self._ytdlp_update_thread.started.connect(self._ytdlp_update_worker.run)
        self._ytdlp_update_worker.finished.connect(self._on_ytdlp_update_finished)
        self._ytdlp_update_worker.finished.connect(self._ytdlp_update_thread.quit)
        self._ytdlp_update_thread.start()

    def _on_ytdlp_update_finished(self, ok: bool, output: str):
        self._ytdlp_update_btn.setEnabled(True)
        self._ytdlp_update_btn.setText("Update yt-dlp")

        if ok:
            from .ytdlp_updater import get_installed_version

            self._ytdlp_version_label.setText(
                f"Installed version: {get_installed_version()}"
            )
            self._ytdlp_update_available_label.setVisible(False)
            self.log_message.emit("[yt-dlp] Update completed successfully.")
        else:
            self.log_message.emit(f"[yt-dlp] Update failed:\n{output}")
            QMessageBox.warning(
                self,
                "yt-dlp Update Failed",
                f"The yt-dlp update failed:\n\n{output}",
            )

    # ── Save / Collect ───────────────────────────────────────────────────

    def set_segmentation_model(self, path: str):
        """Use a (freshly trained) segmentation model for conversions and save."""
        self._conversion_form.set_segmentation_model(path)
        self._save()
        self.log_message.emit(f"[GUI] Segmentation model set to {path}")

    def set_training_running(self, running: bool):
        """Show on the Train button whether a model training is running."""
        self._conversion_form.set_training_running(running)

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
        result["allow_platform_signin"] = self._signin_allowed
        result["proxy_mode"] = self._proxy_mode.currentData()
        result["proxy_url"] = self._proxy_url.text()
        result["proxy_no_proxy"] = self._proxy_no_proxy.text()

        # Serialize providers
        providers = self._llm_provider_list.get_providers()
        result["llm_providers"] = [p.to_dict() for p in providers]

        # Collect API keys (stored in keyring on save)
        api_keys = self._llm_provider_list.get_api_keys()
        for pid, key in api_keys.items():
            result[f"llm_api_key_{pid}"] = key
        # Keys of removed providers: saving an empty key deletes it from the keyring
        for name in self._config:
            if name.startswith("llm_api_key_") and name[len("llm_api_key_"):] not in api_keys:
                result[name] = ""
        # The key of an older version (kept in config.json without a keyring)
        # belongs to the provider made from it; once that is removed or points
        # to another URL, saving deletes the key as well
        if self._config.get("llm_api_key"):
            legacy = next((p for p in providers
                           if p.id == self._config.get("llm_legacy_provider_id")), None)
            if legacy is None or legacy.api_base_url != self._config.get("llm_api_base_url"):
                result["llm_api_key"] = ""

        return result

    # Backward-compatible alias used by main_window.py
    def collect_preferences(self) -> dict:
        """Alias for collect_all()."""
        return self.collect_all()
