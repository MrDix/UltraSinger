"""Widget for managing multiple LLM API providers."""

import json
import logging
import urllib.error
import urllib.request

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..models import LLMProvider

logger = logging.getLogger(__name__)


class _NoScrollComboBox(QComboBox):
    """QComboBox that ignores wheel events unless explicitly focused."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _ModelFetcher(QObject):
    """Fetches available models from an OpenAI-compatible /v1/models endpoint."""

    finished = Signal(list)  # list of model ID strings
    error = Signal(str)

    def __init__(self, base_url: str, api_key: str):
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def run(self):
        url = f"{self._base_url}/models"
        req = urllib.request.Request(url, method="GET")
        req.add_header("User-Agent", "UltraSinger-GUI/1.0")
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = sorted(
                m["id"] for m in data.get("data", []) if m.get("id")
            )
            self.finished.emit(models)
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError) as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(str(e))


class LLMProviderRow(QWidget):
    """A single editable row representing one LLM provider."""

    removed = Signal(str)  # provider_id
    changed = Signal()

    def __init__(self, provider: LLMProvider, parent=None):
        super().__init__(parent)
        self._provider = provider

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        # Header row: radio (default) + name + delete button
        header = QHBoxLayout()
        header.setSpacing(8)

        self._star_btn = QPushButton()
        self._star_btn.setObjectName("starButton")
        self._star_btn.setFixedSize(36, 36)
        self._star_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._star_btn.clicked.connect(self._on_star_clicked)
        self._is_default = provider.is_default
        self._update_star()  # sets text, style, and tooltip
        header.addWidget(self._star_btn)

        self._name_edit = QLineEdit(provider.name)
        self._name_edit.setPlaceholderText("Provider name (e.g. Groq Free)")
        self._name_edit.setStyleSheet("font-weight: bold;")
        self._name_edit.textChanged.connect(self._on_field_changed)
        header.addWidget(self._name_edit, 1)

        delete_btn = QPushButton("\u2715")
        delete_btn.setObjectName("ghostButton")
        delete_btn.setFixedSize(36, 36)
        delete_btn.setStyleSheet("font-size: 18px; padding: 0px;")
        delete_btn.setToolTip("Remove provider")
        delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        delete_btn.clicked.connect(lambda: self.removed.emit(self._provider.id))
        header.addWidget(delete_btn)

        layout.addLayout(header)

        # Fields row 1: Key
        fields_key = QHBoxLayout()
        fields_key.setSpacing(8)

        key_label = QLabel("Key")
        key_label.setFixedWidth(40)
        key_label.setObjectName("caption")
        fields_key.addWidget(key_label)

        self._key_edit = QLineEdit()
        self._key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_edit.setPlaceholderText("gsk_...")
        self._key_edit.textChanged.connect(self._on_field_changed)
        self._key_edit.editingFinished.connect(self._on_key_editing_finished)
        fields_key.addWidget(self._key_edit, 1)

        self._key_toggle = QPushButton("\U0001F441")
        self._key_toggle.setObjectName("ghostButton")
        self._key_toggle.setFixedSize(36, 36)
        self._key_toggle.setStyleSheet("font-size: 18px; padding: 0px;")
        self._key_toggle.setToolTip("Show/hide API key")
        self._key_toggle.clicked.connect(self._toggle_key_visibility)
        fields_key.addWidget(self._key_toggle)

        layout.addLayout(fields_key)

        # Fields row 2: URL
        fields_url = QHBoxLayout()
        fields_url.setSpacing(8)

        url_label = QLabel("URL")
        url_label.setFixedWidth(40)
        url_label.setObjectName("caption")
        fields_url.addWidget(url_label)

        self._url_edit = QLineEdit(provider.api_base_url)
        self._url_edit.setPlaceholderText("https://api.groq.com/openai/v1")
        self._url_edit.textChanged.connect(self._on_field_changed)
        self._url_edit.editingFinished.connect(self._on_url_editing_finished)
        self._last_fetched_url = provider.api_base_url
        fields_url.addWidget(self._url_edit, 1)

        layout.addLayout(fields_url)

        # Fields row 3: Model
        fields_model = QHBoxLayout()
        fields_model.setSpacing(8)

        model_label = QLabel("Model")
        model_label.setFixedWidth(40)
        model_label.setObjectName("caption")
        fields_model.addWidget(model_label)

        self._model_combo = _NoScrollComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._model_combo.lineEdit().setPlaceholderText("qwen/qwen3-32b")
        if provider.default_model:
            self._model_combo.addItem(provider.default_model)
            self._model_combo.setCurrentText(provider.default_model)
        self._model_combo.currentTextChanged.connect(self._on_field_changed)
        fields_model.addWidget(self._model_combo, 1)

        self._fetch_btn = QPushButton("\u21BB")
        self._fetch_btn.setObjectName("ghostButton")
        self._fetch_btn.setFixedSize(36, 36)
        self._fetch_btn.setStyleSheet("font-size: 18px; padding: 0px;")
        self._fetch_btn.setToolTip("Fetch available models from API")
        self._fetch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._fetch_btn.clicked.connect(self._fetch_models)
        fields_model.addWidget(self._fetch_btn)

        self._fetch_thread: QThread | None = None

        layout.addLayout(fields_model)

        # Model hint text
        model_hint = QLabel(
            "Models are fetched automatically when URL and Key are set."
        )
        model_hint.setObjectName("caption")
        model_hint.setStyleSheet(
            "font-size: 10px; color: #605848; background: transparent; "
            "padding: 0px 0px 0px 48px;"
        )
        layout.addWidget(model_hint)

        # Bottom separator
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: rgba(240, 223, 192, 0.06);")
        layout.addWidget(sep)

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    @property
    def provider_id(self) -> str:
        return self._provider.id

    @property
    def is_default(self) -> bool:
        return self._is_default

    def set_default(self, value: bool):
        self._is_default = value
        self._update_star()

    def get_api_key(self) -> str:
        return self._key_edit.text()

    def set_api_key(self, key: str):
        self._key_edit.setText(key)

    def collect(self) -> LLMProvider:
        """Return an updated LLMProvider from current field values."""
        self._provider.name = self._name_edit.text()
        self._provider.api_base_url = self._url_edit.text()
        self._provider.default_model = self._model_combo.currentText()
        self._provider.is_default = self._is_default
        return self._provider

    def _on_field_changed(self):
        self.changed.emit()

    def _on_star_clicked(self):
        if not self._is_default:
            self._is_default = True
            self._update_star()
            self.changed.emit()

    def _update_star(self):
        if self._is_default:
            self._star_btn.setText("\u2605")  # ★ filled
            self._star_btn.setStyleSheet(
                "font-size: 22px; color: #f7d547; padding: 0px; "
                "background: transparent; border: none;"
            )
            self._star_btn.setToolTip("Current default provider")
        else:
            self._star_btn.setText("\u2606")  # ☆ outline
            self._star_btn.setStyleSheet(
                "font-size: 22px; color: #a09888; padding: 0px; "
                "background: transparent; border: none;"
            )
            self._star_btn.setToolTip("Set as default provider")

    def _toggle_key_visibility(self):
        if self._key_edit.echoMode() == QLineEdit.EchoMode.Password:
            self._key_edit.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self._key_edit.setEchoMode(QLineEdit.EchoMode.Password)

    def _on_url_editing_finished(self):
        """Auto-fetch models when the URL field loses focus and changed."""
        url = self._url_edit.text().strip()
        if url and url != self._last_fetched_url:
            self._last_fetched_url = url
            self._fetch_models()

    def _on_key_editing_finished(self):
        """Auto-fetch models when the key field loses focus, if no models loaded yet."""
        if self._model_combo.count() <= 1 and self._url_edit.text().strip():
            self._fetch_models()

    def fetch_models_if_ready(self):
        """Trigger a model fetch if URL and key are available. Called after key is loaded."""
        if self._url_edit.text().strip() and self._key_edit.text().strip():
            self._fetch_models()

    def _fetch_models(self):
        """Fetch available models from the provider's API."""
        url = self._url_edit.text().strip()
        if not url:
            return

        self._fetch_btn.setEnabled(False)
        self._fetch_btn.setToolTip("Fetching models...")

        self._fetch_thread = QThread()
        self._fetcher = _ModelFetcher(url, self._key_edit.text())
        self._fetcher.moveToThread(self._fetch_thread)

        self._fetch_thread.started.connect(self._fetcher.run)
        self._fetcher.finished.connect(self._on_models_fetched)
        self._fetcher.error.connect(self._on_fetch_error)
        self._fetcher.finished.connect(self._cleanup_fetch)
        self._fetcher.error.connect(self._cleanup_fetch)

        self._fetch_thread.start()

    _FALLBACK_MODEL = "qwen/qwen3-32b"

    def _on_models_fetched(self, models: list[str]):
        """Populate the model combobox with fetched models.

        Selection priority: previous model → qwen/qwen3-32b → first model.
        """
        previous = self._model_combo.currentText()
        self._model_combo.clear()
        self._model_combo.addItems(models)

        # Try to keep previous selection
        idx = self._model_combo.findText(previous)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)
        else:
            # Previous model not available — try fallback
            idx = self._model_combo.findText(self._FALLBACK_MODEL)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
            elif models:
                # Neither previous nor fallback found — select first
                self._model_combo.setCurrentIndex(0)

        self._fetch_btn.setToolTip(
            f"Fetch available models from API ({len(models)} found)"
        )

    def _on_fetch_error(self, error: str):
        logger.warning("Failed to fetch models: %s", error)
        self._fetch_btn.setToolTip(f"Fetch failed: {error}")

    def _cleanup_fetch(self):
        self._fetch_btn.setEnabled(True)
        if self._fetch_thread:
            self._fetch_thread.quit()
            self._fetch_thread.wait(2000)
            self._fetch_thread = None


class LLMProviderListWidget(QWidget):
    """Widget for managing multiple LLM providers with add/edit/remove."""

    providers_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[LLMProviderRow] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Container for provider rows
        self._rows_container = QVBoxLayout()
        self._rows_container.setContentsMargins(0, 0, 0, 0)
        self._rows_container.setSpacing(2)
        layout.addLayout(self._rows_container)

        # Empty state
        self._empty_label = QLabel("No LLM providers configured")
        self._empty_label.setObjectName("caption")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(
            "color: #605848; font-size: 12px; padding: 12px;"
        )
        layout.addWidget(self._empty_label)

        # Add button
        add_btn = QPushButton("+ Add LLM Provider")
        add_btn.setObjectName("ghostButton")
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.clicked.connect(self._add_default_provider)
        layout.addWidget(add_btn)

        self._update_empty_state()

    def set_providers(self, providers: list[LLMProvider], api_keys: dict[str, str] | None = None):
        """Populate with a list of providers. api_keys maps provider_id -> key."""
        # Clear existing rows
        for row in self._rows:
            self._rows_container.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

        for provider in providers:
            self._add_row(provider)
            if api_keys and provider.id in api_keys:
                self._rows[-1].set_api_key(api_keys[provider.id])

        self._update_empty_state()

        # Auto-fetch models for rows that have URL + key but no model list
        for row in self._rows:
            row.fetch_models_if_ready()

    def get_providers(self) -> list[LLMProvider]:
        """Return the current list of providers from the UI."""
        return [row.collect() for row in self._rows]

    def get_api_keys(self) -> dict[str, str]:
        """Return provider_id -> api_key mapping."""
        return {row.provider_id: row.get_api_key() for row in self._rows}

    def get_default_provider_id(self) -> str:
        """Return the ID of the default provider, or empty string."""
        for row in self._rows:
            if row.is_default:
                return row.provider_id
        return ""

    def _add_row(self, provider: LLMProvider):
        """Add a provider row to the UI."""
        row = LLMProviderRow(provider, self)
        row.removed.connect(self._remove_provider)
        row.changed.connect(self._on_row_changed)
        self._rows.append(row)
        self._rows_container.addWidget(row)

    def _on_row_changed(self):
        """Ensure only one star is active (exclusive default selection)."""
        sender = self.sender()
        if isinstance(sender, LLMProviderRow) and sender.is_default:
            for row in self._rows:
                if row is not sender and row.is_default:
                    row.set_default(False)
        self.providers_changed.emit()

    def _unique_name(self, base: str) -> str:
        """Generate a unique provider name by appending a number if needed."""
        existing = {row.collect().name for row in self._rows}
        if base not in existing:
            return base
        n = 2
        while f"{base} ({n})" in existing:
            n += 1
        return f"{base} ({n})"

    def _add_default_provider(self):
        """Add a new provider with Groq defaults."""
        is_first = len(self._rows) == 0
        provider = LLMProvider(
            name=self._unique_name("Groq"),
            api_base_url="https://api.groq.com/openai/v1",
            default_model="qwen/qwen3-32b",
            is_default=is_first,
        )
        self._add_row(provider)
        self._update_empty_state()
        self.providers_changed.emit()

    def _remove_provider(self, provider_id: str):
        """Remove a provider row by ID."""
        for i, row in enumerate(self._rows):
            if row.provider_id == provider_id:
                was_default = row.is_default
                self._rows_container.removeWidget(row)
                row.deleteLater()
                self._rows.pop(i)

                # If we removed the default, make the first remaining one default
                if was_default and self._rows:
                    self._rows[0].set_default(True)

                self._update_empty_state()
                self.providers_changed.emit()
                return

    def _update_empty_state(self):
        self._empty_label.setVisible(len(self._rows) == 0)
