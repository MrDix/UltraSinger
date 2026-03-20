"""Per-song settings override dialog."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QScrollArea,
    QVBoxLayout,
)

from .models import LLMProvider
from .settings_tab import ConversionSettingsForm


class PerSongSettingsDialog(QDialog):
    """Dialog for overriding conversion settings for a single queue item.

    Shows the full ``ConversionSettingsForm`` pre-populated with the
    global config merged with any existing per-song overrides.  On
    accept, the caller can retrieve the *overrides-only* dict (values
    that differ from the global config).
    """

    def __init__(
        self,
        global_config: dict,
        overrides: dict,
        llm_providers: list[LLMProvider],
        title: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Settings: {title}" if title else "Per-Song Settings")
        self.setMinimumSize(700, 600)
        self.resize(800, 700)

        self._global_config = global_config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Scroll area wrapping the form
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Merge global + overrides for initial display
        merged = {**global_config, **overrides}

        self._form = ConversionSettingsForm(merged)
        self._form.set_llm_providers(
            llm_providers,
            selected_id=overrides.get("llm_provider_id", ""),
        )

        scroll.setWidget(self._form)
        layout.addWidget(scroll, 1)

        # Dialog buttons — styled consistently with app theme
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn:
            ok_btn.setObjectName("primaryButton")
            ok_btn.setText("Save")

        cancel_btn = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_btn:
            cancel_btn.setObjectName("ghostButton")

        restore_btn = buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults)
        if restore_btn:
            restore_btn.setToolTip("Clear all per-song overrides (use global defaults)")
            restore_btn.clicked.connect(self._restore_defaults)

        layout.addWidget(buttons)

        self._cleared = False

    def get_overrides(self) -> dict:
        """Return only the settings that differ from the global config.

        If the user clicked "Restore Defaults", returns an empty dict.
        """
        if self._cleared:
            return {}

        form_values = self._form.collect_config()
        overrides = {}
        for key, value in form_values.items():
            global_value = self._global_config.get(key)
            if value != global_value:
                overrides[key] = value
        return overrides

    def _restore_defaults(self):
        """Mark all overrides as cleared and accept."""
        self._cleared = True
        self.accept()
