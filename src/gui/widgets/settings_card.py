"""Reusable card container for grouping settings."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class SettingsCard(QWidget):
    """A styled card container for a group of settings controls."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("settingsCard")

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(20, 16, 20, 16)
        self._layout.setSpacing(12)

        if title:
            title_label = QLabel(title)
            title_label.setObjectName("subsectionHeader")
            self._layout.addWidget(title_label)

    def add_widget(self, widget):
        """Add a widget to the card layout."""
        self._layout.addWidget(widget)

    def add_layout(self, layout):
        """Add a sub-layout to the card."""
        self._layout.addLayout(layout)

    def add_row(self, label_text: str, widget, tooltip: str = ""):
        """Add a labeled row with a widget."""
        from PySide6.QtWidgets import QHBoxLayout

        row = QHBoxLayout()
        row.setSpacing(12)

        label = QLabel(label_text)
        label.setMinimumWidth(180)
        if tooltip:
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
        row.addWidget(label)
        row.addWidget(widget, 1)
        self._layout.addLayout(row)

    def add_toggle_row(self, label_text: str, toggle, tooltip: str = ""):
        """Add a row with a label and toggle switch (right-aligned toggle)."""
        from PySide6.QtWidgets import QHBoxLayout

        row = QHBoxLayout()
        row.setSpacing(12)

        label = QLabel(label_text)
        if tooltip:
            label.setToolTip(tooltip)
            toggle.setToolTip(tooltip)
        row.addWidget(label, 1)
        row.addWidget(toggle)
        self._layout.addLayout(row)

    def add_info(self, text: str):
        """Add an info label to the card."""
        info = QLabel(text)
        info.setObjectName("infoLabel")
        info.setWordWrap(True)
        self._layout.addWidget(info)

    def remove_last_item(self):
        """Remove and clean up the last item from the card layout."""
        if self._layout.count() == 0:
            return
        last = self._layout.takeAt(self._layout.count() - 1)
        if last.widget():
            last.widget().hide()
            last.widget().deleteLater()
        elif last.layout():
            while last.layout().count():
                child = last.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

    def add_separator(self):
        """Add a subtle horizontal separator."""
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: rgba(255, 255, 255, 0.06);")
        self._layout.addWidget(sep)
