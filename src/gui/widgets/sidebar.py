"""Sidebar navigation widget with file drop zone and convert button."""

import importlib.metadata
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .file_drop_zone import FileDropZone


class SidebarButton(QPushButton):
    """A sidebar navigation button with icon text and label."""

    def __init__(self, icon_text: str, label: str, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.icon_text = icon_text
        self.label = label
        self.setText(f"  {icon_text}  {label}")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(44)


class Sidebar(QWidget):
    """Sidebar navigation panel with drop zone and convert button."""

    section_changed = Signal(int)
    convert_requested = Signal()  # user wants to start conversion

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 12)
        layout.setSpacing(2)

        # Logo / title
        logo = QLabel("\U0001F3A4 UltraSinger")
        logo.setObjectName("sidebarLogo")
        layout.addWidget(logo)
        layout.addSpacing(8)

        # ── File Drop Zone ─────────────────────────────────────────────
        self._drop_zone = FileDropZone()
        self._drop_zone.setMinimumHeight(90)
        self._drop_zone.setMaximumHeight(110)
        layout.addWidget(self._drop_zone)

        # ── Input indicator (shows current source) ─────────────────────
        self._input_label = QLabel("")
        self._input_label.setObjectName("caption")
        self._input_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._input_label.setWordWrap(True)
        self._input_label.setStyleSheet(
            "color: #e91e63; font-size: 11px; padding: 2px 4px;"
        )
        self._input_label.hide()
        layout.addWidget(self._input_label)

        # ── Convert Button ─────────────────────────────────────────────
        self._convert_btn = QPushButton("\u25B6  Start Conversion")
        self._convert_btn.setObjectName("primaryButton")
        self._convert_btn.setMinimumHeight(40)
        self._convert_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._convert_btn.hide()
        self._convert_btn.clicked.connect(self.convert_requested.emit)
        layout.addWidget(self._convert_btn)

        layout.addSpacing(8)

        # ── Navigation Buttons ─────────────────────────────────────────
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._buttons: list[SidebarButton] = []

        # Track input source
        self._current_input: str = ""
        self._input_type: str = ""  # "file" or "youtube"

        # Wire drop zone
        self._drop_zone.file_selected.connect(self._on_file_dropped)

    @property
    def drop_zone(self) -> FileDropZone:
        """Access the sidebar's file drop zone."""
        return self._drop_zone

    def add_section(self, icon_text: str, label: str) -> int:
        """Add a navigation section. Returns the section index."""
        btn = SidebarButton(icon_text, label, self)
        index = len(self._buttons)
        self._button_group.addButton(btn, index)
        self._buttons.append(btn)
        self.layout().addWidget(btn)
        btn.clicked.connect(lambda _checked=False, idx=index: self.section_changed.emit(idx))
        if index == 0:
            btn.setChecked(True)
        return index

    def finalize(self):
        """Call after all sections are added to push remaining space down."""
        self.layout().addStretch(1)

        # Version label at bottom
        try:
            _version = importlib.metadata.version("ultrasinger")
        except importlib.metadata.PackageNotFoundError:
            _version = "dev"
        version_label = QLabel(f"v{_version}")
        version_label.setObjectName("caption")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout().addWidget(version_label)

    def set_active(self, index: int):
        """Programmatically set the active section."""
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)
            self.section_changed.emit(index)

    # ── Input Source Management ─────────────────────────────────────────

    def set_youtube_input(self, url: str):
        """Set a YouTube URL as the input source."""
        self._current_input = url
        self._input_type = "youtube"
        # Show a compact display: "YT: /watch?v=abc123"
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            path_and_query = parsed.path
            if parsed.query:
                path_and_query += "?" + parsed.query
            display = f"YT: {path_and_query}"
        except Exception:
            display = url
            if len(display) > 35:
                display = display[:32] + "..."
        self._input_label.setText(f"\U0001F310 {display}")
        self._input_label.show()
        self._convert_btn.show()

    def get_input_source(self) -> str:
        """Return the current input source (URL or file path)."""
        return self._current_input

    def get_input_type(self) -> str:
        """Return the input type: 'file', 'youtube', or ''."""
        return self._input_type

    def _on_file_dropped(self, path: str):
        """Handle file selected via drop zone."""
        self._current_input = path
        self._input_type = "file"
        name = Path(path).name
        self._input_label.setText(f"\U0001F3B5 {name}")
        self._input_label.show()
        self._convert_btn.show()

    def clear_input(self):
        """Reset input state after conversion or cancel."""
        self._current_input = ""
        self._input_type = ""
        self._input_label.hide()
        self._convert_btn.hide()
        self._drop_zone.set_file("")
