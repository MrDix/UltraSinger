"""Sidebar navigation widget with file drop zone and conversion queue."""

import importlib.metadata
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .file_drop_zone import FileDropZone
from .queue_list import QueueListWidget


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
    """Sidebar navigation panel with drop zone and conversion queue."""

    section_changed = Signal(int)
    start_all_requested = Signal()
    file_dropped = Signal(str)  # file path from drop zone

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 12)
        layout.setSpacing(2)

        # Logo image
        logo = QLabel()
        logo.setObjectName("sidebarLogo")
        logo_path = Path(__file__).parent.parent / "resources" / "icons" / "logo.jpg"
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            scaled = pixmap.scaledToWidth(
                196, Qt.TransformationMode.SmoothTransformation
            )
            logo.setPixmap(scaled)
        else:
            logo.setText("UltraSinger")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo)
        layout.addSpacing(8)

        # ── File Drop Zone ─────────────────────────────────────────────
        self._drop_zone = FileDropZone()
        self._drop_zone.setMinimumHeight(90)
        self._drop_zone.setMaximumHeight(110)
        layout.addWidget(self._drop_zone)

        # Wire drop zone → emit file_dropped signal
        self._drop_zone.file_selected.connect(self._on_file_dropped)

        # ── Queue List ─────────────────────────────────────────────────
        self._queue_list = QueueListWidget()
        layout.addWidget(self._queue_list)

        # ── Queue Action Buttons ───────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self._start_btn = QPushButton("\u25B6  Start All")
        self._start_btn.setObjectName("primaryButton")
        self._start_btn.setMinimumHeight(36)
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self.start_all_requested.emit)
        btn_row.addWidget(self._start_btn, 1)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("ghostButton")
        self._clear_btn.setMinimumHeight(36)
        self._clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_btn.setEnabled(False)
        btn_row.addWidget(self._clear_btn)

        layout.addLayout(btn_row)
        layout.addSpacing(8)

        # ── Navigation Buttons ─────────────────────────────────────────
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._buttons: list[SidebarButton] = []

    @property
    def drop_zone(self) -> FileDropZone:
        """Access the sidebar's file drop zone."""
        return self._drop_zone

    @property
    def queue_list(self) -> QueueListWidget:
        """Access the sidebar's queue list widget."""
        return self._queue_list

    @property
    def clear_button(self) -> QPushButton:
        """Access the Clear button for external wiring."""
        return self._clear_btn

    def add_section(self, icon_text: str, label: str) -> int:
        """Add a navigation section. Returns the section index."""
        btn = SidebarButton(icon_text, label, self)
        index = len(self._buttons)
        self._button_group.addButton(btn, index)
        self._buttons.append(btn)
        self.layout().addWidget(btn)
        btn.clicked.connect(
            lambda _checked=False, idx=index: self.section_changed.emit(idx)
        )
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

    def update_queue_buttons(self, has_pending: bool, is_running: bool):
        """Update Start All / Clear button states."""
        self._start_btn.setEnabled(has_pending and not is_running)
        self._clear_btn.setEnabled(not is_running)
        if is_running:
            self._start_btn.setText("\u23F3  Running...")
        else:
            self._start_btn.setText("\u25B6  Start All")

    def _on_file_dropped(self, path: str):
        """Forward file selection to parent via signal."""
        self.file_dropped.emit(path)
        # Reset drop zone visual (file is now in the queue)
        self._drop_zone.set_file("")
