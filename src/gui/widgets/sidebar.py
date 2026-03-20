"""Sidebar navigation widget with file drop zone and conversion queue."""

import importlib.metadata
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
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
        self.setMinimumHeight(40)


class Sidebar(QWidget):
    """Sidebar navigation panel with drop zone and conversion queue.

    Layout (top to bottom):
      Logo → Drag Files zone → Convert Queue (fills space) →
      Start/Clear buttons → Navigation buttons → Version
    """

    section_changed = Signal(int)
    start_all_requested = Signal()
    file_dropped = Signal(str)  # file path from drop zone

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(250)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Logo image — full width, no margins
        logo = QLabel()
        logo.setObjectName("sidebarLogo")
        logo_path = Path(__file__).parent.parent / "resources" / "icons" / "logo.png"
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            scaled = pixmap.scaledToWidth(
                250, Qt.TransformationMode.SmoothTransformation
            )
            logo.setPixmap(scaled)
        else:
            logo.setText("UltraSinger")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(logo)

        # Content area with side margins
        self._content_layout = QVBoxLayout()
        self._content_layout.setContentsMargins(6, 4, 6, 8)
        self._content_layout.setSpacing(4)
        outer.addLayout(self._content_layout, 1)
        layout = self._content_layout

        # ── Drag Files Zone ───────────────────────────────────────────
        drop_frame = _SidebarSection("Drop Files")
        self._drop_zone = FileDropZone()
        self._drop_zone.setMinimumHeight(70)
        self._drop_zone.setMaximumHeight(90)
        drop_frame.add_widget(self._drop_zone)
        layout.addWidget(drop_frame)

        # Wire drop zone → emit file_dropped signal
        self._drop_zone.file_selected.connect(self._on_file_dropped)

        # ── Convert Queue (fills available space) ─────────────────────
        queue_frame = _SidebarSection("Convert Queue")
        self._queue_list = QueueListWidget()
        queue_frame.add_widget(self._queue_list, stretch=1)
        layout.addWidget(queue_frame, 1)  # stretch=1 → takes all space

        # ── Queue Action Buttons ──────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self._start_btn = QPushButton("\u25B6  Start All")
        self._start_btn.setObjectName("primaryButton")
        self._start_btn.setMinimumHeight(34)
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(lambda: self.start_all_requested.emit())
        btn_row.addWidget(self._start_btn, 1)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("ghostButton")
        self._clear_btn.setMinimumHeight(34)
        self._clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_btn.setEnabled(False)
        btn_row.addWidget(self._clear_btn)

        layout.addLayout(btn_row)

        # ── Navigation Buttons (at the bottom) ────────────────────────
        self._nav_container = QVBoxLayout()
        self._nav_container.setSpacing(2)
        layout.addLayout(self._nav_container)

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
        self._nav_container.addWidget(btn)
        btn.clicked.connect(
            lambda _checked=False, idx=index: self.section_changed.emit(idx)
        )
        if index == 0:
            btn.setChecked(True)
        return index

    def finalize(self):
        """Call after all sections are added."""
        # Version label at very bottom
        try:
            _version = importlib.metadata.version("ultrasinger")
        except importlib.metadata.PackageNotFoundError:
            _version = "dev"
        version_label = QLabel(f"v{_version}")
        version_label.setObjectName("caption")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._content_layout.addWidget(version_label)

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


class _SidebarSection(QFrame):
    """A labeled frame container for sidebar sections."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebarSection")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 4, 6, 6)
        self._layout.setSpacing(2)

        label = QLabel(title)
        label.setObjectName("caption")
        label.setStyleSheet(
            "font-size: 11px; font-weight: 600; color: #a09888; "
            "letter-spacing: 0.5px; text-transform: uppercase; "
            "background: transparent;"
        )
        self._layout.addWidget(label)

    def add_widget(self, widget, stretch=0):
        self._layout.addWidget(widget, stretch)
