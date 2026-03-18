"""Sidebar navigation widget with icon + label buttons."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


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
    """Sidebar navigation panel with icon+label buttons."""

    section_changed = Signal(int)

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
        layout.addSpacing(12)

        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._buttons: list[SidebarButton] = []

    def add_section(self, icon_text: str, label: str) -> int:
        """Add a navigation section. Returns the section index."""
        btn = SidebarButton(icon_text, label, self)
        index = len(self._buttons)
        self._button_group.addButton(btn, index)
        self._buttons.append(btn)
        # Insert before the spacer (if exists) at the end
        self.layout().addWidget(btn)
        btn.clicked.connect(lambda: self.section_changed.emit(index))
        if index == 0:
            btn.setChecked(True)
        return index

    def finalize(self):
        """Call after all sections are added to push remaining space down."""
        self.layout().addStretch(1)

        # Version label at bottom
        version_label = QLabel("v0.0.13.dev16")
        version_label.setObjectName("caption")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout().addWidget(version_label)

    def set_active(self, index: int):
        """Programmatically set the active section."""
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)
            self.section_changed.emit(index)
