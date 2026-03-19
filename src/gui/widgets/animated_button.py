"""Styled primary action button (CSS hover transitions applied via QSS)."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton


class AnimatedButton(QPushButton):
    """A styled primary action button with CSS hover transitions."""

    def __init__(self, text: str, primary: bool = True, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if primary:
            self.setObjectName("primaryButton")
        self.setMinimumHeight(38)
