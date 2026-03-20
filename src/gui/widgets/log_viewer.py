"""Color-coded log viewer widget for conversion output."""

import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit


# Strip ANSI escape sequences (SGR codes, cursor movement, etc.)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Log level patterns and their colors (matched AFTER ANSI stripping)
_PATTERNS = [
    (re.compile(r"\[error\]|error:|traceback|exception", re.IGNORECASE), QColor("#ef5350")),
    (re.compile(r"\[warn(?:ing)?\]|warn(?:ing)?:", re.IGNORECASE), QColor("#ffa726")),
    (re.compile(r"\[success\]|completed|done|finished", re.IGNORECASE), QColor("#4caf50")),
    (re.compile(r"\[UltraSinger\]", re.IGNORECASE), QColor("#00d4d4")),
]
_DEFAULT_COLOR = QColor("#a09888")


class LogViewer(QPlainTextEdit):
    """A read-only, color-coded log output widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("logOutput")
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Auto-scroll control
        self._auto_scroll = True
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)

    def append_line(self, text: str):
        """Append a colored log line.

        ANSI escape codes from the CLI subprocess are stripped before
        display.  Color is determined by keyword matching instead.
        """
        text = _ANSI_RE.sub("", text)

        color = _DEFAULT_COLOR
        for pattern, c in _PATTERNS:
            if pattern.search(text):
                color = c
                break

        fmt = QTextCharFormat()
        fmt.setForeground(color)

        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text + "\n", fmt)

        if self._auto_scroll:
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().maximum()
            )

    def clear_log(self):
        """Clear all log content."""
        self.clear()

    def _on_scroll(self, value):
        """Track whether the user has scrolled away from the bottom."""
        sb = self.verticalScrollBar()
        self._auto_scroll = value >= sb.maximum() - 10
