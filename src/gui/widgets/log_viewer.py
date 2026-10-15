"""Color-coded log viewer widget for conversion output."""

import re
from collections import deque
from itertools import groupby

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit


# Strip ANSI escape sequences (SGR codes, cursor movement, etc.)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# The lines the viewer keeps: the last lines, and apart from them the last
# warnings and errors
_MAX_LINES = 5000

# Log levels; the first pattern that matches a line (AFTER ANSI stripping)
# gives its level
_ERROR, _WARNING, _SUCCESS, _HEAD, _PLAIN = "error", "warning", "success", "head", "plain"
_ISSUES = (_ERROR, _WARNING)
_PATTERNS = [
    (re.compile(r"\[error\]|error:|traceback|exception", re.IGNORECASE), _ERROR),
    (re.compile(r"\[warn(?:ing)?\]|warn(?:ing)?:", re.IGNORECASE), _WARNING),
    # A step that failed ("Remote STT failed: ...", "<song>: failed (exit code 1)"),
    # after the warnings, which may say that something failed
    (re.compile(r"\bfailed\s*[:(]", re.IGNORECASE), _ERROR),
    (re.compile(r"\[success\]|completed|done|finished", re.IGNORECASE), _SUCCESS),
    (re.compile(r"\[UltraSinger\]", re.IGNORECASE), _HEAD),
]
_COLORS = {
    _ERROR: QColor("#ef5350"),
    _WARNING: QColor("#ffa726"),
    _SUCCESS: QColor("#4caf50"),
    _HEAD: QColor("#00d4d4"),
    _PLAIN: QColor("#a09888"),
}
# The first line of a Python traceback. Its frames are indented, and one
# that starts a line ends with a line that names the exception.
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):")
# A warning of Python's warnings module ("<file>:<line>: UserWarning: ..."),
# followed by its indented source line
_PYTHON_WARNING_RE = re.compile(r":\d+: \w*Warning: ")


class LogViewer(QPlainTextEdit):
    """A read-only, color-coded log output widget.

    It keeps the lines it was given, so that it can show only the warnings
    and errors among them (``set_issues_only``) and all lines again.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("logOutput")
        self.setReadOnly(True)
        self.setMaximumBlockCount(_MAX_LINES)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # The lines as (text, level); the warnings and errors are kept apart
        # as well, so that the filter still shows those older than the last
        # lines (a song prints thousands of progress lines)
        self._lines: deque[tuple[str, str]] = deque(maxlen=_MAX_LINES)
        self._issues: deque[tuple[str, str]] = deque(maxlen=_MAX_LINES)
        self._issues_only = False
        self._formats = {level: _char_format(color) for level, color in _COLORS.items()}
        # A traceback is kept together, a Python warning with its source line
        self._in_traceback = False
        self._exception_due = False
        self._warning_source_due = False

        # Auto-scroll control
        self._auto_scroll = True
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)

    def append_line(self, text: str):
        """Append a colored log line.

        ANSI escape codes from the CLI subprocess are stripped before
        display.  Color is determined by keyword matching instead.  While
        only the warnings and errors are shown, another line is kept but
        not shown.
        """
        text = _ANSI_RE.sub("", text)
        level = self._level(text)
        self._lines.append((text, level))
        if level in _ISSUES:
            self._issues.append((text, level))
        elif self._issues_only:
            return

        self._insert([(text, level)])

        if self._auto_scroll:
            self._scroll_to_end()

    def set_issues_only(self, issues_only: bool):
        """Show only the warnings and errors, also those appended before, or all lines again."""
        if issues_only == self._issues_only:
            return
        self._issues_only = issues_only
        self.setPlaceholderText("No warnings or errors." if issues_only else "")
        self.clear()
        self._insert(self._issues if issues_only else self._lines)
        self._scroll_to_end()

    def issues_only(self) -> bool:
        """Whether only the warnings and errors are shown."""
        return self._issues_only

    def clear_log(self):
        """Clear all log content."""
        self.clear()
        self._lines.clear()
        self._issues.clear()
        self._in_traceback = self._exception_due = self._warning_source_due = False

    def _level(self, text: str) -> str:
        """The level of a line; the lines of a traceback are an error together."""
        indented = text[:1].isspace() and bool(text.strip())
        if indented and self._in_traceback:
            return _ERROR
        if indented and self._warning_source_due:
            self._warning_source_due = False
            return _WARNING
        # The exception that ends a traceback, also one without "Error" in its
        # name ("KeyboardInterrupt")
        exception = self._exception_due and bool(text.strip())
        self._in_traceback = self._exception_due = self._warning_source_due = False
        if exception:
            return _ERROR

        level = next((level for pattern, level in _PATTERNS if pattern.search(text)), _PLAIN)
        if _TRACEBACK_RE.search(text):
            self._in_traceback = True
            # An indented one is part of an exception group: it ends with its frames
            self._exception_due = not text[:1].isspace()
        else:
            self._warning_source_due = _PYTHON_WARNING_RE.search(text) is not None
        return level

    def _insert(self, lines):
        """Append lines of (text, level), each in the color of its level."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.beginEditBlock()
        for level, same_level in groupby(lines, key=lambda line: line[1]):
            cursor.insertText("".join(text + "\n" for text, _ in same_level), self._formats[level])
        cursor.endEditBlock()

    def _scroll_to_end(self):
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum()
        )

    def _on_scroll(self, value):
        """Track whether the user has scrolled away from the bottom."""
        sb = self.verticalScrollBar()
        self._auto_scroll = value >= sb.maximum() - 10


def _char_format(color: QColor) -> QTextCharFormat:
    fmt = QTextCharFormat()
    fmt.setForeground(color)
    return fmt
