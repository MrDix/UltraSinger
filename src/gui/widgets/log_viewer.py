"""Color-coded log viewer widget for conversion output."""

import re
from collections import deque
from itertools import groupby, islice

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit


# Strip ANSI escape sequences (SGR codes, cursor movement, etc.)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# The lines the viewer keeps: the last lines, and apart from them the last
# warnings and errors
_MAX_LINES = 5000

# Log levels (matched AFTER ANSI stripping)
_ERROR, _WARNING, _SUCCESS, _HEAD, _PLAIN = "error", "warning", "success", "head", "plain"
_ISSUES = (_ERROR, _WARNING)
# Warnings and errors: the first of their keywords in a line gives its level
# ("WARNING: [download] Got error: ..." is a warning)
_ISSUE_PATTERNS = [
    (re.compile(r"\[error\]|error:|\berror\b|traceback|exception", re.IGNORECASE), _ERROR),
    (re.compile(r"\[warn(?:ing)?\]|warn(?:ing)?:", re.IGNORECASE), _WARNING),
    # A step that failed ("Remote STT failed: ...", "<song>: failed (exit code 1)")
    (re.compile(r"\bfailed\s*[:(]", re.IGNORECASE), _ERROR),
]
# The other lines: the first pattern that matches gives the level
_OTHER_PATTERNS = [
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
# The first line of a Python traceback. Its frames are indented, and it ends
# with the exception and its message (but an exception group, whose lines
# are all indented, with its frames).
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):")
# A warning of Python's warnings module ("<file>:<line>: UserWarning: ..."),
# followed by its indented source line
_PYTHON_WARNING_RE = re.compile(r":\d+: \w*Warning: ")
# A progress bar (" 50%|#####     | 5/10"): never part of a warning or an error
_PROGRESS_RE = re.compile(r"\s*\d+%\|")


class LogViewer(QPlainTextEdit):
    """A read-only, color-coded log output widget.

    It keeps the lines it was given, so that it can show only the warnings
    and errors among them (``set_issues_only``) and all lines again.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("logOutput")
        self.setReadOnly(True)
        # The lines and the empty block after the last one
        self.setMaximumBlockCount(_MAX_LINES + 1)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # The lines as (text, level); the warnings and errors are kept apart
        # as well, so that the filter still shows those older than the last
        # lines (a song prints thousands of progress lines)
        self._lines: deque[tuple[str, str]] = deque(maxlen=_MAX_LINES)
        self._issues: deque[tuple[str, str]] = deque(maxlen=_MAX_LINES)
        self._issues_only = False
        self._formats = {level: _char_format(color) for level, color in _COLORS.items()}
        # A traceback is kept together (its frames, the exception and its
        # message), a Python warning with its source line
        self._in_traceback = False
        self._exception_due = False
        self._in_exception = False
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
        entry = (text, self._level(text))
        self._lines.append(entry)
        if entry[1] in _ISSUES:
            self._issues.append(entry)
        elif self._issues_only:
            return

        self._insert([entry])

        if self._auto_scroll:
            self._scroll_to_end()

    def set_issues_only(self, issues_only: bool):
        """Show only the warnings and errors, also those appended before, or all lines again.

        A view that follows the newest lines goes on following them. In a
        view scrolled up, the line at its top stays at the top: a warning,
        when all lines are shown again, among the lines around it; another
        line, when only the warnings and errors are shown, as the first
        warning or error after it.
        """
        if issues_only == self._issues_only:
            return
        top = None if self._auto_scroll else self._top_entry()
        self._issues_only = issues_only
        self.setPlaceholderText("No warnings or errors." if issues_only else "")
        self.clear()
        self._insert(self._issues if issues_only else self._lines)
        row = None if top is None else self._row_for(top)
        if row is None:
            self._scroll_to_end()
        else:
            self._scroll_to_row(row)

    def issues_only(self) -> bool:
        """Whether only the warnings and errors are shown."""
        return self._issues_only

    def clear_log(self):
        """Clear all log content."""
        self.clear()
        self._lines.clear()
        self._issues.clear()
        self._in_traceback = self._exception_due = self._in_exception = False
        self._warning_source_due = False

    def _level(self, text: str) -> str:
        """The level of a line; the lines of a traceback are an error together."""
        stripped = text.strip()
        indented = text[:1].isspace() and bool(stripped)
        if indented and self._in_traceback:
            return _ERROR
        if indented and self._warning_source_due and not _PROGRESS_RE.match(text):
            self._warning_source_due = False
            return _WARNING
        # The exception that ends a traceback, also one without "Error" in its
        # name ("KeyboardInterrupt"), and the further lines of its message, up
        # to an empty line or a line of UltraSinger or the GUI ("[Queue] ...")
        exception = bool(stripped) and (self._exception_due or (
            self._in_exception and not text.startswith("[") and not _PROGRESS_RE.match(text)))
        self._in_traceback = self._exception_due = self._warning_source_due = False
        self._in_exception = exception
        if exception:
            return _ERROR

        level = _keyword_level(text)
        if _TRACEBACK_RE.search(text):
            self._in_traceback = True
            # Not after the header of an exception group ("+ Exception Group
            # Traceback ...") or one in it ("| Traceback ...")
            self._exception_due = not text.lstrip().startswith(("+", "|"))
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

    def _top_entry(self):
        """The kept line at the top of the view (None in an empty view)."""
        shown = self._issues if self._issues_only else self._lines
        row = self.firstVisibleBlock().blockNumber()
        return shown[row] if 0 <= row < len(shown) else None

    def _row_for(self, top) -> int | None:
        """The row of the line to show at the top for the line that was at the top.

        None: none, the newest lines are shown.
        """
        if not self._issues_only:
            # The warning or error, else (older than the last lines) the first line
            return next((row for row, entry in enumerate(self._lines) if entry is top), 0)
        # The first warning or error from the line on
        start = next((row for row, entry in enumerate(self._lines) if entry is top), len(self._lines))
        issue = next((entry for entry in islice(self._lines, start, None) if entry[1] in _ISSUES), None)
        return next((row for row, entry in enumerate(self._issues) if entry is issue), None)

    def _scroll_to_row(self, row: int):
        """Show the line of the row at the top (or the last lines, if it is among them)."""
        self._scroll_to_end()
        # Scrolling up to a line puts it at the top
        self.setTextCursor(QTextCursor(self.document().findBlockByNumber(row)))
        self.ensureCursorVisible()
        sb = self.verticalScrollBar()
        self._auto_scroll = sb.value() >= sb.maximum() - 10

    def _scroll_to_end(self):
        """Show the newest lines, and follow them."""
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum()
        )
        # After a clear, the layout reports an old position on the way
        self._auto_scroll = True

    def _on_scroll(self, value):
        """Track whether the user has scrolled away from the bottom."""
        sb = self.verticalScrollBar()
        self._auto_scroll = value >= sb.maximum() - 10


def _keyword_level(text: str) -> str:
    issues = [(match.start(), level) for pattern, level in _ISSUE_PATTERNS
              if (match := pattern.search(text))]
    if issues:
        return min(issues)[1]
    return next((level for pattern, level in _OTHER_PATTERNS if pattern.search(text)), _PLAIN)


def _char_format(color: QColor) -> QTextCharFormat:
    fmt = QTextCharFormat()
    fmt.setForeground(color)
    return fmt
