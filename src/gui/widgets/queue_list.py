"""Compact queue list widget for the sidebar with drag-and-drop support."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ..models import QueueItem
from .file_drop_zone import ALL_EXTENSIONS

# Simple colored dot for status (no confusing emoji)
_STATUS_COLORS = {
    "pending": "#a09888",
    "running": "#ffa726",
    "done": "#4caf50",
    "failed": "#ef5350",
    "cancelled": "#605848",
}


class QueueItemWidget(QWidget):
    """A single compact row representing a queue item."""

    remove_requested = Signal(str)  # item_id
    settings_requested = Signal(str)  # item_id

    def __init__(self, item: QueueItem, parent=None):
        super().__init__(parent)
        self._item_id = item.id
        self.setFixedHeight(30)
        self.setToolTip(item.input_source)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # Status dot (small colored circle)
        self._status_dot = QLabel("\u2B24")  # ⬤
        self._status_dot.setFixedWidth(14)
        self._status_dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_dot)

        # Title — must shrink so gear+remove buttons stay visible
        self._full_title = item.title
        self._title = _ElidingLabel(item.title)
        self._title.setStyleSheet(
            "font-size: 12px; color: #f0dfc0; background: transparent;"
        )
        sp = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sp.setHorizontalStretch(1)
        self._title.setSizePolicy(sp)
        self._title.setMinimumWidth(30)
        layout.addWidget(self._title)

        # Settings gear button (only for pending items)
        self._gear_btn = QPushButton("\u2699")
        self._gear_btn.setFixedSize(22, 22)
        self._gear_btn.setToolTip("Per-song settings (click to override defaults)")
        self._gear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._gear_btn.setStyleSheet(
            "font-size: 13px; color: #a09888; background: transparent; "
            "border: none; padding: 0px; margin: 0px;"
        )
        self._gear_btn.clicked.connect(
            lambda: self.settings_requested.emit(self._item_id)
        )
        layout.addWidget(self._gear_btn)

        # Remove button — U+00D7 (clean ×, no serifs)
        self._remove_btn = QPushButton("\u00D7")
        self._remove_btn.setFixedSize(22, 22)
        self._remove_btn.setToolTip("Remove from queue")
        self._remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._remove_btn.setStyleSheet(
            "font-size: 15px; color: #a09888; background: transparent; "
            "border: none; padding: 0px; margin: 0px;"
        )
        self._remove_btn.clicked.connect(
            lambda: self.remove_requested.emit(self._item_id)
        )
        layout.addWidget(self._remove_btn)

        self.update_status(item.status)

    @property
    def item_id(self) -> str:
        return self._item_id

    def update_status(self, status: str):
        """Update the visual status of this item."""
        color = _STATUS_COLORS.get(status, "#a09888")
        self._status_dot.setStyleSheet(
            f"font-size: 8px; color: {color}; background: transparent;"
        )

        # Remove button only for pending items; gear always visible
        self._remove_btn.setVisible(status == "pending")

        # Gear: editable icon for pending, read-only info icon for others
        if status == "pending":
            self._gear_btn.setText("\u2699")
            self._gear_btn.setToolTip("Per-song settings (click to override defaults)")
        else:
            self._gear_btn.setText("\u2139")  # ℹ info icon
            self._gear_btn.setToolTip("View settings used for this conversion")

        # Dim completed/cancelled items
        if status in ("done", "failed", "cancelled"):
            self._title.setStyleSheet(
                "font-size: 12px; color: #605848; background: transparent;"
            )
        elif status == "running":
            self._title.setStyleSheet(
                "font-size: 12px; color: #ffa726; font-weight: bold; "
                "background: transparent;"
            )
        else:
            self._title.setStyleSheet(
                "font-size: 12px; color: #f0dfc0; background: transparent;"
            )

    def set_has_overrides(self, has_overrides: bool):
        """Show visual indicator when per-song overrides are active."""
        color = "#00d4d4" if has_overrides else "#a09888"
        self._gear_btn.setStyleSheet(
            f"font-size: 13px; color: {color}; background: transparent; "
            "border: none; padding: 0px; margin: 0px;"
        )
        self._gear_btn.setToolTip(
            "Per-song settings (custom)" if has_overrides
            else "Per-song settings (click to override defaults)"
        )


class _ElidingLabel(QLabel):
    """QLabel that truncates text with '…' when space is tight."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._full_text = text
        self.setWordWrap(False)

    def set_full_text(self, text: str):
        self._full_text = text
        self.update()

    def paintEvent(self, event):
        """Draw elided text instead of default rendering."""
        from PySide6.QtGui import QPainter

        painter = QPainter(self)
        fm = self.fontMetrics()
        elided = fm.elidedText(
            self._full_text, Qt.TextElideMode.ElideRight, self.width()
        )
        painter.setPen(self.palette().windowText().color())
        # Respect stylesheet color by parsing it
        ss = self.styleSheet()
        if "color:" in ss:
            import re

            m = re.search(r"color:\s*(#[0-9a-fA-F]+)", ss)
            if m:
                from PySide6.QtGui import QColor

                painter.setPen(QColor(m.group(1)))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignVCenter, elided)
        painter.end()


class QueueListWidget(QWidget):
    """Scrollable list of queue items with drag-and-drop file support."""

    remove_requested = Signal(str)  # item_id
    settings_requested = Signal(str)  # item_id
    file_dropped = Signal(str)  # file path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._item_widgets: dict[str, QueueItemWidget] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Scroll area for items — no max height, fills available space
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        self._container = QWidget()
        self._items_layout = QVBoxLayout(self._container)
        self._items_layout.setContentsMargins(0, 0, 0, 0)
        self._items_layout.setSpacing(1)
        self._items_layout.addStretch(1)

        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll, 1)  # stretch=1 → fills parent

        # Drop hint (always visible at the bottom)
        self._drop_hint = QLabel(
            "Drop audio, video or .txt files here"
        )
        self._drop_hint.setObjectName("caption")
        self._drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_hint.setCursor(Qt.CursorShape.PointingHandCursor)
        self._drop_hint.setStyleSheet(
            "color: #605848; font-size: 10px; padding: 4px 0px;"
        )
        self._drop_hint.mousePressEvent = self._on_hint_clicked
        layout.addWidget(self._drop_hint)

        self._update_empty_state()

    def add_item(self, item: QueueItem):
        """Add a queue item widget."""
        widget = QueueItemWidget(item, self._container)
        widget.remove_requested.connect(self.remove_requested.emit)
        widget.settings_requested.connect(self.settings_requested.emit)
        self._item_widgets[item.id] = widget

        # Insert before the stretch
        count = self._items_layout.count()
        self._items_layout.insertWidget(count - 1, widget)
        self._update_empty_state()

    def remove_item(self, item_id: str):
        """Remove a queue item widget."""
        widget = self._item_widgets.pop(item_id, None)
        if widget:
            self._items_layout.removeWidget(widget)
            widget.deleteLater()
        self._update_empty_state()

    def update_status(self, item_id: str, status: str):
        """Update the status of a queue item."""
        widget = self._item_widgets.get(item_id)
        if widget:
            widget.update_status(status)

    def set_has_overrides(self, item_id: str, has_overrides: bool):
        """Update the override indicator for a queue item."""
        widget = self._item_widgets.get(item_id)
        if widget:
            widget.set_has_overrides(has_overrides)

    # ── Drag-and-drop support ──────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    ext = Path(url.toLocalFile()).suffix.lower()
                    if ext in ALL_EXTENSIONS:
                        event.acceptProposedAction()
                        self._drop_hint.setStyleSheet(
                            "color: #ffa726; font-size: 10px; "
                            "padding: 4px 0px; font-weight: bold;"
                        )
                        return
        event.ignore()

    def dragLeaveEvent(self, _event):
        self._drop_hint.setStyleSheet(
            "color: #605848; font-size: 10px; padding: 4px 0px;"
        )

    def dropEvent(self, event):
        self._drop_hint.setStyleSheet(
            "color: #605848; font-size: 10px; padding: 4px 0px;"
        )
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    ext = Path(path).suffix.lower()
                    if ext in ALL_EXTENSIONS:
                        self.file_dropped.emit(path)
            event.acceptProposedAction()

    def _on_hint_clicked(self, _event):
        """Open a file dialog when the drop hint is clicked."""
        ext_list = " ".join(f"*{e}" for e in sorted(ALL_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio, Video, or UltraStar TXT File",
            "",
            f"Media & TXT Files ({ext_list});;All Files (*)",
        )
        if path:
            self.file_dropped.emit(path)

    # ── Internal ─────────────────────────────────────────────────

    def _update_empty_state(self):
        """Update the drop hint text based on queue state."""
        has_items = len(self._item_widgets) > 0
        if has_items:
            self._drop_hint.setText(
                "Drop files to add more"
            )
        else:
            self._drop_hint.setText(
                "Drop audio, video or .txt files here"
            )
