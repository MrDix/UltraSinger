"""Compact queue list widget for the sidebar."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ..models import QueueItem

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
        self.setFixedHeight(28)
        self.setToolTip(item.input_source)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # Status dot (small colored circle)
        self._status_dot = QLabel("\u2B24")  # ⬤
        self._status_dot.setFixedWidth(14)
        self._status_dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_dot)

        # Title (just the name, no type emoji)
        self._title = QLabel(item.title)
        self._title.setStyleSheet(
            "font-size: 12px; color: #f0dfc0; background: transparent;"
        )
        self._title.setWordWrap(False)
        layout.addWidget(self._title, 1)

        # Settings gear button (only for pending items)
        self._gear_btn = QPushButton("\u2699")
        self._gear_btn.setObjectName("ghostButton")
        self._gear_btn.setFixedSize(20, 20)
        self._gear_btn.setToolTip("Per-song settings (click to override defaults)")
        self._gear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._gear_btn.setStyleSheet(
            "font-size: 13px; color: #f0dfc0; background: transparent; "
            "padding: 0px;"
        )
        self._gear_btn.clicked.connect(
            lambda: self.settings_requested.emit(self._item_id)
        )
        layout.addWidget(self._gear_btn)

        # Remove button (only for pending items)
        self._remove_btn = QPushButton("\u2715")
        self._remove_btn.setObjectName("ghostButton")
        self._remove_btn.setFixedSize(20, 20)
        self._remove_btn.setToolTip("Remove from queue")
        self._remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._remove_btn.setStyleSheet(
            "font-size: 11px; color: #a09888; background: transparent; "
            "padding: 0px;"
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

        # Only show gear/remove buttons for pending items
        self._gear_btn.setVisible(status == "pending")
        self._remove_btn.setVisible(status == "pending")

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
        color = "#00d4d4" if has_overrides else "#f0dfc0"
        self._gear_btn.setStyleSheet(
            f"font-size: 13px; color: {color}; background: transparent; "
            "padding: 0px;"
        )
        self._gear_btn.setToolTip(
            "Per-song settings (custom)" if has_overrides
            else "Per-song settings (click to override defaults)"
        )


class QueueListWidget(QWidget):
    """Scrollable list of queue items for the sidebar."""

    remove_requested = Signal(str)  # item_id
    settings_requested = Signal(str)  # item_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._item_widgets: dict[str, QueueItemWidget] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Scroll area for items — no max height, fills available space
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._container = QWidget()
        self._items_layout = QVBoxLayout(self._container)
        self._items_layout.setContentsMargins(0, 0, 0, 0)
        self._items_layout.setSpacing(1)
        self._items_layout.addStretch(1)

        scroll.setWidget(self._container)
        layout.addWidget(scroll, 1)  # stretch=1 → fills parent

        # Empty state label
        self._empty_label = QLabel("Drop files or queue from browser")
        self._empty_label.setObjectName("caption")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(
            "color: #605848; font-size: 11px; padding: 8px;"
        )
        layout.addWidget(self._empty_label)

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

    def _update_empty_state(self):
        """Show/hide empty state label."""
        has_items = len(self._item_widgets) > 0
        self._empty_label.setVisible(not has_items)
