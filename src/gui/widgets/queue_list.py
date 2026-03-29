"""Compact queue list widget for the sidebar with drag-and-drop support."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import QDesktopServices, QFontMetrics
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

# Try to get human-readable language names
try:
    from langcodes import Language
    def _lang_display_name(code: str) -> str:
        try:
            return Language.get(code).display_name()
        except Exception:
            return code.upper()
except ImportError:
    def _lang_display_name(code: str) -> str:  # type: ignore[misc]
        return code.upper()

import logging

logger = logging.getLogger(__name__)

# Simple colored dot for status (no confusing emoji)
_STATUS_COLORS = {
    "pending": "#a09888",
    "running": "#ffa726",
    "done": "#4caf50",
    "failed": "#ef5350",
    "cancelled": "#605848",
}


class QueueItemWidget(QWidget):
    """A single compact row representing a queue item.

    After conversion completes, an info line is shown below the title
    with language badge, lyrics source, and a folder-open button.
    """

    remove_requested = Signal(str)  # item_id
    settings_requested = Signal(str)  # item_id
    requeue_requested = Signal(str)  # item_id
    clone_requested = Signal(str)  # item_id

    def __init__(self, item: QueueItem, parent=None):
        super().__init__(parent)
        self._item_id = item.id
        self._output_folder = ""
        # Two-line tooltip: full title on line 1, source URL/path on line 2
        self._tooltip_title = item.title
        self._tooltip_source = item.input_source
        self._update_item_tooltip()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Row 1: title row (fixed 30px) ──────────────────────────
        title_row = QWidget()
        title_row.setFixedHeight(30)
        layout = QHBoxLayout(title_row)
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

        # Clone button — ⧉ (two overlapping squares, clearly recognizable)
        self._clone_btn = QPushButton("\u29C9")
        self._clone_btn.setFixedSize(22, 22)
        self._clone_btn.setToolTip("Clone (duplicate this item with same settings)")
        self._clone_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clone_btn.setStyleSheet(
            "font-size: 13px; color: #a09888; background: transparent; "
            "border: none; padding: 0px; margin: 0px;"
        )
        self._clone_btn.clicked.connect(
            lambda: self.clone_requested.emit(self._item_id)
        )
        layout.addWidget(self._clone_btn)

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

        outer.addWidget(title_row)

        # ── Row 2: info line (hidden until completion) ─────────────
        self._info_row = QWidget()
        self._info_row.setFixedHeight(20)
        self._info_row.setVisible(False)
        info_layout = QHBoxLayout(self._info_row)
        info_layout.setContentsMargins(22, 0, 4, 2)
        info_layout.setSpacing(4)

        # Language badge (e.g. "en" with green/red background)
        self._lang_badge = QLabel("")
        self._lang_badge.setFixedHeight(16)
        self._lang_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lang_badge.setStyleSheet(
            "font-size: 9px; font-weight: bold; color: #fff; "
            "background: #4caf50; border-radius: 3px; "
            "padding: 0px 4px; margin: 0px;"
        )
        self._lang_badge.setVisible(False)
        info_layout.addWidget(self._lang_badge)

        # Lyrics source label
        self._lyrics_label = QLabel("")
        self._lyrics_label.setStyleSheet(
            "font-size: 9px; color: #807060; background: transparent;"
        )
        info_layout.addWidget(self._lyrics_label, 1)

        # Info button (ℹ) — view settings used for this conversion
        self._info_btn = QPushButton("\u2139\uFE0F")  # ℹ️
        self._info_btn.setFixedSize(18, 18)
        self._info_btn.setToolTip("View settings used for this conversion")
        self._info_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._info_btn.setStyleSheet(
            "font-size: 11px; color: #a09888; background: transparent; "
            "border: none; padding: 0px; margin: 0px;"
        )
        self._info_btn.clicked.connect(
            lambda: self.settings_requested.emit(self._item_id)
        )
        info_layout.addWidget(self._info_btn)

        # Folder-open button (cross-platform file manager)
        self._folder_btn = QPushButton("\U0001F4C2")  # 📂
        self._folder_btn.setFixedSize(18, 18)
        self._folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._folder_btn.setStyleSheet(
            "font-size: 11px; background: transparent; "
            "border: none; padding: 0px; margin: 0px;"
        )
        self._folder_btn.clicked.connect(self._open_output_folder)
        self._folder_btn.setVisible(False)
        info_layout.addWidget(self._folder_btn)

        # Re-queue button — shown for completed items where re-conversion
        # with different settings (e.g. manual language) may improve results
        self._requeue_btn = QPushButton("\U0001F504")  # 🔄
        self._requeue_btn.setFixedSize(18, 18)
        self._requeue_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._requeue_btn.setStyleSheet(
            "font-size: 11px; background: transparent; "
            "border: none; padding: 0px; margin: 0px;"
        )
        self._requeue_btn.setToolTip(
            "Re-queue this song.\n"
            "Tip: Use per-song settings (\u2699) to set\n"
            "the language manually for better results."
        )
        self._requeue_btn.clicked.connect(
            lambda: self.requeue_requested.emit(self._item_id)
        )
        self._requeue_btn.setVisible(False)
        info_layout.addWidget(self._requeue_btn)

        outer.addWidget(self._info_row)

        self.update_status(item.status)

    @property
    def item_id(self) -> str:
        return self._item_id

    def _update_item_tooltip(self):
        """Build a two-line tooltip: full title + source URL/path."""
        lines = [self._tooltip_title]
        if self._tooltip_source and self._tooltip_source != self._tooltip_title:
            lines.append(self._tooltip_source)
        self.setToolTip("\n".join(lines))

    def update_status(self, status: str):
        """Update the visual status of this item."""
        color = _STATUS_COLORS.get(status, "#a09888")
        self._status_dot.setStyleSheet(
            f"font-size: 8px; color: {color}; background: transparent;"
        )

        # Remove button and gear only for pending items.
        # Completed items show the ℹ button in the info row instead.
        self._remove_btn.setVisible(status == "pending")
        self._gear_btn.setVisible(status == "pending")
        self._clone_btn.setVisible(status == "pending")

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

    def set_result_info(self, info: dict):
        """Show the info line with conversion result metadata."""
        if not info:
            return

        # Lyrics source determines badge color:
        #   Green  = synced lyrics (reference pipeline, best quality)
        #   Orange = plain lyrics (Whisper + correction, medium quality)
        #   Red    = transcribed only (Whisper, lowest quality)
        source = info.get("lyrics_source", "")
        # Combine source + fallback flag into display key
        if source == "synced" and info.get("whisper_fallback"):
            source = "synced (fallback)"
        _SOURCE_COLORS = {
            "synced": "#4caf50",              # green — best
            "synced (fallback)": "#ffa726",   # orange — had synced but fell back
            "plain": "#ffa726",               # orange — medium
            "transcribed": "#ef5350",         # red — lowest
            "instrumental": "#7e57c2",        # purple — no vocals
        }
        _SOURCE_LABELS = {
            "synced": "Synced lyrics",
            "synced (fallback)": "Synced lyrics (Whisper fallback)",
            "plain": "Plain lyrics",
            "transcribed": "Transcribed",
            "instrumental": "Instrumental",
        }
        badge_bg = _SOURCE_COLORS.get(source, "#a09888")
        source_text = _SOURCE_LABELS.get(source, source)
        self._lyrics_label.setText(source_text)

        # Language badge
        lang = info.get("language", "")
        if lang:
            self._lang_badge.setText(lang.upper())
            self._lang_badge.setStyleSheet(
                f"font-size: 9px; font-weight: bold; color: #fff; "
                f"background: {badge_bg}; border-radius: 3px; "
                f"padding: 0px 4px; margin: 0px;"
            )
            lang_name = _lang_display_name(lang)
            pipeline_name = "Reference" if info.get("pipeline") == "reference" else "Whisper"
            tooltip = f"Language: {lang.upper()} ({lang_name}) — Pipeline: {pipeline_name}"
            initial = info.get("initial_language", "")
            if initial and initial != lang:
                initial_name = _lang_display_name(initial)
                tooltip += (
                    f"\nInitial detection: {initial.upper()} ({initial_name}) (wrong)"
                    f" \u2192 Whisper corrected to {lang.upper()} ({lang_name})"
                )
            if info.get("reference_recovered"):
                tooltip += "\nReference pipeline recovered after correction"
            elif info.get("whisper_fallback"):
                tooltip += "\nReference pipeline fell back to Whisper"
            self._lang_badge.setToolTip(tooltip)
            self._lang_badge.setVisible(True)

        # Folder button — show whenever output_folder is known
        output_folder = info.get("output_folder", "")
        if output_folder:
            self._output_folder = output_folder
            self._folder_btn.setToolTip(output_folder)
            self._folder_btn.setVisible(True)

        # Re-queue button — always visible for completed items
        self._requeue_btn.setVisible(True)

        self._info_row.setVisible(True)

    def _open_output_folder(self):
        """Open the song output folder in the system file manager."""
        folder = self._output_folder
        if not folder or not Path(folder).is_dir():
            return
        folder_path = str(Path(folder).resolve())
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))
        if not opened:
            logger.warning("Failed to open folder: %s", folder_path)

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
    """QLabel that truncates text with '...' when space is tight."""

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
    requeue_requested = Signal(str)  # item_id
    clone_requested = Signal(str)  # item_id
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
        widget.requeue_requested.connect(self.requeue_requested.emit)
        widget.clone_requested.connect(self.clone_requested.emit)
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

    def set_result_info(self, item_id: str, info: dict):
        """Show conversion result info on a completed queue item."""
        widget = self._item_widgets.get(item_id)
        if widget:
            widget.set_result_info(info)

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
