"""Console tab with log output and batch progress."""

import logging
from pathlib import Path

from PySide6.QtCore import QElapsedTimer, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .widgets import AnimatedButton, LogViewer

logger = logging.getLogger(__name__)


class QueueTab(QWidget):
    """Displays conversion progress, log output, and batch status."""

    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._output_folder = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Header
        header = QLabel("Console")
        header.setObjectName("sectionHeader")
        layout.addWidget(header)

        # Status bar
        status_widget = QWidget()
        status_widget.setObjectName("settingsCard")
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(16, 12, 16, 12)

        self._status_icon = QLabel("\u23F9")
        self._status_icon.setStyleSheet("font-size: 20px;")
        status_layout.addWidget(self._status_icon)

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        status_layout.addWidget(self._status_label, 1)

        self._stage_label = QLabel("")
        self._stage_label.setObjectName("caption")
        status_layout.addWidget(self._stage_label)

        self._elapsed_label = QLabel("")
        self._elapsed_label.setObjectName("caption")
        status_layout.addWidget(self._elapsed_label)

        layout.addWidget(status_widget)

        # Log viewer
        self._log = LogViewer()
        layout.addWidget(self._log, 1)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self._cancel_btn = QPushButton("Cancel All")
        self._cancel_btn.setObjectName("dangerButton")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)

        btn_row.addStretch()

        self._clear_log_btn = QPushButton("Clear Log")
        self._clear_log_btn.setObjectName("ghostButton")
        self._clear_log_btn.clicked.connect(self._log.clear_log)
        btn_row.addWidget(self._clear_log_btn)

        self._open_folder_btn = AnimatedButton("Open Output Folder", primary=False)
        self._open_folder_btn.setEnabled(False)
        self._open_folder_btn.clicked.connect(self._open_output_folder)
        btn_row.addWidget(self._open_folder_btn)

        layout.addLayout(btn_row)

        # Timer for elapsed time
        self._elapsed_timer = QElapsedTimer()
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self._update_elapsed)

    def append_log(self, text: str):
        """Append a line to the log viewer (public API for QueueManager)."""
        self._log.append_line(text)

    def on_stage_changed(self, stage: str):
        """Update the current stage label."""
        self._stage_label.setText(stage)

    def on_queue_started(self):
        """Handle batch processing start."""
        self._log.clear_log()
        self._status_icon.setText("\u23F3")
        self._status_icon.setStyleSheet("font-size: 20px; color: #ffa726;")
        self._status_label.setText("Processing queue...")
        self._status_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        self._stage_label.setText("")
        self._cancel_btn.setEnabled(True)
        self._open_folder_btn.setEnabled(False)
        self._elapsed_label.setText("00:00")
        self._elapsed_timer.start()
        self._tick_timer.start()

    def on_queue_finished(self, failed_count: int = 0, cancelled: bool = False):
        """Handle batch processing completion.

        Args:
            failed_count: Number of items that failed.
            cancelled: Whether the queue was cancelled by the user.
        """
        self._tick_timer.stop()
        self._cancel_btn.setEnabled(False)

        if cancelled:
            self._status_icon.setText("\u23F9")
            self._status_icon.setStyleSheet("font-size: 20px;")
            self._status_label.setText("Queue cancelled")
            self._status_label.setStyleSheet(
                "font-size: 16px; font-weight: 600; color: #9e9e9e;"
            )
        elif failed_count > 0:
            self._status_icon.setText("\u26A0")
            self._status_icon.setStyleSheet("font-size: 20px;")
            self._status_label.setText(
                f"Completed with {failed_count} error(s)"
            )
            self._status_label.setStyleSheet(
                "font-size: 16px; font-weight: 600; color: #ef5350;"
            )
        else:
            self._status_icon.setText("\u2705")
            self._status_icon.setStyleSheet("font-size: 20px;")
            self._status_label.setText("Queue completed!")
            self._status_label.setStyleSheet(
                "font-size: 16px; font-weight: 600; color: #4caf50;"
            )
        self._stage_label.setText("")

    def set_output_folder(self, folder: str):
        """Set the output folder for the Open Output Folder button."""
        self._output_folder = folder
        self._open_folder_btn.setEnabled(
            bool(folder) and Path(folder).exists()
        )

    def _on_cancel(self):
        self._status_label.setText("Cancelling...")
        self.cancel_requested.emit()

    def _update_elapsed(self):
        elapsed_ms = self._elapsed_timer.elapsed()
        secs = int(elapsed_ms / 1000)
        mins = secs // 60
        secs = secs % 60
        self._elapsed_label.setText(f"{mins:02d}:{secs:02d}")

    def _open_output_folder(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        folder = self._output_folder
        if not folder or not Path(folder).exists():
            return

        opened = QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(Path(folder).resolve()))
        )
        if not opened:
            logger.error("Failed to open folder %s", folder)
            self._log.append_line("[Error] Could not open folder.")
