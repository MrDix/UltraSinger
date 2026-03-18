"""Conversion queue and progress tab with log output."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .ultrasinger_runner import UltraSingerRunner
from .widgets import AnimatedButton, LogViewer


class QueueTab(QWidget):
    """Displays conversion progress, log output, and completed conversions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._runner = UltraSingerRunner(self)
        self._current_output_folder = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Header
        header = QLabel("Conversion Queue")
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

        self._cancel_btn = QPushButton("Cancel")
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

        # Connect runner signals
        self._runner.line_output.connect(self._log.append_line)
        self._runner.stage_changed.connect(self._on_stage_changed)
        self._runner.finished.connect(self._on_finished)

        # Timer for elapsed time
        from PySide6.QtCore import QTimer, QElapsedTimer

        self._elapsed_timer = QElapsedTimer()
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self._update_elapsed)

    @property
    def runner(self) -> UltraSingerRunner:
        return self._runner

    def append_log(self, text: str):
        """Append a line to the log viewer (public API)."""
        self._log.append_line(text)

    def start_conversion(self, args: list[str], output_folder: str = ""):
        """Start a new conversion with the given CLI arguments."""
        if self._runner.is_running:
            self._log.append_line("[GUI] A conversion is already running!")
            return

        self._current_output_folder = output_folder
        self._log.clear_log()
        self._status_icon.setText("\u23F3")
        self._status_icon.setStyleSheet("font-size: 20px; color: #ffa726;")
        self._status_label.setText("Converting...")
        self._status_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        self._stage_label.setText("")
        self._cancel_btn.setEnabled(True)
        self._open_folder_btn.setEnabled(False)
        self._elapsed_label.setText("00:00")

        self._elapsed_timer.start()
        self._tick_timer.start()

        self._runner.start(args)

    def _on_stage_changed(self, stage: str):
        self._stage_label.setText(stage)

    def _on_cancel(self):
        self._runner.cancel()
        self._status_label.setText("Cancelling...")

    def _on_finished(self, exit_code: int):
        self._tick_timer.stop()
        self._cancel_btn.setEnabled(False)

        if exit_code == 0:
            self._status_icon.setText("\u2705")
            self._status_icon.setStyleSheet("font-size: 20px;")
            self._status_label.setText("Completed!")
            self._status_label.setStyleSheet(
                "font-size: 16px; font-weight: 600; color: #4caf50;"
            )
            self._open_folder_btn.setEnabled(bool(self._current_output_folder))
            self._log.append_line("")
            self._log.append_line("[Success] Conversion completed successfully!")
        elif exit_code == -2:
            self._status_icon.setText("\u23F9")
            self._status_icon.setStyleSheet("font-size: 20px;")
            self._status_label.setText("Cancelled")
            self._status_label.setStyleSheet(
                "font-size: 16px; font-weight: 600; color: #ffa726;"
            )
        else:
            self._status_icon.setText("\u274C")
            self._status_icon.setStyleSheet("font-size: 20px;")
            self._status_label.setText(f"Failed (exit code {exit_code})")
            self._status_label.setStyleSheet(
                "font-size: 16px; font-weight: 600; color: #ef5350;"
            )

    def _update_elapsed(self):
        elapsed_ms = self._elapsed_timer.elapsed()
        secs = int(elapsed_ms / 1000)
        mins = secs // 60
        secs = secs % 60
        self._elapsed_label.setText(f"{mins:02d}:{secs:02d}")

    def _open_output_folder(self):
        import subprocess
        import sys

        folder = self._current_output_folder
        if not folder or not Path(folder).exists():
            return

        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", folder])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except OSError as e:
            logger.error("Failed to open folder %s: %s", folder, e)
            self._log.append_line(f"[Error] Could not open folder: {e}")
