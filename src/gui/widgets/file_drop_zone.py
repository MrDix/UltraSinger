"""Drag-and-drop file input widget."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QFileDialog, QVBoxLayout, QWidget

# Supported file extensions
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm", ".mov"}
ALL_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


class FileDropZone(QWidget):
    """A drag-and-drop zone for selecting audio/video files."""

    file_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("fileDropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Local file picker")
        self.setProperty("dragOver", False)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._icon_label = QLabel("\U0001F4C1")
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_label.setStyleSheet("font-size: 32px; background: transparent;")
        layout.addWidget(self._icon_label)

        self._text_label = QLabel("Drag file here or click to select")
        self._text_label.setObjectName("caption")
        self._text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._text_label.setStyleSheet("background: transparent;")
        layout.addWidget(self._text_label)

        self._file_label = QLabel("")
        self._file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._file_label.setWordWrap(True)
        self._file_label.setStyleSheet("color: #e91e63; font-size: 13px; background: transparent;")
        self._file_label.hide()
        layout.addWidget(self._file_label)

        self._current_file: str = ""

    @property
    def current_file(self) -> str:
        return self._current_file

    def set_file(self, path: str):
        """Set the file path programmatically."""
        self._current_file = path
        if path:
            name = Path(path).name
            self._file_label.setText(name)
            self._file_label.show()
            self._text_label.setText("Click to change file")
            self._icon_label.setText("\U0001F3B5")
        else:
            self._file_label.hide()
            self._text_label.setText("Drag file here or click to select")
            self._icon_label.setText("\U0001F4C1")

    def mousePressEvent(self, _event):
        self._open_file_dialog()

    def keyPressEvent(self, event):
        """Open file dialog on Enter or Space key press."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self._open_file_dialog()
            event.accept()
            return
        else:
            super().keyPressEvent(event)

    def _open_file_dialog(self):
        ext_list = " ".join(f"*{e}" for e in sorted(ALL_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio or Video File",
            "",
            f"Media Files ({ext_list});;All Files (*)",
        )
        if path:
            self.set_file(path)
            self.file_selected.emit(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                ext = Path(url.toLocalFile()).suffix.lower()
                if ext in ALL_EXTENSIONS:
                    event.acceptProposedAction()
                    self.setProperty("dragOver", True)
                    self.style().unpolish(self)
                    self.style().polish(self)
                    return
        event.ignore()

    def dragLeaveEvent(self, _event):
        self.setProperty("dragOver", False)
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, event):
        self.setProperty("dragOver", False)
        self.style().unpolish(self)
        self.style().polish(self)

        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                path = url.toLocalFile()
                self.set_file(path)
                self.file_selected.emit(path)
                event.acceptProposedAction()
