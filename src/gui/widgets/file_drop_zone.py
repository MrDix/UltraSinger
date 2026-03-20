"""Drag-and-drop file input widget."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QFileDialog, QVBoxLayout, QWidget

# Supported file extensions
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm", ".mov"}
TXT_EXTENSIONS = {".txt"}
ALL_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS | TXT_EXTENSIONS


class FileDropZone(QWidget):
    """A drag-and-drop zone for selecting audio/video/txt files."""

    file_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("fileDropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(70)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Local file picker")
        self.setProperty("dragOver", False)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(2)

        # File type hint label
        self._type_label = QLabel("Audio  \u2022  Video  \u2022  TXT")
        self._type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._type_label.setStyleSheet(
            "font-size: 12px; font-weight: 600; color: #a09888; "
            "background: transparent; letter-spacing: 0.5px;"
        )
        layout.addWidget(self._type_label)

        self._text_label = QLabel("click or drag here")
        self._text_label.setObjectName("caption")
        self._text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._text_label.setStyleSheet(
            "font-size: 11px; color: #605848; background: transparent;"
        )
        layout.addWidget(self._text_label)

        self._current_file: str = ""

    @property
    def current_file(self) -> str:
        return self._current_file

    def set_file(self, path: str):
        """Set the file path programmatically."""
        self._current_file = path
        if path:
            name = Path(path).name
            self._type_label.setText(name)
            self._type_label.setStyleSheet(
                "font-size: 12px; font-weight: 600; color: #e91e63; "
                "background: transparent;"
            )
            self._text_label.setText("click to change")
        else:
            self._type_label.setText("Audio  \u2022  Video  \u2022  TXT")
            self._type_label.setStyleSheet(
                "font-size: 12px; font-weight: 600; color: #a09888; "
                "background: transparent; letter-spacing: 0.5px;"
            )
            self._text_label.setText("click or drag here")

    def _validate_and_set(self, path: str):
        """Validate extension, set file, and emit signal."""
        ext = Path(path).suffix.lower()
        if ext in ALL_EXTENSIONS:
            self.set_file(path)
            self.file_selected.emit(path)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._open_file_dialog()
        else:
            super().mousePressEvent(event)

    def keyPressEvent(self, event):
        """Open file dialog on Enter or Space key press."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self._open_file_dialog()
            event.accept()
            return
        super().keyPressEvent(event)

    def _open_file_dialog(self):
        ext_list = " ".join(f"*{e}" for e in sorted(ALL_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio, Video, or UltraStar TXT File",
            "",
            f"Media & TXT Files ({ext_list});;All Files (*)",
        )
        if path:
            self._validate_and_set(path)

    def _set_drag_over(self, active: bool):
        """Update the drag-over visual state."""
        self.setProperty("dragOver", active)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                ext = Path(url.toLocalFile()).suffix.lower()
                if ext in ALL_EXTENSIONS:
                    event.acceptProposedAction()
                    self._set_drag_over(True)
                    return
        event.ignore()

    def dragLeaveEvent(self, _event):
        self._set_drag_over(False)

    def dropEvent(self, event):
        self._set_drag_over(False)
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                self._validate_and_set(url.toLocalFile())
                event.acceptProposedAction()
