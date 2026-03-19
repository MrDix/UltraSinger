"""Custom toggle switch widget with smooth animation."""

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import QAbstractButton


class ToggleSwitch(QAbstractButton):
    """Animated toggle switch that replaces checkboxes for on/off settings."""

    toggled_signal = Signal(bool)

    # Colors
    _BG_OFF = QColor(60, 60, 80)
    _BG_ON = QColor(233, 30, 99)
    _KNOB = QColor(255, 255, 255)
    _KNOB_DISABLED = QColor(120, 120, 140)

    def __init__(self, parent=None, checked: bool = False):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self._offset = 4.0 if not checked else 26.0
        self.setFixedSize(QSize(50, 28))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._animation = QPropertyAnimation(self, b"offset", self)
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

    def get_offset(self) -> float:
        return self._offset

    def set_offset(self, value: float):
        self._offset = value
        self.update()

    offset = Property(float, get_offset, set_offset)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Track
        track_path = QPainterPath()
        track_path.addRoundedRect(0, 0, self.width(), self.height(), 14, 14)
        if self.isEnabled():
            bg = self._BG_ON if self.isChecked() else self._BG_OFF
        else:
            bg = QColor(40, 40, 55)
        p.fillPath(track_path, bg)

        # Knob
        knob_color = self._KNOB if self.isEnabled() else self._KNOB_DISABLED
        p.setBrush(knob_color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(int(self._offset), 4, 20, 20)
        p.end()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._animate()
            self.toggled_signal.emit(self.isChecked())

    def setChecked(self, checked: bool):
        super().setChecked(checked)
        self._offset = 26.0 if checked else 4.0
        self.update()

    def _animate(self):
        self._animation.stop()
        self._animation.setStartValue(self._offset)
        self._animation.setEndValue(26.0 if self.isChecked() else 4.0)
        self._animation.start()

    def sizeHint(self):
        return QSize(50, 28)
