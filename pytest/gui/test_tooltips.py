"""Every control on the Settings page and in the training window has its own tooltip.

Controls inside a row container get the row's tooltip from SettingsCard.add_row, so
hovering them shows the explanation without relying on the tooltip of a parent widget.
"""

import os
import unittest

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
    QPushButton,
    QWidget,
)

from src.gui.models import LLMProvider
from src.gui.preferences_tab import PreferencesTab
from src.gui.training_dialog import TrainingDialog
from src.gui.widgets import SettingsCard

_app = QApplication.instance() or QApplication([])

_CONTROLS = (QLineEdit, QAbstractButton, QComboBox, QAbstractSpinBox)


def _controls_without_tooltip(root: QWidget) -> list[str]:
    missing = []
    for w in root.findChildren(QWidget):
        if not isinstance(w, _CONTROLS):
            continue
        if isinstance(w, QLineEdit) and isinstance(w.parent(), (QComboBox, QAbstractSpinBox)):
            continue  # the text field inside a combo or spin box
        if not w.toolTip():
            text = w.text() if isinstance(w, (QLineEdit, QAbstractButton)) else ""
            hint = w.placeholderText() if isinstance(w, QLineEdit) else ""
            missing.append(f"{type(w).__name__} {text or hint!r}")
    return missing


class TestTooltips(unittest.TestCase):
    def test_settings_page(self):
        provider = LLMProvider(name="Local", api_base_url="http://localhost:1/v1",
                               default_model="m", is_default=True)
        page = PreferencesTab({"llm_providers": [provider.to_dict()]}, None)
        self.assertEqual(_controls_without_tooltip(page), [])

    def test_training_window(self):
        self.assertEqual(_controls_without_tooltip(TrainingDialog({})), [])


class TestRowTooltip(unittest.TestCase):
    def test_controls_in_a_row_container_get_the_row_tooltip(self):
        container = QWidget()
        edit = QLineEdit(container)
        browse = QPushButton("Browse", container)
        browse.setToolTip("Pick a file.")
        card = SettingsCard()  # keeps the row alive
        card.add_row("Path", container, "Where the file is.")
        self.assertEqual(edit.toolTip(), "Where the file is.")
        self.assertEqual(browse.toolTip(), "Pick a file.")  # its own tooltip is kept

    def test_inner_widgets_of_a_control_are_left_alone(self):
        combo = QComboBox()
        combo.setEditable(True)
        card = SettingsCard()
        card.add_row("Model", combo, "Which model.")
        self.assertEqual(combo.toolTip(), "Which model.")
        self.assertEqual(combo.lineEdit().toolTip(), "")
