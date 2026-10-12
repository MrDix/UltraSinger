"""ToggleSwitch reports every change of its state, however it was made."""

import os

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.gui.settings_tab import ConversionSettingsForm
from src.gui.widgets.toggle_switch import ToggleSwitch

_app = QApplication.instance() or QApplication([])


def _recorded(toggle):
    states = []
    toggle.toggled_signal.connect(states.append)
    return states


def test_set_checked_emits():
    toggle = ToggleSwitch()
    states = _recorded(toggle)
    toggle.setChecked(True)
    toggle.setChecked(True)  # no change, no signal
    toggle.setChecked(False)
    assert states == [True, False]


def test_keyboard_toggles_and_emits():
    toggle = ToggleSwitch()
    states = _recorded(toggle)
    QTest.keyClick(toggle, Qt.Key.Key_Space)
    assert toggle.isChecked()
    assert states == [True]


def test_mouse_click_emits_once():
    toggle = ToggleSwitch(checked=True)
    states = _recorded(toggle)
    QTest.mouseClick(toggle, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(10, 10))
    assert not toggle.isChecked()
    assert states == [False]


def test_reset_updates_the_dependent_settings():
    """A reset button calls setChecked; the sub-settings must follow."""
    form = ConversionSettingsForm({"llm_correct": True, "remote_stt": True, "detect_growl": True})
    assert form._llm_provider.isEnabled()
    form._llm_correct.setChecked(False)
    form._remote_stt.setChecked(False)
    form._detect_growl.setChecked(False)
    assert not form._llm_provider.isEnabled()
    assert not form._remote_stt_api_key.isEnabled()
    assert not form._freestyle_harmonicity.isEnabled()
