"""LLM providers on the Settings page: model fetches, the provider selection
and the API keys of removed providers."""

import os
import threading
import time
from unittest.mock import patch

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from src.gui.config import _DEFAULTS
from src.gui.models import LLMProvider
from src.gui.preferences_tab import PreferencesTab
from src.gui.settings_tab import ConversionSettingsForm
from src.gui.widgets import llm_provider_list
from src.gui.widgets.llm_provider_list import LLMProviderListWidget, LLMProviderRow, _ModelFetcher

_app = QApplication.instance() or QApplication([])


def _process_events_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.01)
    return condition()


@pytest.fixture
def blocking_fetch(monkeypatch):
    """_ModelFetcher.run blocks until released, then reports two models; records its URLs."""
    release = threading.Event()
    urls = []

    def run(self):
        urls.append(self._base_url)
        release.wait(5)
        self.finished.emit(["a-model", "b-model"])
        self.finished_raw.emit([{"id": "a-model"}, {"id": "b-model"}])
        self.done.emit()

    monkeypatch.setattr(_ModelFetcher, "run", run)
    yield release, urls
    release.set()
    llm_provider_list.wait_for_model_fetches()


def _row(url="https://one.invalid/v1", model=""):
    return LLMProviderRow(LLMProvider(name="P", api_base_url=url, default_model=model))


class TestModelFetch:
    def test_a_second_fetch_waits_for_the_first(self, blocking_fetch):
        release, urls = blocking_fetch
        row = _row()
        row._fetch_models()
        row._url_edit.setText("https://two.invalid/v1")
        row._fetch_models()  # e.g. editing the URL while the first fetch runs
        assert len(llm_provider_list._running_fetches) == 1
        release.set()
        assert _process_events_until(lambda: len(urls) == 2)
        assert urls == ["https://one.invalid/v1", "https://two.invalid/v1"]

    def test_removing_a_provider_during_a_fetch(self, blocking_fetch):
        release, urls = blocking_fetch
        widget = LLMProviderListWidget()
        widget.set_providers([LLMProvider(name="P", api_base_url="https://one.invalid/v1")])
        widget._rows[0]._fetch_models()
        widget._remove_provider(widget._rows[0].provider_id)
        _app.processEvents()  # deletes the row; its fetch thread is still referenced
        assert llm_provider_list._running_fetches
        release.set()
        llm_provider_list.wait_for_model_fetches()
        assert llm_provider_list._running_fetches == []

    def test_a_typed_model_is_kept_after_a_fetch(self):
        row = _row(model="my-own-model")
        row._on_models_fetched(["a-model", "qwen/qwen3-32b"])
        assert row._model_combo.currentText() == "my-own-model"

    def test_without_a_model_the_fallback_is_selected(self):
        row = _row()
        row._on_models_fetched(["a-model", "qwen/qwen3-32b"])
        assert row._model_combo.currentText() == "qwen/qwen3-32b"

    def test_remote_stt_fetch_runs_once_at_a_time(self, blocking_fetch):
        release, urls = blocking_fetch
        form = ConversionSettingsForm({**_DEFAULTS, "remote_stt": True})
        form._fetch_remote_stt_models()
        form._remote_stt.setChecked(False)
        form._remote_stt.setChecked(True)  # re-enables the other remote settings, not Fetch
        assert not form._remote_stt_model_fetch_btn.isEnabled()
        form._fetch_remote_stt_models()
        assert len(urls) <= 1 and len(llm_provider_list._running_fetches) == 1
        release.set()
        assert _process_events_until(lambda: form._remote_stt_model_fetch_btn.isEnabled())


class TestNoFetchAfterShutdown:
    """A fetch that starts after the close-time join would be a running QThread at exit."""

    @pytest.fixture(autouse=True)
    def _reset_flag(self, monkeypatch):
        monkeypatch.setattr(llm_provider_list, "_shutting_down", False)

    def _running(self):
        return [t for t, _ in llm_provider_list._running_fetches if t.isRunning()]

    def test_queued_refetch_is_dropped(self, blocking_fetch):
        release, urls = blocking_fetch
        row = _row()
        row._fetch_models()
        row._url_edit.setText("https://two.invalid/v1")
        row._fetch_models()  # merged into a re-fetch
        release.set()
        llm_provider_list.shut_down_model_fetches()  # what closeEvent does
        for _ in range(30):  # the queued 'done' of the joined fetch arrives now
            _app.processEvents()
            time.sleep(0.01)
        assert self._running() == []
        assert urls == ["https://one.invalid/v1"]
        assert row._fetch_btn.isEnabled()

    def test_field_edit_ending_after_close_starts_no_fetch(self, blocking_fetch):
        release, urls = blocking_fetch
        row = _row()
        row._key_edit.setText("key")
        llm_provider_list.shut_down_model_fetches()
        row._url_edit.setText("https://two.invalid/v1")
        row._url_edit.editingFinished.emit()  # focus leaves the field as the window hides
        row._fetch_models()
        assert self._running() == [] and urls == []
        assert not row._fetching and row._fetch_btn.isEnabled()

    def test_remote_stt_fetch_after_close(self, blocking_fetch):
        release, urls = blocking_fetch
        form = ConversionSettingsForm({**_DEFAULTS, "remote_stt": True})
        llm_provider_list.shut_down_model_fetches()
        form._fetch_remote_stt_models()
        assert self._running() == [] and urls == []
        assert not form._remote_stt_fetching


def _preferences(config):
    # No model fetch over the network (a plain function: PySide trips over mocks in a QObject class)
    with patch.object(LLMProviderRow, "fetch_models_if_ready", lambda self: None):
        return PreferencesTab(config)


def _providers_config(selected="p2"):
    config = dict(_DEFAULTS)
    config["llm_providers"] = [
        {"id": "p1", "name": "One", "api_base_url": "https://one.invalid/v1",
         "default_model": "m1", "is_default": True},
        {"id": "p2", "name": "Two", "api_base_url": "https://two.invalid/v1",
         "default_model": "m2", "is_default": False},
    ]
    config["llm_provider_id"] = selected
    config["llm_api_key_p1"] = "key-one"
    config["llm_api_key_p2"] = "key-two"
    return config


class TestProviderSelection:
    def test_saved_selection_is_restored(self):
        tab = _preferences(_providers_config("p2"))
        assert tab._conversion_form.get_selected_provider_id() == "p2"

    def test_selection_survives_editing_the_provider_list(self):
        tab = _preferences(_providers_config("p1"))
        form = tab._conversion_form
        form._llm_provider.setCurrentIndex(form._llm_provider.findData("p2"))
        tab._llm_provider_list._rows[0]._name_edit.setText("One renamed")  # a keystroke
        assert form.get_selected_provider_id() == "p2"

    def test_removed_selection_falls_back_to_the_default(self):
        tab = _preferences(_providers_config("p2"))
        tab._llm_provider_list._remove_provider("p2")
        assert tab._conversion_form.get_selected_provider_id() == "p1"


class TestRemovedProviderKey:
    def test_key_of_a_removed_provider_is_deleted_on_save(self):
        tab = _preferences(_providers_config())
        tab._llm_provider_list._remove_provider("p2")
        result = tab.collect_all()
        assert result["llm_api_key_p2"] == ""  # saving an empty key deletes it from the keyring
        assert result["llm_api_key_p1"] == "key-one"


def _wheel(widget):
    event = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)


class TestWheelDoesNotChangeSettings:
    def test_proxy_mode(self):
        tab = _preferences(dict(_DEFAULTS))
        before = tab._proxy_mode.currentIndex()
        _wheel(tab._proxy_mode)
        assert tab._proxy_mode.currentIndex() == before

    def test_retry_spin_boxes(self):
        form = ConversionSettingsForm(dict(_DEFAULTS))
        for spin in (form._llm_retry_wait, form._llm_retry_max,
                     form._remote_stt_retry_wait, form._remote_stt_retry_max):
            spin.setEnabled(True)
            before = spin.value()
            _wheel(spin)
            assert spin.value() == before
