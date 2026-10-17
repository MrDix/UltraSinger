"""A failed or cancelled song can be converted again: ↻ lets it wait in the queue again."""

import os
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.gui.config import _DEFAULTS
from src.gui.models import QueueItem
from src.gui.queue_manager import QueueManager
from src.gui.widgets.queue_list import QueueItemWidget, QueueListWidget

_app = QApplication.instance() or QApplication([])

_URL = "https://video.invalid/watch?v=abc"


def _manager():
    manager = QueueManager()
    manager._runner.start = MagicMock()
    manager.set_global_config({**_DEFAULTS, "cookie_file": ""})
    return manager


class TestRetryItem:
    @pytest.mark.parametrize("status", ["failed", "cancelled"])
    def test_waits_again_in_its_place_with_its_settings(self, status):
        manager = _manager()
        first = manager.add_item(_URL, "url", "First")
        song = manager.add_item("https://video.invalid/watch?v=2", "url", "Second")
        song.status, song.exit_code = status, 1
        song.settings_overrides = {"language_mode": "manual", "language": "de"}
        song.result_info, song.resolved_config = {"language": "en"}, {"language": "en"}
        changes = []
        manager.item_status_changed.connect(lambda item_id, new: changes.append((item_id, new)))

        assert manager.retry_item(song.id)

        assert manager.items == [first, song]
        assert song.status == "pending" and song.exit_code is None
        assert song.result_info == {} and song.resolved_config == {}
        assert song.settings_overrides == {"language_mode": "manual", "language": "de"}
        assert changes == [(song.id, "pending")]
        assert manager.pending_count() == 2

    @pytest.mark.parametrize("status", ["pending", "running", "done"])
    def test_only_a_failed_or_cancelled_song(self, status):
        manager = _manager()
        song = manager.add_item(_URL, "url", "Song Title")
        song.status = status
        changes = []
        manager.item_status_changed.connect(lambda item_id, new: changes.append(new))
        assert not manager.retry_item(song.id)
        assert song.status == status and changes == []

    def test_an_unknown_song(self):
        assert not _manager().retry_item("missing")

    def test_start_all_converts_it(self):
        manager = _manager()
        song = manager.add_item(_URL, "url", "Song Title")
        song.status = "failed"
        manager.retry_item(song.id)
        manager.start_all()
        assert song.status == "running"
        args = manager._runner.start.call_args.args[0]
        assert args[args.index("-i") + 1] == _URL

    def test_a_running_queue_converts_it_too(self):
        manager = _manager()
        failed = manager.add_item(_URL, "url", "Failed")
        running = manager.add_item("https://video.invalid/watch?v=2", "url", "Running")
        failed.status = "failed"
        manager.start_all()
        assert running.status == "running"
        manager.retry_item(failed.id)
        manager._on_item_finished(0)
        assert running.status == "done" and failed.status == "running"
        assert manager.is_running

    def test_a_retry_while_cancel_all_stops_the_running_song_is_kept(self):
        """Cancel All cancels the waiting songs at once; the running one ends later."""
        manager = _manager()
        running = manager.add_item(_URL, "url", "Running")
        waiting = manager.add_item("https://video.invalid/watch?v=2", "url", "Waiting")
        manager.start_all()
        finished = []
        manager.queue_finished.connect(lambda failed, cancelled: finished.append(cancelled))

        manager.cancel_all()
        assert waiting.status == "cancelled"
        manager.retry_item(waiting.id)
        manager._on_item_finished(-2)  # the runner reports the cancelled conversion

        assert running.status == "cancelled" and waiting.status == "pending"
        assert finished == [True] and not manager.is_running
        manager._runner.start.assert_called_once()  # the queue stopped; Start All converts it


class TestRetryButton:
    @pytest.mark.parametrize("status, shown", [("failed", True), ("cancelled", True), ("pending", False),
                                               ("running", False), ("done", False)])
    def test_shown_for_a_failed_or_cancelled_song(self, status, shown):
        widget = QueueItemWidget(QueueItem(input_source=_URL, input_type="url", title="Song Title",
                                           status=status))
        assert widget._retry_btn.isHidden() is not shown

    def test_a_click_asks_for_the_retry_and_the_waiting_song_hides_it(self):
        queue_list = QueueListWidget()
        item = QueueItem(input_source=_URL, input_type="url", title="Song Title", status="failed")
        queue_list.add_item(item)
        asked = []
        queue_list.retry_requested.connect(asked.append)
        widget = queue_list._item_widgets[item.id]

        widget._retry_btn.click()
        assert asked == [item.id]

        queue_list.update_status(item.id, "pending")
        assert widget._retry_btn.isHidden()
        assert not widget._gear_btn.isHidden() and not widget._remove_btn.isHidden()
