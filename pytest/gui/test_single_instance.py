"""Only one GUI runs at a time, whatever the length of its process ID.

Two GUIs would both restore and save the queue file (queue.json) and replace
each other's queue. The lock has to cover the same byte for every instance:
locking the byte after the written PID let an instance with a shorter PID in.
"""

from unittest.mock import patch

import pytest

from src import gui_main


@pytest.fixture
def lock_file(tmp_path, monkeypatch):
    path = tmp_path / "ultrasinger_gui.lock"
    monkeypatch.setattr(gui_main, "_LOCK_FILE", path)
    return path


def _start(pid):
    """The lock handle of an instance with this process ID, or None when it is refused."""
    with patch.object(gui_main.os, "getpid", return_value=pid):
        return gui_main._acquire_instance_lock()


@pytest.mark.parametrize("running, second", [
    (1234, 5678),    # same length
    (1234, 98765),   # longer
    (1234, 987),     # shorter: got in before
    (98765, 4),
])
def test_a_second_instance_is_refused(lock_file, running, second):
    first = _start(running)
    assert first is not None
    try:
        assert _start(second) is None
    finally:
        first.close()
    # The refused instance left the file of the running one alone
    assert lock_file.read_text(encoding="utf-8") == str(running)
    # Once the first one has ended, the next one starts and writes its PID
    again = _start(second)
    assert again is not None
    again.close()
    assert lock_file.read_text(encoding="utf-8") == str(second)


def test_an_older_longer_pid_is_replaced(lock_file):
    lock_file.write_text("123456789", encoding="utf-8")  # left by an ended instance
    handle = _start(42)
    assert handle is not None
    handle.close()
    assert lock_file.read_text(encoding="utf-8") == "42"
