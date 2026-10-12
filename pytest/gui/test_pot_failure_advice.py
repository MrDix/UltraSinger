"""The console advice for PO-token failures that downloads print."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Skip (instead of failing collection) where PySide6 or its native libraries are missing.
main_window = pytest.importorskip("src.gui.main_window", exc_type=ImportError)

MISMATCH = ("WARNING: [pot:bgutil:http] Plugin and HTTP server major versions are mismatched. "
            "Update both the plugin and the HTTP server to the same version to proceed.")
TOKEN_ERROR = 'WARNING: [youtube] Error fetching PO Token from "bgutil:http" provider: HTTP Error 500'


def _window():
    return SimpleNamespace(_pot_failure_handled=False, _queue_tab=MagicMock())


def _advice(window, *lines):
    for line in lines:
        main_window.MainWindow._check_for_pot_failure(window, line)
    return [call.args[0] for call in window._queue_tab.append_log.call_args_list]


def test_version_mismatch_gets_its_own_advice():
    advice = _advice(_window(), MISMATCH, MISMATCH)
    assert len(advice) == 1  # once per queue run
    assert "different major versions" in advice[0]
    assert "update" in advice[0] and "Node.js 22 or newer" in advice[0]


def test_failing_token_generation_keeps_the_network_advice():
    advice = _advice(_window(), TOKEN_ERROR)
    assert len(advice) == 1
    assert "network filter" in advice[0]


def test_other_lines_are_ignored():
    assert _advice(_window(), "[download] 42.0% of 3.10MiB") == []
