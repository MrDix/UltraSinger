"""The console can show only its warnings and errors, and all lines again."""

import os
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.gui.config import _DEFAULTS
from src.gui.queue_manager import QueueManager
from src.gui.queue_tab import QueueTab
from src.gui.widgets import log_viewer
from src.gui.widgets.log_viewer import LogViewer

_app = QApplication.instance() or QApplication([])

_RED, _ORANGE = "#ef5350", "#ffa726"


def _viewer(*lines, shown=False):
    viewer = LogViewer()
    if shown:  # laid out and scrolled as on the screen
        viewer.resize(500, 300)
        viewer.show()
    for line in lines:
        viewer.append_line(line)
    _app.processEvents()
    return viewer


def _at_end(viewer):
    _app.processEvents()
    sb = viewer.verticalScrollBar()
    return sb.maximum() > 0 and sb.value() == sb.maximum()


def _top(viewer):
    _app.processEvents()
    return viewer.firstVisibleBlock().text()


def _shown(viewer):
    return viewer.toPlainText().splitlines()


def _colors(viewer):
    """The color of each line shown."""
    colors = []
    block = viewer.document().begin()
    while block.isValid():
        if block.text():
            colors.append(block.begin().fragment().charFormat().foreground().color().name())
        block = block.next()
    return colors


_RUN = [
    "[UltraSinger] Separating vocals",
    "[UltraSinger] Warning: the lyrics have fewer lines than the song",
    " 50%|#####     | 5/10",
    "[UltraSinger] Error: the output folder could not be created",
    "[Queue] Song Title: completed",
]


class TestFilter:
    def test_lines_printed_before_are_filtered_and_shown_again(self):
        viewer = _viewer(*_RUN)
        viewer.set_issues_only(True)
        assert viewer.issues_only()
        assert _shown(viewer) == [_RUN[1], _RUN[3]]
        assert _colors(viewer) == [_ORANGE, _RED]

        viewer.set_issues_only(False)
        assert _shown(viewer) == _RUN

    def test_new_lines_are_filtered_and_kept(self):
        viewer = _viewer(_RUN[0])
        viewer.set_issues_only(True)
        for line in _RUN[1:]:
            viewer.append_line(line)
        assert _shown(viewer) == [_RUN[1], _RUN[3]]

        viewer.set_issues_only(False)
        assert _shown(viewer) == _RUN

    def test_switching_to_the_same_view_changes_nothing(self):
        viewer = _viewer(*_RUN)
        viewer.set_issues_only(False)
        assert _shown(viewer) == _RUN

    def test_without_warnings_and_errors_the_filter_says_so(self):
        viewer = _viewer(_RUN[0])
        viewer.set_issues_only(True)
        assert _shown(viewer) == []
        assert viewer.placeholderText() == "No warnings or errors."
        viewer.set_issues_only(False)
        assert viewer.placeholderText() == ""

    def test_clearing_the_log_clears_the_kept_lines(self):
        viewer = _viewer(*_RUN)
        viewer.clear_log()
        viewer.append_line("[UltraSinger] Transcribing")
        viewer.set_issues_only(True)
        assert _shown(viewer) == []
        viewer.set_issues_only(False)
        assert _shown(viewer) == ["[UltraSinger] Transcribing"]

    def test_warnings_older_than_the_last_lines_are_kept(self, monkeypatch):
        monkeypatch.setattr(log_viewer, "_MAX_LINES", 10)
        viewer = _viewer("WARNING: [download] retrying", *[f"{i}%" for i in range(20)])
        assert "WARNING: [download] retrying" not in _shown(viewer)
        viewer.set_issues_only(True)
        assert _shown(viewer) == ["WARNING: [download] retrying"]

    def test_all_the_last_lines_are_shown(self, monkeypatch):
        monkeypatch.setattr(log_viewer, "_MAX_LINES", 10)
        lines = [f"line {i}" for i in range(15)]
        viewer = _viewer(*lines)
        assert _shown(viewer) == lines[-10:]
        viewer.set_issues_only(True)
        viewer.set_issues_only(False)
        assert _shown(viewer) == lines[-10:]

    def test_ansi_codes_are_stripped(self):
        viewer = _viewer("\x1b[92m[UltraSinger]\x1b[0m \x1b[93mWarning:\x1b[0m no cover found")
        viewer.set_issues_only(True)
        assert _shown(viewer) == ["[UltraSinger] Warning: no cover found"]


class TestWhatIsAWarningOrAnError:
    def test_failed_steps_are_errors(self):
        failed = ["[Queue] Song Title: failed (exit code 1)", "[UltraSinger] Remote STT failed: timed out",
                  "[Queue] Browser download failed: ffmpeg failed (rc=1)"]
        viewer = _viewer(*failed, "[Queue] Restored 3 song(s) from the last session (1 of them failed).")
        viewer.set_issues_only(True)
        assert _shown(viewer) == failed
        assert _colors(viewer) == [_RED] * 3

    def test_a_warning_that_something_failed_stays_a_warning(self):
        viewer = _viewer("[UltraSinger] Warning: lead-vocal separation failed (ValueError())")
        assert _colors(viewer) == [_ORANGE]

    def test_a_traceback_is_kept_whole(self):
        traceback = [
            "Traceback (most recent call last):",
            '  File "src/UltraSinger.py", line 10, in <module>',
            "    main()",
            "    ^^^^^^",
            "KeyboardInterrupt",
        ]
        viewer = _viewer("[UltraSinger] Pitching", *traceback, "[UltraSinger] Cleaning up", "  indented")
        viewer.set_issues_only(True)
        assert _shown(viewer) == traceback
        assert _colors(viewer) == [_RED] * len(traceback)

    def test_chained_tracebacks_are_kept(self):
        chained = [
            "Traceback (most recent call last):",
            '  File "a.py", line 1, in f',
            "ValueError: bad value",
            "",
            "During handling of the above exception, another exception occurred:",
            "",
            "Traceback (most recent call last):",
            '  File "b.py", line 2, in g',
            "RuntimeError: worse",
        ]
        viewer = _viewer(*chained)
        viewer.set_issues_only(True)
        assert _shown(viewer) == [line for line in chained if line]

    def test_a_python_warning_keeps_its_source_line_but_not_the_progress_after_it(self):
        warning = ["C:\\venv\\site-packages\\lib\\io.py:43: UserWarning: the backend is deprecated",
                   "  torchaudio.set_audio_backend('soundfile')"]
        viewer = _viewer(*warning, "  0%|          | 0/12", " 50%|#####     | 6/12", "100%|##########| 12/12")
        viewer.set_issues_only(True)
        assert _shown(viewer) == warning
        assert _colors(viewer) == [_ORANGE, _ORANGE]

    def test_progress_after_a_warning_is_not_part_of_it(self):
        viewer = _viewer("WARNING: [youtube] Falling back to a lower format", "  0%|          | 0/12")
        viewer.set_issues_only(True)
        assert _shown(viewer) == ["WARNING: [youtube] Falling back to a lower format"]

    def test_a_python_warning_without_its_source_line(self):
        viewer = _viewer("C:\\venv\\lib\\io.py:43: UserWarning: deprecated", "  0%|          | 0/12")
        viewer.set_issues_only(True)
        assert _shown(viewer) == ["C:\\venv\\lib\\io.py:43: UserWarning: deprecated"]

    def test_the_first_keyword_gives_the_level(self):
        viewer = _viewer("WARNING: [download] Got error: HTTP Error 503. Retrying (1/10)...",
                         "[UltraSinger] Warning: an exception in the cover lookup",
                         "[UltraSinger] Error: the warning: could not be shown")
        assert _colors(viewer) == [_ORANGE, _ORANGE, _RED]

    def test_error_as_a_word_is_an_error(self):
        errors = ["[UltraSinger] Error in hyphenation for language xx, maybe you want to disable it?",
                  "[UltraSinger] Error! The provided text file does not have a reference to the audio file",
                  "[UltraSinger] Remote STT HTTP error 429"]
        viewer = _viewer(*errors, "[UltraSinger] 3 lines checked, 0 errors", "[UltraSinger] error_tolerance=2")
        viewer.set_issues_only(True)
        assert _shown(viewer) == errors

    def test_the_message_of_the_exception_is_kept(self):
        error = ["Traceback (most recent call last):",
                 '  File "src/UltraSinger.py", line 10, in <module>',
                 "torch.AcceleratorError: CUDA error: unspecified launch failure",
                 "CUDA kernel errors might be asynchronously reported at some other API call.",
                 "For debugging consider passing CUDA_LAUNCH_BLOCKING=1"]
        viewer = _viewer(*error, "", "after the run", "[Queue] Song Title: failed (exit code 1)")
        viewer.set_issues_only(True)
        assert _shown(viewer) == [*error, "[Queue] Song Title: failed (exit code 1)"]

    @pytest.mark.parametrize("after", ["[UltraSinger] Retrying on the CPU", "100%|##########| 10/10"])
    def test_the_message_ends_at_a_tagged_line_or_a_progress_bar(self, after):
        viewer = _viewer("Traceback (most recent call last):", "ValueError: bad value", after, "plain")
        viewer.set_issues_only(True)
        assert _shown(viewer) == ["Traceback (most recent call last):", "ValueError: bad value"]

    def test_a_traceback_after_an_unfinished_progress_bar(self):
        traceback = [" 50%|#####     | 5/10 [00:01<00:01]Traceback (most recent call last):",
                     '  File "src/UltraSinger.py", line 10, in <module>',
                     "KeyboardInterrupt"]
        viewer = _viewer(*traceback, "", "plain")
        viewer.set_issues_only(True)
        assert _shown(viewer) == traceback


class TestScrolling:
    _MIXED = [f"Warning: {i} " + "a warning that wraps " * 8 if i % 4 == 0 else f"line {i}" for i in range(160)]
    # 60 warnings: their view scrolls far enough not to follow the end at the 5th
    _LINES = [f"Warning: w{i}" if i % 10 == 0 else f"line {i}" for i in range(600)]

    def test_the_view_follows_new_lines_after_switching(self):
        viewer = _viewer(*self._MIXED, shown=True)
        viewer.set_issues_only(True)
        for i in range(10):
            viewer.append_line(f"Warning: new {i}")
        assert _at_end(viewer)
        viewer.set_issues_only(False)
        for i in range(20):
            viewer.append_line(f"line new {i}")
        assert _at_end(viewer)

    def test_switching_back_shows_the_warning_at_the_top_among_its_lines(self):
        viewer = _viewer(*self._LINES, shown=True)
        viewer.set_issues_only(True)
        viewer.verticalScrollBar().setValue(5)
        assert _top(viewer) == "Warning: w50"
        viewer.set_issues_only(False)
        assert _top(viewer) == "Warning: w50"
        viewer.append_line("line new")
        assert _top(viewer) == "Warning: w50"  # the view stays there

    def test_switching_to_the_warnings_starts_at_the_line_at_the_top(self):
        viewer = _viewer(*self._LINES, shown=True)
        viewer.verticalScrollBar().setValue(55)
        assert _top(viewer) == "line 55"
        viewer.set_issues_only(True)
        assert _top(viewer) == "Warning: w60"
        viewer.set_issues_only(False)
        assert _top(viewer) == "Warning: w60"

    def test_a_view_at_the_end_stays_at_the_end(self):
        viewer = _viewer(*self._LINES, shown=True)
        assert _at_end(viewer)
        viewer.set_issues_only(True)
        assert _at_end(viewer)
        viewer.set_issues_only(False)
        assert _at_end(viewer)

    @pytest.mark.parametrize("status", ["expired", "missing"])
    def test_a_download_without_the_browser_stream_is_a_warning(self, tmp_path, status):
        manager = QueueManager()
        manager._runner.start = MagicMock()
        manager.set_global_config({**_DEFAULTS, "output_folder": str(tmp_path), "cookie_file": ""})
        manager.set_media_interceptor(MagicMock(
            get_po_token=MagicMock(return_value=""),
            get_stream_with_status=MagicMock(return_value=(None, status)),
        ))
        item = manager.add_item("https://video.invalid/watch?v=abc", "url", "Song Title")
        item.video_id = "abc"
        viewer = LogViewer()
        manager.line_output.connect(viewer.append_line)
        manager.start_all()
        viewer.set_issues_only(True)
        assert [line[:17] for line in _shown(viewer)] == ["[Queue] WARNING: "]
        assert "bot-detection risk" in _shown(viewer)[0]


class TestConsoleTab:
    def test_the_switch_filters_the_console(self):
        tab = QueueTab()
        for line in _RUN:
            tab.append_log(line)
        tab._issues_only.setChecked(True)
        assert _shown(tab._log) == [_RUN[1], _RUN[3]]
        tab._issues_only.setChecked(False)
        assert _shown(tab._log) == _RUN

    def test_the_filter_stays_on_for_the_next_run(self):
        tab = QueueTab()
        tab._issues_only.setChecked(True)
        tab.on_queue_started()
        for line in _RUN:
            tab.append_log(line)
        assert _shown(tab._log) == [_RUN[1], _RUN[3]]
        assert tab._issues_only.toolTip()
