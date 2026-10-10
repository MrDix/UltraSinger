"""Tests for the GUI training window (src/gui/training_dialog.py) and its Train button."""

import os
import sys
import unittest
from pathlib import Path

import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.gui.config import _DEFAULTS
from src.gui.settings_tab import ConversionSettingsForm
from src.gui.training_dialog import (
    TrainingDialog,
    TrainingWorker,
    build_training_commands,
    parse_progress,
    validate_inputs,
)

_app = QApplication.instance() or QApplication([])
REPO = Path(__file__).resolve().parents[2]


class TestHelpers(unittest.TestCase):
    def test_parse_progress(self):
        self.assertEqual(parse_progress("[3/120] tr_00002: ok fit=0.81 offset=+20 ms"), ("extract", 3, 120))
        self.assertEqual(parse_progress("[4/120] tr_00003: ERROR boom"), ("extract", 4, 120))
        self.assertEqual(parse_progress("epoch 7/30 loss 1.2 validation agreement 70.0"), ("train", 7, 30))
        self.assertIsNone(parse_progress("100 songs, 0 already extracted"))

    def test_commands(self):
        extract, train = build_training_commands(REPO, "LIB", "WORK", "M.pt", "EX.json", 12)
        self.assertTrue(extract[-6].endswith("train_segmentation.py"))
        self.assertEqual(extract[-5:], ["extract", "LIB", "WORK", "--exclude", "EX.json"])
        self.assertEqual(train[-6:], ["train", "WORK", "--out", "M.pt", "--epochs", "12"])
        self.assertNotIn("UltraSinger.py", " ".join(extract))

    def test_commands_without_exclude(self):
        extract, _ = build_training_commands(REPO, "LIB", "WORK", "M.pt")
        self.assertNotIn("--exclude", extract)

    def test_validate_inputs(self):
        import tempfile
        with tempfile.TemporaryDirectory() as lib:
            outside = Path(lib) / "work"
            self.assertIn("library", validate_inputs("", str(outside), "m.pt", REPO))
            self.assertIn("work folder", validate_inputs(lib, "", "m.pt", REPO))
            self.assertIn("model file", validate_inputs(lib, str(outside), "", REPO))
            self.assertIn("outside", validate_inputs(lib, str(REPO / "data"), str(outside / "m.pt"), REPO))
            self.assertIn("outside", validate_inputs(lib, str(outside), str(REPO / "m.pt"), REPO))
            self.assertIsNone(validate_inputs(lib, str(outside), str(outside / "m.pt"), REPO))


def _run_worker(commands):
    worker = TrainingWorker(commands, str(REPO))
    lines, progress, finished = [], [], []
    worker.line_output.connect(lines.append)
    worker.progress.connect(lambda *a: progress.append(a))
    worker.finished.connect(finished.append)
    worker.run()  # synchronously, no thread needed for the test
    return lines, progress, finished


class TestWorker(unittest.TestCase):
    def test_runs_commands_in_order_and_reports_progress(self):
        first = [sys.executable, "-c", "print('[1/2] a: ok'); print('[2/2] b: ok')"]
        second = [sys.executable, "-c", "print('epoch 1/1 loss 1.0 validation agreement 50.0')"]
        lines, progress, finished = _run_worker([first, second])
        self.assertEqual(progress, [("extract", 1, 2), ("extract", 2, 2), ("train", 1, 1)])
        self.assertEqual(finished, [0])
        self.assertIn("[2/2] b: ok", lines)

    def test_stops_after_a_failing_command(self):
        failing = [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"]
        never = [sys.executable, "-c", "print('should not run')"]
        lines, _, finished = _run_worker([failing, never])
        self.assertEqual(finished, [3])
        self.assertNotIn("should not run", lines)

    def test_cancel_before_start(self):
        worker = TrainingWorker([[sys.executable, "-c", "print('x')"]], str(REPO))
        finished = []
        worker.finished.connect(finished.append)
        worker.cancel()
        worker.run()
        self.assertEqual(finished, [-2])

    def test_cancel_while_the_next_command_starts(self):
        # cancel() runs between the end of one command and the start of the next
        from unittest.mock import patch
        import src.gui.training_dialog as tt
        cmd = [sys.executable, "-c", "import time; time.sleep(5); print('ran to the end')"]
        worker = TrainingWorker([cmd], str(REPO))
        finished, lines, real_popen = [], [], tt.subprocess.Popen

        def popen_then_cancel(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            if args[0] == cmd:  # not for the kill command itself
                worker.cancel()  # still sees no running process
            return proc

        worker.finished.connect(finished.append)
        worker.line_output.connect(lines.append)
        with patch.object(tt.subprocess, "Popen", popen_then_cancel):
            worker.run()
        self.assertEqual(finished, [-2])
        self.assertNotIn("ran to the end", lines)  # the new process was stopped


class TestTrainingDialog(unittest.TestCase):
    def _start_sleeping_training(self, dialog, seconds=60):
        from unittest.mock import patch
        sleep = [[sys.executable, "-c", f"import time; time.sleep({seconds})"]]
        with (patch("src.gui.training_dialog.validate_inputs", return_value=None),
              patch("src.gui.training_dialog.build_training_commands", return_value=sleep),
              patch("src.gui.config.save_config")):
            dialog._on_start()

    def test_shutdown_waits_for_the_worker_thread(self):
        import time
        dialog = TrainingDialog({})
        self._start_sleeping_training(dialog)
        deadline = time.time() + 20
        while dialog._worker._process is None and time.time() < deadline:
            time.sleep(0.05)
        thread = dialog._thread
        dialog.shutdown()
        self.assertFalse(thread.isRunning())  # never destroyed while running

    def test_not_modal_and_closing_keeps_the_training_running(self):
        dialog = TrainingDialog({})
        self.assertFalse(dialog.isModal())
        dialog.show()
        self._start_sleeping_training(dialog)
        dialog.close()
        self.assertFalse(dialog.isVisible())
        self.assertTrue(dialog.is_running)
        dialog.shutdown()

    def test_inputs_locked_while_running(self):
        dialog = TrainingDialog({})
        running = []
        dialog.running_changed.connect(running.append)
        self._start_sleeping_training(dialog)
        self.assertFalse(dialog._library.isEnabled())
        self.assertFalse(dialog._model_out.isEnabled())
        self.assertFalse(dialog._epochs.isEnabled())
        self.assertFalse(dialog._start.isEnabled())
        self.assertTrue(dialog._stop.isEnabled())
        dialog.shutdown()
        dialog._on_finished(-2)
        self.assertTrue(dialog._library.isEnabled())
        self.assertTrue(dialog._epochs.isEnabled())
        self.assertEqual(running, [True, False])

    def test_defaults(self):
        for key in ("training_library", "training_workdir", "training_exclude", "training_model"):
            self.assertEqual(_DEFAULTS[key], "")
        self.assertEqual(_DEFAULTS["training_epochs"], 30)
        dialog = TrainingDialog({})
        self.assertEqual(dialog.values()["training_epochs"], 30)
        self.assertFalse(dialog.is_running)

    def test_values_from_config(self):
        cfg = {"training_library": "L", "training_workdir": "W", "training_exclude": "E",
               "training_model": "M.pt", "training_epochs": 7}
        self.assertEqual(TrainingDialog(cfg).values(), cfg)

    def test_invalid_inputs_do_not_start(self):
        dialog = TrainingDialog({})
        dialog._on_start()
        self.assertFalse(dialog.is_running)
        self.assertIn("library", dialog._status.text())

    def test_enter_in_a_field_presses_no_button(self):
        from PySide6.QtWidgets import QPushButton
        for button in TrainingDialog({}).findChildren(QPushButton):
            self.assertFalse(button.autoDefault(), button.text())


class TestFinishedTraining(unittest.TestCase):
    """A finished training hands the model to the settings without another click."""

    def _finish(self, code, model_exists=True, edited_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "trained.pt")
            if model_exists:
                open(model, "wb").close()
            dialog = TrainingDialog({"training_model": model})
            got, ended = [], []
            dialog.model_ready.connect(got.append)
            dialog.ended.connect(ended.append)
            dialog._model_in_training = model  # set by _on_start
            if edited_path:
                dialog._model_out.setText(edited_path)  # edited while the training ran
            dialog._on_finished(code)
            return model, got, ended, dialog

    def test_success_sets_the_trained_model(self):
        model, got, ended, dialog = self._finish(0)
        self.assertEqual(got, [model])
        self.assertIn("set as Segmentation Model", dialog._status.text())
        self.assertEqual(ended, [dialog._status.text()])

    def test_success_uses_the_path_that_was_trained(self):
        model, got, _, _ = self._finish(0, edited_path="elsewhere.pt")
        self.assertEqual(got, [model])

    def test_failure_cancel_and_missing_file_keep_the_settings(self):
        for code, exists, text in ((3, True, "failed"), (-2, True, "Stopped"), (0, False, "without writing")):
            _, got, ended, dialog = self._finish(code, model_exists=exists)
            self.assertEqual(got, [], code)
            self.assertIn(text, dialog._status.text())
            self.assertEqual(len(ended), 1)


class TestTrainButton(unittest.TestCase):
    def test_only_on_the_settings_page(self):
        self.assertIsNone(ConversionSettingsForm({})._train_button)  # per-song dialogs
        form = ConversionSettingsForm({}, allow_training=True)
        self.assertEqual(form._train_button.text(), "Train...")
        requested = []
        form.training_requested.connect(lambda: requested.append(True))
        form._train_button.click()
        self.assertEqual(requested, [True])

    def test_shows_a_running_training(self):
        form = ConversionSettingsForm({}, allow_training=True)
        form.set_training_running(True)
        self.assertEqual(form._train_button.text(), "Training...")
        form.set_training_running(False)
        self.assertEqual(form._train_button.text(), "Train...")
        ConversionSettingsForm({}).set_training_running(True)  # no button, no error

    def test_settings_page_forwards_the_request(self):
        from src.gui.preferences_tab import PreferencesTab
        page = PreferencesTab({}, None)
        requested = []
        page.training_requested.connect(lambda: requested.append(True))
        page._conversion_form._train_button.click()
        self.assertEqual(requested, [True])


class TestSettingsReceivesModel(unittest.TestCase):
    def test_set_segmentation_model(self):
        form = ConversionSettingsForm({})
        form.set_segmentation_model("D:/models/new.pt")
        self.assertEqual(form.collect_config()["segmentation_model"], "D:/models/new.pt")
