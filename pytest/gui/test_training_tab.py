"""Tests for the GUI training page (src/gui/training_tab.py)."""

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
from src.gui.training_tab import (
    TrainingTab,
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
        import src.gui.training_tab as tt
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


class TestTrainingTab(unittest.TestCase):
    def test_defaults(self):
        for key in ("training_library", "training_workdir", "training_exclude", "training_model"):
            self.assertEqual(_DEFAULTS[key], "")
        self.assertEqual(_DEFAULTS["training_epochs"], 30)
        tab = TrainingTab({})
        self.assertEqual(tab.values()["training_epochs"], 30)
        self.assertFalse(tab._use.isEnabled())
        self.assertFalse(tab.is_running)

    def test_values_from_config(self):
        cfg = {"training_library": "L", "training_workdir": "W", "training_exclude": "E",
               "training_model": "M.pt", "training_epochs": 7}
        self.assertEqual(TrainingTab(cfg).values(), cfg)

    def test_invalid_inputs_do_not_start(self):
        tab = TrainingTab({})
        tab._on_start()
        self.assertFalse(tab.is_running)
        self.assertIn("library", tab._status.text())

    def test_use_model_emits_path(self):
        tab = TrainingTab({"training_model": "trained.pt"})
        got = []
        tab.model_ready.connect(got.append)
        tab._use.setEnabled(True)
        tab._use.click()
        self.assertEqual(got, ["trained.pt"])


class TestSettingsReceivesModel(unittest.TestCase):
    def test_set_segmentation_model(self):
        form = ConversionSettingsForm({})
        form.set_segmentation_model("D:/models/new.pt")
        self.assertEqual(form.collect_config()["segmentation_model"], "D:/models/new.pt")
