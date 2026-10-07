"""Training tab: train a note segmentation model on the user's own song library.

Runs ``tools/train_segmentation.py extract`` and then ``train`` in a background
process, shows progress and the log, and can hand the finished model to the
conversion settings.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .ultrasinger_runner import ConversionWorker, _build_command, _find_project_root
from .widgets.log_viewer import LogViewer
from .widgets.settings_card import SettingsCard

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"
_EXTRACT_RE = re.compile(r"^\[(\d+)/(\d+)\]")
_EPOCH_RE = re.compile(r"\bepoch (\d+)/(\d+)\b")


def build_training_commands(project_root: Path, library: str, workdir: str, model_out: str,
                            exclude: str = "", epochs: int = 30) -> list[list[str]]:
    """The two tool invocations (extract, then train) for the given inputs."""
    tool = str(Path(project_root) / "tools" / "train_segmentation.py")
    base = _build_command(Path(project_root))[:-1]  # interpreter part, without UltraSinger.py
    extract = base + [tool, "extract", library, workdir]
    if exclude:
        extract += ["--exclude", exclude]
    train = base + [tool, "train", workdir, "--out", model_out, "--epochs", str(int(epochs))]
    return [extract, train]


def parse_progress(line: str) -> tuple[str, int, int] | None:
    """``("extract"|"train", current, total)`` for a progress line of the tool, else ``None``."""
    m = _EXTRACT_RE.match(line.strip())
    if m:
        return "extract", int(m.group(1)), int(m.group(2))
    m = _EPOCH_RE.search(line)
    if m:
        return "train", int(m.group(1)), int(m.group(2))
    return None


def validate_inputs(library: str, workdir: str, model_out: str, project_root: Path) -> str | None:
    """Error message for invalid inputs, or ``None`` when they can be used."""
    if not library or not os.path.isdir(library):
        return "Choose an existing song library folder."
    if not workdir:
        return "Choose a work folder for the extracted training data."
    if not model_out:
        return "Choose where to save the model file."
    root = Path(project_root).resolve()
    for label, p in (("work folder", workdir), ("model file", model_out)):
        try:
            Path(p).resolve().relative_to(root)
        except ValueError:
            continue
        return f"The {label} must be outside the UltraSinger folder (it holds data derived from your library)."
    return None


class TrainingWorker(QObject):
    """Runs the training commands one after another and streams their output."""

    line_output = Signal(str)
    progress = Signal(str, int, int)  # stage, current, total
    finished = Signal(int)  # exit code (0 ok, -2 cancelled)

    def __init__(self, commands: list[list[str]], cwd: str, parent=None):
        super().__init__(parent)
        self._commands = commands
        self._cwd = cwd
        self._process: subprocess.Popen | None = None
        self._cancelled = False

    def run(self):
        exit_code = 0
        for cmd in self._commands:
            if self._cancelled:
                exit_code = -2
                break
            self.line_output.emit(f"[GUI] Running: {' '.join(cmd)}")
            kwargs: dict = {}
            if _IS_WINDOWS:
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            try:
                self._process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                    cwd=self._cwd, encoding="utf-8", errors="replace", env=env, **kwargs,
                )
                if self._cancelled:
                    # cancel() ran before the new process was stored and found nothing to stop
                    ConversionWorker._kill_tree(self._process)
                    self._process.wait()
                    exit_code = -2
                    break
                for line in self._process.stdout:
                    line = line.rstrip("\n\r")
                    self.line_output.emit(line)
                    parsed = parse_progress(line)
                    if parsed:
                        self.progress.emit(*parsed)
                self._process.wait()
                exit_code = self._process.returncode
            except (OSError, subprocess.SubprocessError) as e:
                self.line_output.emit(f"[Error] {e}")
                exit_code = -1
            if self._cancelled:
                exit_code = -2
            if exit_code != 0:
                break
        if exit_code == -2:
            self.line_output.emit("[GUI] Training cancelled by user.")
        self.finished.emit(exit_code)

    def cancel(self):
        self._cancelled = True
        if self._process and self._process.poll() is None:
            ConversionWorker._kill_tree(self._process)
            threading.Thread(target=self._process.wait, daemon=True).start()


class TrainingTab(QWidget):
    """Page with inputs, progress and log for training a segmentation model."""

    model_ready = Signal(str)  # path of a finished model the user wants to use

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config
        self._thread: QThread | None = None
        self._worker: TrainingWorker | None = None
        self._project_root = _find_project_root()

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        scroll.setWidget(inner)

        card = SettingsCard("Train a Segmentation Model (experimental)")
        card.add_info(
            "Learns from the hand-made charts in your song library where notes start, how long "
            "they are and which passages are charted at all. Every song is separated once (GPU "
            "recommended, about 20-40 s per song), then the model is trained (about 30 minutes "
            "for 1000 songs on a GPU). Use well-timed charts: the model learns their style. The "
            "work folder and the model are derived from your library - keep them private."
        )
        self._library = self._path_row(card, "Song Library", "training_library", folder=True,
                                       tooltip="Folder with your UltraStar songs (searched recursively). "
                                               "Solo charts with at least 100 notes and their audio are used.")
        self._workdir = self._path_row(card, "Work Folder", "training_workdir", folder=True,
                                       tooltip="Where the extracted training data is stored (several GB for "
                                               "1000 songs). An interrupted extraction continues here.")
        self._exclude = self._path_row(card, "Exclude Songs", "training_exclude", folder=False,
                                       file_filter="Song list (songs.json)",
                                       tooltip="Optional: songs.json of a chart benchmark sample - these songs are never "
                                               "trained on, so the benchmark stays meaningful.")
        self._model_out = self._path_row(card, "Model File", "training_model", folder=False, save=True,
                                         file_filter="Segmentation model (*.pt)",
                                         tooltip="Where the trained model is written.")
        self._epochs = QSpinBox()
        self._epochs.setRange(1, 200)
        self._epochs.setValue(int(config.get("training_epochs", 30)))
        card.add_row("Epochs", self._epochs, "Training passes; the best one on held-out songs is kept.")
        layout.addWidget(card)

        buttons = QHBoxLayout()
        self._start = QPushButton("Start Training")
        self._start.clicked.connect(self._on_start)
        self._cancel = QPushButton("Cancel")
        self._cancel.setEnabled(False)
        self._cancel.clicked.connect(self._on_cancel)
        self._use = QPushButton("Use This Model")
        self._use.setEnabled(False)
        self._use.setToolTip("Set the trained model as Segmentation Model in the conversion settings.")
        self._use.clicked.connect(lambda: self.model_ready.emit(self._model_out.text().strip()))
        for b in (self._start, self._cancel, self._use):
            buttons.addWidget(b)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self._status = QLabel("")
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._status)
        layout.addWidget(self._progress)
        self._log = LogViewer()
        layout.addWidget(self._log, 1)

    # ── helpers ─────────────────────────────────────────────────────────

    def _path_row(self, card, label, key, folder, save=False, file_filter="", tooltip=""):
        edit = QLineEdit(self._config.get(key, ""))
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        browse = QPushButton("Browse")

        def pick():
            if folder:
                path = QFileDialog.getExistingDirectory(self, label, edit.text())
            elif save:
                path, _ = QFileDialog.getSaveFileName(self, label, edit.text(), file_filter)
            else:
                path, _ = QFileDialog.getOpenFileName(self, label, edit.text(), file_filter)
            if path:
                edit.setText(path)

        browse.clicked.connect(pick)
        row.addWidget(browse)
        container = QWidget()
        container.setLayout(row)
        card.add_row(label, container, tooltip)
        return edit

    def values(self) -> dict:
        return {
            "training_library": self._library.text().strip(),
            "training_workdir": self._workdir.text().strip(),
            "training_exclude": self._exclude.text().strip(),
            "training_model": self._model_out.text().strip(),
            "training_epochs": self._epochs.value(),
        }

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    # ── actions ─────────────────────────────────────────────────────────

    def _on_start(self):
        v = self.values()
        error = validate_inputs(v["training_library"], v["training_workdir"], v["training_model"],
                                self._project_root)
        if error:
            self._status.setText(error)
            return
        self._config.update(v)
        try:
            from .config import save_config
            save_config(self._config)
        except Exception:  # noqa: BLE001 - remembering the inputs is a convenience only
            logger.debug("could not save training settings", exc_info=True)
        commands = build_training_commands(self._project_root, v["training_library"], v["training_workdir"],
                                           v["training_model"], v["training_exclude"], v["training_epochs"])
        self._log.clear_log()
        self._use.setEnabled(False)
        self._start.setEnabled(False)
        self._cancel.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._status.setText("Preparing...")
        self._thread = QThread()
        self._worker = TrainingWorker(commands, str(self._project_root))
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.line_output.connect(self._log.append_line)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

    def _on_cancel(self):
        if self._worker:
            self._worker.cancel()
        self._cancel.setEnabled(False)

    def shutdown(self):
        """Stop a running training (called when the window closes)."""
        if self._worker:
            self._worker.cancel()
        if self._thread:
            self._thread.quit()
            self._thread.wait(10000)

    def _on_progress(self, stage: str, current: int, total: int):
        self._progress.setRange(0, max(total, 1))
        self._progress.setValue(current)
        name = "Extracting songs" if stage == "extract" else "Training epoch"
        self._status.setText(f"{name} {current} / {total}")

    def _on_finished(self, code: int):
        self._start.setEnabled(True)
        self._cancel.setEnabled(False)
        self._progress.setVisible(False)
        if code == 0 and os.path.isfile(self._model_out.text().strip()):
            self._status.setText("Model trained. Measure it with the chart benchmark before relying on it.")
            self._use.setEnabled(True)
        elif code == -2:
            self._status.setText("Cancelled. Starting again continues the extraction where it stopped.")
        else:
            self._status.setText(f"Training failed (exit code {code}) - see the log.")
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
