"""Training window: train a note segmentation model on the user's own song library.

Opened with "Train..." next to the Segmentation Model setting. Runs
``tools/train_segmentation.py extract`` and then ``train`` in a background
process, shows progress and the log, and sets the finished model as
Segmentation Model in the settings.
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
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
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


class TrainingDialog(QDialog):
    """Window with inputs, progress and log for training a segmentation model.

    Not modal: a training runs for a long time and the main window stays usable.
    Closing the window only hides it; a running training continues, and the
    finished model is handed to the settings through ``model_ready``.
    """

    model_ready = Signal(str)  # path of a model that has just been trained
    running_changed = Signal(bool)
    ended = Signal(str)  # status message when a training run has ended

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train a Segmentation Model")
        self.setModal(False)
        self.setMinimumSize(760, 560)
        self.resize(900, 720)
        self._config = config
        self._thread: QThread | None = None
        self._worker: TrainingWorker | None = None
        self._model_in_training = ""
        self._project_root = _find_project_root()
        self._input_rows: list[QWidget] = []

        layout = QVBoxLayout(self)

        card = SettingsCard("Train a Segmentation Model (experimental)")
        card.add_info(
            "Learns from the hand-made charts in your song library where notes start, how long "
            "they are and which passages are charted at all. Every song is separated once (GPU "
            "recommended, about 20-40 s per song), then the model is trained (about 30 minutes "
            "for 1000 songs on a GPU). Use well-timed charts: the model learns their style. The "
            "work folder and the model are derived from your library - keep them private. "
            "You can close this window while the training runs; it continues in the background, "
            "and the finished model is set as Segmentation Model in the settings."
        )
        self._library = self._path_row(
            card, "Song Library", "training_library", folder=True,
            tooltip="Folder with your UltraStar songs (searched recursively). "
                    "Solo charts with at least 100 notes and their audio are used.",
            browse_tooltip="Pick the folder with your UltraStar songs.")
        self._workdir = self._path_row(
            card, "Work Folder", "training_workdir", folder=True,
            tooltip="Where the extracted training data is stored (several GB for 1000 songs), "
                    "outside the UltraSinger folder. An interrupted extraction continues here.",
            browse_tooltip="Pick a folder for the extracted training data.")
        self._exclude = self._path_row(
            card, "Exclude Songs", "training_exclude", folder=False,
            file_filter="Song list (songs.json)",
            tooltip="Optional: songs.json of a chart benchmark sample - these songs are never "
                    "trained on, so the benchmark stays meaningful.",
            browse_tooltip="Pick the songs.json of a chart benchmark sample.")
        self._model_out = self._path_row(
            card, "Model File", "training_model", folder=False, save=True,
            file_filter="Segmentation model (*.pt)",
            tooltip="Where the trained model is written, outside the UltraSinger folder. When the "
                    "training has finished, this file is set as Segmentation Model in the settings.",
            browse_tooltip="Choose where to save the trained model.")
        self._epochs = QSpinBox()
        self._epochs.setRange(1, 200)
        self._epochs.setValue(int(config.get("training_epochs", 30)))
        card.add_row("Epochs", self._epochs, "Training passes over the extracted songs; the model "
                     "of the pass that does best on held-out songs is kept.")
        self._input_rows.append(self._epochs)
        layout.addWidget(card)

        buttons = QHBoxLayout()
        self._start = QPushButton("Start Training")
        self._start.setToolTip("Extract the training data from the song library (an interrupted "
                               "extraction continues), then train the model and save it as Model File.")
        self._start.clicked.connect(self._on_start)
        self._stop = QPushButton("Stop Training")
        self._stop.setToolTip("Stop the extraction or training. Starting again continues the "
                              "extraction where it stopped; the training itself starts over.")
        self._stop.setEnabled(False)
        self._stop.clicked.connect(self._on_stop)
        buttons.addWidget(self._start)
        buttons.addWidget(self._stop)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._status)
        layout.addWidget(self._progress)
        self._log = LogViewer()
        layout.addWidget(self._log, 1)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self._close = QPushButton("Close")
        self._close.setToolTip("Close this window. A running training continues in the background.")
        self._close.clicked.connect(self.close)
        bottom.addWidget(self._close)
        layout.addLayout(bottom)

        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)  # Enter in a path field must not press a button

    # ── helpers ─────────────────────────────────────────────────────────

    def _path_row(self, card, label, key, folder, save=False, file_filter="", tooltip="",
                  browse_tooltip=""):
        edit = QLineEdit(self._config.get(key, ""))
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        browse = QPushButton("Browse")
        browse.setToolTip(browse_tooltip)

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
        self._input_rows.append(container)
        return edit

    def _set_inputs_enabled(self, enabled: bool):
        for row in self._input_rows:
            row.setEnabled(enabled)

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
        self._model_in_training = v["training_model"]
        self._log.clear_log()
        self._set_inputs_enabled(False)
        self._start.setEnabled(False)
        self._stop.setEnabled(True)
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
        self.running_changed.emit(True)

    def _on_stop(self):
        if self._worker:
            self._worker.cancel()
        self._stop.setEnabled(False)

    def shutdown(self):
        """Stop a running training (called when the main window closes)."""
        if self._worker:
            self._worker.cancel()  # stops the process tree, so the worker's run() returns
        if self._thread:
            self._thread.quit()
            # No timeout: a QThread destroyed while still running aborts the
            # application, and cancel() has already stopped what run() waits for.
            self._thread.wait()

    def _on_progress(self, stage: str, current: int, total: int):
        self._progress.setRange(0, max(total, 1))
        self._progress.setValue(current)
        name = "Extracting songs" if stage == "extract" else "Training epoch"
        self._status.setText(f"{name} {current} / {total}")

    def _on_finished(self, code: int):
        self._start.setEnabled(True)
        self._stop.setEnabled(False)
        self._set_inputs_enabled(True)
        self._progress.setVisible(False)
        model = self._model_in_training
        trained = code == 0 and os.path.isfile(model)
        if trained:
            self._status.setText("Model trained and set as Segmentation Model in the settings. "
                                 "Measure it with the chart benchmark before relying on it.")
        elif code == 0:
            self._status.setText("The training ended without writing the model file - see the training log.")
        elif code == -2:
            self._status.setText("Stopped. Starting again continues the extraction where it stopped.")
        else:
            self._status.setText(f"Training failed (exit code {code}) - see the training log.")
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        self.running_changed.emit(False)
        if trained:
            self.model_ready.emit(model)
        self.ended.emit(self._status.text())
