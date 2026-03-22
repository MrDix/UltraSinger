"""Entry point for the UltraSinger PySide6 GUI."""

import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path

# Set Windows AppUserModelID so the taskbar shows our icon instead of Python's.
# Must be called before QApplication is created.
if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("UltraSinger.GUI")

# Suppress noisy third-party warnings that fire at import time.
# Must run before any transitive imports of requests, torchaudio, pyannote, etc.
warnings.filterwarnings("ignore", module="requests")
warnings.filterwarnings("ignore", module="pyannote")
warnings.filterwarnings("ignore", message="In 2\\.9.*torchaudio\\.save_with_torchcodec")

# Lock file path — shared across all invocations to detect duplicate instances.
_LOCK_FILE = Path(tempfile.gettempdir()) / "ultrasinger_gui.lock"


def _acquire_instance_lock():
    """Try to acquire a file-based single-instance lock.

    Returns the open file handle on success (keep it alive for the
    process lifetime) or *None* if another instance already holds it.

    On Windows we use ``msvcrt.locking``; on Unix ``fcntl.flock``.
    Both are advisory but sufficient for our purposes.
    """
    try:
        # Open (or create) the lock file in write mode.  We intentionally
        # do NOT close it — the OS releases the lock when the process exits
        # (even on crash), which is exactly the behaviour we want.
        fh = open(_LOCK_FILE, "w", encoding="utf-8")  # noqa: SIM115
        fh.write(str(os.getpid()))
        fh.flush()

        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)

        return fh
    except (OSError, IOError):
        return None


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Suppress noisy Chromium stderr messages (cache errors, GPU warnings,
    # quota DB resets).  These are harmless but alarm users.
    # FATAL=3 means only show crashes; ERROR/WARNING/INFO are suppressed.
    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--log-level=3")

    # PySide6 imports here to give a clear error if not installed
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        from PySide6.QtCore import Qt
    except ImportError:
        print(
            "Error: PySide6 is not installed.\n"
            "Install the GUI dependencies with:\n"
            "  uv sync --extra gui\n"
            "or:\n"
            "  pip install PySide6 PySide6-WebEngine",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
    except ImportError:
        print(
            "Error: PySide6 WebEngine (part of PySide6-Addons) is not available.\n"
            "Install with:\n"
            "  uv sync --extra gui\n"
            "or:\n"
            "  pip install PySide6",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Single-instance guard ─────────────────────────────────────────
    # Must happen after PySide6 is imported (for QMessageBox) but before
    # QWebEngine initialises (to avoid Chromium DB lock conflicts).
    lock_handle = _acquire_instance_lock()
    if lock_handle is None:
        # Another instance is running — show a user-friendly dialog.
        # We need a temporary QApplication just to display the message.
        _guard_app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.warning(
            None,
            "UltraSinger",
            "UltraSinger is already running.\n\n"
            "Please close the other instance first.\n"
            "Running multiple instances simultaneously can cause\n"
            "database lock errors in the embedded browser.",
        )
        sys.exit(0)

    from PySide6.QtGui import QFont, QIcon

    # Load stylesheet and icon
    resources = Path(__file__).parent / "gui" / "resources"
    qss_path = resources / "styles.qss"
    icons_dir = resources / "icons"

    # Use .ico on Windows (taskbar/title bar), .png elsewhere
    if sys.platform == "win32":
        icon_path = icons_dir / "logo.ico"
    else:
        icon_path = icons_dir / "logo.png"

    app = QApplication(sys.argv)
    app.setApplicationName("UltraSinger")
    app.setOrganizationName("UltraSinger")

    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Set default font — ensures tooltips and all widgets use consistent sizing
    font = QFont("Segoe UI Variable", 10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    if qss_path.exists():
        qss = qss_path.read_text(encoding="utf-8")
        # Inject absolute icon paths (QSS can't resolve relative paths)
        chevron_path = (icons_dir / "chevron-down.svg").as_posix()
        qss = qss.replace("{{CHEVRON_DOWN}}", chevron_path)
        app.setStyleSheet(qss)

    from gui.main_window import MainWindow

    window = MainWindow()
    window.show()

    exit_code = app.exec()

    # Ensure Chromium's renderer subprocess terminates and flushes
    # cookies/storage to disk.  Destroying the window triggers
    # BrowserTab.shutdown() → QWebEnginePage/View deletion → Chromium
    # IPC shutdown.  processEvents() drains the event queue so the
    # subprocess has time to write before Python exits.
    del window
    for _ in range(10):
        app.processEvents()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
