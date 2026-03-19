"""Entry point for the UltraSinger PySide6 GUI."""

import logging
import sys
from pathlib import Path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # PySide6 imports here to give a clear error if not installed
    try:
        from PySide6.QtWidgets import QApplication
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

    # Load stylesheet
    qss_path = Path(__file__).parent / "gui" / "resources" / "styles.qss"

    app = QApplication(sys.argv)
    app.setApplicationName("UltraSinger")
    app.setOrganizationName("UltraSinger")

    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    from gui.main_window import MainWindow

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
