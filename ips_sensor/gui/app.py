"""Application entry point kept separate so the portable core has no Qt import."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 is required to run the desktop GUI. Install with: pip install -e .[gui]")
        return 1
    from .main_window import SensorMainWindow

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Inductive Sensor Footprint Generator")
    window = SensorMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
