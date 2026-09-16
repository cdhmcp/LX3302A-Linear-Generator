"""PyInstaller-friendly launcher for the desktop application."""

from ips_sensor.gui.app import main


if __name__ == "__main__":
    raise SystemExit(main())
