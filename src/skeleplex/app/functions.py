"""Functions for the app."""

from qtpy.QtWidgets import QApplication


def close():
    """Close the active window."""
    QApplication.activeWindow().close()
    print("close")
