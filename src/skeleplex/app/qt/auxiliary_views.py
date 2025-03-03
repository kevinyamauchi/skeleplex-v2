"""Dock widget to hold additional views of the data."""

from qtpy.QtWidgets import QDockWidget, QLabel, QWidget

from skeleplex.app.qt.styles import DOCK_WIDGET_STYLE


class AuxiliaryViews(QDockWidget):
    """A dock widget for the auxiliary views.

    This will hold things like a view of slices along a spline.
    """

    MINIMUM_WIDTH: int = 200

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        self.setStyleSheet(DOCK_WIDGET_STYLE)

        self.label = QLabel("I'm the Auxiliary Views")
        self.label.setMinimumWidth(self.MINIMUM_WIDTH)

        self.setWidget(self.label)
