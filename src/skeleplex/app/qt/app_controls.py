"""Dock widget for the Application Controls."""

from pathlib import Path

from magicgui import magicgui
from qtpy.QtWidgets import QDockWidget, QVBoxLayout, QWidget

from skeleplex.app.data import SkeletonDataPaths


class AppControlsWidget(QWidget):
    """A widget for the application controls.

    This is the widget embedded in the AppControlsDock.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)

        self.load_data_widget = magicgui(self._load_data_gui)

        layout = QVBoxLayout()
        layout.addWidget(self.load_data_widget.native)
        layout.addStretch()
        self.setLayout(layout)

    def _load_data_gui(
        self,
        image_path: Path | None = None,
        segmentation_path: Path | None = None,
        skeleton_graph_path: Path | None = None,
    ) -> SkeletonDataPaths:
        """Magicgui callable to generate loading widget.

        This is used to generate a magicgui widget
        """
        return SkeletonDataPaths(
            image=image_path,
            segmentation=segmentation_path,
            skeleton_graph=skeleton_graph_path,
        )


class AppControlsDock(QDockWidget):
    """A dock widget for the application controls.

    This will be used as a container GUI elements
    for controlling the state of the application.
    """

    MINIMUM_WIDTH: int = 200

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)

        self.setWidget(AppControlsWidget(parent=self))
