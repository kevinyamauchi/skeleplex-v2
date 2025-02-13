"""Dock widget for the Application Controls."""

from pathlib import Path

from magicgui import magicgui
from qtpy.QtWidgets import (
    QButtonGroup,
    QDockWidget,
    QGroupBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from skeleplex.app.data import SkeletonDataPaths
from skeleplex.app.qt.flat_group_box import FlatGroupBox
from skeleplex.app.qt.styles import (
    DOCK_WIDGET_STYLE,
)

VIEW_BUTTON_STYLE = """
QGroupBox {
    background-color: #f3f3f3;
    border: 1px solid black;
    margin-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 7px;
    padding: 0px 5px 0px 5px;
    background-color: #f3f3f3;
}
QRadioButton {
    background-color: #f3f3f3;
}
"""


class DataViewWidget(FlatGroupBox):
    """A widget for selecting which regions of the data are in view."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(title="Data View", accent_color="#b7e2d8", parent=parent)

        # buttons for the mode
        self.mode_buttons = QButtonGroup(parent=self)
        self.all_button = QRadioButton("All", parent=self)
        self.bounding_box_button = QRadioButton("Bounding box", parent=self)
        self.node_button = QRadioButton("Node", parent=self)
        self.mode_buttons.addButton(self.all_button)
        self.mode_buttons.addButton(self.bounding_box_button)
        self.mode_buttons.addButton(self.node_button)
        self.mode_buttons.setExclusive(True)
        self.button_box = QGroupBox(title="View mode", parent=self)
        self.button_box.setStyleSheet(VIEW_BUTTON_STYLE)
        layout = QVBoxLayout()
        layout.addWidget(self.all_button)
        layout.addWidget(self.bounding_box_button)
        layout.addWidget(self.node_button)
        self.button_box.setAutoFillBackground(True)
        self.button_box.setLayout(layout)

        # button to render the view
        self.render_button = QPushButton("Render", parent=self)

        # Add the widgets
        self.add_widget(self.button_box)
        self.add_widget(self.render_button)


class DataSelectorWidget(FlatGroupBox):
    """A widget for selecting data from the main viewer."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(title="Data Selector", accent_color="#cab8c4", parent=parent)

        # buttons for the mode
        self.mode_buttons = QButtonGroup(parent=self)
        self.image_button = QRadioButton("Image", parent=self)
        self.segmentation_button = QRadioButton("Segmentation", parent=self)
        self.skeleton_button = QRadioButton("Skeleton", parent=self)
        self.mode_buttons.addButton(self.image_button)
        self.mode_buttons.addButton(self.segmentation_button)
        self.mode_buttons.addButton(self.skeleton_button)
        self.mode_buttons.setExclusive(True)
        self.button_box = QGroupBox(title="Data type", parent=self)
        self.button_box.setStyleSheet(VIEW_BUTTON_STYLE)
        layout = QVBoxLayout()
        layout.addWidget(self.image_button)
        layout.addWidget(self.segmentation_button)
        layout.addWidget(self.skeleton_button)
        self.button_box.setAutoFillBackground(True)
        self.button_box.setLayout(layout)

        # Add the widgets
        self.add_widget(self.button_box)


class AppControlsWidget(QWidget):
    """A widget for the application controls.

    This is the widget embedded in the AppControlsDock.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)

        self.load_data_widget = magicgui(self._load_data_gui)
        stores_box = FlatGroupBox("Data Stores", accent_color="#b7e2d8", parent=self)
        stores_box.add_widget(self.load_data_widget.native)

        # widget for selecting the data view
        self.view_box = DataViewWidget(parent=self)

        # widget for selecting the data selection mode
        self.selection_box = DataSelectorWidget(parent=self)

        layout = QVBoxLayout()

        layout.addWidget(stores_box)
        layout.addWidget(self.view_box)
        layout.addWidget(self.selection_box)

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
        self.setStyleSheet(DOCK_WIDGET_STYLE)
        self.setWidget(AppControlsWidget(parent=self))
