"""Dock widget for the Application Controls."""

import sys
from pathlib import Path

import numpy as np
from magicgui import magicgui
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDockWidget,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from skeleplex.app._data import (
    AllViewRequest,
    BoundingBoxViewRequest,
    EdgeSelectionPasteRequest,
    NodeSelectionPasteRequest,
    NoneViewRequest,
    SkeletonDataPaths,
    ViewRequest,
)
from skeleplex.app.qt.flat_group_box import FlatHGroupBox, FlatVGroupBox
from skeleplex.app.qt.styles import (
    DOCK_WIDGET_STYLE,
)

GROUP_BOX_STYLE = """
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


class ViewAllModeControls(QGroupBox):
    """A widget for controlling the view all mode."""

    render_requested = Signal(AllViewRequest)

    def __init__(self, parent=None):
        super().__init__(title="View all controls", parent=parent)

        # button to render the view
        self.render_button = QPushButton("Render", parent=self)

        # connect the button click event
        self.render_button.clicked.connect(self._on_render_button_clicked)

        # make the layout
        layout = QVBoxLayout()
        layout.addWidget(self.render_button)
        self.setLayout(layout)

        # set the style
        self.setStyleSheet(GROUP_BOX_STYLE)

    def _on_render_button_clicked(self):
        """Handle the render button click event."""
        # Emit a signal to request rendering the view
        self.render_requested.emit(AllViewRequest())


class ViewNoneModeControls(QGroupBox):
    """A widget for controlling the view none mode."""

    render_requested = Signal(NoneViewRequest)

    def __init__(self, parent=None):
        super().__init__(title="View none controls", parent=parent)

        # button to render the view
        self.render_button = QPushButton("Render", parent=self)

        # connect the button click event
        self.render_button.clicked.connect(self._on_render_button_clicked)

        # make the layout
        layout = QVBoxLayout()
        layout.addWidget(self.render_button)
        self.setLayout(layout)

        # set the style
        self.setStyleSheet(GROUP_BOX_STYLE)

    def _on_render_button_clicked(self):
        """Handle the render button click event."""
        # Emit a signal to request rendering the view
        self.render_requested.emit(NoneViewRequest())


class ViewBoundingBoxControls(QGroupBox):
    """A widget for controlling the bounding box view mode."""

    def __init__(self, parent=None):
        super().__init__(title="Bounding box controls", parent=parent)

        # make widget for setting bounding box
        self.bounding_box_widget = magicgui(
            self.update_bounding_box,
            call_button="Render",
            minimum={"options": {"max": sys.float_info.max}},
            maximum={"options": {"max": sys.float_info.max}},
        )

        # make the layout
        layout = QVBoxLayout()
        layout.addWidget(self.bounding_box_widget.native)
        self.setLayout(layout)

        # set the style
        self.setMaximumWidth(350)
        self.setStyleSheet(GROUP_BOX_STYLE)

    def update_bounding_box(
        self, minimum: tuple[float, float, float], maximum: tuple[float, float, float]
    ) -> BoundingBoxViewRequest:
        """Update the bounding box controls with new minimum and maximum values."""
        # This method can be extended to update the bounding box controls
        # based on the provided minimum and maximum coordinates.
        # For now, it is a placeholder.
        return BoundingBoxViewRequest(
            minimum=np.asarray(minimum), maximum=np.asarray(maximum)
        )


class SkeletonDataViewWidget(FlatHGroupBox):
    """A widget for selecting which regions of the data are in view."""

    # signal gets emitted when a view request is made
    view_requested = Signal(ViewRequest)

    def __init__(self, collapsible: bool = False, parent: QWidget | None = None):
        super().__init__(
            title="Skeleton View",
            accent_color="#b7e2d8",
            collapsible=collapsible,
            parent=parent,
        )

        # buttons for the mode
        self.mode_buttons = QButtonGroup(parent=self)
        self.all_button = QRadioButton("All", parent=self)
        self.all_button.setChecked(True)
        self.bounding_box_button = QRadioButton("Bounding box", parent=self)
        self.mode_buttons.addButton(self.all_button)
        self.mode_buttons.addButton(self.bounding_box_button)
        self.mode_buttons.setExclusive(True)
        self.button_box = QGroupBox(title="View mode", parent=self)
        self.button_box.setStyleSheet(GROUP_BOX_STYLE)
        layout = QVBoxLayout()
        layout.addWidget(self.all_button)
        layout.addWidget(self.bounding_box_button)
        self.button_box.setAutoFillBackground(True)
        self.button_box.setLayout(layout)

        # Make the view all widget
        self.view_all_controls = ViewAllModeControls(parent=self)

        # make the view bounding box widget
        self.view_bounding_box_controls = ViewBoundingBoxControls(parent=self)
        self.view_bounding_box_controls.setVisible(False)

        # connect the view all event
        self.view_all_controls.render_requested.connect(self._on_view_requested)

        # connect the view bounding box event
        self.view_bounding_box_controls.bounding_box_widget.called.connect(
            self._on_view_requested
        )

        # connect the mode buttons
        self.mode_buttons.buttonClicked.connect(self._on_mode_changed)

        # Add the widgets
        self.add_widget(self.button_box)
        self.add_widget(self.view_all_controls)
        self.add_widget(self.view_bounding_box_controls)

    def _on_mode_changed(self):
        if self.all_button.isChecked():
            # View all controls selected
            self.view_all_controls.setVisible(True)
            self.view_bounding_box_controls.setVisible(False)
        else:
            # Bounding box controls selected
            self.view_all_controls.setVisible(False)
            self.view_bounding_box_controls.setVisible(True)

    def _on_view_requested(self, request: ViewRequest):
        """Relay the view request when one of the subwidgets make a request.

        Parameters
        ----------
        request : ViewRequest
            The view request to relay.
        """
        self.view_requested.emit(request)


class SegmentationDataViewWidget(FlatHGroupBox):
    """A widget for selecting which regions of the data are in view."""

    # signal gets emitted when a view request is made
    view_requested = Signal(ViewRequest)

    def __init__(self, collapsible: bool = False, parent: QWidget | None = None):
        super().__init__(
            title="Segmentation View",
            accent_color="#b7e2d8",
            collapsible=collapsible,
            parent=parent,
        )

        # buttons for the mode
        self.mode_buttons = QButtonGroup(parent=self)
        self.none_button = QRadioButton("None", parent=self)
        self.none_button.setChecked(True)
        self.all_button = QRadioButton("All", parent=self)
        self.bounding_box_button = QRadioButton("Bounding box", parent=self)
        self.mode_buttons.addButton(self.none_button)
        self.mode_buttons.addButton(self.all_button)
        self.mode_buttons.addButton(self.bounding_box_button)
        self.mode_buttons.setExclusive(True)
        self.button_box = QGroupBox(title="View mode", parent=self)
        self.button_box.setStyleSheet(GROUP_BOX_STYLE)
        layout = QVBoxLayout()
        layout.addWidget(self.none_button)
        layout.addWidget(self.all_button)
        layout.addWidget(self.bounding_box_button)
        self.button_box.setAutoFillBackground(True)
        self.button_box.setLayout(layout)

        # make the view none widget
        self.view_none_controls = ViewNoneModeControls(parent=self)

        # Make the view all widget
        self.view_all_controls = ViewAllModeControls(parent=self)
        self.view_all_controls.setVisible(False)

        # make the view bounding box widget
        self.view_bounding_box_controls = ViewBoundingBoxControls(parent=self)
        self.view_bounding_box_controls.setVisible(False)

        # connect the view none event
        self.view_none_controls.render_requested.connect(self._on_view_requested)

        # connect the view all event
        self.view_all_controls.render_requested.connect(self._on_view_requested)

        # connect the view bounding box event
        self.view_bounding_box_controls.bounding_box_widget.called.connect(
            self._on_view_requested
        )

        # connect the mode buttons
        self.mode_buttons.buttonClicked.connect(self._on_mode_changed)

        # Add the widgets
        self.add_widget(self.button_box)
        self.add_widget(self.view_none_controls)
        self.add_widget(self.view_all_controls)
        self.add_widget(self.view_bounding_box_controls)

    def _on_mode_changed(self):
        if self.none_button.isChecked():
            # No segmentation selected
            self.view_none_controls.setVisible(True)

            # set the other controls to invisible
            self.view_all_controls.setVisible(False)
            self.view_bounding_box_controls.setVisible(False)
        elif self.all_button.isChecked():
            # View all controls selected
            self.view_all_controls.setVisible(True)

            # set the other controls to invisible
            self.view_none_controls.setVisible(False)
            self.view_bounding_box_controls.setVisible(False)
        else:
            # Bounding box controls selected
            self.view_bounding_box_controls.setVisible(True)

            # set the other controls to invisible
            self.view_none_controls.setVisible(False)
            self.view_all_controls.setVisible(False)

    def _on_view_requested(self, request: ViewRequest):
        """Relay the view request when one of the subwidgets make a request.

        Parameters
        ----------
        request : ViewRequest
            The view request to relay.
        """
        self.view_requested.emit(request)


class SelectionModeWidget(QGroupBox):
    """Widget for controlling a selection mode."""

    def __init__(self, title: str = "", parent: QWidget | None = None):
        super().__init__(title=title, parent=parent)

        self.enable_checkbox = QCheckBox("Enable")
        self.selection_box = QLineEdit()

        # Make the layout
        layout = QHBoxLayout()
        layout.addWidget(self.enable_checkbox)
        layout.addWidget(self.selection_box)
        self.setLayout(layout)

        # set the style
        self.setStyleSheet(GROUP_BOX_STYLE)


class DataSelectorWidget(FlatHGroupBox):
    """A widget for selecting data from the main viewer."""

    def __init__(self, collapsible: bool = False, parent: QWidget | None = None):
        super().__init__(
            title="Data Selector",
            accent_color="#cab8c4",
            collapsible=collapsible,
            parent=parent,
        )

        self.edge_mode_box = SelectionModeWidget(title="Edge", parent=self)
        self.node_mode_box = SelectionModeWidget(title="Node", parent=self)

        # Add the widgets
        self.add_widget(self.edge_mode_box)
        self.add_widget(self.node_mode_box)

    def _on_edge_selection_change(self, event):
        """Update the GUI when the selected edges change."""
        if len(event) == 0:
            self.edge_mode_box.selection_box.setText("")
        else:
            self.edge_mode_box.selection_box.setText(str(event))

    def _on_node_selection_change(self, event):
        """Update the GUI when the selected nodes change."""
        if len(event) == 0:
            self.node_mode_box.selection_box.setText("")
        else:
            self.node_mode_box.selection_box.setText(str(event))

    def _on_edge_paste_request(self, paste_request: EdgeSelectionPasteRequest):
        """Handle a request to paste edge selection data.

        This pastes the edges from the paste request into the currently
        selected widget if the widget is a LineEdit.
        """
        if not isinstance(paste_request, EdgeSelectionPasteRequest):
            # if not a paste request, do nothing
            return

        selected_widget = QApplication.focusWidget()
        if not isinstance(selected_widget, QLineEdit):
            # if the selected widget isn't a QLineEdit, do nothing
            return

        # paste the text
        selected_widget.setText(str(paste_request.edge_keys))

    def _on_node_paste_request(self, paste_request: NodeSelectionPasteRequest):
        """Handle a request to paste node selection data.

        This pastes the nodes from the paste request into the currently
        selected widget if the widget is a LineEdit.
        """
        if not isinstance(paste_request, NodeSelectionPasteRequest):
            # if not a paste request, do nothing
            return

        selected_widget = QApplication.focusWidget()
        if not isinstance(selected_widget, QLineEdit):
            # if the selected widget isn't a QLineEdit, do nothing
            return

        # paste the text
        selected_widget.setText(str(paste_request.node_keys))


class AppControlsWidget(QWidget):
    """A widget for the application controls.

    This is the widget embedded in the AppControlsDock.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)

        self.load_data_widget = magicgui(self._load_data_gui)
        stores_box = FlatVGroupBox(
            "Data Stores", accent_color="#b7e2d8", collapsible=True, parent=self
        )
        stores_box.add_widget(self.load_data_widget.native)

        # widget for selecting the skeleton data view
        self.skeleton_view_box = SkeletonDataViewWidget(
            collapsible=True,
            parent=self,
        )

        # widget for selecting the segmentation data view
        self.segmentation_view_box = SegmentationDataViewWidget(
            collapsible=True,
            parent=self,
        )

        # widget for selecting the data selection mode
        self.selection_box = DataSelectorWidget(collapsible=True, parent=self)

        # make the layout
        layout = QVBoxLayout()
        layout.addWidget(stores_box)
        layout.addWidget(self.skeleton_view_box)
        layout.addWidget(self.segmentation_view_box)
        layout.addWidget(self.selection_box)
        layout.addStretch()

        layout.setAlignment(Qt.AlignTop)

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
