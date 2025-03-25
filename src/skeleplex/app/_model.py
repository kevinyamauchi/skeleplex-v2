"""The main application model."""

import logging

import numpy as np
from app_model import Application
from app_model.types import Action, MenuRule

from skeleplex.app._constants import CommandId, MenuId
from skeleplex.app._data import (
    DataManager,
    SelectionManager,
    SkeletonDataPaths,
)
from skeleplex.app._viewer_controller import ViewerController
from skeleplex.app.actions import ACTIONS
from skeleplex.app.qt import MainWindow

log = logging.getLogger(__name__)


class SkelePlexApp(Application):
    """The main application class."""

    def __init__(
        self, data: DataManager | None = None, selection: SelectionManager | None = None
    ) -> None:
        super().__init__("SkelePlex")

        self._main_window = MainWindow(
            app=self,
        )
        # This will build a menu bar based on these menus
        self._main_window.setModelMenuBar([MenuId.FILE, MenuId.EDIT, MenuId.DATA])

        # make the data model
        if data is None:
            data = DataManager(file_paths=SkeletonDataPaths(), selection=selection)
        self._data = data

        # make the viewer model
        self._viewer = ViewerController(parent_widget=self._main_window)

        for canvas in self._viewer._backend._canvas_widgets.values():
            # add the canvas widgets
            self._main_window._set_main_viewer_widget(canvas)

        # ACTIONS is a list of Action objects.
        for action in ACTIONS:
            self.register_action(action)
        self._register_data_actions()

        # connect the data events
        self._connect_data_events()

        # connect the selection events
        self._connect_selection_events()

        # update the data view
        self.data.view.update()

    @property
    def data(self) -> DataManager:
        """Get the data manager."""
        return self._data

    def load_main_viewer(self) -> None:
        """Add the data to the main viewer."""
        log.debug("Loading data into the main viewer...")
        if self.data.skeleton_graph is None:
            log.debug("No skeleton graph loaded.")
            return

        self._viewer.main_canvas.update_skeleton_geometry(
            node_coordinates=self.data.view.node_coordinates,
            edge_coordinates=self.data.view.edge_coordinates,
        )

    def look_at_skeleton(self) -> None:
        """Set the camera in the main viewer to look at the skeleton."""
        self._viewer.main_canvas.look_at_skeleton()

    def show(self) -> None:
        """Show the app."""
        self._main_window.show()

    def _on_edge_selection_change(self, event) -> None:
        """Handle a change in the edge selection."""
        print(event)
        if len(event) == 0:
            coordinates = np.empty((0, 3))
            highlighted_edge_keys = np.empty((0, 2))
        else:
            coordinates = []
            highlighted_edge_keys = []
            for edge_index in event:
                print(edge_index)
                edge_mask = np.all(
                    np.equal(self.data.view.edge_keys, np.asarray(edge_index)), axis=1
                )
                coordinates.append(self.data.view.edge_coordinates[edge_mask])
                highlighted_edge_keys.append(self.data.view.edge_keys[edge_mask])
            coordinates = np.concatenate(coordinates)
            highlighted_edge_keys = np.concatenate(highlighted_edge_keys)

        # set the highlight in the rendered scene
        self._viewer.main_canvas.set_edge_highlight(edge_coordinates=coordinates)

        # store the highlighted edge keys
        self.data.view._highlighted_edge_keys = highlighted_edge_keys

    def _register_data_actions(self) -> None:
        """Register actions for adding/removing data to/from the viewer."""
        # add points
        self.register_action(
            Action(
                id=CommandId.LOAD_DATA,
                title="Load data",
                icon="fa6-solid:folder-open",
                callback=self.load_main_viewer,
                menus=[MenuRule(id=MenuId.DATA)],
            )
        )
        self.register_action(
            Action(
                id=CommandId.LOOK_AT_SKELETON,
                title="Look at skeleton",
                icon="fa6-solid:house",
                callback=self._viewer.main_canvas.look_at_skeleton,
                menus=[MenuRule(id=MenuId.DATA)],
            )
        )

    def _connect_data_events(self) -> None:
        """Connect the events for handling changes in the data."""
        # event for when the data loading button is pressed
        # this updates the data paths and loads the data.
        self._main_window.app_controls.widget().load_data_widget.called.connect(
            self.data._update_paths_load_data
        )

        # event for updating the view when the render button is pressed
        self._main_window.app_controls.widget().view_box.render_button.clicked.connect(
            self.data.view.update
        )

        # event for updating the main viewer when the data paths are updated
        self.data.view.events.data.connect(self.load_main_viewer)

    def _connect_selection_events(self) -> None:
        """Connect the events for handling changes to the data selection state.

        These all interface with the SelectionManager.
        """
        # the widget containing all the selection GUI
        selection_widget = self._main_window.app_controls.widget().selection_box

        # events for synchronizing the edge selection enabled state with the GUI.
        selection_widget.edge_mode_box.enable_checkbox.stateChanged.connect(
            self.data.selection._on_edge_enabled_update
        )

        # event for synchronizing the edge selection values with the viewer.
        self.data.selection.edge.events.values.connect(self._on_edge_selection_change)

        # event for attaching/detaching the edge selection callback.
        self.data.selection.edge.events.enabled.connect(
            self._on_edge_selection_enabled_changed
        )

        # event for updating the edge selection GUI when the selection changes.
        self.data.selection.edge.events.values.connect(
            selection_widget._on_edge_selection_change
        )

    def _on_edge_selection_enabled_changed(self, enabled: bool):
        """Callback to update the viewer when the edge selection is enabled/disabled.

        This attached/detaches the edge selection callback from the main viewer.
        """
        if enabled:
            # attach the selection callback
            self._viewer.main_canvas.add_skeleton_edge_callback(
                callback=self.data._on_edge_selection_click,
            )

        else:
            # detach the selection callback
            self._viewer.main_canvas.remove_skeleton_edge_callback(
                callback=self.data._on_edge_selection_click,
            )
