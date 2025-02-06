"""The main application model."""

import logging

from app_model import Application
from app_model.types import Action, MenuRule
from cellier.models.data_stores.lines import LinesMemoryStore
from cellier.models.data_stores.points import PointsMemoryStore
from cellier.models.nodes.lines_node import LinesNode, LinesUniformMaterial
from cellier.models.nodes.points_node import PointsNode, PointsUniformMaterial

from skeleplex.app.actions import ACTIONS
from skeleplex.app.cellier import make_viewer_controller, make_viewer_model
from skeleplex.app.constants import CommandId, MenuId
from skeleplex.app.data import DataManager, SkeletonDataPaths
from skeleplex.app.qt import MainWindow

log = logging.getLogger(__name__)


class SkelePlexApp(Application):
    """The main application class."""

    def __init__(self) -> None:
        super().__init__("SkelePlex")

        # ACTIONS is a list of Action objects.
        for action in ACTIONS:
            self.register_action(action)
        self._register_data_actions()

        self._main_window = MainWindow(
            app=self,
        )
        # This will build a menu bar based on these menus
        self._main_window.setModelMenuBar([MenuId.FILE, MenuId.EDIT, MenuId.DATA])

        # make the data model
        self._data = DataManager(file_paths=SkeletonDataPaths())
        # self._data.file_paths.skeleton_graph = "./scripts/e16_skeleplex_v2.json"
        # self._data.load()

        # make the viewer model
        self._viewer_model, self._main_viewer_scene_id = make_viewer_model()
        self._viewer_controller = make_viewer_controller(
            viewer_model=self._viewer_model, parent_widget=self._main_window
        )

        for canvas in self._viewer_controller._canvas_widgets.values():
            # add the canvas widgets
            self._main_window.setCentralWidget(canvas)

        # connect the data events
        self._connect_data_events()

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

        # get the node coordinates
        coordinates = self.data.node_coordinates

        # make the points store
        points_store = PointsMemoryStore(coordinates=coordinates)

        # make the points material
        points_material_3d = PointsUniformMaterial(
            size=3, color=(1, 1, 1, 1), size_coordinate_space="data"
        )

        # make the points model
        points_visual_3d = PointsNode(
            name="points_node_3d",
            data_store_id=points_store.id,
            material=points_material_3d,
        )

        # add the points to the viewer
        self._viewer_controller.add_data_store(data_store=points_store)
        self._viewer_controller.add_visual(
            visual_model=points_visual_3d, scene_id=self._main_viewer_scene_id
        )

        # get the edge lines
        edge_coordinates = self.data.edge_coordinates

        # make the lines store
        lines_store = LinesMemoryStore(coordinates=edge_coordinates)

        # make the lines material
        lines_material_3d = LinesUniformMaterial(
            color=(0, 0, 1, 1), size=2, size_coordinate_space="data"
        )

        # make the lines model
        lines_visual_3d = LinesNode(
            name="lines_node_3d",
            data_store_id=lines_store.id,
            material=lines_material_3d,
        )

        self._viewer_controller.add_data_store(data_store=lines_store)
        self._viewer_controller.add_visual(
            visual_model=lines_visual_3d, scene_id=self._main_viewer_scene_id
        )

        self._viewer_controller.reslice_scene(scene_id=self._main_viewer_scene_id)

        self.points = points_visual_3d

    def look_at_skeleton(self) -> None:
        """Set the camera in the main viewer to look at the skeleton.

        todo: add guard for when called and no skeleton is loaded.
        """
        self._viewer_controller.look_at_visual(
            visual_id=self.points.id,
            view_direction=(0, 0, 1),
            up=(0, 1, 0),
        )

    def show(self) -> None:
        """Show the app."""
        self._main_window.show()

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
                callback=self.look_at_skeleton,
                menus=[MenuRule(id=MenuId.DATA)],
            )
        )

    def _connect_data_events(self) -> None:
        """Connect the events for handling changes in the data."""
        # event for when the data loading button is pressed
        # this updates the data paths
        self._main_window.app_controls.widget().load_data_widget.called.connect(
            self._load_data_clicked
        )

        # event for updating the main viewer when the data paths are updated
        self.data.events.data.connect(self.load_main_viewer)

    def _load_data_clicked(
        self,
        new_data_paths: SkeletonDataPaths,
    ) -> None:
        """Load data from the GUI.

        This is a method intended to be used to generate a magicgui widget
        for the GUI.
        """
        self.data.file_paths.image = new_data_paths.image
        self.data.file_paths.segmentation = new_data_paths.segmentation
        self.data.file_paths.skeleton_graph = new_data_paths.skeleton_graph
        self.data.load()
