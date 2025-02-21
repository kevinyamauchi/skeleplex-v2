"""Classes for interfacing with the viewer."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from cellier.models.data_stores.lines import LinesMemoryStore
from cellier.models.data_stores.points import PointsMemoryStore
from cellier.models.nodes.lines_node import LinesNode, LinesUniformMaterial
from cellier.models.nodes.points_node import PointsNode, PointsUniformMaterial
from cellier.viewer_controller import ViewerController as CellierViewerController

from skeleplex.app.cellier.utils import make_viewer_controller, make_viewer_model

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget


@dataclass
class RenderedSkeletonComponents:
    """A class for storing the components for a rendered skeleton.

    These data are used for accessing the rendered skeleton in the viewer backend.
    """

    nodes_store: PointsMemoryStore | None = None
    nodes_visual: PointsNode | None = None
    edges_store: LinesMemoryStore | None = None
    edges_visual: LinesNode | None = None
    edge_highlight_store: LinesMemoryStore | None = None
    edge_highlight_visual: LinesNode | None = None

    def populated(self) -> bool:
        """Returns True if all the components are populated."""
        return all(
            [
                self.nodes_store is not None,
                self.nodes_visual is not None,
                self.edges_store is not None,
                self.edges_visual is not None,
                self.edge_highlight_store is not None,
                self.edge_highlight_visual is not None,
            ]
        )


class MainCanvasController:
    """A class for controlling the main canvas."""

    def __init__(self, scene_id: str, backend: CellierViewerController):
        self._scene_id = scene_id
        self._backend = backend

        # this will store the rendered skeleton components
        self._skeleton = RenderedSkeletonComponents()

    @property
    def scene_id(self) -> str:
        """Get the scene ID of the main canvas."""
        return self._scene_id

    def update_skeleton_geometry(
        self, edge_coordinates: np.ndarray, node_coordinates: np.ndarray
    ):
        """Update the geometry of the skeleton in the viewer."""
        # make the highlight lines store
        edge_highlight_store = LinesMemoryStore(coordinates=np.empty((0, 3)))

        # make the highlight lines material
        edge_highlight_material_3d = LinesUniformMaterial(
            color=(1, 0, 1, 1), size=3, size_coordinate_space="data", opacity=1.0
        )

        # make the highlight lines model
        edge_highlight_visual = LinesNode(
            name="edge_highlight",
            data_store_id=edge_highlight_store.id,
            material=edge_highlight_material_3d,
            pick_write=True,
        )

        self._backend.add_data_store(data_store=edge_highlight_store)
        self._backend.add_visual(
            visual_model=edge_highlight_visual, scene_id=self.scene_id
        )

        # make the lines store
        edge_lines_store = LinesMemoryStore(coordinates=edge_coordinates)

        # make the lines material
        edge_lines_material_3d = LinesUniformMaterial(
            color=(0, 0, 1, 1), size=2, size_coordinate_space="data"
        )

        # make the lines model
        edge_lines_visual = LinesNode(
            name="edge_lines",
            data_store_id=edge_lines_store.id,
            material=edge_lines_material_3d,
        )

        self._backend.add_data_store(data_store=edge_lines_store)
        self._backend.add_visual(visual_model=edge_lines_visual, scene_id=self.scene_id)

        # make the points store
        points_store = PointsMemoryStore(coordinates=node_coordinates)

        # make the points material
        points_material_3d = PointsUniformMaterial(
            size=3, color=(0, 0, 0, 1), size_coordinate_space="data"
        )

        # make the points model
        points_visual_3d = PointsNode(
            name="node_points",
            data_store_id=points_store.id,
            material=points_material_3d,
        )

        # add the points to the viewer
        self._backend.add_data_store(data_store=points_store)
        self._backend.add_visual(visual_model=points_visual_3d, scene_id=self.scene_id)

        # store the rendered skeleton components
        self._skeleton = RenderedSkeletonComponents(
            nodes_store=points_store,
            nodes_visual=points_visual_3d,
            edges_store=edge_lines_store,
            edges_visual=edge_lines_visual,
            edge_highlight_store=edge_highlight_store,
            edge_highlight_visual=edge_highlight_visual,
        )

        # reslice the scene
        self._backend.reslice_scene(scene_id=self.scene_id)

    def look_at_skeleton(
        self,
        view_direction: tuple[int, int, int] = (0, 0, 1),
        up: tuple[int, int, int] = (0, 1, 0),
    ):
        """Adjust the camera to look at the skeleton."""
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        self._backend.look_at_visual(
            visual_id=self._skeleton.nodes_visual.id,
            view_direction=view_direction,
            up=up,
        )

    def set_edge_highlight(
        self,
        edge_coordinates: np.ndarray,
    ):
        """Set the edge highlight coordinates.

        Parameters
        ----------
        edge_coordinates : np.ndarray
            The coordinates of the edge to highlight.
        """
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        self._skeleton.edge_highlight_store.coordinates = edge_coordinates.astype(
            np.float32
        )
        self._backend.reslice_scene(scene_id=self.scene_id)

    def add_skeleton_edge_callback(
        self, callback: Callable, callback_type: tuple[str, ...]
    ):
        """Add a callback to the skeleton edge visual.

        Parameters
        ----------
        callback : Callable
            The callback function.
        callback_type : tuple[str, ...]
            The type of callback. See the pygfx documentation for event types.
        """
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        # add for the data visual
        self._backend.add_visual_callback(
            visual_id=self._skeleton.edges_visual.id,
            callback=partial(callback, click_source="data"),
            callback_type=callback_type,
        )

        # add for the highlight visual
        self._backend.add_visual_callback(
            visual_id=self._skeleton.edge_highlight_visual.id,
            callback=partial(callback, click_source="highlight"),
            callback_type=callback_type,
        )

    def add_skeleton_node_callback(
        self, callback: Callable, callback_type: tuple[str, ...]
    ):
        """Add a callback to the skeleton node visual.

        Parameters
        ----------
        callback : Callable
            The callback function.
        callback_type : tuple[str, ...]
            The type of callback. See the pygfx documentation for event types.
        """
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        self._backend.add_visual_callback(
            visual_id=self._skeleton.nodes_visual_visual.id,
            callback=callback,
            callback_type=callback_type,
        )

    def remove_skeleton_edge_callback(
        self, callback: Callable, callback_type: tuple[str, ...]
    ):
        """Remove a callback from the skeleton edge visual.

        Parameters
        ----------
        callback : Callable
            The callback function.
        callback_type : tuple[str, ...]
            The type of callback. See the pygfx documentation for event types.
        """
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        self._backend.remove_visual_callback(
            visual_id=self._skeleton.edges_visual.id,
            callback=callback,
            callback_type=callback_type,
        )

    def remove_skeleton_nodes_callback(
        self, callback: Callable, callback_type: tuple[str, ...]
    ):
        """Remove a callback from the skeleton node visual.

        Parameters
        ----------
        callback : Callable
            The callback function.
        callback_type : tuple[str, ...]
            The type of callback. See the pygfx documentation for event types.
        """
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        self._backend.remove_visual_callback(
            visual_id=self._skeleton.nodes_visual.id,
            callback=callback,
            callback_type=callback_type,
        )


class ViewerController:
    """A class for controlling the viewer backend."""

    def __init__(self, parent_widget: "QWidget"):
        viewer_model, main_canvas_scene_id = make_viewer_model()
        self._backend = make_viewer_controller(
            viewer_model=viewer_model, parent_widget=parent_widget
        )

        # make the main canvas controller
        self._main_canvas = MainCanvasController(
            scene_id=main_canvas_scene_id, backend=self._backend
        )

    @property
    def main_canvas(self):
        """Get the controller for the main canvas."""
        return self._main_canvas
