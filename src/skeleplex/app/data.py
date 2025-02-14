"""Module for handling data in the SkelePlex application."""

import logging
from enum import Enum

import numpy as np
from psygnal import EventedModel, Signal, SignalGroup
from pydantic.types import FilePath

from skeleplex.graph import SkeletonGraph
from skeleplex.visualize.spline import line_segment_coordinates_from_spline

log = logging.getLogger(__name__)


class SkeletonDataPaths(EventedModel):
    """A class storing the state of the skeleton dataset.

    Parameters
    ----------
    image : FilePath | None
        The path to the image file.
    segmentation : FilePath | None
        The path to the segmentation image file.
    skeleton_graph : FilePath
        The path to the skeleton graph file.
    """

    image: FilePath | None = None
    segmentation: FilePath | None = None
    skeleton_graph: FilePath | None = None

    def has_paths(self) -> bool:
        """Returns true if any of the paths are set."""
        return any([self.image, self.segmentation, self.skeleton_graph])


class ViewMode(Enum):
    """The different viewing modes.

    ALL: Show all data.
    BOUNDING_BOX: Show data in a specified bounding box.
    NODE: Show data around a specified node.
    """

    ALL = "all"
    BOUNDING_BOX = "bounding_box"
    NODE = "node"


class ViewEvents(SignalGroup):
    """Events for the DataManager class."""

    data = Signal()


class DataView:
    """A class to manage the current view on the data."""

    events = ViewEvents()

    def __init__(
        self, data_manager: "DataManager", mode: ViewMode = ViewMode.ALL
    ) -> None:
        self._data_manager = data_manager
        self._mode = mode

        # initialize the data
        self._edge_coordinates: np.ndarray | None = None
        self._edge_indices: np.ndarray | None = None
        self._node_coordinates: np.ndarray | None = None

    @property
    def mode(self) -> ViewMode:
        """Get the current view mode."""
        return self._mode

    @mode.setter
    def mode(self, mode: ViewMode | str) -> None:
        """Set the current view mode."""
        if not isinstance(mode, ViewMode):
            mode = ViewMode(mode)
        self._mode = mode
        self.update()

    @property
    def node_coordinates(self) -> np.ndarray | None:
        """Get the coordinates of the current view of the nodes in the skeleton graph.

        (n_nodes, 3) array of node coordinates.
        """
        return self._node_coordinates

    @property
    def edge_coordinates(self) -> np.ndarray | None:
        """Get the coordinates of the current view of the edges in the skeleton graph.

        (n_edges x 2 x n_points_per_edge, 3) array of edge coordinates.
        """
        return self._edge_coordinates

    @property
    def edge_indices(self) -> np.ndarray | None:
        """Get the indices of the current view of the edges in the skeleton graph.

        (n_edges x 2 x n_points_per_edge,) array of edge indices.
        """
        return self._edge_indices

    def _get_view_all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the view for all data."""
        return (
            self._data_manager.node_coordinates,
            self._data_manager.edge_coordinates,
            self._data_manager.edge_indices,
        )

    def update(self) -> None:
        """Update the data for the currently specified view.

        This updates the edge coordinates, edge indices, and node indices.
        """
        if self._data_manager.skeleton_graph is None:
            # if the data isn't loaded, nothing to update
            return
        if self._mode == ViewMode.ALL:
            self._node_coordinates, self._edge_coordinates, self._edge_indices = (
                self._get_view_all()
            )
        else:
            raise NotImplementedError(f"View mode {self._mode} not implemented.")

        # Emit signal that the view data has been updated
        self.events.data.emit()


class DataEvents(SignalGroup):
    """Events for the DataManager class."""

    data = Signal()


class DataManager:
    """A class to manage data."""

    events = DataEvents()

    def __init__(
        self,
        file_paths: SkeletonDataPaths,
        load_data: bool = True,
    ) -> None:
        self._file_paths = file_paths

        self._view = DataView(data_manager=self, mode=ViewMode.ALL)

        # initialize the data
        self._skeleton_graph: SkeletonGraph | None = None
        self._node_coordinates: np.ndarray | None = None
        self._edge_coordinates: np.ndarray | None = None
        self._edge_indices: np.ndarray | None = None

        if self.file_paths.has_paths() and load_data:
            self.load()

    @property
    def file_paths(self) -> SkeletonDataPaths:
        """Get the file paths."""
        return self._file_paths

    @property
    def view(self) -> DataView:
        """Get the current data view."""
        return self._view

    @property
    def skeleton_graph(self) -> SkeletonGraph:
        """Get the skeleton graph."""
        return self._skeleton_graph

    @property
    def node_coordinates(self) -> np.ndarray | None:
        """Get the coordinates of the nodes in the skeleton graph.

        (n_nodes, 3) array of node coordinates.
        """
        return self._node_coordinates

    @property
    def edge_coordinates(self) -> np.ndarray | None:
        """Get the coordinates of the edges in the skeleton graph.

        (n_edges x 2 x n_points_per_edge, 3) array of edge coordinates.
        """
        return self._edge_coordinates

    @property
    def edge_indices(self) -> np.ndarray | None:
        """Get the indices of the edges in the skeleton graph.

        (n_edges x 2 x n_points_per_edge,) array of edge indices.
        """
        return self._edge_indices

    def load(self) -> None:
        """Load data."""
        # load the skeleton graph
        if self.file_paths.skeleton_graph:
            log.info(f"Loading skeleton graph from {self.file_paths.skeleton_graph}")
            self._skeleton_graph = SkeletonGraph.from_json_file(
                self.file_paths.skeleton_graph
            )
            self._update_node_coordinates()
            self._update_edge_coordinates()
        else:
            log.info("No skeleton graph loaded.")
            self._skeleton_graph = None

        self.events.data.emit()

    def to_dict(self) -> dict:
        """Convert to json-serializable dictionary."""
        return self._data.to_dict()

    def _update_node_coordinates(self) -> None:
        """Get and store the node coordinates from the skeleton graph.

        todo: make it possible to update without recompute everything
        """
        if self._skeleton_graph is None:
            return None
        self._node_coordinates = self.skeleton_graph.node_coordinates_array

    def _update_edge_coordinates(self) -> None:
        """Get and store the edge spline coordinates from the skeleton graph.

        todo: make it possible to update without recomputing everything
        """
        if self._skeleton_graph is None:
            return None

        edge_coordinates = []
        edge_indices = []
        for edge_index, edge_spline in enumerate(
            self.skeleton_graph.edge_splines.values()
        ):
            # get the edge coordinates
            new_coordinates = line_segment_coordinates_from_spline(edge_spline)
            edge_coordinates.append(new_coordinates)

            # get the edge indices
            n_coordinates = len(new_coordinates)
            edge_indices.append(np.full((n_coordinates,), edge_index))

        self._edge_coordinates = np.concatenate(edge_coordinates)
        self._edge_indices = np.concatenate(edge_indices)

    def _update_paths_load_data(
        self,
        new_data_paths: SkeletonDataPaths,
    ) -> None:
        """Update the file paths and load the new data.

        This is a method intended to be used to generate a magicgui widget
        for the GUI.
        """
        self.file_paths.image = new_data_paths.image
        self.file_paths.segmentation = new_data_paths.segmentation
        self.file_paths.skeleton_graph = new_data_paths.skeleton_graph
        self.load()


class EdgeSelectionManager(EventedModel):
    """Class to manage selection of edge in the viewer.

    Parameters
    ----------
    enabled : bool
        Set to true if the edge selection is enabled.
        The default value is False.
    values : set[tuple[int, int]] | None
        The selected edges.
    """

    enabled: bool
    values: set[tuple[int, int]]


class SelectionManager(EventedModel):
    """Class to manage selection of data in the viewer."""

    edge: EdgeSelectionManager

    def _on_edge_enabled_update(self, event):
        self.edge.enabled = event > 0
