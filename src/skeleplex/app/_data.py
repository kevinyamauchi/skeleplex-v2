"""Module for handling data in the SkelePlex application."""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
from cellier.types import MouseButton, MouseCallbackData, MouseEventType, MouseModifiers
from psygnal import EventedModel, Signal, SignalGroup
from pydantic.types import FilePath

from skeleplex.graph import SkeletonGraph
from skeleplex.graph.constants import NODE_COORDINATE_KEY
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
        self._edge_keys: np.ndarray | None = None
        self._highlighted_edge_keys: np.ndarray | None = None
        self._node_coordinates: np.ndarray | None = None
        self._node_keys: np.ndarray | None = None
        self._highlighted_node_keys: np.ndarray | None = None

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
    def node_keys(self) -> np.ndarray | None:
        """Get the keys of the nodes in the rendered graph.

        (n_nodes,) array of node keys. These are index-matched with
        the node_coordinates array.
        """
        return self._node_keys

    @property
    def highlighted_node_keys(self) -> np.ndarray | None:
        """Get the indices of the highlighted nodes in the rendered graph.

        (n_highlighted_nodes,) array of node indices.
        """
        return self._highlighted_node_keys

    @property
    def edge_coordinates(self) -> np.ndarray | None:
        """Get the coordinates of the current view of the edges in the rendered graph.

        (n_edges x 2 x n_points_per_edge, 3) array of edge coordinates.
        """
        return self._edge_coordinates

    @property
    def edge_indices(self) -> np.ndarray | None:
        """Get the indices of the current view of the edges in the rendered graph.

        (n_edges x 2 x n_points_per_edge,) array of edge indices.
        """
        return self._edge_indices

    @property
    def edge_keys(self) -> np.ndarray | None:
        """Get the keys of the edge for each edge coordinate in the rendered graph.

        (n_edges x 2 x n_points_per_edge,) array of edge keys.
        """
        return self._edge_keys

    @property
    def highlighted_edge_keys(self) -> np.ndarray | None:
        """Get keys of the highlighted edges for each coordinate in the rendered graph.

        (n_edges x 2 x n_points_per_edge,) array of highlighted edge keys.
        """
        return self._highlighted_edge_keys

    def _get_view_all(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the view for all data.

        Returns
        -------
        node_coordinates : np.ndarray
            (n_nodes,3) array of the coordinates of
            the rendered nodes in the skeleton graph.
        node_keys : np.ndarray
            (n_nodes) array of the keys of the nodes in the skeleton graph.
        edge_coordinates : np.ndarray
            (n_edges x 2 x n_points_per_edge, 3) array of the coordinates of
            the rendered edges in the skeleton graph.
        edge_indices : np.ndarray
            The indices of the edges in the skeleton graph.
        edge_keys : np.ndarray
            The keys of the edge for each edge coordinate in the skeleton graph.

        """
        return (
            self._data_manager.node_coordinates,
            self._data_manager.node_keys,
            self._data_manager.edge_coordinates,
            self._data_manager.edge_indices,
            self._data_manager.edge_keys,
        )

    def update(self) -> None:
        """Update the data for the currently specified view.

        This updates the edge coordinates, edge indices, and node indices.
        """
        if self._data_manager.skeleton_graph is None:
            # if the data isn't loaded, nothing to update
            return
        if self._mode == ViewMode.ALL:
            (
                self._node_coordinates,
                self._node_keys,
                self._edge_coordinates,
                self._edge_indices,
                self._edge_keys,
            ) = self._get_view_all()
            self._highlighted_edge_keys = np.empty((0, 2))
        else:
            raise NotImplementedError(f"View mode {self._mode} not implemented.")

        # Emit signal that the view data has been updated
        self.events.data.emit()


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


class NodeSelectionManager(EventedModel):
    """Class to manage selection of nodes in the viewer.

    Parameters
    ----------
    enabled : bool
        Set to true if the edge selection is enabled.
        The default value is False.
    values : set[int] | None
        The selected nodes.
    """

    enabled: bool
    values: set[int]


@dataclass(frozen=True)
class EdgeSelectionPasteRequest:
    """Selected edges to paste.

    This is used for passing selected edges to the paste operation.
    For example, when pasting edges from the selection to a
    GUI widget.

    Parameters
    ----------
    edge_key : set[tuple[int, int]]
        The keys of the selected edges.
    """

    edge_keys: set[tuple[int, int]]


@dataclass(frozen=True)
class NodeSelectionPasteRequest:
    """Selected nodes to paste.

    This is used for passing selected nodes to the paste operation.
    For example, when pasting nodes from the selection to a
    GUI widget.

    Parameters
    ----------
    node_keys : set[int]
        The keys of the selected nodes.
    """

    node_keys: set[int]


class SelectionManager(EventedModel):
    """Class to manage selection of data in the viewer."""

    edge: EdgeSelectionManager
    node: NodeSelectionManager

    def _make_edge_selection_paste_request(self):
        """Create a paste request for the selected edges.

        This emits the request as a SelectionManager.events.edge signal.
        """
        request = EdgeSelectionPasteRequest(edge_keys=self.edge.values)
        self.events.edge.emit(request)

    def _make_node_selection_paste_request(self):
        """Create a paste request for the selected nodes.

        This emits the request as a SelectionManager.events.node signal.
        """
        request = NodeSelectionPasteRequest(node_keys=self.node.values)
        self.events.node.emit(request)

    def _on_edge_enabled_update(self, event):
        """Callback when the UI updates the edge selection enabled state."""
        self.edge.enabled = event > 0

    def _on_node_enabled_update(self, event):
        """Callback when the UI updates the node selection enabled state."""
        self.node.enabled = event > 0


class DataEvents(SignalGroup):
    """Events for the DataManager class."""

    data = Signal()


class DataManager:
    """A class to manage data.

    Parameters
    ----------
    file_paths : SkeletonDataPaths
        The paths to the data files.
    selection : SelectionManager | None
        The selection manager.

    Attributes
    ----------
    events : DataEvents
        The events for the DataManager class.
    node_coordinates : np.ndarray | None
        The coordinates of the nodes in the skeleton graph.
        This is None when the skeleton hasn't been loaded.
    node_keys : np.ndarray | None
        The keys for the nodes in the skeleton graph.
        This is index-matched with the node_coordinates array.
        This is None when the skeleton hasn't been loaded.
    edge_coordinates : np.ndarray | None
        The coordinates for rendering the edges in the skeleton graph
        as line segments. This is None when the skeleton hasn't been loaded.
    edge_indices : np.ndarray | None
        The indices for the edges in the skeleton graph.
        This is None when the skeleton hasn't been loaded.
    edge_keys : np.ndarray | None
        The keys for edges of each edge coordinate in the skeleton graph.
        This is None when the skeleton hasn't been loaded.
    """

    events = DataEvents()

    def __init__(
        self,
        file_paths: SkeletonDataPaths,
        selection: SelectionManager | None = None,
        load_data: bool = True,
    ) -> None:
        self._file_paths = file_paths

        self._view = DataView(data_manager=self, mode=ViewMode.ALL)

        # make the selection model
        if selection is None:
            selection = SelectionManager(
                edge=EdgeSelectionManager(enabled=False, values=set()),
                node=NodeSelectionManager(enabled=False, values=set()),
            )
        self._selection = selection

        # initialize the data
        self._skeleton_graph: SkeletonGraph | None = None
        self._node_coordinates: np.ndarray | None = None
        self._edge_coordinates: np.ndarray | None = None
        self._edge_indices: np.ndarray | None = None
        self._edge_keys: np.ndarray | None = None

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
    def selection(self) -> SelectionManager:
        """Get the current data selection."""
        return self._selection

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
    def node_keys(self) -> np.ndarray | None:
        """Get the keys of the nodes in the skeleton graph.

        (n_nodes,) array of node keys. These are index-matched with
        the node_coordinates array.
        """
        return self._node_keys

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

    @property
    def edge_keys(self) -> np.ndarray | None:
        """Get the keys of the edge for each edge coordinate in the skeleton graph.

        (n_edges x 2 x n_points_per_edge,) array of edge keys.
        """
        return self._edge_keys

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
            # don't do anything if the skeleton graph is not loaded
            return

        node_coordinates = []
        node_keys = []
        for key, node_data in self.skeleton_graph.graph.nodes(data=True):
            node_keys.append(key)
            node_coordinates.append(node_data[NODE_COORDINATE_KEY])
        self._node_coordinates = np.array(node_coordinates, dtype=np.float32)
        self._node_keys = np.array(node_keys, dtype=np.int32)

    def _update_edge_coordinates(self) -> None:
        """Get and store the edge spline coordinates from the skeleton graph.

        todo: make it possible to update without recomputing everything
        """
        if self._skeleton_graph is None:
            return None

        edge_coordinates = []
        edge_indices = []
        edge_keys = []
        for edge_index, (edge_key, edge_spline) in enumerate(
            self.skeleton_graph.edge_splines.items()
        ):
            # get the edge coordinates
            new_coordinates = line_segment_coordinates_from_spline(edge_spline)
            edge_coordinates.append(new_coordinates)

            # get the edge indices
            n_coordinates = len(new_coordinates)
            edge_indices.append(np.full((n_coordinates,), edge_index))

            # get the edge keys
            edge_keys.append(np.tile(edge_key, (n_coordinates, 1)))

        self._edge_coordinates = np.concatenate(edge_coordinates)
        self._edge_indices = np.concatenate(edge_indices)
        self._edge_keys = np.concatenate(edge_keys)

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

    def _on_edge_selection_click(
        self, event: MouseCallbackData, click_source: str = "data"
    ):
        """Callback for the edge picking event from the renderer.

        Parameters
        ----------
        event : pygfx.PointerEvent
            The event data from the pygfx click event.
        click_source : str
            The source of the click event. Should be either "data" (the main visual)
            or "highlight" (the highlight visual).
        """
        if (
            (MouseModifiers.CTRL not in event.modifiers)
            or (event.button != MouseButton.LEFT)
            or (event.type != MouseEventType.PRESS)
        ):
            # only pick with control + LMB
            return

        # get the index of the vertex the click was close to
        vertex_index = event.pick_info["vertex_index"]

        # get the edge key from the vertex index
        if click_source == "data":
            edge_key_numpy = self.view.edge_keys[vertex_index]
            edge_key = (int(edge_key_numpy[0]), int(edge_key_numpy[1]))
        elif click_source == "highlight":
            edge_key = tuple(self.view.highlighted_edge_keys[vertex_index])
        else:
            raise ValueError(f"Unknown click source: {click_source}")

        if edge_key in self.selection.edge.values:
            # if the edge is already selected, deselect it.
            self.selection.edge.values.remove(edge_key)
        else:
            # if the edge is not selected, select it.
            if MouseModifiers.SHIFT not in event.modifiers:
                # if shift is not pressed, clear the selection
                self.selection.edge.values.clear()
            self.selection.edge.values.add(edge_key)
        self.selection.edge.events.values.emit(self.selection.edge.values)

    def _on_node_selection_click(
        self, event: MouseCallbackData, click_source: str = "data"
    ):
        """Callback for the node picking event from the renderer.

        Parameters
        ----------
        event : pygfx.PointerEvent
            The event data from the pygfx click event.
        click_source : str
            The source of the click event. Should be either "data" (the main visual)
            or "highlight" (the highlight visual).
        """
        if (
            (MouseModifiers.CTRL not in event.modifiers)
            or (event.button != MouseButton.LEFT)
            or (event.type != MouseEventType.PRESS)
        ):
            # only pick with control + LMB
            return

        # get the index of the vertex the click was close to
        vertex_index = event.pick_info["vertex_index"]

        # get the edge key from the vertex index
        if click_source == "data":
            node_key = int(self.view.node_keys[vertex_index])
        elif click_source == "highlight":
            node_key = int(self.view.highlighted_node_keys[vertex_index])
        else:
            raise ValueError(f"Unknown click source: {click_source}")

        if node_key in self.selection.node.values:
            # if the edge is already selected, deselect it.
            self.selection.node.values.remove(node_key)
        else:
            # if the edge is not selected, select it.
            if MouseModifiers.SHIFT not in event.modifiers:
                # if shift is not pressed, clear the selection
                self.selection.node.values.clear()
            self.selection.node.values.add(node_key)
        self.selection.node.events.values.emit(self.selection.node.values)
