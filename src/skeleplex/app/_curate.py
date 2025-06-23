import numpy as np
import sys
from collections import deque
from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Any
from skeleplex.graph.constants import (
    EDGE_SPLINE_KEY,
    NODE_COORDINATE_KEY
)
from magicgui import magicgui

from skeleplex.graph.modify_graph import (
    connect_without_merging,
    delete_edge,
    merge_nodes,
    split_edge
)

if TYPE_CHECKING:
    # prevent circular import
    from skeleplex.app._data import DataManager

import ast


def edge_string_to_key(edge_string: str) -> set[tuple[int, ...]]:
    """Parse a string representation of a set of tuples back into a Python set.

    This function safely converts string representations of sets containing tuples
    back into their original Python data structure. It handles the case where
    the string was created using str() on a set of tuples.

    Parameters
    ----------
    edge_string : str
        String representation of a set of tuples, typically created by
        calling str() on a set object containing tuples.

    Returns
    -------
    set[tuple[int, ...]]
        A set containing tuples parsed from the input string.

    Raises
    ------
    ValueError
        If the string cannot be safely parsed as a set of tuples.
    SyntaxError
        If the string contains invalid Python syntax.
    """
    try:
        # parse the string to convert it back to a set of tuples
        parsed_result = ast.literal_eval(edge_string)

        # Verify that the result is a set
        if not isinstance(parsed_result, set):
            raise ValueError(f"Expected a set, but got {type(parsed_result).__name__}")

        # Verify that all elements are tuples
        for element in parsed_result:
            if not isinstance(element, tuple):
                raise ValueError("Expected all elements must be tuples")

        return parsed_result

    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Could not parse edge string '{edge_string}': {e}") from e


def node_string_to_node_keys(node_string: str) -> set[int]:
    """Parse a string representation of a set of integers back into a Python set.

    This function safely converts string representations of sets containing integers
    back into their original Python data structure. It handles the case where
    the string was created using str() on a set of integers.

    Parameters
    ----------
    node_string : str
        String representation of a set of integers, typically created by
        calling str() on a set object containing integers.

    Returns
    -------
    set[int]
        A set containing integers parsed from the input string.

    Raises
    ------
    ValueError
        If the string cannot be safely parsed as a set of integers,
        or if the string exceeds the maximum allowed length.
    SyntaxError
        If the string contains invalid Python syntax.
    """
    try:
        parsed_result = ast.literal_eval(node_string)

        # Verify that the result is a set
        if not isinstance(parsed_result, set):
            raise ValueError(f"Expected a set, but got {type(parsed_result).__name__}")

        # Verify that all elements are integers
        for element in parsed_result:
            if not isinstance(element, int):
                raise ValueError("All elements must be integers")

        return parsed_result

    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Could not parse node string '{node_string}': {e}") from e
    

class LIFOBuffer:
    """A last-in-first-out buffer with a maximum size.

    This buffer automatically removes the oldest items when the maximum
    size is exceeded, maintaining LIFO ordering for retrieval.

    Parameters
    ----------
    max_size : int
        The maximum number of items the buffer can hold.
        Must be greater than 0.

    Raises
    ------
    ValueError
        If max_size is not a positive integer.
    """

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be greater than 0")

        self._buffer = deque(maxlen=max_size)
        self._max_size = max_size

    def push(self, item: Any) -> None:
        """Add an item to the buffer.

        If the buffer is at maximum capacity, the oldest item
        will be automatically removed.

        Parameters
        ----------
        item : Any
            The item to add to the buffer.
        """
        self._buffer.append(item)

    def pop(self) -> Any:
        """Remove and return the most recently added item.

        Returns
        -------
        Any
            The most recently added item.

        Raises
        ------
        IndexError
            If the buffer is empty.
        """
        if not self._buffer:
            raise IndexError("pop from empty buffer")
        return self._buffer.pop()

    @property
    def max_size(self) -> int:
        """Get the maximum capacity of the buffer.

        Returns
        -------
        int
            The maximum number of items the buffer can hold.
        """
        return self._max_size

    def clear(self) -> None:
        """Remove all items from the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        """Return the number of items in the buffer."""
        return len(self._buffer)


class CurationManager:
    def __init__(
        self,
        data_manager: "DataManager",
    ):
        self._data = data_manager

        # buffers for undo and redo operations
        self._undo_buffer = LIFOBuffer(max_size=10)
        self._redo_buffer = LIFOBuffer(max_size=10)

    def delete_edge(
        self,
        edge: Annotated[set[tuple[int, int]] | str, {"widget_type": "LineEdit"}],
        redraw: bool = True,
    ) -> None:
        """Delete an edge from the skeleton graph.

        Parameters
        ----------
        edge : tuple[int, int] | None
            The edge to delete, represented as a tuple of node IDs.
            If None, no action is taken.
        redraw : bool
            Flag set to True to redraw the graph after deletion.
            Defaults value is True.
        """
        if len(edge) == 0:
            # if no edge is selected, do nothing
            return

        # parse the edge if it is a string
        edges = edge_string_to_key(edge) if isinstance(edge, str) else edge

        # store the previous state in the undo buffer
        self._undo_buffer.push(deepcopy(self._data.skeleton_graph))

        # delete the edge from the skeleton graph
        for edge in edges:
            delete_edge(skeleton_graph=self._data.skeleton_graph, edge=edge)

        if redraw:
            # redraw the graph
            self._update_and_request_redraw()

    def connect_without_merging(
        self,
        start_node: Annotated[int, {"widget_type": "LineEdit"}],
        end_node: Annotated[int, {"widget_type": "LineEdit"}],
        redraw: bool = True,
    ) -> None:
        """Connect two nodes in the skeleton graph without merging them.

        This method connects two nodes in the skeleton graph by creating an edge
        between them. If the nodes are already connected, no action is taken.
        The connection does not merge the nodes, preserving their individual identities.

        Parameters
        ----------
        start_node : int
            The ID of the first node to connect.
        end_node : int
            The ID of the second node to connect.
        redraw : bool
            Flag set to True to redraw the graph after connecting.
            Defaults value is True.
        """
        start_node = node_string_to_node_keys(start_node)
        end_node = node_string_to_node_keys(end_node)
        start_node = next(iter(start_node), None)
        end_node = next(iter(end_node), None)
        if start_node is None or end_node is None:
            # if either node is None, do nothing
            return
        if start_node == end_node:
            # if both nodes are the same, do nothing
            return

        # store the previous state in the undo buffer
        self._undo_buffer.push(deepcopy(self._data.skeleton_graph))

        # connect the nodes without merging
        connect_without_merging(
            skeleton_graph=self._data.skeleton_graph, node1=start_node, node2=end_node
        )

        if redraw:
            # redraw the graph
            self._update_and_request_redraw()

    def connect_with_merging(
        self,
        node_to_keep: Annotated[int, {"widget_type": "LineEdit"}],
        node_to_merge: Annotated[int, {"widget_type": "LineEdit"}],
        redraw: bool = True,
    ) -> None:
        """Connect two nodes in the skeleton graph by merging them.

        This method connects two nodes in the skeleton graph by merging one node
        into another. The node to keep will retain its identity, while the other
        node will be merged into it, effectively removing it from the graph.

        Parameters
        ----------
        node_to_keep : int
            The ID of the nod e to keep after merging.
        node_to_merge : int
            The ID of the node to merge into the first node.
        redraw : bool
            Flag set to True to redraw the graph after connecting.
            Defaults value is True.
        """
        node_to_keep = node_string_to_node_keys(node_to_keep)
        node_to_merge = node_string_to_node_keys(node_to_merge)
        node_to_keep = next(iter(node_to_keep), None)
        node_to_merge = next(iter(node_to_merge), None)

        if node_to_keep == node_to_merge:
            # if both nodes are the same, do nothing
            return

        # store the previous state in the undo buffer
        self._undo_buffer.push(deepcopy(self._data.skeleton_graph))

        # merge the nodes in the skeleton graph
        merge_nodes(
            skeleton_graph=self._data.skeleton_graph,
            node_to_keep=node_to_keep,
            node_to_merge=node_to_merge,
        )

        # connect the nodes with merging
        if redraw:
            # redraw the graph
            self._update_and_request_redraw()
    
    def undo(self, redraw: bool = True) -> None:
        """Undo the last action performed on the skeleton graph.

        This method restores the skeleton graph to its previous state
        using the undo buffer. If there are no actions to undo, it does nothing.

        Parameters
        ----------
        redraw : bool
            Flag set to True to redraw the graph after undoing.
            Defaults value is True.
        """
        if len(self._undo_buffer) == 0:
            # if there are no actions to undo, do nothing
            return

        # store the current state in the redo buffer
        self._redo_buffer.push(deepcopy(self._data.skeleton_graph))

        # restore the previous state from the undo buffer
        previous_state = self._undo_buffer.pop()
        self._data._skeleton_graph = previous_state

        if redraw:
            # redraw the graph
            self._update_and_request_redraw()

    def redo(self, redraw: bool = True) -> None:
        """Redo the last undone action on the skeleton graph.

        This method restores the skeleton graph to the next state in the undo buffer.
        If there are no actions to redo, it does nothing.

        Parameters
        ----------
        redraw : bool
            Flag set to True to redraw the graph after redoing.
            Defaults value is True.
        """
        if len(self._redo_buffer) == 0:
            # if there are no actions to redo, do nothing
            return

        # store the current state in the undo buffer
        self._undo_buffer.push(deepcopy(self._data.skeleton_graph))

        # restore the next state from the redo buffer
        next_state = self._redo_buffer.pop()
        self._data._skeleton_graph = next_state

        if redraw:
            # redraw the graph
            self._update_and_request_redraw()


    def render_around_node(
            self,
            node_id = int,
            bounding_box_width: int = 100,
        ):
        """Render a bounding box around the specified node.

        Parameters
        ----------
        node_id : int
            The ID of the node to render around.
        bounding_box_width : int
            The width of the bounding box to render around the node.
            Default is 100.
        """
        # get the coordinate of the node
        node_id = node_string_to_node_keys(node_id)
        node_id = next(iter(node_id), None)
        graph_object = self._data.skeleton_graph.graph
        node_coordinate = graph_object.nodes[node_id][NODE_COORDINATE_KEY]

        # get the minimum and maximum coordinates for the bounding box
        half_width = bounding_box_width / 2
        min_coordinate = node_coordinate - half_width
        max_coordinate = node_coordinate + half_width

        # set the bounding box in the viewer
        self._data.view.bounding_box._min_coordinate = min_coordinate
        self._data.view.bounding_box._max_coordinate = max_coordinate

        # set the render mode to bounding box
        self._data.view.mode = "bounding_box"


    def _update_and_request_redraw(
        self, clear_edge_selection: bool = True, clear_node_selection: bool = True
    ) -> None:
        """Update the rendered graph data and request a redraw."""
        # Clear the selection if specified
        if clear_edge_selection:
            self._data.selection.edge.values = set()
        if clear_node_selection:
            self._data.selection.node.values = set()

        # Update the skeleton graph data
        self._data._update_edge_coordinates()
        self._data._update_edge_colors()
        self._data._update_node_coordinates()

        # Update the data view
        self._data.view.update()


def make_split_edge_widget(viewer):
    """Create a widget for splitting edges in the skeleton graph."""
    @magicgui(
        edge_to_split_ID={"widget_type": "LineEdit"},
        split_pos={"widget_type": "FloatSlider", "min": 0.0,
                   "max": 1.0,
                   "step": 0.01,
                   "value": 0.5},
    )
    def split_edge_widget(edge_to_split_ID: int, split_pos: float = 0.5):
        """Widget to split an edge in the skeleton graph.

        Parameters
        ----------
        edge_to_split_ID : str
            The ID of the edge to split, represented as a string.
            This should be a string representation of a set of tuples,
            e.g. "{(1, 2), (2, 3)}".
        split_pos : float
            The position to split the edge at, between 0 and 1.
            Default value is 0.5, which means the edge will be split in the middle
            of its length.
        """
        edge_key = next(iter(edge_string_to_key(edge_to_split_ID)))
        viewer.curate._undo_buffer.push(deepcopy(viewer.curate._data.skeleton_graph))
        split_edge(viewer.curate._data.skeleton_graph, edge_key, split_pos)
        viewer.curate._update_and_request_redraw()

    def preview_split():
        """Preview the split edge operation.

        This function is connected to the split_pos widget to update the preview
        of the split edge in the viewer.
        It calculates the position of the split point based on the current
        split_pos value and the spline of the edge to be split.
        The calculated point position is then set in the point_store and made visible.
        """
        edge_key = next(iter(edge_string_to_key(
            split_edge_widget.edge_to_split_ID.value)
            ))
        spline = viewer.data.skeleton_graph.graph.edges[edge_key][EDGE_SPLINE_KEY]
        point_pos = spline.eval(split_edge_widget.split_pos.value)

        split_edge_widget.point_store.coordinates = np.array([point_pos],
                                                             dtype=np.float32)
        split_edge_widget.point_visual.appearance.visible = True
        viewer._viewer._backend.reslice_all()

    split_edge_widget.split_pos.changed.connect(preview_split)
    split_edge_widget.point_visual, split_edge_widget.point_store = viewer.add_points()
    split_edge_widget.point_visual.appearance.visible = False

    return split_edge_widget