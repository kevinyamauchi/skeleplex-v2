from collections import deque
from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Any

from skeleplex.graph.modify_graph import delete_edge

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
        self._data._update_node_coordinates()

        # Update the data view
        self._data.view.update()
