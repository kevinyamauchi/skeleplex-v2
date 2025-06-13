from skeleplex.app import DataManager
from skeleplex.graph.modify_graph import delete_edge


class CurationManager:
    def __init__(
        self,
        data_manager: DataManager,
    ):
        self._data = data_manager

    def delete_edge(self, edge: tuple[int, int] | None, redraw: bool = True) -> None:
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
        if edge is None:
            # if no edge is selected, do nothing
            return

        # delete the edge from the skeleton graph
        delete_edge(skeleton_graph=self._data.skeleton_graph, edge=edge)

        if redraw:
            # redraw the graph
            self._update_and_request_redraw()

    def _update_and_request_redraw(self):
        """Update the rendered graph data and request a redraw."""
        # Update the skeleton graph data
        self._data._update_edge_coordinates()
        self._data._update_node_coordinates()

        # Request a redraw of the skeleton graph
        self._data.events.data.emit()
