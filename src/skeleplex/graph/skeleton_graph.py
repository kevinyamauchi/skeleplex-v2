"""Data class for a skeleton graph."""

import json
import logging

import networkx as nx
import numpy as np
from splinebox import Spline as SplineboxSpline
from splinebox.spline_curves import _prepared_dict_for_constructor

from skeleplex.graph.constants import (
    EDGE_COORDINATES_KEY,
    EDGE_SPLINE_KEY,
    NODE_COORDINATE_KEY,
)
from skeleplex.graph.image_to_graph import image_to_graph_skan
from skeleplex.graph.spline import B3Spline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def skeleton_graph_encoder(object_to_encode):
    """JSON encoder for the networkx skeleton graph.

    This function is to be used with the Python json.dump(s) functions
    as the `default` keyword argument.
    """
    if isinstance(object_to_encode, np.ndarray):
        return object_to_encode.tolist()
    elif isinstance(object_to_encode, SplineboxSpline):
        spline_dict = object_to_encode._to_dict(version=2)
        if "__class__" in spline_dict:
            raise ValueError(
                "The Spline object to encode already has a '__class__' key."
            )
        spline_dict.update({"__class__": "splinebox.Spline"})
        return spline_dict
    elif isinstance(object_to_encode, B3Spline):
        return object_to_encode.to_json_dict()
    raise TypeError(f"Object of type {type(object_to_encode)} is not JSON serializable")


def skeleton_graph_decoder(json_object):
    """JSON decoder for the networkx skeleton graph.

    This function is to be used with the Python json.load(s) functions
    as the `object_hook` keyword argument.
    """
    if "__class__" in json_object:
        # all custom classes are identified by the __class__ key
        if json_object["__class__"] == "splinebox.Spline":
            json_object.pop("__class__")
            spline_kwargs = _prepared_dict_for_constructor(json_object)
            return SplineboxSpline(**spline_kwargs)
        if json_object["__class__"] == "skeleplex.B3Spline":
            return B3Spline.from_json_dict(json_object)
    return json_object


def make_graph_directed(graph: nx.Graph, origin: int) -> nx.DiGraph:
    """Return a directed graph from an undirected graph.

    The directed graph has the same nodes and edges as the undirected graph.
    If the graph is fragmented, meaning has multiple unconnected subgraphs,
    the function will choose the node with the highest degree as the origin node
    for each fragment.

    Parameters
    ----------
    graph : nx.Graph
        The undirected graph to convert to a directed graph.
    origin : int
        The node to use as the origin node for the directed graph.
        The origin node will have no incoming edges.
    """
    if isinstance(graph, nx.DiGraph):
        logger.info("The input graph is already a directed graph.")
        return graph
    if len(list(nx.connected_components(graph))) > 1:
        logger.warning("""
        The input graph is not connected.
        The unconnected components might lose edges
        """)
        origin_part = nx.node_connected_component(graph, origin)
        fragments = graph.subgraph(set(graph.nodes()) - origin_part)
        graph = graph.subgraph(origin_part)
    else:
        fragments = None

    di_graph = nx.DiGraph(graph)
    di_graph.remove_edges_from(di_graph.edges - nx.bfs_edges(di_graph, origin))

    if fragments:
        # choose a node with the highest degree as the origin node
        # Do this for each fragment
        for fragment in nx.connected_components(fragments):
            fragment_subgraph = fragments.subgraph(fragment)
            """Choose a origin of the fragment with the highest degree.
            This is arbitrary but finding a better node
            without knowledge were the network broke is hard"""
            origin = max(fragment_subgraph.degree, key=lambda x: x[1])[0]
            di_fragment = nx.DiGraph(fragment_subgraph)
            di_fragment.remove_edges_from(
                di_fragment.edges - nx.bfs_edges(di_fragment, origin)
            )
            di_graph.add_edges_from(di_fragment.edges(data=True))
            di_graph.add_nodes_from(di_fragment.nodes(data=True))

    return di_graph


def get_next_node_key(graph: nx.Graph) -> int:
    """Return the next available node key in the graph.

    This function assumes the graph node keys are integers.

    Parameters
    ----------
    graph : nx.Graph
        The graph to get the next node key from.

    Returns
    -------
    int
        The next available node key.
        This is the maximum node key + 1.
        If there are no nodes, the function returns 0.
    """
    return int(np.max(graph.nodes)) + 1 if graph.nodes else 0


def orient_splines(graph: nx.DiGraph) -> nx.DiGraph:
    """Checks if the splines are oriented correctly.

    If the beginning of the spline is closer to the end node than the start node,
    it gets flipped.
    Also checks if the edge coordinates are aligend with the spline.
    This only checks, if the splines are correctly connected to the nodes,
    not the order in the Graph. Best used on a directed graph.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to orient the splines in.

    Returns
    -------
    nx.DiGraph
        The graph with the splines oriented correctly.

    """
    edge_spline_dict = {}
    edge_coordinates_dict = {}

    for u, v, attr in graph.edges(data=True):
        spline = attr[EDGE_SPLINE_KEY]
        u_coord = graph.nodes[u][NODE_COORDINATE_KEY]
        spline_coordinates = spline.eval(np.array([0, 1]))
        # check if spline evaluation is closer to the start or end node
        if np.linalg.norm(u_coord - spline_coordinates[0]) > np.linalg.norm(
            u_coord - spline_coordinates[-1]
        ):
            logger.info(f"Flipped spline of edge ({u,v}).")
            edge_coordinates = attr[EDGE_COORDINATES_KEY]
            # check if path is inverse to spline
            if np.linalg.norm(
                edge_coordinates[0] - spline_coordinates[0]
            ) > np.linalg.norm(edge_coordinates[-1] - spline_coordinates[-1]):
                edge_coordinates = edge_coordinates[::-1]

            flipped_spline, flipped_cords = spline.flip_spline(edge_coordinates)
            edge_spline_dict[(u, v)] = flipped_spline
            edge_coordinates_dict[(u, v)] = flipped_cords

    nx.set_edge_attributes(graph, edge_spline_dict, EDGE_SPLINE_KEY)
    nx.set_edge_attributes(graph, edge_coordinates_dict, EDGE_COORDINATES_KEY)

    return graph

class SkeletonGraph:
    """Data class for a skeleton graph.

    Parameters
    ----------
    graph : nx.Graph
        The skeleton graph.
    """

    _backend = "networkx"

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    @property
    def backend(self) -> str:
        """Return the backend used to store the graph."""
        return self._backend

    @property
    def nodes(self):
        """Return a list of nodes."""
        return self.graph.nodes()

    @property
    def node_coordinates(self) -> dict:
        """Return a dictionary of node coordinates."""
        node_coordinates = {}
        for node, node_data in self.graph.nodes(data=True):
            node_coordinates[node] = node_data[NODE_COORDINATE_KEY]
        return node_coordinates

    @property
    def node_coordinates_array(self) -> np.ndarray:
        """Return a numpy array of node coordinates.

        The array is of shape (n_nodes, n_dimensions).
        The order of the nodes is the same as the order of the nodes attribute.
        """
        return np.array(
            [
                node_data[NODE_COORDINATE_KEY]
                for _, node_data in self.graph.nodes(data=True)
            ]
        )

    @property
    def edges(self):
        """Return a list of edges."""
        return self.graph.edges()

    @property
    def edge_splines(self) -> dict:
        """Return a list of edge splines."""
        edge_splines = {}
        for edge_start, edge_end, edge_data in self.graph.edges(data=True):
            edge_splines[(edge_start, edge_end)] = edge_data[EDGE_SPLINE_KEY]
        return edge_splines

    def to_json_file(self, file_path: str):
        """Return a JSON representation of the graph."""
        graph_dict = nx.node_link_data(self.graph, edges="edges")
        object_dict = {"graph": graph_dict}

        with open(file_path, "w") as file:
            json.dump(object_dict, file, indent=2, default=skeleton_graph_encoder)

    @classmethod
    def from_json_file(cls, file_path: str):
        """Return a SkeletonGraph from a JSON file."""
        with open(file_path) as file:
            object_dict = json.load(file, object_hook=skeleton_graph_decoder)
        graph = nx.node_link_graph(object_dict["graph"], edges="edges")
        return cls(graph=graph)

    @classmethod
    def from_skeleton_image(
        cls, skeleton_image: np.ndarray, max_spline_knots: int = 10
    ) -> "SkeletonGraph":
        """Return a SkeletonGraph from a skeleton image.

        Parameters
        ----------
        skeleton_image : np.ndarray
            The skeleton image to convert to a graph.
        max_spline_knots : int
            The maximum number of knots to use for the spline fit to the branch path.
            If the number of data points in the branch is less than this number,
            the spline will use n_data_points - 1 knots.
            See the splinebox Spline class docs for more information.
        """
        graph = image_to_graph_skan(
            skeleton_image=skeleton_image, max_spline_knots=max_spline_knots
        )
        return cls(graph=graph)

    @classmethod
    def from_graph(
        cls, graph, edge_coordinate_key, node_coordinate_key
    ) -> "SkeletonGraph":
        """Return a SkeletonGraph from a graph.

        Parameters
        ----------
        graph : nx.Graph
            The graph to convert to a SkeletonGraph.
        edge_coordinate_key : str
            The key to use for the edge coordinates.
        node_coordinate_key : str
            The key to use for the node coordinates.
        """
        for _, _, attr in graph.edges(data=True):
            attr[EDGE_COORDINATES_KEY] = attr.pop(edge_coordinate_key)
            # add spline
            spline = B3Spline.from_points(attr[EDGE_COORDINATES_KEY])
            attr[EDGE_SPLINE_KEY] = spline
        for _, node_data in graph.nodes(data=True):
            node_data[NODE_COORDINATE_KEY] = node_data.pop(node_coordinate_key)
        return cls(graph=graph)

    def __eq__(self, other: "SkeletonGraph"):
        """Check if two SkeletonGraph objects are equal."""
        if set(self.nodes) != set(other.nodes):
            # check if the nodes are the same
            return False
        elif set(self.edges) != set(other.edges):
            # check if the edges are the same
            return False
        else:
            return True

    def to_directed(self, origin: int) -> nx.DiGraph:
        """Return a directed graph from the skeleton graph.

        The directed graph has the same nodes and edges as the skeleton graph.

        Parameters
        ----------
        origin : int
            The node to use as the origin node for the directed graph.
            The origin node will have no incoming edges.
        """
        self.graph = make_graph_directed(self.graph, origin)
        return self.graph

    def orient_splines(self) -> nx.DiGraph:
        """Orient the splines in the graph."""
        self.graph = orient_splines(self.graph)
        return self.graph

    def compute_edge_lengths(self) -> dict:
        """Return a dictionary of edge lengths."""
        edge_lengths = {}
        for u, v, attr in self.graph.edges(data=True):
            edge_lengths[(u, v)] = attr[EDGE_SPLINE_KEY].arc_length()

        nx.set_edge_attributes(self.graph, edge_lengths, "length")
        return edge_lengths