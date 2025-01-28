"""Data class for a skeleton graph."""

import json

import networkx as nx
import numpy as np
from splinebox import Spline as SplineboxSpline
from splinebox.spline_curves import _prepared_dict_for_constructor

from skeleplex.graph.constants import EDGE_SPLINE_KEY, NODE_COORDINATE_KEY
from skeleplex.graph.image_to_graph import image_to_graph_skan
from skeleplex.graph.spline import B3Spline


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
