"""Data class for a skeleton graph."""

import json
import logging
import os

import h5py
import networkx as nx
import numpy as np
from splinebox import Spline as SplineboxSpline
from splinebox.spline_curves import _prepared_dict_for_constructor

from skeleplex.graph.constants import (
    DAUGHTER_EDGES_KEY,
    EDGE_COORDINATES_KEY,
    EDGE_SPLINE_KEY,
    GENERATION_KEY,
    LENGTH_KEY,
    NODE_COORDINATE_KEY,
    PARENT_EDGE_KEY,
    SISTER_EDGE_KEY,
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

    # Convert lists back to numpy arrays
    for key, value in json_object.items():
        if isinstance(value, list):
            if key in [NODE_COORDINATE_KEY, EDGE_COORDINATES_KEY]:
                try:
                    json_object[key] = np.array(value)
                except Exception:
                    pass  # If conversion fails, keep it as a list

            if key in [SISTER_EDGE_KEY, DAUGHTER_EDGES_KEY, PARENT_EDGE_KEY]:
                if key == DAUGHTER_EDGES_KEY:
                    json_object[key] = [tuple(edge) for edge in value]
                else:
                    json_object[key] = tuple(value)

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
        If there are no nodes, the function returns 0.
    """
    node_numbers = list(graph.nodes)
    node_numbers.sort()

    if len(node_numbers) == 0:
        return 0

    free_node = node_numbers[0] + 1
    while free_node in node_numbers:
        node_numbers.pop(0)
        free_node = node_numbers[0] + 1
    return free_node


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

    def __init__(
        self,
        graph: nx.Graph,
        origin: int | None = None,
        image_path: str | None = None,
        image_key: str | None = None,
        voxel_size: float | None = None,
    ):
        self.graph = graph
        self.origin = origin
        self.image_path = image_path
        self.image_key = image_key
        self.voxel_size = voxel_size

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
        # if one of the attributes is not None, add it to the dict
        # if not add a placeholder

        object_dict = {
            "graph": graph_dict,
            "origin": self.origin,
            "image_path": self.image_path,
            "image_key": self.image_key,
            "voxel_size": self.voxel_size,
        }

        with open(file_path, "w") as file:
            json.dump(object_dict, file, indent=2, default=skeleton_graph_encoder)

    @classmethod
    def from_json_file(cls, file_path: str):
        """Return a SkeletonGraph from a JSON file."""
        with open(file_path) as file:
            object_dict = json.load(file, object_hook=skeleton_graph_decoder)
        graph = nx.node_link_graph(object_dict["graph"], edges="edges")
        skeleton_object = cls(graph=graph)
        # do only if keys exist
        if "origin" in object_dict:
            skeleton_object.origin = object_dict["origin"]
        if "image_path" in object_dict:
            skeleton_object.image_path = object_dict["image_path"]
        if "image_key" in object_dict:
            skeleton_object.image_key = object_dict["image_key"]
        if "voxel_size" in object_dict:
            skeleton_object.voxel_size = tuple(object_dict["voxel_size"])
        return skeleton_object

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
        """Return a SkeletonGraph from a networkx graph.

        The edges and nodes need to have an attribute with the specified keys
        containing the coordinates of the nodes and edges as an np.ndarray.
        Requires edge coordinates of length greater than 4
        to successfully create a spline.

        Parameters
        ----------
        graph : nx.Graph
            The graph to convert to a SkeletonGraph.
        edge_coordinate_key : str
            The key to use for the edge coordinates.
        node_coordinate_key : str
            The key to use for the node coordinates.
        """
        graph_mod = graph.copy()
        for _, _, attr in graph_mod.edges(data=True):
            attr[EDGE_COORDINATES_KEY] = attr.pop(edge_coordinate_key)
            # add spline
            spline = B3Spline.from_points(attr[EDGE_COORDINATES_KEY])
            attr[EDGE_SPLINE_KEY] = spline
        for _, node_data in graph_mod.nodes(data=True):
            node_data[NODE_COORDINATE_KEY] = node_data.pop(node_coordinate_key)
        return cls(graph=graph_mod)

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
        Stores the origin node as an attribute.

        Parameters
        ----------
        origin : int
            The node to use as the origin node for the directed graph.
            The origin node will have no incoming edges.
        """
        self.graph = make_graph_directed(self.graph, origin)
        self.origin = origin
        return self.graph

    def orient_splines(self) -> nx.DiGraph:
        """Orient the splines in the graph."""
        self.graph = orient_splines(self.graph)
        return self.graph

    def compute_branch_lengths(self) -> dict:
        """Return a dictionary of edge lengths.

        The keys of the dictionary are the edge tuples, the values are arc lengths
        of the fitted splines. Units will be the same as voxel scale.
        """
        edge_lengths = {}
        for u, v, attr in self.graph.edges(data=True):
            edge_lengths[(u, v)] = attr[EDGE_SPLINE_KEY].arc_length

        nx.set_edge_attributes(self.graph, edge_lengths, LENGTH_KEY)
        return edge_lengths

    def sample_volume_slices_from_spline(
        self,
        volume: np.ndarray,
        filepath: str,
        slice_prefix: str,
        slice_spacing: float,
        slice_size: int,
        interpolation_order: int = 3,
        max_generation: int | None = None,
        segmentation: np.ndarray | None = None,
    ):
        """Sample volume slices from the splines in the graph.

        Parameters
        ----------
        volume : np.ndarray
            The volume to sample slices from.
        filepath : str
            The path to save the slices.
        slice_prefix : str
            The prefix to use for the slice filenames.
        slice_spacing : float
            The spacing between slices. Normalized between 0 and 1.
        slice_size : int
            The size of the slices in pixels.
        voxel_size : float
            The size of the pixel in the slice in units of the volume.
        interpolation_order : int
            The order of the interpolation to use for the spline.
            For labels use 0
        max_generation : int
            The maximum generation of the spline to sample.
            If None, all levels are sampled.
        segmentation : np.ndarray | None
            The segmentation to sample slices from.
            If None, only the image is sampled.
        """
        image_voxel_size = self.voxel_size

        origin = self.origin
        if not origin:
            raise ValueError("No origin node provided. Please set origin.")

        if not image_voxel_size:
            logger.warning("No voxel size provided. Assuming pixel size is 1 µm.")
            image_voxel_size = (1, 1, 1)

        if not os.path.exists(filepath):
            os.makedirs(filepath)
            logger.info(f"Created directory {filepath}")

        graph = self.graph.copy()
        generation_dict = nx.get_edge_attributes(graph, GENERATION_KEY)
        spline_dict = nx.get_edge_attributes(graph, EDGE_SPLINE_KEY)

        # Scale the pixel size to the image voxel size
        pixel_size = 1 / image_voxel_size[0]

        for u, v in nx.breadth_first_search.bfs_edges(graph, source=origin):
            if max_generation and (generation_dict[(u, v)] >= max_generation):
                break
            spline = spline_dict[(u, v)]
            positions = np.linspace(0.1, 0.9, np.ceil(1 / slice_spacing).astype(int))
            image_slice = spline.sample_volume_2d(
                volume,
                positions,
                grid_shape=(slice_size, slice_size),
                grid_spacing=(pixel_size, pixel_size),
                sample_interpolation_order=interpolation_order,
                image_voxel_size=image_voxel_size,
            )

            if segmentation is not None:
                segmentation_slice = spline.sample_volume_2d(
                    segmentation,
                    positions,
                    grid_shape=(slice_size, slice_size),
                    grid_spacing=(pixel_size, pixel_size),
                    sample_interpolation_order=0,
                    image_voxel_size=image_voxel_size,
                )

            slice_path = os.path.join(
                filepath,
                f"{slice_prefix}sn_{u}_en_{v}_level_" f"{generation_dict[(u, v)]}.h5",
            )
            with h5py.File(slice_path, "w") as f:
                f.create_dataset("image", data=image_slice)
                if segmentation is not None:
                    f.create_dataset("segmentation", data=segmentation_slice)
            logger.info(f"Saved slice to {slice_path}")
