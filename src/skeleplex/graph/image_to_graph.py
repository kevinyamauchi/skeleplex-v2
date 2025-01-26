"""Utilities to convert a skeleton image to a graph."""

import networkx as nx
import numpy as np
import splinebox
from skan.csr import Skeleton as SkanSkeleton
from skan.csr import summarize

from skeleplex.graph.constants import NODE_COORDINATE_KEY


def image_to_graph_skan(
    skeleton_image: np.ndarray, max_spline_knots: int = 10
) -> nx.MultiGraph:
    """Convert a skeleton image to a graph using skan.

    Parameters
    ----------
    skeleton_image : np.ndarray
        The image to convert to a skeleton graph.
        The image should be a binary image and already skeletonized.
    max_spline_knots : int
        The maximum number of knots to use for the spline fit to the branch path.
        If the number of data points in the branch is less than this number,
        the spline will use n_data_points - 1 knots.
        See the splinebox Spline class docs for more information.
    """
    # make the skeleton
    skeleton = SkanSkeleton(skeleton_image=skeleton_image)
    summary_table = summarize(skeleton, separator="_")

    # get all of the nodes
    # this might be slow - may need to speed up
    # source_nodes = set(summary_table["node_id_src"])
    # destination_nodes = set(summary_table["node_id_dst"])
    # all_nodes = source_nodes.union(destination_nodes)

    skeleton_graph = nx.MultiGraph()
    for row in summary_table.itertuples(name="Edge"):
        # Iterate over the rows in the table.
        # Each row is an edge in the graph
        index = row.Index
        i = row.node_id_src
        j = row.node_id_dst

        # fit a spline to the path
        # todo: factor our to spline module
        # todo: reconsider how the number of knots is set
        spline_path = skeleton.path_coordinates(index)
        n_points = len(spline_path)
        if n_points <= max_spline_knots:
            n_spline_knots = n_points - 1
        else:
            n_spline_knots = max_spline_knots
        basis_function = splinebox.B3()
        spline = splinebox.Spline(n_spline_knots, basis_function)
        spline.fit(spline_path)

        # Nodes are added if they don't exist so only need to add edges
        skeleton_graph.add_edge(
            i,
            j,
            **{"path": spline_path, "spline": spline},
        )

    # add the node coordinates
    new_node_data = {}
    for node_index, node_data in skeleton_graph.nodes(data=True):
        node_data[NODE_COORDINATE_KEY] = np.asarray(skeleton.coordinates[node_index])
        new_node_data[node_index] = node_data

    nx.set_node_attributes(skeleton_graph, new_node_data)

    return skeleton_graph
