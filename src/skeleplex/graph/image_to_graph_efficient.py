"""Conversion of a skeletonized image into a graph representation."""

import networkx as nx
import numpy as np
from skan import summarize

from skeleplex.graph.adjacency_matrix import construct_matrix
from skeleplex.graph.constants import NODE_COORDINATE_KEY
from skeleplex.graph.skeleton_object import create_skeleton_object
from skeleplex.graph.skeleton_processing import compute_degrees, label_skeleton


def image_to_graph_dask(skel):
    """
     Convert a skeletonized image into a graph representation.

    Parameters
    ----------
    skel : dask.array.Array, optional
        The skeleton image to process.

    Returns
    -------
    tuple
        (skeleton_graph, skelint) -> The NetworkX skeleton graph and skeleton image.
    """
    # Step 1: Skeleton Processing
    skelint = label_skeleton(skel)
    degrees_image = compute_degrees(skel)

    # Step 2: Creating the adjacency matrix
    graph = construct_matrix(skelint)
    # visualize_graph(graph)

    # Step 3: Creating a skeleton object to do the summary for analysis
    skel_obj = create_skeleton_object(skel, skelint, degrees_image, graph)
    summary_table = summarize(skel_obj, separator="_")
    print(summary_table)

    skeleton_graph = nx.MultiGraph()

    for row in summary_table.itertuples(name="Edge"):
        # Extract nodes
        index = row.Index
        i = row.node_id_src
        j = row.node_id_dst

        # Extract the path between nodes
        path = skel_obj.path_coordinates(index)

        skeleton_graph.add_edge(i, j, **{"path": path})

    # Assign Node Coordinates
    node_data = {
        node_index: {NODE_COORDINATE_KEY: np.asarray(skel_obj.coordinates[node_index])}
        for node_index in skeleton_graph.nodes
    }

    nx.set_node_attributes(skeleton_graph, node_data)

    print(
        f"""Constructed skeleton graph with {len(skeleton_graph.nodes)} nodes
        and {len(skeleton_graph.edges)} edges"""
    )
    print(skelint.chunks)

    return skeleton_graph, skelint
