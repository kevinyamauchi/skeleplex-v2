"""Conversion of a skeletonized image into a graph representation chunk-wise.

These functions are useful for converting images that are too large to fit in memory.
"""

import logging

import dask.array as da
import networkx as nx
import numpy as np
from skan import summarize

from skeleplex.graph.adjacency_matrix import construct_matrix
from skeleplex.graph.constants import NODE_COORDINATE_KEY
from skeleplex.graph.skeleton_object import create_skeleton_object
from skeleplex.graph.skeleton_processing import compute_degrees, label_skeleton

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def image_to_graph_chunked(skeleton_image: da.Array) -> tuple[nx.MultiGraph, da.Array]:
    """
     Convert a skeletonized image into a graph representation.

    Parameters
    ----------
    skeleton_image : dask.array.Array, optional
        The skeleton image to process.

    Returns
    -------
    tuple
        (skeleton_graph, skelint) -> The NetworkX skeleton graph and skeleton image.
    """
    if not isinstance(skeleton_image, da.Array):
        raise TypeError("Input must be a dask.array.Array.")

    if skeleton_image.ndim != 3:
        raise ValueError("Input must be a 3D image.")

    logger.info("Starting skeleton image processing...")

    # step 1: Skeleton Processing
    labeled_skeleton_image = label_skeleton(skeleton_image)
    degrees_image = compute_degrees(skeleton_image)

    # step 2: Creating the adjacency matrix
    logger.info("Constructing adjacency matrix...")
    adjacency_matrix = construct_matrix(labeled_skeleton_image)

    # step 3: Creating a skeleton object to do the summary for analysis
    logger.info("Creating skeleton object and generating summary...")
    skeleton_object = create_skeleton_object(
        skeleton_image, labeled_skeleton_image, degrees_image, adjacency_matrix
    )

    summary_table = summarize(skeleton_object, separator="_")

    skeleton_graph = nx.MultiGraph()

    for row in summary_table.itertuples(name="Edge"):
        # extract nodes
        index = row.Index
        i = row.node_id_src
        j = row.node_id_dst

        # extract the path between nodes
        path = skeleton_object.path_coordinates(index)

        skeleton_graph.add_edge(i, j, **{"path": path})

    # assign Node Coordinates
    node_data = {
        node_index: {
            NODE_COORDINATE_KEY: np.asarray(skeleton_object.coordinates[node_index])
        }
        for node_index in skeleton_graph.nodes
    }

    nx.set_node_attributes(skeleton_graph, node_data)

    logger.info(
        f"Constructed skeleton graph with {len(skeleton_graph.nodes)} nodes "
        f"and {len(skeleton_graph.edges)} edges."
    )
    logger.info(f"Labeled skeleton image chunks: {labeled_skeleton_image.chunks}")

    return skeleton_graph, labeled_skeleton_image
