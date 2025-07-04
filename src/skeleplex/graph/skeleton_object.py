"""Construction of a Skeleton object which enables further analysis and processing.

This function is adapted from Genevieve Buckley's distributed-skeleton-analysis repo:
https://github.com/GenevieveBuckley/distributed-skeleton-analysis
"""

import dask.array as da
import numpy as np
import scipy.sparse
from skan import Skeleton
from skan.csr import _build_skeleton_path_graph, csr_to_nbgraph


def create_skeleton_object(
    skeleton_image: da.Array,
    labeled_skeleton_image: da.Array,
    degrees_image: da.Array,
    adjacency_matrix: scipy.sparse.csr_matrix,
) -> Skeleton:
    """
    Constructs a Skeleton object from a skeletonized image.

    This function processes a labeled skeleton, computes the degrees of nodes,
    and constructs a graph representation using SciPy sparse matrices and
    skan's Skeleton object.

    Parameters
    ----------
    skeleton_image : dask.array.Array
        The original skeletonized image.
    labeled_skeleton_image : dask.array.Array
        The labeled skeleton image where each connected component has a unique ID.
    degrees_image : dask.array.Array
        An image representing the degree (connectivity) of each skeleton pixel.
    adjacency_matrix : scipy.sparse.csr_matrix
        The adjacency matrix representing the skeleton graph.

    Returns
    -------
    skeleton_object : skan.Skeleton
        A Skeleton object containing the graph, node coordinates, and paths.
    """
    ndim = skeleton_image.ndim

    skeleton_object = Skeleton(np.eye(5))
    skeleton_object.skeleton_image = skeleton_image
    skeleton_object.spacing = [1] * skeleton_image.ndim
    skeleton_object.graph = adjacency_matrix
    skeleton_object.degrees_image = degrees_image

    nonzero_degree_values = degrees_image[degrees_image > 0].compute()
    degrees = np.concatenate((np.array([0]), nonzero_degree_values))
    skeleton_object.degrees = degrees

    labeled_pixel_ids = [labeled_skeleton_image > 0]
    labeled_pixel_ids = np.asarray(labeled_pixel_ids)
    sorted_indices = np.argsort(
        labeled_skeleton_image[labeled_skeleton_image > 0].compute()
    )

    raw_data = skeleton_image
    nonzero_pixel_intensity = raw_data[labeled_skeleton_image > 0]

    nonzero_pixel_intensity.compute_chunk_sizes()

    node_props = nonzero_pixel_intensity[sorted_indices].compute()
    node_props = np.concatenate((np.array([0]), node_props))

    nbgraph = csr_to_nbgraph(adjacency_matrix, node_props=node_props)

    skeleton_object.nbgraph = nbgraph

    skelint_np = labeled_skeleton_image.compute()

    pixel_coords = np.argwhere(skelint_np > 0)
    pixel_labels = skelint_np[skelint_np > 0]

    sorted_indices = np.argsort(pixel_labels)
    pixel_coords = pixel_coords[sorted_indices]
    pixel_labels = pixel_labels[sorted_indices]

    skeleton_object.coordinates = np.concatenate(([[0.0] * ndim], pixel_coords), axis=0)

    paths = _build_skeleton_path_graph(nbgraph)
    skeleton_object.paths = paths
    skeleton_object.n_paths = paths.shape[0]

    skeleton_object._distances_initialized = False
    skeleton_object.distances = np.empty(skeleton_object.n_paths, dtype=float)
    skeleton_object.path_lengths()

    return skeleton_object
