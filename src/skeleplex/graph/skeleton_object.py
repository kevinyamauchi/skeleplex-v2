"""Construction of a Skeleton object which enables further analysis and processing."""

import numpy as np
from skan import Skeleton
from skan.csr import _build_skeleton_path_graph, csr_to_nbgraph


def create_skeleton_object(skel, skelint, degrees_image, graph):
    """
    Constructs a Skeleton object from a skeletonized image.

    This function processes a labeled skeleton, computes the degrees of nodes,
    and constructs a graph representation using SciPy sparse matrices and
    skan's Skeleton object.

    Parameters
    ----------
    skel : dask.array.Array
        The original skeletonized image.
    skelint : dask.array.Array
        The labeled skeleton image where each connected component has a unique ID.
    degrees_image : dask.array.Array
        An image representing the degree (connectivity) of each skeleton pixel.
    graph : scipy.sparse.csr_matrix
        The adjacency matrix representing the skeleton graph.

    Returns
    -------
    skel_obj : skan.Skeleton
        A Skeleton object containing the graph, node coordinates, and paths.
    """
    ndim = skel.ndim

    # Initialize Skeleton object
    skel_obj = Skeleton(np.eye(5))  # Dummy initialization
    skel_obj.skeleton_image = skel
    skel_obj.spacing = [1] * skel.ndim  # Pixel spacing
    skel_obj.graph = graph
    skel_obj.degrees_image = degrees_image

    # Calculate the degrees attribute
    nonzero_degree_values = degrees_image[
        degrees_image > 0
    ].compute()  # triggers Dask computation
    degrees = np.concatenate((np.array([0]), nonzero_degree_values))
    skel_obj.degrees = degrees

    # Create a numba-compatible version of the skeleton graph adjacency matrix
    # node_prop for Skeleton class, so we can get the NBGraph (numba-ified graph)
    nonzero_pixel_ids = skelint[skelint > 0]
    nonzero_pixel_ids = np.asarray(nonzero_pixel_ids)  # coerces to a numpy array
    sorted_indices = np.argsort(nonzero_pixel_ids)  # Dask doesn't implement argsort

    raw_data = skel
    nonzero_pixel_intensity = raw_data[skelint > 0]

    # important, otherwise the indexing with sorted_indices thinks it's out of bounds
    nonzero_pixel_intensity.compute_chunk_sizes()

    node_props = nonzero_pixel_intensity[
        sorted_indices
    ].compute()  # trigger Dask computation
    node_props = np.concatenate((np.array([0]), node_props))  # add a dummy index

    nbgraph = csr_to_nbgraph(
        graph, node_props=node_props
    )  # node_props=None is the default
    # nbgraph = csr_to_nbgraph(graph, node_props=node_props)

    skel_obj.nbgraph = nbgraph

    # We also need to tell skan the non-zero pixel locations from our skeleton image.
    # Compute skelint first to ensure known chunk sizes
    skelint_np = skelint.compute()

    # Extract labeled pixel coordinates (z, y, x) and their corresponding labels
    pixel_coords = np.argwhere(skelint_np > 0)  # Get skeleton pixel positions
    pixel_labels = skelint_np[skelint_np > 0]  # Get corresponding labels

    # Sort by label to ensure consistent mapping
    sorted_indices = np.argsort(pixel_labels)
    pixel_coords = pixel_coords[sorted_indices]  # Reorder coordinates
    pixel_labels = pixel_labels[sorted_indices]  # Reorder labels

    # Assign sorted coordinates (ensuring node IDs align with pixel locations)
    skel_obj.coordinates = np.concatenate(([[0.0] * ndim], pixel_coords), axis=0)

    paths = _build_skeleton_path_graph(nbgraph)
    skel_obj.paths = paths
    skel_obj.n_paths = paths.shape[0]

    # MUST reset distances_initialized AND the empty numpy array to calc branch length
    skel_obj._distances_initialized = False
    skel_obj.distances = np.empty(skel_obj.n_paths, dtype=float)
    skel_obj.path_lengths()

    return skel_obj
