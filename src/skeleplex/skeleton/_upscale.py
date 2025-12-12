"""Upscale a skeleton image while preserving topology."""

import numpy as np
from skimage.draw import line_nd
from skimage.graph import pixel_graph


def upscale_skeleton(
    skeleton: np.ndarray,
    scale_factors: tuple[int, int, int],
) -> np.ndarray:
    """Upscale a 3D skeleton image while maintaining 1-voxel width.

    This function upscales a skeleton by scaling the coordinates of skeleton
    voxels and drawing lines between voxels that were originally connected.

    Parameters
    ----------
    skeleton : np.ndarray
        3D boolean array representing the skeleton, where True indicates
        skeleton voxels. Must be 3D.
    scale_factors : tuple[int, int, int]
        Integer scaling factors for each dimension (z, y, x). Must be positive
        integers.

    Returns
    -------
    upscaled_skeleton : np.ndarray
        Boolean array of the upscaled skeleton with shape
        (skeleton.shape[0] * scale_factors[0],
         skeleton.shape[1] * scale_factors[1],
         skeleton.shape[2] * scale_factors[2]).

    Raises
    ------
    ValueError
        If skeleton is not 3D, if scale_factors are not integers,
        or if scale_factors are not positive.
    """
    # Validate inputs
    if skeleton.ndim != 3:
        raise ValueError(f"Skeleton must be 3D, got {skeleton.ndim}D")

    if len(scale_factors) != 3:
        raise ValueError(f"scale_factors must have length 3, got {len(scale_factors)}")

    # Check that scale factors are integers
    if not all(isinstance(s, int | np.integer) for s in scale_factors):
        raise ValueError(f"scale_factors must be integers, got {scale_factors}")

    # Check that scale factors are positive
    if not all(s > 0 for s in scale_factors):
        raise ValueError(
            f"scale_factors must be positive integers, got {scale_factors}"
        )

    # Calculate upscaled shape
    upscaled_shape = tuple(skeleton.shape[i] * scale_factors[i] for i in range(3))

    # Create output array
    upscaled_skeleton = np.zeros(upscaled_shape, dtype=bool)

    # Get the connectivity graph of the original skeleton
    # Use connectivity=3 for 26-connectivity (includes diagonals)
    edges, nodes = pixel_graph(skeleton.astype(bool), connectivity=3)

    if len(nodes) == 0:
        # Empty skeleton, return empty upscaled skeleton
        return upscaled_skeleton

    # Convert the raveled node indices back to coordinates in the original scale
    node_coordinates = np.array(np.unravel_index(nodes, skeleton.shape)).T

    # Scale the coordinates and clip to array boundaries
    scaled_node_coords = np.round(node_coordinates * np.array(scale_factors)).astype(
        int
    )

    # Clip to ensure coordinates are within bounds
    for i in range(3):
        scaled_node_coords[:, i] = np.clip(
            scaled_node_coords[:, i], 0, upscaled_shape[i] - 1
        )

    # Set the scaled coordinates to True
    upscaled_skeleton[
        scaled_node_coords[:, 0],
        scaled_node_coords[:, 1],
        scaled_node_coords[:, 2],
    ] = True

    # Convert edges to COO format arrays for iteration
    # edges is a sparse matrix where
    # entry (i,j) means nodes[i] and nodes[j] are connected
    edge_indices = np.array(edges.nonzero()).T

    # Draw lines between connected voxels in the upscaled image
    for i in range(edge_indices.shape[0]):
        # Get the indices into the nodes array
        idx1 = edge_indices[i, 0]
        idx2 = edge_indices[i, 1]

        # Get the scaled coordinates of the two connected voxels
        coord1 = scaled_node_coords[idx1]
        coord2 = scaled_node_coords[idx2]

        # Draw a line between them
        # line_nd returns indices for each dimension
        line_indices = line_nd(coord1, coord2, endpoint=True)

        # Set all voxels along the line to True
        upscaled_skeleton[line_indices] = True

    return upscaled_skeleton
