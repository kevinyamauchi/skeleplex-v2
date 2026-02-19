"""Upscale a skeleton image while preserving topology."""

import numba
import numpy as np
from scipy.sparse import csr_matrix
from skimage.draw import line_nd
from skimage.graph import pixel_graph


@numba.njit(cache=True)
def _has_edge(indptr: np.ndarray, indices: np.ndarray, u: int, v: int) -> bool:
    """Check whether an edge (u, v) exists in the CSR graph.

    Parameters
    ----------
    indptr : np.ndarray
        The indptr array of the CSR adjacency matrix.
    indices : np.ndarray
        The indices array of the CSR adjacency matrix.
    u : int
        Source node index.
    v : int
        Target node index.

    Returns
    -------
    exists : bool
        True if edge (u, v) exists, False otherwise.
    """
    for k in range(indptr[u], indptr[u + 1]):
        if indices[k] == v:
            return True
    return False


@numba.njit(cache=True)
def _find_nodes_to_remove(
    indptr: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Find all nodes that are redundant shortcuts and can be safely removed.

    A node v is removable if, for every pair of its neighbours (a, b), a and
    b are still connected without going through v — either via a direct edge
    between them, or via a shared neighbour other than v.

    This is exactly the set of nodes that are "inside" a triangle cluster and
    whose removal does not disconnect any pair of their neighbours. In a valid
    1-voxel-wide skeleton these are the diagonal shortcut pixels introduced by
    26-connectivity at 90-degree corners.

    Only nodes of degree >= 2 are considered, since tips and isolated nodes
    cannot be part of a triangle.

    Parameters
    ----------
    indptr : np.ndarray
        The indptr array of the CSR adjacency matrix. Shape (n_nodes + 1,).
    indices : np.ndarray
        The indices array of the CSR adjacency matrix. Shape (nnz,).

    Returns
    -------
    nodes_to_remove : np.ndarray
        1D array of node indices that are safe to remove.
    """
    n_nodes = indptr.shape[0] - 1
    result = np.empty(n_nodes, dtype=np.int64)
    count = 0

    for v in range(n_nodes):
        deg = indptr[v + 1] - indptr[v]
        if deg < 2:
            continue

        nbrs_start = indptr[v]
        nbrs_end = indptr[v + 1]

        # Check every pair of v's neighbours (a, b).
        # The pair is "covered" (connected without v) if:
        #   (1) a and b share a direct edge, OR
        #   (2) a and b share a common neighbour other than v.
        # If all pairs are covered, v is redundant.
        removable = True
        for i in range(nbrs_start, nbrs_end):
            a = indices[i]
            for j in range(i + 1, nbrs_end):
                b = indices[j]

                # (1) Direct edge a-b?
                if _has_edge(indptr, indices, a, b):
                    continue

                # (2) Common neighbour of a and b, other than v?
                found_common = False
                for ka in range(indptr[a], indptr[a + 1]):
                    na = indices[ka]
                    if na == v:
                        continue
                    for kb in range(indptr[b], indptr[b + 1]):
                        if indices[kb] == na:
                            found_common = True
                            break
                    if found_common:
                        break

                if not found_common:
                    removable = False
                    break

            if not removable:
                break

        if removable:
            result[count] = v
            count += 1

    return result[:count]


def collapse_false_junctions(
    graph: csr_matrix,
) -> csr_matrix:
    """Collapse false junction shortcuts in a skeleton pixel graph.

    A removable node is one where every pair of its neighbours is already
    connected without going through it — either by a direct edge or via a
    shared neighbor. Such nodes are pure diagonal shortcuts introduced by
    26-connectivity at 90-degree corners; they add spurious degree-3 nodes
    without representing real bifurcations.

    The node is simply deleted. No rewiring is needed because all pairwise
    connectivity between its neighbours is preserved by other existing edges.

    Parameters
    ----------
    graph : csr_matrix
        Symmetric CSR adjacency matrix representing the skeleton pixel graph,
        as returned by ``skimage.graph.pixel_graph``. Edge weights are
        Euclidean distances between connected voxels.

    Returns
    -------
    cleaned_graph : csr_matrix
        A new CSR adjacency matrix with all pure-shortcut nodes removed.
        The matrix preserves the same node count as the input so that flat
        pixel indices remain valid; removed nodes become all-zero rows and
        columns.
    """
    graph_csr = graph.tocsr()
    indptr = graph_csr.indptr.astype(np.int64)
    indices = graph_csr.indices.astype(np.int64)

    nodes_to_remove = _find_nodes_to_remove(indptr, indices)

    if nodes_to_remove.shape[0] == 0:
        return graph_csr.copy()

    # Build a boolean mask of removed nodes for fast lookup
    n = graph_csr.shape[0]
    removed = np.zeros(n, dtype=bool)
    removed[nodes_to_remove] = True

    # Keep all edges where neither endpoint is removed
    orig_rows, orig_cols = graph_csr.nonzero()
    orig_data = np.asarray(graph_csr[orig_rows, orig_cols]).ravel()

    keep_mask = ~(removed[orig_rows] | removed[orig_cols])

    cleaned = csr_matrix(
        (orig_data[keep_mask], (orig_rows[keep_mask], orig_cols[keep_mask])),
        shape=(n, n),
    )

    return cleaned


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

    # Remove false junction shortcuts
    edges = collapse_false_junctions(edges)

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
