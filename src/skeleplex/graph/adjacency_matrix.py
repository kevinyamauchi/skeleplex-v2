"""Conversion of a skeletonized image into a sparse graph representation using Dask.

These functions are adapted from Genevieve Buckley's distributed-skeleton-analysis repo:
https://github.com/GenevieveBuckley/distributed-skeleton-analysis
"""

import functools
import operator
from itertools import product

import dask.array as da
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from dask.array.slicing import cached_cumsum
from dask.delayed import delayed
from scipy import sparse
from skan.csr import _write_pixel_graph
from skan.nputil import raveled_steps_to_neighbors


@delayed
def skeleton_graph_func(
    labeled_skeleton_chunk: np.ndarray, spacing: float = 1
) -> pd.DataFrame:
    """
    Converts a skeleton chunk into a adjacency matrix representation.

    Parameters
    ----------
    labeled_skeleton_chunk : np.ndarray
        A skeletonized binary image chunk.
    spacing : float, optional
        Spacing between pixels in the skeleton, by default 1.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the graph edges with 'row', 'col', and 'data' columns.
    """
    ndim = labeled_skeleton_chunk.ndim
    spacing = np.ones(ndim, dtype=float) * spacing
    num_edges = _num_edges(labeled_skeleton_chunk.astype(bool))
    padded_chunk = np.pad(labeled_skeleton_chunk, 1)
    steps, distances = raveled_steps_to_neighbors(
        padded_chunk.shape, ndim, spacing=spacing
    )

    row = np.empty(num_edges, dtype=int)
    col = np.empty(num_edges, dtype=int)
    data = np.empty(num_edges, dtype=float)
    _write_pixel_graph(padded_chunk, steps, distances, row, col, data)

    return pd.DataFrame({"row": row, "col": col, "data": data})


def _num_edges(skeleton_chunk: np.ndarray) -> int:
    """
    Computes the total number of edges in a skeletonized image.

    Parameters
    ----------
    skeleton_chunk : np.ndarray
        The skeletonized binary image.

    Returns
    -------
    int
        The total number of edges in the skeleton.
    """
    ndim = skeleton_chunk.ndim
    degree_kernel = np.ones((3,) * ndim)
    degree_kernel[(1,) * ndim] = 0
    degree_image = (
        scipy.ndimage.convolve(
            skeleton_chunk.astype(int), degree_kernel, mode="constant"
        )
        * skeleton_chunk
    )
    num_edges = np.sum(degree_image)
    return int(num_edges)


def slices_from_chunks_overlap(
    chunks: tuple[tuple[int, ...], ...], array_shape: tuple[int, ...], depth: int = 1
) -> list[tuple[slice, ...]]:
    """Translate chunks tuple to a set of slices in product order.

    Parameters
    ----------
    chunks : tuple
        The chunks of the corresponding dask array.
    array_shape : tuple
        Shape of the corresponding dask array.
    depth : int
        The number of pixels to overlap, providing we're not at the array edge.

    Example
    -------
    >>> slices_from_chunks_overlap(
    ...     ((4,), (7, 7)), (4, 14), depth=1
    ... )  # doctest: +NORMALIZE_WHITESPACE
     [(slice(0, 5, None), slice(0, 8, None)),
      (slice(0, 5, None), slice(6, 15, None))]
    """
    cumdims = [cached_cumsum(bds, initial_zero=True) for bds in chunks]

    slices = []
    for starts, shapes in zip(cumdims, chunks, strict=False):
        inner_slices = []
        for s, dim, maxshape in zip(starts, shapes, array_shape, strict=False):
            slice_start = s
            slice_stop = s + dim
            if slice_start > 0:
                slice_start -= depth
            if slice_stop >= maxshape:
                slice_stop += depth
            inner_slices.append(slice(slice_start, slice_stop))
        slices.append(inner_slices)

    return list(product(*slices))


def construct_matrix(labeled_skeleton_image: da.Array) -> scipy.sparse.csr_matrix:
    """
    Constructs a sparse adjacency matrix from a skeletonized image.

    This function processes skeleton image chunks using Dask and builds a graph
    representation of the structure.

    Parameters
    ----------
    labeled_skeleton_image : dask.array.Array
        The labeled skeletonized image.

    Returns
    -------
    adjacency_matrix : scipy.sparse.csr_matrix
        A sparse adjacency matrix representing the skeleton graph.
    """
    chunk_iterator = zip(
        np.ndindex(*labeled_skeleton_image.numblocks),
        map(
            functools.partial(operator.getitem, labeled_skeleton_image),
            slices_from_chunks_overlap(
                labeled_skeleton_image.chunks, labeled_skeleton_image.shape, depth=1
            ),
        ),
        strict=False,
    )

    meta = dd.utils.make_meta(
        [("row", np.int64), ("col", np.int64), ("data", np.float64)]
    )
    chunk_graphs = [
        dd.from_delayed(skeleton_graph_func(block), meta=meta)
        for _, block in chunk_iterator
    ]
    graph_edges_df = dd.concat(chunk_graphs)

    graph_edges_df = graph_edges_df.drop_duplicates()

    k = len(graph_edges_df)
    row = np.array(graph_edges_df["row"])
    col = np.array(graph_edges_df["col"])
    data = np.array(graph_edges_df["data"])

    adjacency_matrix = sparse.coo_matrix((data[:k], (row[:k], col[:k]))).tocsr()

    return adjacency_matrix


def visualize_graph(adjacency_matrix: scipy.sparse.csr_matrix) -> None:
    """
    Visualizes the adjacency matrix of a skeleton graph.

    Parameters
    ----------
    adjacency_matrix : scipy.sparse.csr_matrix
        The adjacency matrix representing the skeleton graph.
    """
    plt.xticks(ticks=np.arange(0, adjacency_matrix.shape[1], 1))
    plt.yticks(ticks=np.arange(0, adjacency_matrix.shape[0], 1))
    plt.imshow(adjacency_matrix.todense(), cmap="viridis")
    plt.colorbar(label="Graph Connectivity")
    plt.title("Sparse Graph Representation")
    plt.show()
