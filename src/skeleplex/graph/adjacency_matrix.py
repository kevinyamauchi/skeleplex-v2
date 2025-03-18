"""Conversion of a skeletonized image into a sparse graph representation using Dask."""

import functools
import operator
from itertools import product

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
def skeleton_graph_func(skelint, spacing=1):
    """
    Converts a skeleton chunk into a graph representation.

    Parameters
    ----------
    skelint : np.ndarray
        A skeletonized binary image chunk.
    spacing : float, optional
        Spacing between pixels in the skeleton, by default 1.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the graph edges with 'row', 'col', and 'data' columns.
    """
    ndim = skelint.ndim
    spacing = np.ones(ndim, dtype=float) * spacing
    num_edges = _num_edges(skelint.astype(bool))
    padded_skelint = np.pad(skelint, 1)  # pad image to prevent looparound errors
    steps, distances = raveled_steps_to_neighbors(
        padded_skelint.shape, ndim, spacing=spacing
    )

    # from function skan.csr._pixel_graph
    row = np.empty(num_edges, dtype=int)
    col = np.empty(num_edges, dtype=int)
    data = np.empty(num_edges, dtype=float)
    _write_pixel_graph(padded_skelint, steps, distances, row, col, data)

    return pd.DataFrame({"row": row, "col": col, "data": data})


def _num_edges(skel):
    """
    Computes the total number of edges in a skeletonized image.

    Parameters
    ----------
    skel : np.ndarray
        The skeletonized binary image.

    Returns
    -------
    int
        The total number of edges in the skeleton.
    """
    ndim = skel.ndim
    degree_kernel = np.ones((3,) * ndim)
    degree_kernel[(1,) * ndim] = 0  # remove centre pixel
    degree_image = (
        scipy.ndimage.convolve(skel.astype(int), degree_kernel, mode="constant") * skel
    )
    num_edges = np.sum(degree_image)
    return int(num_edges)


def slices_from_chunks_overlap(chunks, array_shape, depth=1):
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


def construct_matrix(skelint):
    """
    Constructs a sparse adjacency matrix from a skeletonized image.

    This function processes skeleton image chunks using Dask and builds a graph
    representation of the structure.

    Parameters
    ----------
    skelint : dask.array.Array
        The labeled skeletonized image.

    Returns
    -------
    scipy.sparse.csr_matrix
        A sparse adjacency matrix representing the skeleton graph.
    """
    image = skelint

    block_iter = zip(
        np.ndindex(*image.numblocks),
        map(
            functools.partial(operator.getitem, image),
            slices_from_chunks_overlap(image.chunks, image.shape, depth=1),
        ),
        strict=False,
    )

    meta = dd.utils.make_meta(
        [("row", np.int64), ("col", np.int64), ("data", np.float64)]
    )  # it's very important to include meta
    intermediate_results = [
        dd.from_delayed(skeleton_graph_func(block), meta=meta)
        for _, block in block_iter
    ]
    results = dd.concat(intermediate_results)

    # drop duplicates from the results
    results = results.drop_duplicates()

    k = len(results)
    row = np.array(results["row"])
    col = np.array(results["col"])
    data = np.array(results["data"])

    graph = sparse.coo_matrix((data[:k], (row[:k], col[:k]))).tocsr()

    return graph


def visualize_graph(graph):
    """
    Visualizes the adjacency matrix of a skeleton graph.

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        The adjacency matrix representing the skeleton graph.
    """
    plt.xticks(ticks=np.arange(0, graph.shape[1], 1))
    plt.yticks(ticks=np.arange(0, graph.shape[0], 1))
    plt.imshow(graph.todense(), cmap="viridis")
    plt.colorbar(label="Graph Connectivity")
    plt.title("Sparse Graph Representation")
    plt.show()
