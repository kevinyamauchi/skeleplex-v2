"""Labeling skeletonized structures and computing pixel connectivity degrees.

These functions are adapted from Genevieve Buckley's distributed-skeleton-analysis repo:
https://github.com/GenevieveBuckley/distributed-skeleton-analysis
"""

import dask.array as da
import numpy as np
from dask_image.ndfilters import convolve
from dask_image.ndmeasure import label


def label_skeleton(skeleton_image: da.Array) -> da.Array:
    """
    Labels the connected components in a skeletonized image.

    Parameters
    ----------
    skeleton_image : dask.array.Array
        The input skeletonized image.

    Returns
    -------
    labeled_skeleton_image : dask.array.Array
        A labeled skeleton image where each connected component has a unique label.
    """
    ndim = skeleton_image.ndim
    structure_kernel = np.zeros((3,) * ndim)
    structure_kernel[(1,) * ndim] = 1
    labeled_skeleton_image, num_features = label(
        skeleton_image, structure=structure_kernel
    )

    return labeled_skeleton_image


def compute_degrees(skeleton_image: da.Array) -> da.Array:
    """
    Computes the degree of each skeleton pixel based on its connectivity.

    The degree is determined by counting the number of neighboring pixels
    that are part of the skeleton.

    Parameters
    ----------
    skeleton_image : dask.array.Array
        The input skeletonized image.

    Returns
    -------
    degrees_image : dask.array.Array
        An image where each skeleton pixel is assigned its degree value.
    """
    ndim = skeleton_image.ndim
    degree_kernel = np.ones((3,) * ndim)
    degree_kernel[(1,) * ndim] = 0

    degrees_image = (
        convolve(skeleton_image.astype(int), degree_kernel, mode="constant")
        * skeleton_image
    )

    return degrees_image
