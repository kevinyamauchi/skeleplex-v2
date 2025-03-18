"""Labeling skeletonized structures and computing pixel connectivity degrees."""

import numpy as np
from dask_image.ndfilters import convolve
from dask_image.ndmeasure import label


def label_skeleton(skel):
    """
    Labels the connected components in a skeletonized image.

    Parameters
    ----------
    skel : dask.array.Array
        The input skeletonized image.

    Returns
    -------
    skelint : dask.array.Array
        A labeled skeleton image where each connected component has a unique label.
    """
    ndim = skel.ndim
    structure_kernel = np.zeros((3,) * ndim)
    structure_kernel[(1,) * ndim] = 1  # add centre pixel
    skelint, num_features = label(skel, structure=structure_kernel)

    return skelint


def compute_degrees(skel):
    """
    Computes the degree of each skeleton pixel based on its connectivity.

    The degree is determined by counting the number of neighboring pixels
    that are part of the skeleton.

    Parameters
    ----------
    skel : dask.array.Array
        The input skeletonized image.

    Returns
    -------
    degrees_image : dask.array.Array
        An image where each skeleton pixel is assigned its degree value.
    """
    ndim = skel.ndim
    degree_kernel = np.ones((3,) * ndim)
    degree_kernel[(1,) * ndim] = 0  # Remove center pixel

    degrees_image = convolve(skel.astype(int), degree_kernel, mode="constant") * skel

    return degrees_image
