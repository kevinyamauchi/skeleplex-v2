"""Apply 3D skeletonization to a binary image using Dask."""

import dask.array as da
import numpy as np
from skimage.morphology import skeletonize


def apply_skeletonization_chunked(
    binary_image: da.Array, factor: tuple[int, int, int] = (2, 2, 2), depth: int = 1
) -> da.Array:
    """
    Apply 3D skeletonization to a binary image using Dask.

    Parameters
    ----------
    binary_image : dask.array.Array
        The binary 3D image to be skeletonized.
    factor : tuple of int
        The factor by which the image shape is divided to determine chunk sizes.
    depth : int, optional
        Number of overlapping pixels between chunks to maintain skeleton continuity.

    Returns
    -------
    skeleton_image : dask.array.Array
        A lazily evaluated Dask array containing the skeletonized image.
    """
    if not isinstance(binary_image, da.Array):
        raise TypeError("Input must be a dask.array.Array.")

    if binary_image.ndim != 3:
        raise ValueError("Input must be a 3D image.")
    desired_chunksize = np.array(binary_image.shape) // factor

    rechunked_image = binary_image.rechunk(desired_chunksize)

    skeleton_image = da.map_overlap(
        skeletonize,
        rechunked_image,
        depth=depth,
    )

    return skeleton_image
