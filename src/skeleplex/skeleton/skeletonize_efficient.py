"""Apply 3D skeletonization to a binary image using Dask."""

import dask.array as da
import numpy as np
from skimage.morphology import skeletonize


def apply_skeletenization_efficient(image, factor=(2, 2, 2), depth=1):
    """
    Apply 3D skeletonization to a binary image using Dask.

    Parameters
    ----------
    image : dask.array.Array
        The binary 3D image to be skeletonized.
    factor : tuple of int
        The factor by which the image shape is divided to determine chunk sizes.
    depth : int, optional
        Number of overlapping pixels between chunks to maintain skeleton continuity.

    Returns
    -------
    dask.array.Array
        A lazily evaluated Dask array containing the skeletonized image.
    """
    # Compute controlled chunk size based on the image shape
    desired_chunksize = np.array(image.shape) // factor

    # Rechunk the image for consistency
    rechunked_image = image.rechunk(desired_chunksize)

    # Apply skeletonization using Dask with overlap
    dask_skeleton = da.map_overlap(
        skeletonize,  # Function to apply
        rechunked_image,  # Rechunked input image
        depth=depth,  # Overlap pixels
    )

    return dask_skeleton
