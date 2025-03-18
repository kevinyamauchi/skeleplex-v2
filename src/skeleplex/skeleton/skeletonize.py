"""Function for skeletonization of segmented branching structures."""

import numpy as np
from skimage.morphology import skeletonize


def apply_skeletonization(image):
    """
    Function to perform skeletonization on a 3D binary image.

    Parameters
    ----------
    image : np.ndarray
        A 3D binary NumPy array representing the segmented structure.

    Returns
    -------
    np.ndarray
        A 3D NumPy array of the same shape as the input, containing
        the skeletonized version of the binary structure.
    """
    # Ensure input is a 3D NumPy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    if image.ndim != 3:
        raise ValueError("Input must be a 3D binary image.")

    # Ensure image is binary
    binary_image = image > 0

    # Apply skeletonization
    skeleton = skeletonize(binary_image)

    return skeleton
