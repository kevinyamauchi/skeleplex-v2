"""Module for skeletonization of segmented branching structures."""

import numpy as np
from skimage.morphology import skeletonize


def apply_skeletonization(image):
    """Apply 3D skeletonization to a binary image."""
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
