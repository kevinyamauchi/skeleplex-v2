"""
Functions for computing normalized distance transform.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, label



def local_normalized_distance(
    image: np.ndarray,  
    max_ball_radius: int = 30,
) -> np.ndarray:
    
    """
    Compute normalized distance transform for a binary image.
    
    Parameters
    ----------
    image : np.ndarray
        Binary array where non-zero values are interpreted as foreground.
    max_ball_radius : int
        Maximum radius of the ball used for maximum filtering.
        Default is 30.
        
    Returns
    -------
    np.ndarray
        Array of same shape as input image, containing normalized distance values.
    """        

    image = np.asarray(image)
    binary = image > 0
    labeled, num_labels = label(binary)
    normalized_distance = np.zeros_like(image, dtype=np.float32)

    for i in range(1, num_labels + 1):
        mask = labeled == i

        distance = distance_transform_edt(mask)

        local_max = np.max(distance)
        radius = min(int(local_max / 2), max_ball_radius)

        # apply maximum filter to normalize distances locally
        local_max_distance = maximum_filter(distance, size=radius * 2 + 1)

        normalized_distance[mask] = distance[mask] / (local_max_distance[mask])

    return normalized_distance




def local_normalized_distance_gpu(
    image: np.ndarray,
    max_ball_radius: int = 30,
) -> np.ndarray:
    """
    Compute normalized distance transform for a binary image on the GPU using CuPy.

    Parameters
    ----------
    image : np.ndarray
        Binary array where non-zero values are interpreted as foreground.
    max_ball_radius : int
        Maximum radius for the structuring element used in the maximum filter.
        Default is 30.

    Returns
    -------
    np.ndarray
        Array of same shape as input image, containing normalized distance values.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as distance_transform_edt_gpu
    from cupyx.scipy.ndimage import maximum_filter as maximum_filter_gpu, label

    image = cp.asarray(image)  # move to GPU
    binary = image > 0
    labeled, num_labels = label(binary)
    normalized_distance = cp.zeros_like(image, dtype=cp.float32)

    for i in range(1, num_labels + 1):
        mask = labeled == i

        distance = distance_transform_edt_gpu(mask)

        local_max = cp.max(distance)
        radius = min(int(local_max / 2), max_ball_radius)

        # apply maximum filter to normalize distances locally
        local_max_distance = maximum_filter_gpu(distance, size=radius * 2 + 1)

        normalized_distance[mask] = distance[mask] / (local_max_distance[mask])

    return cp.asnumpy(normalized_distance)