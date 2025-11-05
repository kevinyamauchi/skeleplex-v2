"""Compute the radius map for a segmented image using GPU acceleration."""

import dask
import dask.array as da
import numpy as np


def radius_map_generator_gpu(
    image: np.ndarray,
    max_ball_radius: int = 30,
) -> np.ndarray:
    """
    Compute max radius map for a segmented image.

    This algorithm computes the maximal radius for each connected component
    of a segmented image in a given radius and labels the image according to
    radius size in voxels.

    This function is accelerated on the GPU using CuPy.

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
        Array of same shape as input image, containing distance values.
    """
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import (
            distance_transform_edt as distance_transform_edt_gpu,
        )
        from cupyx.scipy.ndimage import label
        from cupyx.scipy.ndimage import maximum_filter as maximum_filter_gpu

    except ImportError as err:
        raise ImportError(
            "local_normalized_distance_gpu requires CuPy. "
            "Please install it by following the CuPy "
            "installation instructions for your GPU."
        ) from err

    image = cp.asarray(image)  # move to GPU
    binary = image > 0
    labeled, num_labels = label(binary)
    local_max_distance = cp.zeros_like(image, dtype=cp.float32)

    for i in range(1, num_labels + 1):
        mask = labeled == i

        distance = distance_transform_edt_gpu(mask)

        local_max = cp.max(distance)
        radius = min(int(local_max / 2), max_ball_radius)

        # apply maximum filter to normalize distances locally
        local_max_distance[mask] = maximum_filter_gpu(distance, size=radius * 3)[mask]

    return cp.asnumpy(local_max_distance)


if __name__ == "__main__":
    lung_image = da.from_zarr("bifurcating_tree.zarr")
    with dask.config.set(num_workers=1):
        lung_image_radius_map = lung_image.map_overlap(
            radius_map_generator_gpu, depth=30, boundary=0, dtype=np.float32
        )

        lung_image_radius_map.to_zarr("radius_map.zarr", overwrite=True)
