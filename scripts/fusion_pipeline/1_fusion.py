"""Fusion Part 1: Generate Radius Map and Scale Map."""

##################################################################################################
#                                           IMPORTS
##################################################################################################
import time
from functools import partial

import dask
import dask.array as da
import numpy as np


##################################################################################################
#                                           FUNCTIONS
##################################################################################################
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


def scale_map_generator_gpu(radius_map: np.ndarray, scale_ranges: dict) -> np.ndarray:
    """
    Generate the scales map for the fusion algorithm.

    Each value in the radius map is mapped to a certain scale based on the range
    that it falls into in the provided scale range dictionary.

    It is accelerated on the GPU using CuPy.

    Parameters
    ----------
    radius_map : np.ndarray
        Array with non-zero values documenting the radius of each tube.
    scale_ranges : dict
        This dictionary is used to map the scales to the radii in the radius_map.

    Returns
    -------
    np.ndarray
        Array of same shape as input image, containing scale mapped values.
    """
    try:
        import cupy as cp

    except ImportError as err:
        raise ImportError(
            "local_normalized_distance_gpu requires CuPy. "
            "Please install it by following the CuPy "
            "installation instructions for your GPU."
        ) from err

    radius_map = cp.asarray(radius_map)
    mask = radius_map > 0

    scale_map = cp.zeros_like(radius_map, dtype=np.float32)

    for key, (start, end) in scale_ranges.items():
        mask = (radius_map >= start) & (radius_map < end)
        scale_map[mask] = key

    return cp.asnumpy(scale_map)


##################################################################################################
#                                           RUNNING FUSION
##################################################################################################

# Define the image prefix used to name the files
image_prefix = "IMAGE_PREFIX"  # ADAPT HERE

# Example: define scales and their valid ranges
scale_ranges_manual = {
    0: (1, 5),
    -1: (5, 12),
    -2: (12, 30),
    -3: (30, 150),
}  # ADAPT HERE


# Load the initial image (here: label)
lung_image = da.from_zarr("data/IMAGE_NAME.zarr")  # ADAPT HERE
lung_image = lung_image.rechunk((94, 94, 94))


################ Fusion Part 1 ################

# Generate Radius Map

start_time = time.time()
with dask.config.set(num_workers=1):
    lung_image_radius_map = lung_image.map_overlap(
        radius_map_generator_gpu, depth=30, boundary=0, dtype=np.float32
    )

    lung_image_radius_map.to_zarr(
        f"data/{image_prefix}_radius_map.zarr/scale_original", overwrite=True
    )

lung_image_radius_map = da.from_zarr(
    f"data/{image_prefix}_radius_map.zarr/scale_original"
)
lung_image_radius_map = lung_image_radius_map.rechunk((94, 94, 94))
print(f"--- Radius map creation took {time.time()
                                      - start_time} seconds ---")


# Generate Scale Map

start_time = time.time()
with dask.config.set(num_workers=1):
    prefunction_scale_map_generator_gpu = partial(
        scale_map_generator_gpu, scale_ranges=scale_ranges_manual
    )

    lung_image_scale_map = lung_image_radius_map.map_overlap(
        prefunction_scale_map_generator_gpu,
        depth=10,
        boundary=0,
        dtype=lung_image_radius_map.dtype,
    )

    lung_image_scale_map = lung_image_scale_map.rechunk((94, 94, 94))
    lung_image_scale_map.to_zarr(
        f"data/{image_prefix}_image_scale_map.zarr/scale_original", overwrite=True
    )

print(f"--- Scale map creation took {time.time()
                                     - start_time} seconds ---")
print("End of Fusion part 1.")
