import dask.array as da  # noqa
import numpy as np
import pyclesperanto as cle
from scipy.ndimage import distance_transform_edt


def distance_transform(img):
    """Compute the distance transform of a binary image."""
    return distance_transform_edt(img, return_distances=True, return_indices=False)


def max_dist_gpu(img, ball_radius=6):
    """
    Compute the maximum distance transform on GPU using pyclesperanto.

    Parameters
    ----------
    img : da.Array
        Input image to compute the maximum distance transform for.
    ball_radius : int, optional
        Radius of the ball used for local maximum computation, by default 6.

    Returns
    -------
    da.Array
        Maximum distance transform of the input image.
    """
    local_maximum_cle = cle.maximum_sphere(
        img, radius_x=ball_radius, radius_y=ball_radius, radius_z=ball_radius
    )
    print(local_maximum_cle.shape)

    local_maximum = np.asarray(local_maximum_cle)

    return local_maximum


def normalized_distance_transform(
    img: da.Array, depth: int, min_ball_radius: int = 6
) -> da.Array:
    """
    Compute the normalized distance transform of a binary image.

    Parameters
    ----------
    img : da.Array
        Binary image to compute the distance transform for.
    depth : int
        Depth for the map_overlap operation.
    min_ball_radius : int, optional
        Minimum radius of the ball used for local maximum computation, by default 6.
        Depends on GPU memory.

    Returns
    -------
    da.Array
        Normalized distance transform of the input image.
    """
    # Compute the distance transform
    distance_image = img.map_overlap(distance_transform, depth=depth, boundary="none")

    # Compute the maximum distance
    max_distance = da.max(distance_image)
    ball_radius = min(int(max_distance / 2), min_ball_radius)

    # Compute the local maximum image
    # Here the depth is fixed to the kernel,
    # as we need to handle the boundaries correctly.
    local_maximum_image = distance_image.map_overlap(
        max_dist_gpu, depth=ball_radius * 2, boundary="none", dtype=np.float32
    )

    # Normalize the local maximum image
    local_maximum_image_norm = np.zeros_like(local_maximum_image)
    local_maximum_image_norm = da.where(img, distance_image / local_maximum_image, 0.0)

    return local_maximum_image_norm
