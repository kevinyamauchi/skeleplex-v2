"""Drawing tubes in a volume as example data."""

import numpy as np
from skimage.draw import disk


def draw_tube(
    image: np.ndarray,
    center: tuple[int, int] = (50, 50),
    diameter: int = 20,
    z_span: tuple[int, int] = (25, 75),
) -> np.ndarray:
    """
    Add a tube aligned along the z-axis into an existing 3D volume.

    Parameters
    ----------
    image : ndarray
        3D volume (z, y, x) that will be modified in place.
    center : tuple of int
        (row, col) center of the tube in each xy slice.
    diameter : int
        Diameter of the circular cross-section.
    z_span : tuple of int
        (start, end) range of z-slices to fill.

    Returns
    -------
    image : ndarray
        The input image with the new tube added.
    """
    radius = diameter // 2
    rr, cc = disk(center, radius, shape=image.shape[1:])

    for z in range(z_span[0], z_span[1]):
        image[z, rr, cc] = 1

    return image


def draw_tubes_image_example() -> np.ndarray:
    """
    Creates a 3D image with multiple tubes for testing.

    First and empty 100x100x100 volume is created, then seven tubes with different
    centers, diameters, and z-spans are added to imitate different scales.

    Returns
    -------
    image : ndarray
        The output image is a 3D numpy array of shape 100x100x100 containing
        seven tubes along the z-axis with various diameters.

    """
    tubes_image = np.zeros((100, 100, 100), dtype=np.uint8)

    draw_tube(tubes_image, center=(70, 70), diameter=45, z_span=(45, 95))
    draw_tube(tubes_image, center=(20, 65), diameter=30, z_span=(25, 65))
    draw_tube(tubes_image, center=(55, 25), diameter=25, z_span=(10, 45))
    draw_tube(tubes_image, center=(25, 30), diameter=20, z_span=(45, 75))
    draw_tube(tubes_image, center=(40, 85), diameter=14, z_span=(45, 60))
    draw_tube(tubes_image, center=(80, 20), diameter=10, z_span=(10, 25))
    draw_tube(tubes_image, center=(10, 10), diameter=10, z_span=(60, 75))

    return tubes_image
