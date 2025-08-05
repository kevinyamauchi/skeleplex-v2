"""Creating biforcating trees as example data."""

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.draw import line
from skimage.morphology import ball


def draw_branch_2d(
    image,
    start,
    length,
    angle,
    level,
    max_levels,
    z_layer=10,
    left_angle=30,
    right_angle=30,
    length_ratio=0.7,
):
    """
    Adds branches to a bifurcating tree in a single 2D layer.

    Parameters
    ----------
    image : np.ndarray
        The 3D binary image where the tree is drawn.
    start : tuple[int, int]
        The (y, x) starting position of the branch.
    length : int
        The length of the current branch.
    angle : float
        The angle (in degrees) at which the branch grows.
    level : int
        The current recursion depth (bifurcation level).
    max_levels : int
        The maximum number of bifurcation levels.
    z_layer : int, optional
        The z-plane in which the tree is drawn, by default 10.
    left_angle : float, optional
        The angle offset for left branches, by default 30 degrees.
    right_angle : float, optional
        The angle offset for right branches, by default 30 degrees.
    length_ratio : float, optional
        The ratio by which branch length decreases in each bifurcation, by default 0.7.

    Returns
    -------
    None
        Modifies the input image in place.
    """
    if level > max_levels or length <= 0:
        return

    # Calculate the end point of the branch
    y_end = start[0] + int(length * np.cos(np.radians(angle)))
    x_end = start[1] + int(length * np.sin(np.radians(angle)))

    # Clip endpoints to stay within the image bounds
    y_end = np.clip(y_end, 0, image.shape[1] - 1)
    x_end = np.clip(x_end, 0, image.shape[2] - 1)

    # Draw the branch in the specific z-layer
    rr, cc = line(start[0], start[1], y_end, x_end)
    image[z_layer, rr, cc] = 1

    # New starting point for daughter branches
    new_start = (y_end, x_end)

    # Add left and right daughter branches
    draw_branch_2d(
        image,
        new_start,
        int(length * length_ratio),
        angle - left_angle,
        level + 1,
        max_levels,
        z_layer,
        left_angle,
        right_angle,
        length_ratio,
    )  # Left branch

    draw_branch_2d(
        image,
        new_start,
        int(length * length_ratio),
        angle + right_angle,
        level + 1,
        max_levels,
        z_layer,
        left_angle,
        right_angle,
        length_ratio,
    )  # Right branch


def generate_tree_3d(
    shape=(100, 100, 100),
    num_bifurcations=1,
    branch_length=20,
    z_layer=10,
    left_angle=30,
    right_angle=30,
    length_ratio=0.7,
):
    """
    Generate a 3D bifurcating tree in a specified z-layer of a 3D image.

    Parameters
    ----------
    shape : tuple[int, int, int], optional
        The dimensions of the 3D image, by default (100, 100, 100).
    num_bifurcations : int, optional
        The number of bifurcation levels, by default 1.
    branch_length : int, optional
        The length of the initial trunk, by default 20.
    z_layer : int, optional
        The specific z-plane where the tree is generated, by default 10.
    left_angle : float, optional
        The angle offset for left branches, by default 30 degrees.
    right_angle : float, optional
        The angle offset for right branches, by default 30 degrees.
    length_ratio : float, optional
        The ratio by which branch length decreases per bifurcation, by default 0.7.

    Returns
    -------
    np.ndarray
        A 3D binary image with the generated tree structure.
    """
    # Create an empty black 3D image
    image = np.zeros(shape, dtype=bool)

    # Starting point of the tree (top center of the chosen z-layer)
    # with a border of 20 pixels
    start = (20, shape[1] // 2)

    # Add the main trunk growing downward in the specified z-layer
    draw_branch_2d(
        image,
        start,
        branch_length,
        angle=0,
        level=0,
        max_levels=num_bifurcations,
        z_layer=z_layer,
        left_angle=left_angle,
        right_angle=right_angle,
        length_ratio=length_ratio,
    )

    return image


def apply_dilation_3d(image, dilation_radius=3):
    """
    Apply morphological dilation to thicken the tree structure in 3D.

    Parameters
    ----------
    image : np.ndarray
        A binary 3D image containing the tree structure.
    dilation_radius : int, optional
        The radius of the structuring element for dilation, by default 3.

    Returns
    -------
    np.ndarray
        A dilated version of the input tree image.
    """
    structuring_element = ball(dilation_radius)
    dilated_image = binary_dilation(image, structure=structuring_element)

    return dilated_image
