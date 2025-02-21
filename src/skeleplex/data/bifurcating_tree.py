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
    length_ratio=0.5,
):
    """Add branches to the tree in a single 2D layer (z_layer)."""
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
    nodes=0,
    branch_length=20,
    z_layer=10,
    left_angle=30,
    right_angle=30,
    length_ratio=0.5,
):
    """Generate a bifurcating tree inside a specific z-layer of a 3D image."""
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
        max_levels=nodes,
        z_layer=z_layer,
        left_angle=left_angle,
        right_angle=right_angle,
        length_ratio=length_ratio,
    )

    return image


def apply_dilation_3d(image, dilation_radius=3):
    """Apply morphological dilation to the binary tree image in 3D."""
    structuring_element = ball(dilation_radius)
    dilated_image = binary_dilation(image, structure=structuring_element)
    return dilated_image
