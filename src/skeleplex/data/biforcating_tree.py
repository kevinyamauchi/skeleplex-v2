"""Creating biforcating trees as example data."""

import matplotlib.pyplot as plt
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
    """Recursively add branches to the tree in a single 2D layer (z_layer)."""
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


def visualize_2d_slice(image, depth_layer=None):
    """Visualize a single depth layer of a 3D binary image."""
    if depth_layer is None:
        depth_layer = image.shape[0] // 2  # Default to middle slice

    plt.imshow(image[depth_layer], cmap="gray")
    plt.title(f"2D Slice at Depth Layer {depth_layer}")
    plt.axis("off")
    plt.show()


def save_tree(image, filename):
    """Save the 3D binary tree or dilated image to a .npy file."""
    np.save(filename, image)
    print(f"Tree saved to {filename}")


if __name__ == "__main__":
    # Parameters
    shape = (100, 100, 100)  # Size of the 3D image
    nodes = 8  # Number of bifurcation levels (0 = trunk only)
    branch_length = 30  # Length of each branch
    z_layer = 10  # The specific layer in which the tree will grow
    dilation_radius = 3  # Radius for morphological dilation
    left_angle = 40  # Angular deviation for left branches
    right_angle = 60  # Angular deviation for right branches
    length_ratio = 0.7  # Ratio for branch length decrease

    # Generate the 3D tree skeleton in a specific z-layer
    tree = generate_tree_3d(
        shape, nodes, branch_length, z_layer, left_angle, right_angle, length_ratio
    )

    # Apply morphological dilation to create the fake segmentation
    dilated_tree = apply_dilation_3d(tree, dilation_radius)

    # Save the tree and dilated tree
    save_tree(tree, ".//src//skeleplex//data//skeleton2.npy")
    save_tree(dilated_tree, ".//src//skeleplex//data//segmentation2.npy")

    # Visualize the chosen z-layer (original and dilated)
    visualize_2d_slice(tree, depth_layer=z_layer)  # Visualize the skeleton
    visualize_2d_slice(dilated_tree, depth_layer=z_layer)  # Visualize the segmentation
