"""Tests for the bifurcating tree generator."""

import numpy as np

from skeleplex.data.bifurcating_tree import apply_dilation_3d, generate_tree_3d


def test_generate_tree_3d():
    """Test the generation of bifurcating trees."""

    # Create a basic bifurcating tree
    shape = (100, 100, 100)
    num_bifurcations = 3  # Number of bifurcation levels
    branch_length = 20
    z_layer = 10

    tree = generate_tree_3d(shape, num_bifurcations, branch_length, z_layer)

    # Check if output is a NumPy array
    assert isinstance(tree, np.ndarray), "Output should be a NumPy array"

    # Check the dimensions
    assert tree.shape == shape, "Generated tree should match the input shape"

    # Ensure that at the tree exists
    assert tree.sum() > 0, "Tree should not be empty"

    # Ensure that the tree exists **only in one z-layer**
    nonzero_layers = np.count_nonzero(tree.sum(axis=(1, 2)))
    assert nonzero_layers == 1, "Tree should be contained within a single z-layer"


def test_apply_dilation_3d():
    """Test the morphological dilation function."""

    # Create a small synthetic binary image
    image = np.zeros((20, 20, 20), dtype=bool)
    image[10, 10, 10] = 1  # Single voxel in the middle

    # Apply dilation
    dilated_image = apply_dilation_3d(image, dilation_radius=3)

    # Check if output is a NumPy array
    assert isinstance(
        dilated_image, np.ndarray
    ), "Dilation output should be a NumPy array"

    # Ensure dilation increases the number of nonzero pixels
    assert dilated_image.sum() > image.sum(), "Dilation should increase the object size"

    # Ensure that dilation retains the original nonzero points
    assert image[10, 10, 10] == 1, "Original center voxel should still be present"
