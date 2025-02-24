"""Test for 3D skeletonization using a bifurcating tree."""

import numpy as np

from skeleplex.data.bifurcating_tree import apply_dilation_3d, generate_tree_3d
from skeleplex.skeleton.skeletonize import apply_skeletonization_3d


def test_skeletonize():
    """Test skeletonization on a generated bifurcating tree."""

    # Generate a synthetic bifurcating tree
    shape = (100, 100, 100)
    tree = generate_tree_3d(shape, nodes=3, branch_length=30, z_layer=10)

    # Apply morphological dilation to simulate segmentation
    segmented_tree = apply_dilation_3d(tree, dilation_radius=3)

    # Apply skeletonization
    skeleton = apply_skeletonization_3d(segmented_tree)

    # Check if output is a NumPy array
    assert isinstance(skeleton, np.ndarray), "Output should be a NumPy array"

    # Check if shape remains unchanged
    assert (
        skeleton.shape == segmented_tree.shape
    ), "Output shape should match input shape"

    # Check that the skeleton is not empty
    assert np.any(skeleton), "Skeletonization failed: skeleton is empty"

    # Check that the skeleton has fewer nonzero pixels than the segmentation
    assert np.count_nonzero(skeleton) < np.count_nonzero(
        segmented_tree
    ), "Skeleton should be a thinner version of the input segmentation"
