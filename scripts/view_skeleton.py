"""Script to view the skeletonization of a real segmentation.

Supports viewing from an .h5 file or a synthetic bifurcating tree.
"""

import argparse

import h5py
import napari

from skeleplex.data.bifurcating_tree import apply_dilation_3d, generate_tree_3d
from skeleplex.skeleton.skeletonize import apply_skeletonization


def load_h5_segmentation(filepath, dataset_name="segmentation"):
    """Load a 3D segmentation from an HDF5 (.h5) file."""
    with h5py.File(filepath, "r") as f:
        segmentation = f[dataset_name][:]

    print(f"Loaded .h5 file: {filepath} | Dataset shape: {segmentation.shape}")
    return segmentation


# Argument parser for selecting input type
parser = argparse.ArgumentParser(
    description="View skeletonization of either .h5 " "segmentation or synthetic tree."
)
parser.add_argument("--h5", type=str, help="Path to an .h5 segmentation file.")
args = parser.parse_args()


if args.h5:
    # Load segmentation from .h5 file
    segmented_tree = load_h5_segmentation(args.h5)
else:
    # Generate synthetic bifurcating tree (if no .h5 file is provided)
    print("No .h5 file provided. Generating synthetic bifurcating tree...")
    shape = (100, 100, 100)
    nodes = 4
    branch_length = 25
    z_layer = 10
    left_angle = 40
    right_angle = 60
    length_ratio = 0.8
    dilation_radius = 3

    segmented_tree = generate_tree_3d(
        shape, nodes, branch_length, z_layer, left_angle, right_angle, length_ratio
    )

    segmented_tree = apply_dilation_3d(segmented_tree, dilation_radius)

# Apply skeletonization
skeletonized_tree = apply_skeletonization(segmented_tree)

# Open Napari Viewer
viewer = napari.Viewer()
viewer.add_image(segmented_tree, name="Segmented Image", colormap="gray")
viewer.add_image(skeletonized_tree, name="Skeletonized Image", colormap="blue")

if __name__ == "__main__":
    napari.run()
