"""Script to view the graph representation of a real segmentation.

Supports viewing from an .h5 file or a synthetic bifurcating tree.
"""

import argparse

import h5py
import napari
import numpy as np

from skeleplex.data.bifurcating_tree import generate_tree_3d
from skeleplex.graph.image_to_graph import image_to_graph_skan
from skeleplex.visualize.spline import line_segment_coordinates_from_spline


def load_h5_segmentation(filepath, dataset_name="segmentation"):
    """Load a 3D segmentation from an HDF5 (.h5) file."""
    with h5py.File(filepath, "r") as f:
        segmentation = f[dataset_name][:]

    print(f"Loaded .h5 file: {filepath} | Dataset shape: {segmentation.shape}")
    return segmentation


# Argument parser for selecting input type
parser = argparse.ArgumentParser(
    description="View graph representation of either .h5 segmentation "
    "or synthetic tree."
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
    nodes = 1
    branch_length = 30
    z_layer = 10
    left_angle = 40
    right_angle = 60
    length_ratio = 0.8

    segmented_tree = generate_tree_3d(
        shape, nodes, branch_length, z_layer, left_angle, right_angle, length_ratio
    )

# Convert the skeleton image to a graph
skeleton_graph = image_to_graph_skan(segmented_tree)

# Extract node positions
node_positions = np.array(
    [skeleton_graph.nodes[n]["node_coordinate"] for n in skeleton_graph.nodes]
)

# Extract edges using splines
edges = []
for _u, _v, edge_data in skeleton_graph.edges(data=True):
    if "spline" in edge_data:
        spline = edge_data["spline"]
        edge_segments = line_segment_coordinates_from_spline(spline, n_line_segments=30)

        # Ensure correct shape (N, 2, 3)
        edge_pairs = edge_segments.reshape(-1, 2, 3)
        edges.extend(edge_pairs)

edges = np.array(edges)  # Convert to NumPy array


# Visualize in Napari
viewer = napari.Viewer()

viewer.add_image(
    segmented_tree, name="Segmented Image", colormap="gray"
)  # Original segmentation

viewer.add_points(node_positions, name="Graph Nodes", size=5, face_color="red")  # Nodes

viewer.add_shapes(
    edges, shape_type="line", name="Graph Edges", edge_color="blue"
)  # Edges

if __name__ == "__main__":
    napari.run()
