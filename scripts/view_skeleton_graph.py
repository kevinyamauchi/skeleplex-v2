"""Convert a 3D bifurcating tree into a graph, and visualize it in Napari."""

import napari
import numpy as np

from skeleplex.data.bifurcating_tree import generate_tree_3d
from skeleplex.graph.image_to_graph import image_to_graph_skan
from skeleplex.visualize.spline import line_segment_coordinates_from_spline

tree = generate_tree_3d(
    shape=(200, 200, 200),
    nodes=3,
    length_ratio=0.8,
    left_angle=30,
    right_angle=30,
    branch_length=50,
)


# Convert the skeleton image to a graph
skeleton_graph = image_to_graph_skan(tree)

# Extract node positions
node_positions = np.array(
    [skeleton_graph.nodes[n]["node_coordinate"] for n in skeleton_graph.nodes]
)

# Extract edges
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
viewer.dims.ndisplay = 3  # Enables 3D visualization

viewer.add_image(tree, name="Tree", colormap="gray")  # Original tree

viewer.add_points(node_positions, name="Graph Nodes", size=5, face_color="red")  # Nodes

viewer.add_shapes(
    edges, shape_type="line", name="Graph Edges", edge_color="blue"
)  # Edges


napari.run()
