"""Script to view the bifurcating tree skeleton image and its graph representation."""

import napari
import numpy as np

from skeleplex.data.bifurcating_tree import generate_tree_3d
from skeleplex.graph.image_to_graph import image_to_graph_skan

# Parameters for the bifurcating tree
shape = (100, 100, 100)
nodes = 3
branch_length = 30
z_layer = 10
left_angle = 40
right_angle = 60
length_ratio = 0.8


# Generate the bifurcating tree skeleton
skeleton_image = generate_tree_3d(
    shape, nodes, branch_length, z_layer, left_angle, right_angle, length_ratio
)

# Convert the skeleton image to a graph
skeleton_graph = image_to_graph_skan(skeleton_image)

# Print node attributes to check available keys
for n in skeleton_graph.nodes:
    print(f"Node {n} attributes: {skeleton_graph.nodes[n]}")


# Extract node positions
node_positions = np.array(
    [skeleton_graph.nodes[n]["node_coordinate"] for n in skeleton_graph.nodes]
)

# Extract edges
edges = []
for u, v, _ in skeleton_graph.edges:
    pos_u = skeleton_graph.nodes[u]["node_coordinate"]
    pos_v = skeleton_graph.nodes[v]["node_coordinate"]
    edges.append([pos_u, pos_v])  # Each edge is a line between two nodes

edges = np.array(edges)  # Convert to NumPy array for Napari

# Visualize in Napari
viewer = napari.Viewer()
viewer.add_image(skeleton_image, name="Bifurcating Tree Skeleton")  # Original skeleton
viewer.add_points(node_positions, name="Graph Nodes", size=5, face_color="red")  # Nodes
viewer.add_shapes(
    edges, shape_type="line", name="Graph Edges", edge_color="blue"
)  # Edges

if __name__ == "__main__":
    napari.run()
