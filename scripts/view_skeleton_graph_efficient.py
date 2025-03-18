"""Extract the skeleton graph from a 3D skeleton and visualize it in Napari."""

import dask.array as da
import matplotlib.pyplot as plt
import napari
import numpy as np

from skeleplex.data.bifurcating_tree import generate_tree_3d
from skeleplex.graph.constants import NODE_COORDINATE_KEY
from skeleplex.graph.image_to_graph_efficient import image_to_graph_dask
from skeleplex.skeleton.skeletonize_efficient import apply_skeletenization_efficient

# Generate synthetic tree
tree = generate_tree_3d(
    shape=(200, 200, 200),
    nodes=3,
    length_ratio=0.8,
    left_angle=30,
    right_angle=30,
    branch_length=50,
)

dask_tree = da.from_array(tree)

# Apply skeletonization
skel = apply_skeletenization_efficient(dask_tree, factor=(2, 2, 2))

# Generate the graph and get skeleton image
skeleton_graph, skelint = image_to_graph_dask(skel=skel)

# Visualize skeleton slice
plt.imshow(skelint[10, :, :].compute(), cmap="viridis")
plt.colorbar(label="Degree Value")
plt.title("Skeleton Degrees Visualization")
plt.show()

# Ensure skelint is a NumPy array
skelint_np = skelint.compute() if not isinstance(skelint, np.ndarray) else skelint

# Extract node coordinates
node_positions = np.array(
    [data[NODE_COORDINATE_KEY] for _, data in skeleton_graph.nodes(data=True)]
)

# Extract edges as line segments
edges = [data["path"] for _, _, data in skeleton_graph.edges(data=True)]

# Open Napari viewer
viewer = napari.Viewer()
viewer.dims.ndisplay = 3  # Enables 3D visualization

# Add original skeleton image
viewer.add_image(
    skelint_np, colormap="gray", blending="additive", name="Skeleton Image"
)

# Add nodes as points
viewer.add_points(node_positions, size=2, face_color="red", name="Graph Nodes")

# Add edges as line segments
viewer.add_shapes(edges, shape_type="path", edge_color="blue", name="Graph Edges")

napari.run()
