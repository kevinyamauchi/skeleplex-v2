"""Example of fitting a skeleton graph to a skeleton image."""

from itertools import cycle

import napari
import numpy as np

from skeleplex.data import big_t
from skeleplex.graph import SkeletonGraph

# load an example skeleton image
skeleton_image = big_t()

# construct the skeleton graph
skeleton_graph = SkeletonGraph.from_skeleton_image(skeleton_image)

# get the coordinates of each node
node_coordinates = skeleton_graph.node_coordinates_array

# make the napari viewer
viewer = napari.view_image(skeleton_image)
viewer.add_points(node_coordinates)

# draw the edge splines
sample_points = np.linspace(0, 1, 5, endpoint=True)
colors = cycle(["magenta", "green", "blue", "yellow", "purple"])
for edge_key, edge_spline in skeleton_graph.edge_splines.items():
    # get points along the edge spline
    edge_points = edge_spline.eval(sample_points)

    # add a shapes layer for the edge
    viewer.add_shapes(
        edge_points,
        shape_type="path",
        edge_color=next(colors),
        edge_width=0.2,
        name=f"edge {edge_key}",
    )

viewer.dims.ndisplay = 3


if __name__ == "__main__":
    napari.run()
