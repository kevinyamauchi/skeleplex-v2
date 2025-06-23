"""Example script of launching the viewer and updating the edge colors."""

import numpy as np
from magicgui import magicgui

import skeleplex
from skeleplex.app import view_skeleton
from skeleplex.visualize import EdgeColormap

# path_to_graph = "e13_skeleton_graph_image_skel_clean_new_model_v2.json"
path_to_graph = "../scripts/e16_skeleplex_v2.json"

viewer = view_skeleton(graph_path=path_to_graph)


@magicgui
def randomly_color_edges():
    """Example widget the updates the point coordinates."""
    # make an edge colormap with some random colors
    # the colormap is a dictionary mapping edge keys to RGBA colors
    colormap = {
        edge_key: np.random.rand(4) for edge_key in viewer.data.skeleton_graph.edges
    }
    edge_colormap = EdgeColormap.from_arrays(
        colormap=colormap, default_color=np.array((0.0, 0.0, 0.0, 1.0))
    )

    viewer.data.edge_colormap = edge_colormap


viewer.add_auxiliary_widget(randomly_color_edges.native, name="Color edges")


# start the GUI event loop and block until the application is closed
skeleplex.app.run()
