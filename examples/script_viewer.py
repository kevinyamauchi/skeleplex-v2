"""Example script of launching the viewer for a skeleton graph."""

from magicgui import magicgui

import skeleplex
from skeleplex.app import view_skeleton

path_to_graph = "e13_skeleton_graph_image_skel_clean_new_model_v2.json"

viewer = view_skeleton(graph_path=path_to_graph)

# make the undo widget
undo_widget = magicgui(viewer.curate.undo)
viewer.add_auxiliary_widget(undo_widget.native, name="Undo")

# make the edge deletion widget
edge_deletion_widget = magicgui(
    viewer.curate.delete_edge,
)
viewer.add_auxiliary_widget(edge_deletion_widget.native, name="Delete edge")

# start the GUI event loop and block until the application is closed
skeleplex.app.run()
