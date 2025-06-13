"""Example script of launching the viewer for a skeleton graph."""

from magicgui import magicgui
from qtpy.QtWidgets import QDockWidget

import skeleplex
from skeleplex.app import view_skeleton

path_to_graph = "e13_skeleton_graph_image_skel_clean_new_model_v2.json"

viewer = view_skeleton(graph_path=path_to_graph)


undo_widget = magicgui(viewer.curate.undo)
test_dock = QDockWidget("Undo")
test_dock.setWidget(undo_widget.native)
viewer.add_auxiliary_widget(test_dock)

edge_deletion_widget = magicgui(
    viewer.curate.delete_edge,
)
dock_widget = QDockWidget("Delete edge")
dock_widget.setWidget(edge_deletion_widget.native)
viewer.add_auxiliary_widget(dock_widget)

# start the GUI event loop and block until the application is closed
skeleplex.app.run()
