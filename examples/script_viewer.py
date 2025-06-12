"""Example script of launching the viewer for a skeleton graph."""

import skeleplex
from skeleplex.app import view_skeleton

path_to_graph = "../scripts/e16_skeleplex_v2.json"

viewer = view_skeleton(graph_path=path_to_graph)

# start the GUI event loop and block until the application is closed
skeleplex.app.run()
