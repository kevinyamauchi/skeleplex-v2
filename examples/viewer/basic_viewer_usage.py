"""Example script of launching the viewer for a skeleton graph."""

import skeleplex
from skeleplex.app import view_skeleton

# path_to_graph = "../e13_skeleton_graph_image_skel_clean_new_model_v2.json"
# path_to_graph = "../../scripts/e16_skeleplex_v2.json"
path_to_graph = "../example_data/skeleton_graph.json"
path_to_segmentation = "../example_data/bifurcating_tree.zarr"

viewer = view_skeleton(
    graph_path=path_to_graph,
    segmentation_path=path_to_segmentation,
    segmentation_voxel_size_um=(1.0, 1.0, 1.0),
)

# start the GUI event loop and block until the application is closed
skeleplex.app.run()
