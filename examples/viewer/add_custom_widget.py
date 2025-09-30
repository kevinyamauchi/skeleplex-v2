"""Example script of launching the viewer for a skeleton graph."""

import sys

from magicgui import magicgui

import skeleplex
from skeleplex.app import view_skeleton
from skeleplex.graph.constants import NODE_COORDINATE_KEY

# path_to_graph = "e13_skeleton_graph_image_skel_clean_new_model_v2.json"
path_to_graph = "../scripts/e16_skeleplex_v2.json"

viewer = view_skeleton(graph_path=path_to_graph)


# we can also create widgets with new functions using
# the instantiated viewer
@magicgui(
    node_id={"max": sys.float_info.max},
    bounding_box_width={"min": 0, "max": sys.float_info.max},
)
def render_around_node(node_id: int, bounding_box_width: int = 100):
    """Render a bounding box around the specified node.

    Parameters
    ----------
    node_id : int
        The ID of the node to render around.
    bounding_box_width : int
        The width of the bounding box to render around the node.
        Default is 100.
    """
    # get the coordinate of the node
    graph_object = viewer.data.skeleton_graph.graph
    node_coordinate = graph_object.nodes[node_id][NODE_COORDINATE_KEY]

    # get the minimum and maximum coordinates for the bounding box
    half_width = bounding_box_width / 2
    min_coordinate = node_coordinate - half_width
    max_coordinate = node_coordinate + half_width

    # set the bounding box in the viewer
    viewer.data.view.bounding_box._min_coordinate = min_coordinate
    viewer.data.view.bounding_box._max_coordinate = max_coordinate

    # set the render mode to bounding box
    viewer.data.view.mode = "bounding_box"


viewer.add_auxiliary_widget(render_around_node.native, name="Render around node")


# start the GUI event loop and block until the application is closed
skeleplex.app.run()
