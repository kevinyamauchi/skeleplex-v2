"""Example script of launching the viewer and adding points to it."""

import numpy as np
from cellier.models.data_stores import PointsMemoryStore
from cellier.models.visuals import PointsUniformAppearance, PointsVisual
from magicgui import magicgui

import skeleplex
from skeleplex.app import view_skeleton

# path_to_graph = "e13_skeleton_graph_image_skel_clean_new_model_v2.json"
path_to_graph = "../scripts/e16_skeleplex_v2.json"

viewer = view_skeleton(graph_path=path_to_graph)


def add_points_to_viewer():
    """Add points to the viewer."""
    # create a list of points to add
    # note these must be Float32
    point_coordinates = np.array(
        [
            [1000, 1000, 1000],
            [1500, 1500, 1500],
        ],
        dtype=np.float32,
    )

    # make the data store for the points
    new_points_store = PointsMemoryStore(
        coordinates=point_coordinates,
    )

    # set up the points appearance
    points_appearance = PointsUniformAppearance(
        size=50, color=(0, 1, 0, 1), size_coordinate_space="data"
    )

    # make the highlight points model
    points_visual = PointsVisual(
        name="node_highlight_points",
        data_store_id=new_points_store.id,
        appearance=points_appearance,
    )

    # add the data and visual to the viewer backend (cellier)
    viewer._viewer._backend.add_data_store(data_store=new_points_store)
    viewer._viewer._backend.add_visual(
        visual_model=points_visual, scene_id=viewer._viewer._main_canvas.scene_id
    )

    # reslice the viewer to update the display
    viewer._viewer._backend.reslice_all()
    return points_visual, new_points_store


points_visual, points_store = add_points_to_viewer()

# set the points visibility to False
points_visual.appearance.visible = False


@magicgui
def update_points():
    """Example widget the updates the point coordinates."""
    # make new point coordinates
    new_point_coordinates = np.random.uniform(0, 5000, (100, 3)).astype(np.float32)

    # set the new coordinates in the points store
    points_store.coordinates = new_point_coordinates

    # set the points visibility to True
    points_visual.appearance.visible = True

    # reslice the viewer to update the display
    viewer._viewer._backend.reslice_all()


viewer.add_auxiliary_widget(update_points.native, name="Update points")


# start the GUI event loop and block until the application is closed
skeleplex.app.run()
