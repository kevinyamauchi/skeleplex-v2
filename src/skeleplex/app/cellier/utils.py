"""Viewer controller for the Cellier renderer."""

from cellier.models.data_manager import DataManager
from cellier.models.scene import (
    AxisAlignedRegionSelector,
    Canvas,
    CoordinateSystem,
    DimsManager,
    PerspectiveCamera,
    RangeTuple,
    Scene,
    TrackballCameraController,
)
from cellier.models.viewer import SceneManager, ViewerModel
from cellier.slicer.slicer import SlicerType
from cellier.types import CoordinateSpace
from cellier.viewer_controller import CellierController
from qtpy.QtWidgets import QWidget


def make_viewer_model() -> tuple[ViewerModel, str]:
    """Make the viewer controller."""
    # make the data manager (empty for now)
    data_manager = DataManager(stores={})

    # make the scene coordinate system
    coordinate_system_3d = CoordinateSystem(
        name="scene_3d", axis_labels=("z", "y", "x")
    )
    dims_3d = DimsManager(
        coordinate_system=coordinate_system_3d,
        range=(RangeTuple(0, 100, 1), RangeTuple(0, 100, 1), RangeTuple(0, 100, 1)),
        selection=AxisAlignedRegionSelector(
            space_type=CoordinateSpace.WORLD,
            ordered_dims=(0, 1, 2),
            n_displayed_dims=3,
            index_selection=(slice(None, None), slice(None, None), slice(None, None)),
        ),
    )

    # make the canvas
    controller = TrackballCameraController(enabled=True)
    camera_3d = PerspectiveCamera(controller=controller)
    canvas_3d = Canvas(camera=camera_3d)

    # make the scene
    main_viewer_scene = Scene(
        dims=dims_3d, visuals=[], canvases={canvas_3d.id: canvas_3d}
    )

    scene_manager = SceneManager(scenes={main_viewer_scene.id: main_viewer_scene})

    # make the viewer model
    viewer_model = ViewerModel(data=data_manager, scenes=scene_manager)

    return viewer_model, main_viewer_scene.id


def make_viewer_controller(
    viewer_model: ViewerModel, parent_widget: QWidget
) -> CellierController:
    """Make the viewer controller."""
    return CellierController(
        model=viewer_model,
        slicer_type=SlicerType.ASYNCHRONOUS,
        widget_parent=parent_widget,
    )
