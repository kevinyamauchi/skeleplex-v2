from IPython import get_ipython
from qtpy.QtWidgets import QApplication

from skeleplex.app import DataManager, SkelePlexApp, SkeletonDataPaths

# store reference to QApplication to prevent garbage collection
_app_ref: QApplication | None = None


def view_skeleton(
    graph_path: str,
):
    """Launch the skeleton viewer application.

    Parameters
    ----------
    graph_path : str
        Path to the skeleton graph JSON file.
    """
    return view_skeleton_ipython(graph_path)


def view_skeleton_ipython(
    graph_path: str,
) -> SkelePlexApp:
    """View a skeleton in an IPython environment.

    This works for both jupyter and ipython console environments.

    Parameters
    ----------
    graph_path : str
        Path to the skeleton graph JSON file.

    Returns
    -------
    SkelePlexApp
        The SkelePlex application instance for viewing the skeleton.
    """
    # set the IPython Qt GUI event loop
    ipython = get_ipython()
    ipython.enable_gui("qt")

    # get the qapplication instance
    qapp = QApplication.instance() or QApplication([])

    # load the data
    data_manager = DataManager(file_paths=SkeletonDataPaths(skeleton_graph=graph_path))

    # make the viewer
    viewer = SkelePlexApp(data=data_manager)
    viewer.show()

    # Store reference to prevent garbage collection
    _app_ref = qapp

    return viewer
