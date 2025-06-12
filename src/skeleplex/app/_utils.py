from IPython import get_ipython
from qtpy.QtCore import QTimer
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

    Returns
    -------
    SkelePlexApp
        The SkelePlex application instance for viewing the skeleton.
    """
    global _app_ref

    # get the qapplication instance
    qapp = QApplication.instance() or QApplication([])

    # Store reference to prevent garbage collection
    _app_ref = qapp

    # load the data
    data_manager = DataManager(file_paths=SkeletonDataPaths(skeleton_graph=graph_path))

    # make the viewer
    viewer = SkelePlexApp(data=data_manager)
    viewer.show()

    # Wait a short time for things to load and then look at the skeleton
    # this is a hack...do something smarter later
    timer = QTimer()
    timer.singleShot(100, viewer.look_at_skeleton)

    # start the Qt event loop if in Jupyter/IPython
    if should_launch_ipython_event_loop():
        start_qt_loop_ipython()

    return viewer


def start_qt_loop_ipython():
    """Start the Qt event loop in an IPython environment.

    This works for both jupyter and ipython console environments.
    """
    ipython = get_ipython()
    ipython.enable_gui("qt")


def should_launch_ipython_event_loop() -> bool:
    """
    Check if the IPython Qt event loop should be launched.

    This means that we are both in an IPython environment and the
    event loop is not already running.

    Returns
    -------
    bool
        True if running in IPython and the loop is needed.
        False otherwise.
    """
    shell = get_ipython()

    if not shell:
        # not in IPython environment
        return False

    return not shell.active_eventloop == "qt"


def run():
    """Start the Qt application event loop.

    This is meant to be used in a script.
    This should be called after the viewer is set up.
    """
    # get the qapplication instance
    qapp = QApplication.instance() or QApplication([])

    # start the Qt application event loop
    qapp.exec_()
