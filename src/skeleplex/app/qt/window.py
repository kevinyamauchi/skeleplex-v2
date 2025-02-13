"""Qt implementation of the main window for the application."""

from app_model import Application
from app_model.backends.qt import QModelMainWindow
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QStatusBar, QWidget

from skeleplex.app.qt import AppControlsDock, AuxiliaryViews
from skeleplex.app.qt.main_viewer import MainViewerWidget

MIN_WINDOW_WIDTH = 1000
MIN_WINDOW_HEIGHT = 600

MAIN_WINDOW_STYLE = """
background-color: white;
QMainWindow {
    background-color: white;
}

QToolBar {
    background: white;
    border: none;
}
"""


class MainWindow(QModelMainWindow):
    """Qt + app-model implementation of the main window for the application."""

    def __init__(self, app: Application):
        super().__init__(app)
        # set the background color
        self.setStyleSheet(MAIN_WINDOW_STYLE)

        # self.tool_bar = self.addModelToolBar(MenuId.FILE, exclude={CommandId.OPEN})
        # self.tool_bar.setStyleSheet("background: white;")

        # set the central widget
        self.setCentralWidget(QLabel("I'm the Central Widget"))

        # set the minimum window size - app will launch with this size.
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        # Create the app controls as a dock widget (left)
        self._create_app_controls()

        # create the auxiliary views as a dock widget (right)
        self._create_auxiliary_views()

        # create the status bar at the bottom of the window
        self._create_status_bar()

    def _create_app_controls(self):
        self.app_controls = AppControlsDock(parent=self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.app_controls)

    def _create_auxiliary_views(self):
        self.auxiliary_views = AuxiliaryViews(parent=self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.auxiliary_views)

    def _create_status_bar(self):
        status = QStatusBar()
        status.showMessage("I'm the Status Bar")
        self.setStatusBar(status)

    def _set_main_viewer_widget(self, canvas_widget: QWidget):
        """Set the main viewer widget."""
        self.setCentralWidget(
            MainViewerWidget(canvas_widget=canvas_widget, parent=self)
        )
