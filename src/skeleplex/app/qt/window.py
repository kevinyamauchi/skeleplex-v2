"""Qt implementation of the main window for the application."""

from app_model import Application
from app_model.backends.qt import QModelMainWindow
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QStatusBar

from skeleplex.app.constants import CommandId, MenuId
from skeleplex.app.qt import AppControlsDock, AuxiliaryViews

MIN_WINDOW_WIDTH = 1000
MIN_WINDOW_HEIGHT = 600


class MainWindow(QModelMainWindow):
    """Qt + app-model implementation of the main window for the application."""

    def __init__(self, app: Application):
        super().__init__(app)
        self.addModelToolBar(MenuId.FILE, exclude={CommandId.OPEN})

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
