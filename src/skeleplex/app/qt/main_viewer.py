"""Widgets for the main viewer."""

from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from skeleplex.app.qt.flat_group_box import FlatGroupBox


class MainViewerFrame(QFrame):
    """A frame for the main viewer canvas."""

    MINIMUM_HEIGHT = 400

    def __init__(self, canvas_widget: QWidget, parent: QWidget):
        super().__init__(parent=parent)

        self.setStyleSheet("border: 1px solid black;")

        # set the minimum height
        self.setMinimumHeight(self.MINIMUM_HEIGHT)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas_widget)
        self.setLayout(layout)


class ImageControls(FlatGroupBox):
    """Control UI for the skeleton."""

    def __init__(self, parent: QWidget):
        super().__init__(
            title="Image Appearance", accent_color="#92a8d7", parent=parent
        )

        self.add_widget(QLabel("test"))


class SkeletonControls(FlatGroupBox):
    """Control UI for the skeleton."""

    def __init__(self, parent: QWidget):
        super().__init__(
            title="Skeleton Appearance", accent_color="#92a8d7", parent=parent
        )

        self.add_widget(QLabel("test"))


class MainViewerControls(QWidget):
    """A widget for the main viewer controls."""

    def __init__(self, parent: QWidget):
        super().__init__(parent=parent)
        self.skeleton_controls = SkeletonControls(parent=self)
        self.image_controls = ImageControls(parent=self)

        # set the maximum height
        self.setMaximumHeight(100)

        # layout
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_controls)
        layout.addWidget(self.skeleton_controls)
        self.setLayout(layout)


class MainViewerWidget(QWidget):
    """A widget for the main viewer."""

    def __init__(self, canvas_widget: QWidget, parent: QWidget):
        super().__init__(parent=parent)
        self.main_viewer_frame = MainViewerFrame(
            canvas_widget=canvas_widget, parent=self
        )
        self.main_viewer_controls = MainViewerControls(parent=self)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 5, 10, 0)
        layout.addWidget(self.main_viewer_frame)
        layout.addWidget(self.main_viewer_controls)
        self.setLayout(layout)
