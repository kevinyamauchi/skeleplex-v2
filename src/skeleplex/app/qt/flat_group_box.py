"""A flat-styled group box widget."""

from qtpy.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from skeleplex.app.qt.styles import FLAT_FRAME_STYLE, FLAT_TITLE_STYLE


class FlatFrame(QFrame):
    """A flat-styled frame widget.

    Parameters
    ----------
    parent : QWidget | None
        The parent widget. The default is None.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)

        # set the styling
        self.setStyleSheet(FLAT_FRAME_STYLE)

        self.setLayout(QVBoxLayout())

    def add_widget(self, widget: QWidget):
        """Add a widget to the frame.

        Parameters
        ----------
        widget : QWidget
            The widget to add.
        """
        self.layout().addWidget(widget)


class FlatGroupBox(QWidget):
    """A flat-styled group box widget.

    Parameters
    ----------
    title : str
        The title of the group box.
        The default is "".
    accent_color : str
        The accent color for the group box. This is
        used for the title bar and other accents.
        Color should be a hex string.
        The default is "#b7e2d8".
    background_color : str
        The background color for the group box.
        Color should be a hex string.
        The default is "#f3f3f3".
    parent : QWidget | None
        The parent widget. The default is None.
    """

    def __init__(
        self,
        title: str = "",
        accent_color: str = "#b7e2d8",
        background_color: str = "#f3f3f3",
        parent: QWidget | None = None,
    ):
        super().__init__(parent=parent)

        # set the background color
        self.setStyleSheet(f"background-color: {background_color};")

        self.title_widget = QLabel(title, parent=self)
        self.title_widget.setStyleSheet(
            FLAT_TITLE_STYLE.format(accent_color=accent_color)
        )
        self.frame = FlatFrame(parent=self)

        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.title_widget)
        layout.addWidget(self.frame)
        layout.addStretch()
        self.setLayout(layout)

    def add_widget(self, widget: QWidget):
        """Add a widget to the group box.

        Parameters
        ----------
        widget : QWidget
            The widget to add.
        """
        self.frame.add_widget(widget)
