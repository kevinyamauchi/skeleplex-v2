"""QSS styles for the application."""

# style for the flat group box title
# accent_color is the color of the title bar.
# background_color is the color of the group box.
FLAT_TITLE_STYLE = """
QLabel {{
    qproperty-alignment: AlignCenter;
	border: 1px solid {accent_color};
	background-color: {accent_color};
	padding: 5px 0px;
	color: black;
	max-height: 25px;
        font-size: 14px;
}}
"""

FLAT_BUTTON_STYLE = """
QPushButton {{
    qproperty-alignment: AlignCenter;
	border: 1px solid {accent_color};
	background-color: {accent_color};
	padding: 5px 0px;
	color: black;
	max-height: 25px;
        font-size: 14px;
}}
"""

FLAT_FRAME_STYLE = """

QPushButton {
    border: None;
    background-color: #95a7ae;
    padding-left: 5px;
    padding-right: 5px;
    padding-top: 2px;
    padding-bottom: 2px;
}

"""

DOCK_WIDGET_STYLE = """
QDockWidget {
    border: None
}
QDockWidget:title {
    background: #D5D6D7;
}

QDockWidget::close-button, QDockWidget::float-button {
    border: 0px solid transparent;
    background: #D5D6D7;
    padding: 0px;
}

QDockWidget::close-button:hover, QDockWidget::float-button:hover {
    background: #D5D6D7;
}
"""
