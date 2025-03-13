from io import BytesIO  # noqa
import napari
from skeleplex.data.skeletons import generate_toy_skeleton_graph_symmetric_branch_angle
from skeleplex.measurements.graph_properties import compute_level

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from skeleplex.graph.constants import (
    EDGE_SPLINE_KEY,
    GENERATION_KEY,
)
from skeleplex.graph.skeleton_graph import SkeletonGraph
import logging

logging.getLogger().setLevel(logging.CRITICAL)
# redraw current layer to update the color of the edges


def change_color_attr(
    viewer,
    skeleton: SkeletonGraph,
    edge_attribute: str,
    cmap=plt.cm.viridis,
    levels: int | None = None,
):
    """Change the color of the edges based on an attribute."""
    current_layer = next(iter(viewer.layers.selection)).name

    color_dict = nx.get_edge_attributes(skeleton.graph, edge_attribute)
    norm = plt.Normalize(vmin=min(color_dict.values()), vmax=max(color_dict.values()))
    # Map each float value to a hex color
    color_dict_hex = {k: mcolors.rgb2hex(cmap(norm(v))) for k, v in color_dict.items()}

    generation_dict = nx.get_edge_attributes(skeleton.graph, GENERATION_KEY)

    if not levels:
        levels = max(generation_dict.values())

    color_list = []
    for edge in skeleton.graph.edges:
        if generation_dict[edge] > levels:
            continue
        edge_color = color_dict_hex.get(edge, "#FFFFFF")

        color_list.append(edge_color)

    viewer.layers[current_layer].edge_color = color_list
    viewer.layers[current_layer].refresh()


class SkeletonViewer:
    """Class to visualize a skeleton graph in napari."""

    def __init__(
        self,
        skeleton: SkeletonGraph,
        viewer=None,
        edge_width: int = 4,
        level_depth: int | None = None,
        num_samples: int = 5,
        edge_color_attr: str = GENERATION_KEY,
    ):
        self.skeleton = skeleton

        if viewer is None:
            self.viewer = napari.Viewer()
        self.viewer = viewer

        if level_depth is None:
            level_depth = max(
                nx.get_edge_attributes(self.skeleton.graph, GENERATION_KEY).values()
            )
        self.level_depth = level_depth
        self.edge_width = edge_width
        self.num_samples = num_samples
        self.sample_points = np.linspace(0.01, 0.99, self.num_samples, endpoint=True)
        self.edge_color_attr = edge_color_attr
        self.cmap = plt.cm.viridis

        self._initialize_viewer()

    def _initialize_viewer(self):
        color_dict = nx.get_edge_attributes(self.skeleton.graph, self.edge_color_attr)
        norm = plt.Normalize(
            vmin=min(color_dict.values()), vmax=max(color_dict.values())
        )
        # Map each float value to a hex color
        color_dict_hex = {
            k: mcolors.rgb2hex(self.cmap(norm(v))) for k, v in color_dict.items()
        }
        generation_dict = nx.get_edge_attributes(self.skeleton.graph, GENERATION_KEY)
        # max generation

        shapes = []
        color_list = []
        for edge in self.skeleton.graph.edges:
            if generation_dict[edge] > self.level_depth:
                continue
            edge_color = color_dict_hex.get(edge, "#FFFFFF")
            spline = self.skeleton.graph.edges[edge][EDGE_SPLINE_KEY]
            try:
                eval_points = spline.eval(self.sample_points, atol=0.1, approx=True)
            except ValueError:
                eval_points = spline.eval(
                    np.linspace(0.01, 0.99, 4), atol=0.1, approx=True
                )
            shapes.append(eval_points)

            color_list.append(edge_color)

        self.viewer.add_shapes(
            shapes, edge_color=color_list, shape_type="path", name="edges", edge_width=4
        )


class ChangeBranchColorWidget(QWidget):
    """Widget to change the color of the edges based on an attribute."""

    def __init__(self, skeleton_viewer: SkeletonViewer):
        super().__init__()
        self.skeleton_viewer = skeleton_viewer
        self.initUI()

    def initUI(self):
        """Initialize the widget layout."""
        layout = QVBoxLayout()

        self.label = QLabel("Select Edge Attribute for Coloring:")
        layout.addWidget(self.label)

        self.comboBox = QComboBox()
        self.comboBox.addItems(self.get_edge_attributes())
        self.comboBox.currentTextChanged.connect(self._change_edge_color)
        layout.addWidget(self.comboBox)

        self.colorbar_label = QLabel()  # Label to display colormap image
        layout.addWidget(self.colorbar_label)

        self.setLayout(layout)

        # Initialize with the first attribute if available
        if self.comboBox.count() > 0:
            self._change_edge_color(GENERATION_KEY)

    def get_edge_attributes(self):
        """Retrieve all edge attributes stored in the skeleton graph."""
        if not self.skeleton_viewer.skeleton.graph.edges:
            return []

        # Get all attributes from all edges
        attribute_set = set()
        for _, _, edge_data in self.skeleton_viewer.skeleton.graph.edges(data=True):
            attribute_set.update(edge_data.keys())

        return list(attribute_set)

    def _change_edge_color(self, attribute_name):
        """Update the edge colors based on the selected attribute."""
        change_color_attr(
            self.skeleton_viewer.viewer,
            self.skeleton_viewer.skeleton,
            edge_attribute=attribute_name,
            cmap=self.skeleton_viewer.cmap,
            levels=self.skeleton_viewer.level_depth,
        )

        self.update_colorbar(attribute_name)

    def update_colorbar(self, attribute_name):
        """Generate and update the colormap image with actual min/max values."""
        edge_values = nx.get_edge_attributes(
            self.skeleton_viewer.skeleton.graph, attribute_name
        ).values()

        if not edge_values:
            return  # Skip if no values found

        min_val, max_val = min(edge_values), max(edge_values)
        cmap = self.skeleton_viewer.cmap
        norm = plt.Normalize(vmin=min_val, vmax=max_val)

        fig, ax = plt.subplots(figsize=(4, 0.4))  # Create colorbar figure
        fig.subplots_adjust(bottom=0.5)

        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation="horizontal",
        )
        cbar.ax.set_xlabel(f"{attribute_name} (Min: {min_val:.2f}, Max: {max_val:.2f})")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)

        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())

        self.colorbar_label.setPixmap(pixmap)  # Update QLabel with colormap


# Visualize an example skeleton graph
skeleton_graph = generate_toy_skeleton_graph_symmetric_branch_angle(19, 27, 20)
skeleton_graph.graph = compute_level(skeleton_graph.graph, origin=-1)

viewer = napari.Viewer()
skeleton_viewer = SkeletonViewer(skeleton_graph, viewer)
viewer.window.add_dock_widget(ChangeBranchColorWidget(skeleton_viewer))
