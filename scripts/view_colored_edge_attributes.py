from io import BytesIO  # noqa
import napari
from skeleplex.data.skeletons import generate_toy_skeleton_graph_symmetric_branch_angle
from skeleplex.measurements.graph_properties import compute_level
from PyQt5.QtWidgets import QDoubleSpinBox, QHBoxLayout

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget, QPushButton

from skeleplex.graph.constants import (
    EDGE_SPLINE_KEY,
    GENERATION_KEY,
)
import numbers
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
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Change the color of the edges in the skeleton viewer based on attribute."""
    current_layer = next(iter(viewer.layers.selection)).name

    color_dict = nx.get_edge_attributes(skeleton.graph, edge_attribute)
    for key, value in color_dict.items():
        if not value:
            color_dict[key] = np.nan

    if vmin is None:
        vmin = np.nanmin(list(color_dict.values()))
    if vmax is None:
        vmax = np.nanmax(list(color_dict.values()))

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
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


def filter_edges_by_attribute(
    viewer,
    skeleton: SkeletonGraph,
    edge_attribute: str,
    cmap=plt.cm.viridis,
    levels: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Filter edges based on a specific attribute and its value range.

    Its better to just render them transparent
    """
    current_layer = next(iter(viewer.layers.selection)).name

    color_dict = nx.get_edge_attributes(skeleton.graph, edge_attribute)
    for key, value in color_dict.items():
        if not value:
            color_dict[key] = np.nan

    if vmin is None:
        vmin = np.nanmin(list(color_dict.values()))
    if vmax is None:
        vmax = np.nanmax(list(color_dict.values()))

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    color_dict_hex = {k: mcolors.rgb2hex(cmap(norm(v))) for k, v in color_dict.items()}

    generation_dict = nx.get_edge_attributes(skeleton.graph, GENERATION_KEY)
    if not levels:
        levels = max(generation_dict.values())

    edges_to_remove = [
        edge
        for edge, value in nx.get_edge_attributes(
            skeleton.graph, edge_attribute
        ).items()
        if not isinstance(value, numbers.Number) or not (vmin <= value <= vmax)
    ]

    color_list = []
    for edge in skeleton.graph.edges:
        if generation_dict[edge] > levels:
            continue
        if edge in edges_to_remove:
            # Render edges that are filtered out as transparent
            edge_color = "#00000000"
        else:
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
            vmin=np.nanmin(list(color_dict.values())),
            vmax=np.nanmax(list(color_dict.values())),
        )
        # Map each float value to a hex color
        color_dict_hex = {
            k: mcolors.rgb2hex(self.cmap(norm(v))) for k, v in color_dict.items()
        }
        generation_dict = nx.get_edge_attributes(self.skeleton.graph, GENERATION_KEY)
        # max generation

        shapes = []
        color_list = []
        start_nodes = []
        end_nodes = []
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
            start_nodes.append(edge[0])
            end_nodes.append(edge[1])
        # edge_ids = np.array(edge_ids)
        self.viewer.add_shapes(
            shapes,
            edge_color=color_list,
            shape_type="path",
            name="edges",
            edge_width=self.edge_width,
            properties={
                "start_node": start_nodes,
                "end_node": end_nodes,
            },
        )


class ChangeBranchColorWidget(QWidget):
    """Widget to change the color of the edges in the skeleton viewer."""

    def __init__(self, skeleton_viewer: SkeletonViewer):
        super().__init__()
        self.skeleton_viewer = skeleton_viewer
        self.min_spin = None
        self.max_spin = None
        self.current_attr = None
        self.initUI()

    def initUI(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        self.label = QLabel("Select Edge Attribute for Coloring:")
        layout.addWidget(self.label)

        self.comboBox = QComboBox()
        self.comboBox.addItems(self.get_edge_attributes())
        self.comboBox.currentTextChanged.connect(self._on_attribute_change)
        layout.addWidget(self.comboBox)

        # Min/Max Controls
        self.min_spin = QDoubleSpinBox()
        self.max_spin = QDoubleSpinBox()
        self.min_spin.setDecimals(2)
        self.max_spin.setDecimals(2)
        self.min_spin.setSingleStep(0.1)
        self.max_spin.setSingleStep(0.1)

        self.min_spin.valueChanged.connect(self._update_coloring)
        self.max_spin.valueChanged.connect(self._update_coloring)

        minmax_layout = QHBoxLayout()
        minmax_layout.addWidget(QLabel("Min:"))
        minmax_layout.addWidget(self.min_spin)
        minmax_layout.addWidget(QLabel("Max:"))
        minmax_layout.addWidget(self.max_spin)
        layout.addLayout(minmax_layout)

        self.colorbar_label = QLabel()
        layout.addWidget(self.colorbar_label)

        if self.comboBox.count() > 0:
            self._on_attribute_change(self.comboBox.currentText())

        self.filter_button = QPushButton("Filter Branches")
        self.filter_button.clicked.connect(self._update_rendering)

        layout.addWidget(self.filter_button)

        self.setLayout(layout)

    def get_edge_attributes(self):
        """Get the edge attributes from the skeleton graph."""
        if not self.skeleton_viewer.skeleton.graph.edges:
            return []
        attribute_set = set()
        for _, _, edge_data in self.skeleton_viewer.skeleton.graph.edges(data=True):
            attribute_set.update(edge_data.keys())
        return list(attribute_set)

    def _on_attribute_change(self, attribute_name):
        self.current_attr = attribute_name
        values = [
            v
            for v in nx.get_edge_attributes(
                self.skeleton_viewer.skeleton.graph, attribute_name
            ).values()
            if isinstance(v, numbers.Number) and not np.isnan(v)
        ]

        if not values:
            return

        min_val, max_val = np.nanmin(values), np.nanmax(values)

        self.min_spin.setMinimum(min_val)
        self.min_spin.setMaximum(max_val)
        self.min_spin.setValue(min_val)

        self.max_spin.setMinimum(min_val)
        self.max_spin.setMaximum(max_val)
        self.max_spin.setValue(max_val)

        self._update_coloring()

    def _update_coloring(self):
        min_val = self.min_spin.value()
        max_val = self.max_spin.value()

        change_color_attr(
            self.skeleton_viewer.viewer,
            self.skeleton_viewer.skeleton,
            edge_attribute=self.current_attr,
            cmap=self.skeleton_viewer.cmap,
            levels=self.skeleton_viewer.level_depth,
            vmin=min_val,
            vmax=max_val,
        )

        self._update_colorbar(self.current_attr, min_val, max_val)

    def _update_colorbar(self, attribute_name, vmin, vmax):
        cmap = self.skeleton_viewer.cmap
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(figsize=(4, 0.4))
        fig.subplots_adjust(bottom=0.5)

        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation="horizontal",
        )
        cbar.ax.set_xlabel(f"{attribute_name} (Min: {vmin:.2f}, Max: {vmax:.2f})")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)

        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        self.colorbar_label.setPixmap(pixmap)

    def _update_rendering(self):
        min_val = self.min_spin.value()
        max_val = self.max_spin.value()

        # Filter edges based on the selected attribute and its value range.
        filter_edges_by_attribute(
            self.skeleton_viewer.viewer,
            self.skeleton_viewer.skeleton,
            edge_attribute=self.current_attr,
            cmap=self.skeleton_viewer.cmap,
            levels=self.skeleton_viewer.level_depth,
            vmin=min_val,
            vmax=max_val,
        )
        self._update_colorbar(self.current_attr, min_val, max_val)


if __name__ == "__main__":
    # Visualize an example skeleton graph
    skeleton_graph = generate_toy_skeleton_graph_symmetric_branch_angle(19, 27, 20)
    skeleton_graph.graph = compute_level(skeleton_graph.graph, origin=-1)

    viewer = napari.Viewer()
    skeleton_viewer = SkeletonViewer(skeleton_graph, viewer)
    viewer.window.add_dock_widget(ChangeBranchColorWidget(skeleton_viewer))
