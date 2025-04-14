"""Tests for the skeleplex.graph.image_to_graph_chunked module."""

import dask.array as da
import networkx as nx
import numpy as np

from skeleplex.data import simple_t
from skeleplex.graph.constants import NODE_COORDINATE_KEY
from skeleplex.graph.image_to_graph_chunked import image_to_graph_chunked
from skeleplex.graph.skeleton_processing import label_skeleton
from skeleplex.skeleton.skeletonize_chunked import apply_skeletonization_chunked


def test_labeling_has_unique_ids():
    """Test that all skeleton pixels have unique labels after labeling."""
    skeleton_image = simple_t()
    skeleton_image = da.from_array(skeleton_image)

    skeleton_dask = apply_skeletonization_chunked(skeleton_image)

    skelint = label_skeleton(skeleton_dask)
    labeled = skelint.compute()

    nonzero_labels = labeled[labeled > 0]
    unique_labels = np.unique(nonzero_labels)

    assert len(nonzero_labels) == len(unique_labels)


def test_image_to_graph_dask_structure():
    """Test graph structure produced by image_to_graph_dask using simple_t."""
    skeleton_image = simple_t()
    skeleton_image = da.from_array(skeleton_image)

    skeleton_dask = apply_skeletonization_chunked(skeleton_image)

    graph, _ = image_to_graph_chunked(skeleton_dask)

    # Y-shape should produce 4 nodes and 3 edges
    assert isinstance(graph, nx.MultiGraph)  # Output should be a NetworkX MultiGraph
    assert graph.number_of_nodes() == 4  # Graph should have 4 nodes
    assert graph.number_of_edges() == 3  # Graph should have 3 edges


def test_image_to_graph_dask_attributes():
    """Test that all nodes and edges have required attributes."""
    skeleton_image = simple_t()
    skeleton_image = da.from_array(skeleton_image)

    skeleton_dask = apply_skeletonization_chunked(skeleton_image)

    graph, _ = image_to_graph_chunked(skeleton_dask)

    # Check node attributes
    for node_id, attrs in graph.nodes(data=True):
        assert (
            NODE_COORDINATE_KEY in attrs
        ), f"Node {node_id} is missing coordinate attribute"

    # Check edge attributes
    for u, v, attrs in graph.edges(data=True):
        assert "path" in attrs, f"Edge ({u}, {v}) is missing path attribute"
        assert len(attrs["path"]) > 0, f"Edge ({u}, {v}) has empty path"
