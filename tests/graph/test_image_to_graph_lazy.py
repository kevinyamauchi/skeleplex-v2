"""Test for the skeleplex.graph.image_to_graph_lazy functions."""

import dask.array as da
import networkx as nx
import numpy as np

from skeleplex.data.bifurcating_tree import generate_tree_3d
from skeleplex.graph.constants import NODE_COORDINATE_KEY
from skeleplex.graph.image_to_graph_lazy import (
    assign_unique_ids,
    compute_degrees,
    construct_dataframe,
    remove_isolated_voxels,
    skeleton_image_to_graph,
)

def test_lazy_graph_construction_tree():
    """Test full lazy image-to-graph pipeline on synthetic tree skeleton."""

    image = generate_tree_3d(
        shape=(5, 100, 100),
        num_bifurcations=2,
        branch_length=40,
        z_layer=2,
        left_angle=60,
        right_angle=60,
    )

    skeleton = da.from_array(image.astype(np.uint8), chunks=(25, 25, 25))

    # compute degrees and remove isolated voxels
    degrees = compute_degrees(skeleton)
    filtered = remove_isolated_voxels(skeleton, degrees)

    # assign unique voxel IDs
    labeled, _ = assign_unique_ids(filtered)

    # construct edges DataFrame
    edges_df = construct_dataframe(labeled).compute()

    graph = skeleton_image_to_graph(
        skeleton_image=filtered,
        degrees_image=degrees,
        graph_edges_df=edges_df,
    )

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 8
    assert graph.number_of_edges() == 7
    
    node_attrs = nx.get_node_attributes(graph, NODE_COORDINATE_KEY)
    assert all(
        isinstance(coord, np.ndarray) and coord.shape == (3,)
        for coord in node_attrs.values()
    )

    for _, _, attrs in graph.edges(data=True):
        assert "path" in attrs
        assert isinstance(attrs["path"], np.ndarray)
        assert attrs["path"].shape[1] == 3