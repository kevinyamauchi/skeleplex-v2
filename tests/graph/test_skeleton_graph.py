"""Tests for the SkeletonGraph class."""

import tempfile

import dask.array as da
import networkx as nx
import numpy as np

from skeleplex.graph.constants import (
    EDGE_COORDINATES_KEY,
    EDGE_SPLINE_KEY,
)
from skeleplex.graph.skeleton_graph import (
    SkeletonGraph,
    get_next_node_key,
    orient_splines,
)


def test_skeleton_graph_equality(simple_t_skeleton_graph):
    """Test the equality of two SkeletonGraph objects."""
    assert simple_t_skeleton_graph == simple_t_skeleton_graph

    # check that changing the nodes makes the graphs not equal
    modified_node_graph = simple_t_skeleton_graph.graph.copy()
    modified_node_graph.add_node(9000)
    new_skeleton_graph = SkeletonGraph(graph=modified_node_graph)
    assert simple_t_skeleton_graph != new_skeleton_graph

    # check that changing the edges makes the graphs not equal
    modified_edge_graph = simple_t_skeleton_graph.graph.copy()
    modified_edge_graph.add_edge(0, 15)
    new_skeleton_graph = SkeletonGraph(graph=modified_edge_graph)
    assert simple_t_skeleton_graph != new_skeleton_graph


def test_skeleton_graph_json_round_trip(simple_t_skeleton_graph, tmp_path):
    """Test writing and reading a SkeletonGraph object"""
    # write the graph to a file
    file_path = tmp_path / "test.json"
    simple_t_skeleton_graph.origin = 0
    simple_t_skeleton_graph.to_json_file(file_path)

    # read the graph from the file
    new_skeleton_graph = SkeletonGraph.from_json_file(file_path)

    # check that the graphs are equal
    assert simple_t_skeleton_graph == new_skeleton_graph
    assert simple_t_skeleton_graph.origin == new_skeleton_graph.origin


def test_skeleton_graph_to_directed(simple_t_skeleton_graph):
    """Test converting a SkeletonGraph to a directed graph."""
    directed_graph = simple_t_skeleton_graph.to_directed(origin=0)
    assert directed_graph.is_directed()

    # check that the directed graph has the same nodes
    assert set(directed_graph.nodes) == set(simple_t_skeleton_graph.graph.nodes)

    # check if the origin node has no incoming edges
    assert len(list(directed_graph.in_edges(0))) == 0

    # check edge directionality
    for u, v in directed_graph.edges:
        assert all(
            edges in directed_graph.out_edges(u) for edges in directed_graph.in_edges(v)
        )


def test_get_next_node_id():
    """Test the get_next_node_id function."""
    # initialize an empty graph
    graph = nx.Graph()

    assert get_next_node_key(graph) == 0

    # add a node to the graph
    graph.add_node(0)
    assert get_next_node_key(graph) == 1

    # add multiple nodes to the graph
    graph.add_nodes_from([1, 2, 3, 10, 23, 65])
    assert get_next_node_key(graph) == 4


def test_skeleton_graph_orient_splines(simple_t_with_flipped_spline):
    """Test orienting the splines in a SkeletonGraph."""
    correct_spline_coordinates = np.linspace([10, 0, 0], [10, 10, 0], 4)
    flipped_edge = (0, 1)

    # reorder the graph
    oriented_graph = orient_splines(simple_t_with_flipped_spline)
    oriented_spline = oriented_graph.edges()[flipped_edge][EDGE_SPLINE_KEY]
    eval_oriented_spline = oriented_spline.eval(np.linspace(0, 1, 4))
    oriented_edge_coordinates = oriented_graph.edges()[flipped_edge][
        EDGE_COORDINATES_KEY
    ]

    # check that the spline is oriented
    np.testing.assert_allclose(
        eval_oriented_spline, correct_spline_coordinates, atol=0.5
    )
    np.testing.assert_allclose(
        oriented_edge_coordinates, correct_spline_coordinates, atol=0.5
    )


def create_straight_edge_volume(scale=1):
    # Create the in-memory array
    volume = np.zeros((50 // scale, 50 // scale, 50 // scale), dtype=np.float32)
    volume[10 // scale : 30 // scale, 0 : 10 // scale, 0 : 10 // scale] = 1
    return da.from_array(volume, chunks=(10, 10, 10))


def test_sample_volume_slices_from_spline(straight_edge_graph):
    straight_edge_graph.origin = 0

    # Create a straight edge volume
    straight_edge_volume = create_straight_edge_volume(scale=2)
    # scale volume to voxel size 2
    straight_edge_graph.voxel_size_um = (2, 2, 2)

    slices_vox2 = straight_edge_graph.sample_volume_slices_from_spline(
        straight_edge_volume,
        slice_spacing=0.3,
        slice_size_um=10,
        sample_grid_spacing_um=2,
        interpolation_order=1,
        approx=True,
    )

    # Create a straight edge volume
    straight_edge_volume = create_straight_edge_volume(scale=1)
    # scale volume to voxel size 2
    straight_edge_graph.voxel_size_um = (1, 1, 1)

    slices_vox1 = straight_edge_graph.sample_volume_slices_from_spline(
        straight_edge_volume,
        slice_spacing=0.3,
        slice_size_um=10,
        sample_grid_spacing_um=2,
        interpolation_order=1,
        approx=True,
    )

    for key, img_slice in slices_vox1.items():
        slice_vox2 = slices_vox2[key]
        np.testing.assert_allclose(img_slice, slice_vox2, atol=1e-5, equal_nan=True)


def test_sample_volume_slices_from_spline_parallel(straight_edge_graph):
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save volume to Zarr store
        straight_edge_volume = create_straight_edge_volume(scale=1)
        straight_edge_volume.to_zarr(tmpdir + "scale_1.zarr")
        straight_edge_graph.origin = 0
        straight_edge_graph.voxel_size_um = (1, 1, 1)
        # Now pass to your function
        slices_vox1 = straight_edge_graph.sample_volume_slices_from_spline_parallel(
            tmpdir + "scale_1.zarr",
            slice_spacing=0.3,
            slice_size_um=10,
            sample_grid_spacing_um=1,
            interpolation_order=1,
            approx=True,
            num_workers=1,
        )

        straight_edge_volume = create_straight_edge_volume(scale=2)
        straight_edge_volume.to_zarr(tmpdir + "scale_2.zarr")
        straight_edge_graph.origin = 0
        straight_edge_graph.voxel_size_um = (2, 2, 2)
        # Now pass to your function
        slices_vox2 = straight_edge_graph.sample_volume_slices_from_spline_parallel(
            tmpdir + "scale_2.zarr",
            slice_spacing=0.3,
            slice_size_um=10,
            sample_grid_spacing_um=1,
            interpolation_order=1,
            approx=True,
            num_workers=1,
        )
        for key, img_slice in slices_vox1.items():
            slice_vox2 = slices_vox2[key]
            np.testing.assert_allclose(img_slice, slice_vox2, atol=1e-5, equal_nan=True)
