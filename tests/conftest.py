"""Fixtures for testing with Pytest."""

import networkx as nx
import numpy as np
import pytest

from skeleplex.graph.constants import (
    EDGE_COORDINATES_KEY,
    EDGE_SPLINE_KEY,
    NODE_COORDINATE_KEY,
)
from skeleplex.graph.skeleton_graph import SkeletonGraph
from skeleplex.graph.spline import B3Spline


@pytest.fixture
def simple_t_skeleton_graph():
    """Return the simple T skeleton as a graph."""
    graph = nx.DiGraph()
    # add nodes
    graph.add_node(0, **{NODE_COORDINATE_KEY: np.array([10, 0, 0])})
    graph.add_node(1, **{NODE_COORDINATE_KEY: np.array([10, 10, 0])})
    graph.add_node(2, **{NODE_COORDINATE_KEY: np.array([0, 10, 0])})
    graph.add_node(3, **{NODE_COORDINATE_KEY: np.array([20, 10, 0])})

    # add edge coordinates
    graph.add_edge(
        0, 1, **{EDGE_COORDINATES_KEY: np.linspace([10, 0, 0], [10, 10, 0], 4)}
    )
    graph.add_edge(
        1, 2, **{EDGE_COORDINATES_KEY: np.linspace([10, 10, 0], [0, 10, 0], 4)}
    )
    graph.add_edge(
        1, 3, **{EDGE_COORDINATES_KEY: np.linspace([10, 10, 0], [20, 10, 0], 4)}
    )

    # add spline
    graph.add_edge(
        0,
        1,
        **{
            EDGE_SPLINE_KEY: B3Spline.from_points(
                np.linspace([10, 0, 0], [10, 10, 0], 4)
            )
        },
    )
    graph.add_edge(
        1,
        2,
        **{
            EDGE_SPLINE_KEY: B3Spline.from_points(
                np.linspace([10, 10, 0], [0, 10, 0], 4)
            )
        },
    )
    graph.add_edge(
        1,
        3,
        **{
            EDGE_SPLINE_KEY: B3Spline.from_points(
                np.linspace([10, 10, 0], [20, 10, 0], 4)
            )
        },
    )

    return SkeletonGraph(graph=graph)


@pytest.fixture
def simple_spline():
    """Return a simple B3 spline.

    The spline goes in straight line from (0, 0, 0) to (1, 0, 0).
    """
    # make three points that go from x=0 to x=1
    points = np.array(
        [
            [0, 0, 0],
            [0.25, 0, 0],
            [0.5, 0, 0],
            [0.75, 0, 0],
            [1, 0, 0],
        ]
    )

    # fit a spline to the points
    return B3Spline.from_points(points)


@pytest.fixture
def simple_t_with_flipped_spline():
    """Return the simple T skeleton as a graph with a flipped spline."""

    graph = nx.DiGraph()
    # add nodes
    graph.add_node(0, **{NODE_COORDINATE_KEY: np.array([10, 0, 0])})
    graph.add_node(1, **{NODE_COORDINATE_KEY: np.array([10, 10, 0])})
    graph.add_node(2, **{NODE_COORDINATE_KEY: np.array([0, 10, 0])})
    graph.add_node(3, **{NODE_COORDINATE_KEY: np.array([20, 10, 0])})

    # add edge coordinates
    # flipped edge
    graph.add_edge(
        0, 1, **{EDGE_COORDINATES_KEY: np.linspace([10, 10, 0], [10, 0, 0], 4)}
    )

    graph.add_edge(
        1, 2, **{EDGE_COORDINATES_KEY: np.linspace([10, 10, 0], [0, 10, 0], 4)}
    )
    graph.add_edge(
        1, 3, **{EDGE_COORDINATES_KEY: np.linspace([10, 10, 0], [20, 10, 0], 4)}
    )

    # add spline
    # flipped spline
    graph.add_edge(
        0,
        1,
        **{
            EDGE_SPLINE_KEY: B3Spline.from_points(
                np.linspace([10, 10, 0], [10, 0, 0], 4)
            )
        },
    )

    graph.add_edge(
        1,
        2,
        **{
            EDGE_SPLINE_KEY: B3Spline.from_points(
                np.linspace([10, 10, 0], [0, 10, 0], 4)
            )
        },
    )
    graph.add_edge(
        1,
        3,
        **{
            EDGE_SPLINE_KEY: B3Spline.from_points(
                np.linspace([10, 10, 0], [20, 10, 0], 4)
            )
        },
    )

    return graph

@pytest.fixture
def straight_edge_graph():
    graph = nx.DiGraph()

    graph.add_node(0, **{NODE_COORDINATE_KEY: np.array([10, 0, 0])})
    graph.add_node(1, **{NODE_COORDINATE_KEY: np.array([20, 0, 0])})
    graph.add_node(2, **{NODE_COORDINATE_KEY: np.array([30, 0, 0])})

    graph.add_edge(
        0, 1, **{EDGE_COORDINATES_KEY: np.linspace([10, 0, 0], [20, 0, 0], 4)}
    )
    graph.add_edge(
        1, 2, **{EDGE_COORDINATES_KEY: np.linspace([20, 0, 0], [30, 0, 0], 4)}
    )

    for edge in graph.edges:
        graph.edges[edge][EDGE_SPLINE_KEY] = B3Spline.from_points(
            graph.edges[edge][EDGE_COORDINATES_KEY]
        )
    
    return SkeletonGraph(graph=graph)
