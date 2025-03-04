import networkx as nx  # noqa: D100
import numpy as np

from skeleplex.graph.constants import (
    BRANCH_ANGLE_EDGE_KEY,
    EDGE_SPLINE_KEY,
    NODE_COORDINATE_KEY,
    ROTATION_ANGLE_EDGE_KEY,
    SIBLING_ANGLE_EDGE_KEY,
    SISTER_EDGE_KEY,
)
from skeleplex.measurements.utils import get_normal_of_plane, rad2deg, unit_vector


def compute_midline_branch_angle_branch_nodes(graph: nx.DiGraph):
    """Calculates the midline branch angle for each branch in the graph.

    Computes the midline anlges for each branch in the graph and returns
    the midline branch angle as an edge attribute.

    To compute the vectors, only the branch nodes are taken in consideration.
    Branches are simplified to a straight line between branch nodes.

    Graph requirements:
    - The graph must be directed
    - The graph must be ordered with the desired hierarchy
    - The graph must have a 'node_coordinate' attribute for each node

    Returns
    -------
    graph : nx.DiGraph
        The input graph with the angles added as edge attributes

    Raises
    ------
    ValueError
        Raises and error if the end point of the parent branch and the
        start point of the daughter branch are not the same
    ValueError
        Raises and error if the length of the midline vector is != 1
    ValueError
        Raises and error if the length of the branch vector is != 1

    """
    tree = graph.copy()

    angle_dict = {}
    center_points = []
    midline_points = []

    node_coordinates = nx.get_node_attributes(tree, "node_coordinate")

    for u, v, _ in tree.edges(data=True):
        edge = (u, v)
        # print(list(tree.in_edges(u)))
        if not list(tree.in_edges(u)):
            continue
        parent_edge = next(iter(tree.in_edges(u)))

        parent_start_node_coordinates = node_coordinates[parent_edge[0]]
        parent_end_node_coordinates = node_coordinates[parent_edge[1]]

        parent_vector = unit_vector(
            parent_start_node_coordinates - parent_end_node_coordinates
        )
        midline_vector = -parent_vector

        start_node_coordinates = node_coordinates[edge[0]]

        if np.all(parent_end_node_coordinates != start_node_coordinates):
            raise ValueError("Branch point ill defined.")

        end_node_coordinates = node_coordinates[edge[1]]
        branch_vector = unit_vector(end_node_coordinates - start_node_coordinates)

        if round(np.linalg.norm(midline_vector)) != 1:
            raise ValueError(f"""Midline vector is not normalized.
                             Its length is {np.linalg.norm(midline_vector)}""")
        if round(np.linalg.norm(branch_vector)) != 1:
            raise ValueError(f"""Branch vector is not normalized.
                             Its length is {np.linalg.norm(branch_vector)}""")

        dot = np.dot(midline_vector, branch_vector)
        angle = np.degrees(np.arccos(dot))

        angle_dict[edge] = angle

        # store for visualization
        center_points.append(parent_end_node_coordinates)
        midline_points.append(parent_end_node_coordinates + (10 * midline_vector))

    nx.set_edge_attributes(tree, angle_dict, BRANCH_ANGLE_EDGE_KEY)

    return tree, center_points, midline_points


def compute_midline_branch_angle_spline(graph: nx.DiGraph, n_samples: int):
    """Calculates the midline branch angle for each branch in the graph.

    Computes the midline anlges for each branch in the graph and returns
    the midline branch angle as an edge attribute.

    To compute the vectors, the spline is used to sample points along the
    branch. Angles are computed between the tangent of the spline of the branch
    and the tangent of the spline of the parent branch. Sampling distance starts
    at the common node and moves towards the end of both branches.

    Graph requirements:
    - The graph must be directed
    - The graph must be ordered with the desired hierarchy
    - The graph must have a 'node_coordinate' attribute for each node

    Returns
    -------
    graph : nx.DiGraph
        The input graph with the angles added as edge attributes

    Raises
    ------
    ValueError
        Raises and error if the end point of the parent branch and the
        start point of the daughter branch are not the same
    ValueError
        Raises and error if the length of the midline vector is != 1
    ValueError
        Raises and error if the length of the branch vector is != 1

    """
    graph = graph.copy()
    angle_dict = {}
    # loop over each edge
    for u, v in graph.edges():
        edge = (u, v)
        parent_edge = list(graph.in_edges(u))
        if not parent_edge:
            continue
        parent_edge = parent_edge[0]
        parent_spline = graph.edges[parent_edge][EDGE_SPLINE_KEY]
        spline = graph.edges[edge][EDGE_SPLINE_KEY]
        sample_positions = np.linspace(0, 1, n_samples)
        parent_tangents = parent_spline.eval(sample_positions, derivative=1)
        tangents = spline.eval(sample_positions, derivative=1)
        # normalize the tangents
        tangents = [unit_vector(t) for t in tangents]
        parent_tangents = [unit_vector(t) for t in parent_tangents]

        angle_list = []

        for i in range(len(tangents)):
            t = tangents[i]
            j = len(parent_tangents) - i - 1
            pt = parent_tangents[j]
            dot = np.dot(t, pt)
            angle = np.degrees(np.arccos(dot))
            angle_list.append(angle)
        # angle_std = np.std(angle_list)
        mean_angle = np.mean(angle_list)
        angle_dict[edge] = mean_angle

    nx.set_edge_attributes(graph, angle_dict, BRANCH_ANGLE_EDGE_KEY)
    return graph


def compute_rotation_angle(graph: nx.DiGraph):
    """Calculates the rotation angle for each edge in the graph.

    Compute the rotation angle between the plane defined by the parent node
    and the plane defined by the edge and its sister.

    The edge attribute is defined in the
    ROTATION_ANGLE_EDGE_KEY constant.

    The input graph should have the following attributes:

    - NODE_COORDINATE_KEY: The node coordinates
    - SISTER_EDGE_KEY: The sister edge
    - Directed graph with the correct orientation
    - Strictly dichotomous tree

    Parameters
    ----------
    graph : nx.DiGraph
        The input graph
    """
    rotation_angle_dict = {}
    graph = graph.copy()
    node_coord = nx.get_node_attributes(graph, NODE_COORDINATE_KEY)
    for edge in graph.edges():
        parent = list(graph.in_edges(edge[0]))

        sister = None
        if SISTER_EDGE_KEY in graph.edges[edge]:
            sister = graph.edges[edge][SISTER_EDGE_KEY]

        if not parent or not sister:
            continue
        parent = parent[0]
        parent_sister = None
        if SISTER_EDGE_KEY in graph.edges[parent]:
            parent_sister = graph.edges[parent][SISTER_EDGE_KEY]

        if not parent_sister:
            continue

        if isinstance(parent_sister[0], list):
            parent_sister = tuple(parent_sister[0])
        if isinstance(sister[0], list):
            sister = tuple(sister[0])

        parent_plane = [
            node_coord[parent[0]],
            node_coord[parent[1]],
            node_coord[parent_sister[1]],
        ]

        edge_plane = [node_coord[edge[0]], node_coord[edge[1]], node_coord[sister[1]]]
        if parent_plane and edge_plane:
            normal_parent = get_normal_of_plane(
                parent_plane[0], parent_plane[1], parent_plane[2]
            )
            normal_edge = get_normal_of_plane(
                edge_plane[0], edge_plane[1], edge_plane[2]
            )
            normal_parent_unit = unit_vector(normal_parent)
            normal_edge_unit = unit_vector(normal_edge)
            rotation_angle = np.arccos(np.dot(normal_parent_unit, normal_edge_unit))
            if rotation_angle > np.pi / 2:
                rotation_angle = np.pi - rotation_angle
            rotation_angle_dict[edge] = rad2deg(rotation_angle)

    nx.set_edge_attributes(graph, rotation_angle_dict, ROTATION_ANGLE_EDGE_KEY)

    return graph


def compute_sibling_angle(graph: nx.DiGraph):
    """Calculates the sibling angle for each edge in the graph.

    Computes the sibling angles for each edge in the graph
    and returns the sibling angle as an edge attribute.

    The sibling angle is the angle between the edge and its sister edge.

    Graph requirements:
    - The graph must be directed
    - The graph must be ordered with the desired hierarchy
    - The graph must have a 'node_coordinate' attribute for each node

    Parameters
    ----------
    graph : nx.DiGraph
        The input graph

    Returns
    -------
    graph : nx.DiGraph
        The input graph with the sibling angle added as edge attributes
    """
    graph = graph.copy()
    angle_dict = {}
    sister_pairs = nx.get_edge_attributes(graph, SISTER_EDGE_KEY)
    sister_pairs = [(edge, sister_pairs[edge]) for edge in sister_pairs]
    # keep only one sister pair as they both have the same angle
    unique_pairs = set()
    for pair in sister_pairs:
        if isinstance(pair[0][0], list) or isinstance(pair[1][0], list):
            continue
        pair = tuple(sorted(pair))
        unique_pairs.add(pair)
    unique_pairs = list(unique_pairs)

    node_coordinates = nx.get_node_attributes(graph, NODE_COORDINATE_KEY)
    for sister_pair in unique_pairs:
        edge = sister_pair[0]
        sister = sister_pair[1]
        shared_node_coord = node_coordinates[edge[0]]
        edge_vector = unit_vector(node_coordinates[edge[1]] - shared_node_coord)
        sister_vector = unit_vector(node_coordinates[sister[1]] - shared_node_coord)

        dot = np.dot(edge_vector, sister_vector)
        angle = np.degrees(np.arccos(dot))
        angle_dict[edge] = angle
        angle_dict[sister] = angle

    nx.set_edge_attributes(graph, angle_dict, SIBLING_ANGLE_EDGE_KEY)

    return graph
