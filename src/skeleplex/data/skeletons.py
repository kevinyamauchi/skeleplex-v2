import networkx as nx  # noqa
import numpy as np

from skeleplex.graph.constants import EDGE_COORDINATES_KEY, NODE_COORDINATE_KEY
from skeleplex.graph.skeleton_graph import SkeletonGraph


def generate_toy_skeleton_graph_symmetric_branch_angle(
    num_nodes: int = 19, angle: int = 72, edge_length: float = 20
) -> SkeletonGraph:
    """Generate a toy skeleton graph with a symmetric branch angle.

    Parameters
    ----------
    num_nodes : int
        The number of nodes in the graph.
    angle : int
        The angle between branches in degrees.
    edge_length : float
        The length of each edge in the graph.

    """
    # Create a directed graph
    graph = nx.DiGraph()

    # Convert angle to radians and divide by 2 for symmetric branching
    angle_rad = np.radians(angle / 2)

    # Initialize node positions dictionary with the root node at the origin
    node_pos_dic = {0: np.array([0, 0, 0])}
    parent_nodes = [0]  # Start with the root node

    # Add trachea node and edge
    trachea_pos = np.array([-edge_length, 0, 0])
    node_pos_dic[-1] = trachea_pos
    graph.add_node(-1, node_coordinate=trachea_pos)
    graph.add_edge(
        -1,
        0,
        **{
            EDGE_COORDINATES_KEY: np.linspace(
                trachea_pos, np.array([0, 0, 0]), 5 + edge_length
            )
        },
    )

    # Initialize the first two branches
    m = edge_length * np.cos(angle_rad)
    n = edge_length * np.sin(angle_rad)
    new_pos = node_pos_dic[0] + np.array([m, n, 0])
    node_pos_dic[1] = new_pos
    edge_coordinates = np.linspace(node_pos_dic[0], new_pos, 5 + edge_length)
    graph.add_node(1)
    graph.add_edge(0, 1, **{EDGE_COORDINATES_KEY: edge_coordinates, "side": "left"})

    m = edge_length * np.cos(-angle_rad)
    n = edge_length * np.sin(-angle_rad)
    new_pos = node_pos_dic[0] + np.array([m, n, 0])
    node_pos_dic[2] = new_pos
    edge_coordinates = np.linspace(node_pos_dic[0], new_pos, 5 + edge_length)
    graph.add_node(2)
    graph.add_edge(0, 2, **{EDGE_COORDINATES_KEY: edge_coordinates, "side": "right"})

    parent_nodes = [1, 2]  # Update parent nodes to the first two branches
    i = 3  # Start adding new nodes from index 3

    while i < num_nodes:
        new_parents = []
        for parent_node in parent_nodes:
            if i < num_nodes:
                # Add the first child node
                angle_rad = np.radians(angle / 2)

                # Get the path to the root node
                # count the number of left vs right edges
                path = nx.shortest_path(graph, 0, parent_node)
                edges = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
                sides = [graph.edges[edge]["side"] for edge in edges]
                left_edges = sides.count("left")
                right_edges = sides.count("right")
                num_rotations = left_edges - right_edges

                # Adjust angle based on the number of rotations
                angle_rad *= num_rotations + 1

                m = edge_length * np.cos(angle_rad)
                n = edge_length * np.sin(angle_rad)
                side = "left"
                new_pos = node_pos_dic[parent_node] + np.array([m, n, 0])

                node_pos_dic[i] = new_pos
                edge_coordinates = np.linspace(
                    node_pos_dic[parent_node], new_pos, 5 + edge_length
                )
                graph.add_node(i)
                graph.add_edge(
                    parent_node,
                    i,
                    **{EDGE_COORDINATES_KEY: edge_coordinates, "side": side},
                )
                new_parents.append(i)
                i += 1

            if i < num_nodes:
                # Add the second child node and rotate in the opposite direction
                angle_rad = np.radians(angle) / 2

                path = nx.shortest_path(graph, 0, parent_node)
                edges = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
                sides = [graph.edges[edge]["side"] for edge in edges]
                left_edges = sides.count("left")
                right_edges = sides.count("right")
                num_rotations = left_edges - right_edges

                angle_rad *= num_rotations - 1

                m = edge_length * np.cos(angle_rad)
                n = edge_length * np.sin(angle_rad)
                side = "right"
                new_pos = node_pos_dic[parent_node] + np.array([m, n, 0])
                node_pos_dic[i] = new_pos
                edge_coordinates = np.linspace(
                    node_pos_dic[parent_node], new_pos, 5 + edge_length
                )
                graph.add_node(i)
                graph.add_edge(
                    parent_node,
                    i,
                    **{EDGE_COORDINATES_KEY: edge_coordinates, "side": side},
                )
                new_parents.append(i)
                i += 1

        parent_nodes = new_parents  # Update parent nodes for the next iteration

    # Set node attributes for the graph
    nx.set_node_attributes(graph, node_pos_dic, NODE_COORDINATE_KEY)

    # Create a SkeletonGraph from the graph
    skeleton_graph = SkeletonGraph.from_graph(
        graph, EDGE_COORDINATES_KEY, NODE_COORDINATE_KEY
    )
    return skeleton_graph
