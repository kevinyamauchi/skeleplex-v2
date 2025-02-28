import networkx as nx  # noqa: D100

from skeleplex.graph.constants import (
    DAUGHTER_EDGES_KEY,
    EDGE_SPLINE_KEY,
    GENERATION_KEY,
    LENGTH_KEY,
    NUMBER_OF_TIPS_KEY,
    PARENT_EDGE_KEY,
    SISTER_EDGE_KEY,
)


def get_sister_edges(graph: nx.DiGraph):
    """Return a graph with sister edges annotated.

    This function identifies sister edges for each edge in the graph.
    Sister edges are edges that share the same start node but are not the same edge.

    Parameters
    ----------
    graph : nx.DiGraph
        The input graph.

    Returns
    -------
    nx.DiGraph
        The graph with sister edges annotated.
    """
    graph = graph.copy()
    sister_dict = {}
    for edge in graph.edges():
        sisters = list(graph.out_edges(edge[0]))
        sister = [s for s in sisters if s != edge]
        if len(sister) == 1:
            sister_dict[edge] = sister[0]
        elif len(sister) > 1:
            sister_dict[edge] = sister
    nx.set_edge_attributes(graph, sister_dict, SISTER_EDGE_KEY)
    return graph


def get_daughter_edges(graph):
    """Return a graph with daughter edges annotated.

    This function identifies daughter edges for each edge in the graph.
    Daughter edges are edges that share the same end node.

    Parameters
    ----------
    graph : nx.DiGraph
        The input graph.

    Returns
    -------
    nx.DiGraph
        The graph with daughter edges annotated.
    """
    graph = graph.copy()
    daughter_dict = {}
    for edge in graph.edges():
        daughters = list(graph.out_edges(edge[1]))
        daughter_dict[edge] = daughters
    nx.set_edge_attributes(graph, daughter_dict, DAUGHTER_EDGES_KEY)
    return graph


def get_parent_edges(graph: nx.DiGraph):
    """Return a graph with parent edges annotated.

    This function identifies parent edges for each edge in the graph.
    Parent edges are edges that share the same start node.

    Parameters
    ----------
    graph : nx.DiGraph
        The input graph.

    Returns
    -------
    nx.DiGraph
        The graph with parent edges annotated.
    """
    graph = graph.copy()
    parent_dict = {}
    for edge in graph.edges():
        parents = list(graph.in_edges(edge[0]))
        parent_dict[edge] = parents
    nx.set_edge_attributes(graph, parent_dict, PARENT_EDGE_KEY)
    return graph


def compute_level(graph: nx.DiGraph, origin: int):
    """Compute the level of each node and edge in the graph.

    The level of a node is the shortest path length from the origin node.
    The level of an edge is the level of its start node.

    Parameters
    ----------
    graph : nx.DiGraph
        The input graph.
    origin : int
        The origin node.

    Returns
    -------
    nx.DiGraph
        The graph with node and edge levels annotated.
    """
    level_dir = {}
    graph = graph.copy()
    for node in graph.nodes():
        if nx.has_path(graph, origin, node):
            level = nx.shortest_path_length(graph, origin, node)
        else:
            level = -1
        level_dir[node] = level
    nx.set_node_attributes(graph, level_dir, GENERATION_KEY)

    # Set edge level to level of start node
    level_dir = {}
    for u, v in graph.edges():
        node_level = graph.nodes[u][GENERATION_KEY]
        level_dir[(u, v)] = node_level
    nx.set_edge_attributes(graph, level_dir, GENERATION_KEY)
    return graph


def compute_branch_length(graph):
    """Compute the length of each branch in the graph.

    The length of a branch is the path length between its start and end nodes.

    Parameters
    ----------
    graph : nx.DiGraph
        The input graph.

    Returns
    -------
    nx.DiGraph
        The graph with branch lengths annotated.
    """
    graph = graph.copy()
    length_dir = {}
    for u, v in graph.edges():
        spline = graph[u][v][EDGE_SPLINE_KEY]
        length = spline.arc_length
        length_dir[(u, v)] = length
    nx.set_edge_attributes(graph, length_dir, LENGTH_KEY)
    return graph


def count_number_of_tips_connected_to_edge(graph, start_node, end_node):
    """Count the number of tips connected to an edge in the graph.

    The number of tips connected to an edge is the number of leaf nodes in the subtree
    rooted at the edge's end node.

    Parameters
    ----------
    graph : nx.DiGraph
        The input graph.
    start_node : int
        The start node of the edge.
    end_node : int
        The end node of the edge.

    Returns
    -------
    int
        The number of tips connected to the edge.
    """
    # Perform a breadth-first search starting from the end_node
    subtree = nx.bfs_tree(graph, end_node)

    # Initialize count of endpoints
    num_endpoints = 0

    # Iterate through nodes in the subtree
    for node in subtree.nodes:
        # Check if the node is a leaf node (degree 1) and is not the start_node
        if subtree.degree(node) == 1 and node != start_node:
            num_endpoints += 1

    return num_endpoints


def compute_number_of_tips_connected_to_edges(graph):
    """Compute the number of tips connected to each edge in the graph.

    The number of tips connected to an edge is the number of leaf nodes in the subtree
    rooted at the edge's end node.

    Parameters
    ----------
    graph : nx.DiGraph
        The input graph.

    Returns
    -------
    nx.DiGraph
        The graph with the number of tips connected to each edge annotated.
    """
    graph = graph.copy()
    num_tips_dir = {}
    for u, v in graph.edges():
        num_tips = count_number_of_tips_connected_to_edge(graph, u, v)
        num_tips_dir[(u, v)] = num_tips
    nx.set_edge_attributes(graph, num_tips_dir, NUMBER_OF_TIPS_KEY)
    return graph
