import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any
from skeleplex.graph.skeleton_graph import SkeletonGraph
import warnings
import logging
from skeleplex.graph.spline import B3Spline
from skeleplex.graph.constants import (
    EDGE_COORDINATES_KEY,
    EDGE_SPLINE_KEY,
    NODE_COORDINATE_KEY,
    GENERATION_KEY,
    VALIDATED_KEY,
    START_NODE_KEY,
    END_NODE_KEY,
    LENGTH_KEY

)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def merge_edge(graph:nx.DiGraph, n1:int, v1:int, n2:int):
    """merge edges in graph and add edge attributes. 
    n1 is merged with n2. v1 is removed.
    """
    graph = graph.copy()

    start_node = graph.nodes(data=True)[n1][NODE_COORDINATE_KEY]
    end_node = graph.nodes(data=True)[n2][NODE_COORDINATE_KEY]
    middle_node = graph.nodes(data=True)[v1][NODE_COORDINATE_KEY]


    edge_attributes1 = graph.get_edge_data(n1,v1)
    edge_attributes2 = graph.get_edge_data(v1,n2)
    graph.remove_edge(n1,v1)
    graph.remove_edge(v1,n2)
    graph.remove_node(v1)
    merge_edge = (n1,n2)
    merge_attributes = {}
    for key in edge_attributes1:
        if key == 'validated':
            if edge_attributes1[key] and edge_attributes2[key] == True:
                merge_attributes[key] = True
            else:
                merge_attributes[key] = False


        if key == EDGE_SPLINE_KEY:

            points1 = edge_attributes1[EDGE_COORDINATES_KEY]
            points2 = edge_attributes2[EDGE_COORDINATES_KEY]

            #start and end node coordinates
            #this checking is probably not necessary as we use directed graphs
            #but just to be sure
            if (np.allclose(points1[0], start_node) & 
                np.allclose(points2[0], middle_node)):
                logger.info('no flip')
                spline_points = np.vstack((points1, points2))
                
            elif (np.allclose(points1[-1], start_node) &
                   np.allclose(points2[0], middle_node)):
                logger.info('flip 1')
                spline_points = np.vstack((np.flip(points1, axis = 0), points2))
            elif (np.allclose(points1[0], start_node) & 
                  np.allclose(points2[-1], middle_node)):
                logger.info('flip 2')
                spline_points = np.vstack((points1, np.flip(points2, axis = 0)))
            elif (np.allclose(points1[-1], start_node) & 
                  np.allclose(points2[-1], middle_node)):
                logger.info('flip both')
                spline_points = np.vstack((np.flip(points1, axis = 0),
                                            np.flip(points2, axis = 0)))    
            else:
                warnings.warn('Warning: Edge splines not connected.')
                spline_points = np.vstack((points1, points2))
            #sanity check
            if np.allclose(spline_points[-1], end_node):
                logger.info('sanity check passed')    

                
            _, idx = np.unique(spline_points, axis=0, return_index=True)
            spline_points = spline_points[np.sort(idx)]
            spline = B3Spline.from_points(spline_points)
            merge_attributes[key] = spline
            merge_attributes[EDGE_COORDINATES_KEY] = spline_points
        if key == START_NODE_KEY:
            merge_attributes[key] = n1
        if key == END_NODE_KEY:
            merge_attributes[key] = n2
        if key == GENERATION_KEY:
            merge_attributes[key] = edge_attributes1[key]
        
        if key == LENGTH_KEY:
            merge_attributes[key] = merge_attributes[EDGE_SPLINE_KEY].arc_length()

        if key not in  [VALIDATED_KEY, 
                            EDGE_COORDINATES_KEY, 
                            EDGE_SPLINE_KEY, 
                            START_NODE_KEY, 
                            END_NODE_KEY, 
                            GENERATION_KEY,
                            LENGTH_KEY]:
            logger.warning(('Warning: Attribute {} not merged. '.format(key) ,
                            'Consider recomputing.'))
            
    graph.add_edge(*merge_edge, **merge_attributes)
    return graph

def delete_edge(SkeletonGraph_obj:SkeletonGraph, edge: Tuple[int, int]):
        """delete edge."""

        #check if directed
        if not SkeletonGraph_obj.graph.is_directed():
            SkeletonGraph_obj.to_directed()

        #copy graph
        graph = SkeletonGraph_obj.graph.copy()
        graph.remove_edge(*edge)

        #detect all changes
        changed_edges = set(SkeletonGraph_obj.graph.edges) - set(graph.edges)
        for edge in changed_edges:
            for node in edge:
                if graph.degree(node) == 0:
                    graph.remove_node(node)
                #merge edges if node has degree 2
                elif graph.degree(node) == 2:
                    #merge
                    in_edge = list(graph.in_edges(node))
                    out_edge = list(graph.out_edges(node))
                    if len(in_edge) == 0:
                        raise ValueError(('Deleting the edge would break the graph'),
                                         'Are you trying to delete the origin?')
                        
                    graph = merge_edge(graph,in_edge[0][0], node, out_edge[0][1])
                    logger.info('merge')


        #check if graph is still connected, if not remove orphaned nodes
        SkeletonGraph_obj.graph.remove_nodes_from(list(nx.isolates(SkeletonGraph_obj.graph)))
        SkeletonGraph_obj.graph = graph


def length_pruning(SkeletonGraph_obj:SkeletonGraph, length_threshold:int):
        """Prune all edges with length below threshold
        
        Parameters
        ----------
        graph : nx.Graph
            The graph to prune.
        length_threshold : int
            The threshold for the length of the edges.
        
        
        """
        graph = SkeletonGraph_obj.graph
        g_unmodified = graph.copy()

        #check if length is already computed
        if  len(nx.get_edge_attributes(graph, 'length')) == 0:
            len_dict = SkeletonGraph.compute_branch_length(graph)
            nx.set_edge_attributes(graph, len_dict, 'length')
        c = 0
        for node, degree in g_unmodified.degree():
            if degree == 1:
                edge = list(graph.edges(node))[0]
                path_length = graph.edges[edge[0], edge[1]].get('length')
                start_node = graph.edges[edge]['start_node']
                if path_length < length_threshold:
                    c+=1
                    #check orientation of edge
                    if start_node == edge[0]:
                        edge =edge[::-1]
                    try:
                        delete_edge(graph, edge)
                        logger.info(f'deleted{edge}')
                    except:
                        logger.info(f'could not delete{edge}')
                        continue