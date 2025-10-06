"""Functions for detecting and fixing breaks in skeletonized segmentations.

This module provides utilities to prune short branches and iteratively fix breaks in
skeletons, using KDTree-based proximity search and background intersection checks.
"""

from collections import defaultdict
from math import floor

import numpy as np
from numba import jit, prange
from numba.typed import List
from numba.types import bool_, float32
from scipy.spatial import KDTree
from skan import Skeleton, summarize
from skimage.morphology import label

from skeleplex.graph.utils import draw_line_segment
from skeleplex.measurements.utils import unit_vector


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def check_background_intersection(
    start_point: np.ndarray,
    end_point: np.ndarray,
    label_image: np.ndarray,
    skeleton_label_image: np.ndarray,
    step_size: float,
) -> tuple[bool, float]:
    """Check if a line segment intersects background in a label image.

    Parameters
    ----------
    start_point : np.ndarray
        (3,) array containing the starting point of the line segment.
    end_point : np.ndarray
        (3,) array containing the end point of the line segment.
    label_image : np.ndarray
        The label image to check for background intersection.
    skeleton_label_image : np.ndarray
        The skeleton label image to check for branch intersection.
    step_size : float
        The distance between points to sample along the line segment.

    Returns
    -------
    intersects_background : bool
        True if the line segment intersects background, False otherwise.
    line_length : float
        The length of the line segment.
    """
    line_length = np.linalg.norm(end_point - start_point)
    num_points = floor(line_length / step_size)

    # unit vector in the direction to check
    unit_vector = (end_point - start_point) / line_length
    step_vector = step_size * unit_vector

    # get the branch label values for the start and end point
    start_skeleton_value = skeleton_label_image[
        int(start_point[0]), int(start_point[1]), int(start_point[2])
    ]
    end_skeleton_value = skeleton_label_image[
        int(end_point[0]), int(end_point[1]), int(end_point[2])
    ]

    for step_index in range(num_points):
        query_point = start_point + step_index * step_vector
        label_value = label_image[
            int(query_point[0]), int(query_point[1]), int(query_point[2])
        ]
        skeleton_value = skeleton_label_image[
            int(query_point[0]), int(query_point[1]), int(query_point[2])
        ]
        if label_value == 0:
            # line intersects background
            return False, line_length
        if not (
            (skeleton_value == 0)
            or (skeleton_value == start_skeleton_value)
            or (skeleton_value == end_skeleton_value)
        ):
            # line intersects a different branch
            return False, line_length

    return True, line_length


@jit(nopython=True, fastmath=True, cache=True)
def filter_points_from_same_branch(
    start_point: np.ndarray, query_points: np.ndarray, skeleton_label_image: np.ndarray
) -> np.ndarray:
    """Filter points that are on the same branch as the start point.

    Parameters
    ----------
    start_point : np.ndarray
        (3,) array of the coordinates of the start point.
    query_points : np.ndarray
        (m, 3) array of the coordinates of the points to check.
    skeleton_label_image : np.ndarray
        The skeleton label image to check for branch intersection.

    Returns
    -------
    np.ndarray
        (m,) boolean array where True indicates the point is on a different branch.
    """
    start_point_label_value = skeleton_label_image[
        int(start_point[0]), int(start_point[1]), int(start_point[2])
    ]

    # loop through the query points
    n_query_points = query_points.shape[0]
    point_in_different_branch = np.zeros((n_query_points,), bool_)
    for query_point_index in range(n_query_points):
        query_label_value = skeleton_label_image[
            int(query_points[query_point_index, 0]),
            int(query_points[query_point_index, 1]),
            int(query_points[query_point_index, 2]),
        ]
        if query_label_value == start_point_label_value:
            point_in_different_branch[query_point_index] = False
        else:
            point_in_different_branch[query_point_index] = True

    return point_in_different_branch


@jit(nopython=True, fastmath=True, cache=True)
def find_nearest_point_in_segmentation(
    start_point: np.ndarray,
    points_to_check: np.ndarray,
    label_image: np.ndarray,
    skeleton_label_image: np.ndarray,
) -> np.ndarray:
    """Find the nearest point in a segmentation for a given start point.

    The function finds points that do not intersect background or other branches,
    and selects the nearest one.

    Parameters
    ----------
    start_point : np.ndarray
        (3,) array of the coordinates of the start point.
    points_to_check : np.ndarray
        (m, 3) array of the coordinates of the points to check.
    label_image : np.ndarray
        The label image to check for background intersection.
    skeleton_label_image : np.ndarray
        The skeleton label image to check for branch intersection.

    Returns
    -------
    np.ndarray
        (3,) array of the coordinates of the nearest point.
    """
    # get the points to query
    points_to_query_mask = filter_points_from_same_branch(
        start_point,
        points_to_check,
        skeleton_label_image,
    )
    points_to_query = points_to_check[points_to_query_mask]

    # loop through array
    n_query_points = points_to_query.shape[0]
    all_intersects_background = np.zeros((n_query_points,), dtype=bool_)
    all_distances = np.zeros((n_query_points,), dtype=float32)
    for query_point_index in range(n_query_points):
        query_point = points_to_query[query_point_index]

        intersects_background, distance = check_background_intersection(
            start_point=start_point,
            end_point=query_point,
            label_image=label_image,
            skeleton_label_image=skeleton_label_image,
            step_size=0.5,
        )

        # store the values
        all_intersects_background[query_point_index] = intersects_background
        all_distances[query_point_index] = distance

    # find the shortest distance that doesn't intersect background
    if all_intersects_background.sum() == 0:
        # return nans if no point passes the filters
        return np.array([np.nan, np.nan, np.nan])
    else:
        shortest_distance_index = all_distances[all_intersects_background].argmin()
        return points_to_query[all_intersects_background][shortest_distance_index]


@jit(nopython=True, fastmath=True, cache=True)
def find_nearest_and_straightest_point_in_segmentation(
    start_point: np.ndarray,
    base_vector: np.ndarray,
    points_to_check: np.ndarray,
    label_image: np.ndarray,
    skeleton_label_image: np.ndarray,
    angle_threshold: float = 90,
    weight: float = 0.5,
) -> np.ndarray:
    """Find the nearest and straightest point in a segmentation for a given start point.

    The function finds points that do not intersect background or other branches,
    and selects the one that minimizes a combined score of distance and angle to the
    base vector.
    The weighting between distance and angle is currently fixed at 50/50.

    Parameters
    ----------
    start_point : np.ndarray
        (3,) array of the coordinates of the start point.
    base_vector : np.ndarray
        (3,) array of the direction vector for the start point.
    points_to_check : np.ndarray
        (m, 3) array of the coordinates of the points to check.
    label_image : np.ndarray
        The label image to check for background intersection.
    skeleton_label_image : np.ndarray
        The skeleton label image to check for branch intersection.
    angle_threshold : float
        The maximum angle (in degrees) allowed between the existing branch direction
        and the direction to the potential connection point. Default is 90.
    weight : float
        The weight given to distance in the combined score. Default is 0.5.

    Returns
    -------
    np.ndarray
        (3,) array of the coordinates of the nearest and straightest point.
    """
    # get the points to query
    points_to_query_mask = filter_points_from_same_branch(
        start_point,
        points_to_check,
        skeleton_label_image,
    )
    points_to_query = points_to_check[points_to_query_mask]

    # loop through array
    n_query_points = points_to_query.shape[0]
    all_intersects_background = np.zeros((n_query_points,), dtype=bool_)
    all_distances = np.zeros((n_query_points,), dtype=float32)
    all_angles = np.zeros((n_query_points,), dtype=float32)
    for query_point_index in range(n_query_points):
        query_point = points_to_query[query_point_index]

        intersects_background, distance = check_background_intersection(
            start_point=start_point,
            end_point=query_point,
            label_image=label_image,
            skeleton_label_image=skeleton_label_image,
            step_size=0.5,
        )

        # compute angle
        query_vector = unit_vector_numba(query_point - start_point)
        cos_angle = np.dot(base_vector, query_vector)
        angle = np.arccos(cos_angle) * 180 / np.pi

        # store the values
        all_intersects_background[query_point_index] = intersects_background
        all_distances[query_point_index] = distance
        all_angles[query_point_index] = angle

    # find the shortest distance that doesn't intersect background
    if all_intersects_background.sum() == 0:
        # return nans if no point passes the filters
        return np.array([np.nan, np.nan, np.nan])
    else:
        scores = weight * (all_distances / all_distances.max())(
            +(1 - weight) * (all_angles / all_angles.max())
        )
        best_index = np.argmin(scores[all_intersects_background])
        # if best_index is using a large angle, return nan
        if all_angles[all_intersects_background][best_index] > angle_threshold:
            return np.array([np.nan, np.nan, np.nan])
        return points_to_query[all_intersects_background][best_index]


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def find_missing_branches(
    start_points: np.ndarray,
    proximal_points: List[np.ndarray],
    skeleton_coordinates: np.ndarray,
    label_image: np.ndarray,
    skeleton_label_image: np.ndarray,
) -> np.ndarray:
    """Find the nearest points in a segmentation for given start points.

    Parameters
    ----------
    start_points : np.ndarray
        (n, 3) array of the coordinates of the start points.
    proximal_points : List[np.ndarray]
        List of arrays containing the indices of points proximal to each start point.
    skeleton_coordinates : np.ndarray
        (m, 3) array of the coordinates of the skeleton points.
    label_image : np.ndarray
        The label image to check for background intersection.
    skeleton_label_image : np.ndarray
        The skeleton label image to check for branch intersection.

    Returns
    -------
    np.ndarray
        (n, 3) array of the coordinates of the nearest points.
    """
    n_proximal_points = len(proximal_points)

    # loop through the start points
    all_nearest_points = np.zeros((n_proximal_points, 3))
    for point_index in prange(n_proximal_points):
        points_in_radius = skeleton_coordinates[proximal_points[point_index]]

        # get the start point
        start_point = start_points[point_index]

        nearest_point = find_nearest_point_in_segmentation(
            start_point=start_point,
            points_to_check=points_in_radius,
            label_image=label_image,
            skeleton_label_image=skeleton_label_image,
        )
        all_nearest_points[point_index, :] = nearest_point

    return all_nearest_points


@jit(parallel=True, fastmath=True, cache=True)
def find_missing_branches_angle(
    start_points: np.ndarray,
    start_point_ids: np.ndarray,  # now row indices (not raw IDs)
    proximal_points,
    vector_array,
    skeleton_coordinates,
    label_image,
    skeleton_label_image,
    angle_threshold,
    weight: float = 0.5,
):
    """Find the nearest and straightest points in a segmentation for given start points.

    Parameters
    ----------
    start_points : np.ndarray
        (n, 3) array of the coordinates of the start points.
    start_point_ids : np.ndarray
        (n,) array of the row indices of the start points in the vector_array.
    proximal_points : List[np.ndarray]
        List of arrays containing the indices of points proximal to each start point.
    vector_array : np.ndarray
        (m, 3) array of the direction vectors for each start point.
    skeleton_coordinates : np.ndarray
        (p, 3) array of the coordinates of the skeleton points.
    label_image : np.ndarray
        The label image to check for background intersection.
    skeleton_label_image : np.ndarray
        The skeleton label image to check for branch intersection.
    angle_threshold : float
        The maximum angle (in degrees) allowed between the existing branch direction
        and the direction to the potential connection point.
    weight : float
        The weight given to distance in the combined score. Default is 0.5.

    Returns
    -------
    np.ndarray
        (n, 3) array of the coordinates of the nearest and straightest points.
    """
    n_proximal_points = len(proximal_points)
    all_nearest_and_straight_points = np.zeros((n_proximal_points, 3), dtype=np.float32)

    for point_index in prange(n_proximal_points):
        points_in_radius = skeleton_coordinates[proximal_points[point_index]]

        row_idx = start_point_ids[point_index]
        base_vector = vector_array[row_idx]
        start_point = start_points[point_index]

        nearest_straight_point = find_nearest_and_straightest_point_in_segmentation(
            start_point=start_point,
            base_vector=base_vector,
            points_to_check=points_in_radius,
            label_image=label_image,
            skeleton_label_image=skeleton_label_image,
            angle_threshold=angle_threshold,
            weight=weight,
        )
        all_nearest_and_straight_points[point_index, :] = nearest_straight_point

    return all_nearest_and_straight_points


def find_breaks_in_skeleton(
    skeleton_obj: Skeleton,
    end_point_radius: float,
    segmentation_label_image: np.ndarray,
    skeleton_label_image: np.ndarray,
    n_workers: int = -1,
    include_angles=True,
    angle_threshold=90,
    weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find breaks in a skeleton and suggest connections to fix them.

    If angles are included, the function will consider the direction of existing
    branches when suggesting connections,
    preferring those that align with the branch direction.
    The weighting between distance and angle is currently fixed at 50/50.
    The points are only suggested if they do not intersect background or other branches.


    Parameters
    ----------
    skeleton_obj : Skeleton
        The skan skeleton object to find breaks in.
    end_point_radius : float
        The radius around each end point to search for potential connections.
    segmentation_label_image : np.ndarray
        The segmentation label image used to check for background intersection.
    skeleton_label_image : np.ndarray
        The skeleton label image used to check for branch intersection.
    n_workers : int
        The number of workers to use for parallel processing.
        Default is -1 (use all available).
    include_angles : bool
        Whether to include angle consideration when finding breaks. Default is True.
    angle_threshold : float
        The maximum angle (in degrees) allowed between the existing branch direction
        and the direction to the potential connection point.
        Only used if include_angles is True. Default is 90.
    weight : float
        The weight given to distance in the combined score. Default is 0.5.

    Returns
    -------
    node_ids : np.ndarray
        The IDs of the nodes in the skeleton that are end points with detected breaks.
    source_coordinates : np.ndarray
        The coordinates of the end points with detected breaks.
    destination_coordinates : np.ndarray
        The coordinates of the suggested connection points to fix the breaks.
    """
    # build the tree for finding points in radius
    skeleton_coordinates = skeleton_obj.coordinates.astype(float)
    skeleton_tree = KDTree(skeleton_obj.coordinates)

    # get points proximal to end points
    degree_1_nodes = _get_degree_1_nodes(skeleton_obj)

    proximal_points = skeleton_tree.query_ball_point(
        degree_1_nodes, r=end_point_radius, workers=n_workers
    )

    points_numba = List(np.array(indices) for indices in proximal_points)

    if include_angles:
        degree_1_nodes_ids = _get_degree_1_nodes_ids(skeleton_obj)
        degree_1_vectors = _get_edge_vectors_degree1(skeleton_obj)
        edge_vectors, row_indices = _vector_dict_to_array(
            degree_1_nodes_ids, degree_1_vectors
        )

        nearest_points = find_missing_branches_angle(
            start_points=degree_1_nodes,
            start_point_ids=row_indices,
            proximal_points=points_numba,
            vector_array=edge_vectors,
            skeleton_coordinates=skeleton_coordinates,
            label_image=segmentation_label_image,
            skeleton_label_image=skeleton_label_image,
            angle_threshold=angle_threshold,
            weight=weight,
        )
    else:
        nearest_points = find_missing_branches(
            start_points=degree_1_nodes,
            proximal_points=points_numba,
            skeleton_coordinates=skeleton_coordinates,
            label_image=segmentation_label_image,
            skeleton_label_image=skeleton_label_image,
        )

    to_join_mask = np.logical_not(np.any(np.isnan(nearest_points), axis=1))
    source_coordinates = degree_1_nodes[to_join_mask]
    destination_coordinates = nearest_points[to_join_mask]

    # get the ids of the nodes of the source_coordinates
    degree_1_node_ids = np.squeeze(np.argwhere(skeleton_obj.degrees == 1))
    node_ids = degree_1_node_ids[to_join_mask]
    return node_ids, source_coordinates, destination_coordinates


def prune_and_fix_skeleton(
    skeplex_skeleton, segmentation, branch_trimming_len=10, break_distance=20
):
    """
    Iteratively prunes short branches and fixes breaks in a skeleton until none remain.

    Parameters
    ----------
    skeplex_skeleton : ndarray
        Binary skeleton image.
    segmentation : ndarray
        Segmentation image used for break detection.
    branch_trimming_len : float
        Maximum length of branches to prune.
    break_distance : float
        Maximum distance for detecting breaks.

    Returns
    -------
    Skeleton
        The final pruned and fixed Skeleton object.
    """
    skeleton_obj = Skeleton(skeplex_skeleton)
    total_pruned = 0
    total_breaks_fixed = 0

    while True:
        # --- Step 1: Prune short branches ---
        df = summarize(skeleton_obj)
        print(
            "num_deg1 pre prune",
            len(skeleton_obj.coordinates[skeleton_obj.degrees == 1]),
        )

        short_branches = df.loc[
            (df["branch-distance"] < branch_trimming_len) & (df["branch-type"] != 2)
        ]
        short_branch_indices = short_branches.index.to_list()

        if short_branch_indices:
            skeleton_obj = skeleton_obj.prune_paths(short_branch_indices)
            total_pruned += len(short_branch_indices)
            print(f"{len(short_branch_indices)} branches pruned")
        else:
            print("No short branches left to prune.")
        # --- Step 2: Detect and fix breaks ---
        skeleton_label_fill = skeleton_obj.skeleton_image.copy()
        skelplex_fill = label(skeleton_label_fill)

        node_ids, source_coords, target_coords = find_breaks_in_skeleton(
            skeleton_obj, break_distance, segmentation, skelplex_fill
        )
        print(f"Found {len(node_ids)} breaks in skeleton")

        if len(node_ids) > 0:
            for source, target in zip(source_coords, target_coords, strict=False):
                draw_line_segment(source, target, skelplex_fill)
            total_breaks_fixed += len(node_ids)
            skeleton_obj = Skeleton(skelplex_fill > 0)  # Update skeleton
        else:
            print("No breaks found.")
        # --- Exit condition ---
        if not short_branch_indices and not len(node_ids) > 0:
            print("Pruning and break fixing complete. ")
            break

    return skelplex_fill


def _precompute_edges(summary):
    adj = defaultdict(list)
    for src, dst in zip(summary["node-id-src"], summary["node-id-dst"], strict=False):
        adj[src].append(dst)
        adj[dst].append(src)  # undirected graph
    return adj


def _get_edge_vectors_degree1(skeleton_obj):
    vectors = {}
    summary = summarize(skeleton_obj)
    coords = skeleton_obj.coordinates
    adj = _precompute_edges(summary)

    for node_id in _get_degree_1_nodes_ids(skeleton_obj):
        neighbor = adj[node_id][0]  # degree-1 → exactly one neighbor
        edge_vector = coords[node_id] - coords[neighbor]
        vectors[node_id] = unit_vector(edge_vector)
    return vectors


def _get_degree_1_nodes(skeleton_obj: Skeleton) -> np.ndarray:
    """Get the end points of the skeleton.

    End points are defined as nodes with degree 1.

    Parameters
    ----------
    skeleton_obj : Skeleton
        The skan skeleton object that you want to get the end points from.

    Returns
    -------
    np.ndarray
        (n x 3) array of the coordinates of the end points of the skeleton.
    """
    return skeleton_obj.coordinates[skeleton_obj.degrees == 1]


def _get_degree_1_nodes_ids(skeleton_obj):
    """Get the IDs of the end points of the skeleton.

    End points are defined as nodes with degree 1.

    Parameters
    ----------
    skeleton_obj : Skeleton
        The skan skeleton object that you want to get the end points from.

    Returns
    -------
    np.ndarray
        (n,) array of the IDs of the end points of the skeleton.
    """
    return np.where(skeleton_obj.degrees == 1)[0]


def _vector_dict_to_array(degree_1_nodes_ids, vector_dict):
    id_to_index = {id_: i for i, id_ in enumerate(degree_1_nodes_ids)}
    vector_array = np.vstack([vector_dict[id_] for id_ in degree_1_nodes_ids]).astype(
        np.float64
    )

    row_indices = np.array(
        [id_to_index[id_] for id_ in degree_1_nodes_ids], dtype=np.int64
    )
    return vector_array, row_indices


@jit(fastmath=True, cache=True)
def unit_vector_numba(vector):
    """Compute the unit vector of a given vector.

    Is numba compatible.

    Parameters
    ----------
    vector : np.ndarray
        (3,) array of the vector to normalize.

    Returns
    -------
    np.ndarray
        (3,) array of the unit vector.
    """
    norm = np.sqrt(np.sum(vector * vector))
    if norm == 0:
        return vector
    return vector / norm
