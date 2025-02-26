import networkx as nx  # noqa: D100
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation as R


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return np.squeeze(np.asarray(vector / np.linalg.norm(vector)))


def get_normal_of_plane(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    """Get the normal vector of a plane defined by three points.

    Parameters
    ----------
    p1 : np.ndarray
        First point.
    p2 : np.ndarray
        Second point.
    p3 : np.ndarray
        Third point.

    """
    v1 = p2 - p1
    v2 = p3 - p1
    cp = np.cross(v1, v2)
    if all(cp == 0):
        ValueError("The points are colinear")
    return cp


def ensure_same_normal_direction(normals: dict, reference_direction):
    """Ensure that all normals have the same direction."""
    for key, normal in normals.items():
        if np.sign(normal[0]) != reference_direction:
            print("flip")
            normals[key] = -normal  # Reverse the direction of the normal
    return normals


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray):
    """Compute the rotation matrix that rotates unit vector a onto unit vector b.

    Parameters
    ----------
    a : numpy.ndarray
        The initial unit vector.
    b : numpy.ndarray
        The target unit vector.

    Returns
    -------
    numpy.ndarray
        The rotation matrix.
    """
    # Compute the cross product and its magnitude
    v = np.cross(a, b)
    s = np.linalg.norm(v)

    # Compute the dot product
    c = np.dot(a, b)

    # Skew-symmetric cross-product matrix
    v_cross = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    # Rotation matrix
    if s != 0:
        rm = np.eye(3) + v_cross + np.dot(v_cross, v_cross) * ((1 - c) / (s**2))
    else:
        rm = np.eye(3)

    return R.from_matrix(rm)


def rad2deg(rad):
    """Convert radians to degrees."""
    return rad * 180 / np.pi


def deg2rad(deg):
    """Convert degrees to radians."""
    return deg * np.pi / 180


def graph_attributes_to_df(graph: nx.Graph):
    """Converts all edge attributes of a graph to a pandas dataframe.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to convert

    Returns
    -------
    attr_df : pd.DataFrame
        A pandas dataframe with all edge attributes

    """
    attr_dict = {}
    for u, v, attr in graph.edges(data=True):
        attr_dict[(u, v)] = attr

    attr_df = pd.DataFrame.from_dict(attr_dict, orient="index").reset_index(drop=True)

    return attr_df


def get_normal_of_closest_point(mesh: trimesh.Trimesh, points: np.ndarray):
    """Computes the normal of the surface at the closest point to a set of points.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh representing the surface.
    points : np.ndarray
        An array of shape (n_points, 3) containing the 3D coordinates of the points.

    Returns
    -------
    normal_dict : dict
        A dictionary mapping each point to its corresponding normal vector.
    distance_dict : dict
        A dictionary mapping each point to its distance to the surface.
    """
    _, distance, closest_triangle = mesh.nearest.on_surface(points)
    normals = trimesh.triangles.normals(mesh.triangles)[0]
    normal_dict = {}
    distance_dict = {}
    for i in range(len(points)):
        normal_dict[tuple(points[i])] = normals[closest_triangle[i]]
        distance_dict[tuple(points[i])] = distance[i]

    return normal_dict, distance_dict
