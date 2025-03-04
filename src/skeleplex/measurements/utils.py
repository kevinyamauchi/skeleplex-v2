import networkx as nx  # noqa: D100
import numpy as np
import pandas as pd
import trimesh
import trimesh as tri
from scipy.interpolate import Rbf
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA


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


def extract_central_region(points, percentile=50):
    """
    Extracts the central region of the point cloud using PCA and a distance threshold.

    Parameters
    ----------
    points : np.ndarray
        (n_points, 3) array containing the 3D point cloud.
    percentile : float
        The percentage of points to keep (default 50% around the mean).

    Returns
    -------
    central_points : np.ndarray
        The extracted central points.
    pca : PCA
        The PCA transformation object.
    mean : np.ndarray
        Mean of the original point cloud.
    components : np.ndarray
        The principal component axes.
    """
    # Center the points
    mean = np.mean(points, axis=0)
    centered_points = points - mean

    # Apply PCA
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    components = pca.components_  # Principal axes

    # Transform points into PCA space
    aligned_points = centered_points @ components.T  # Manual projection

    # Compute distances in PCA space (only in the XY plane)
    distances = np.linalg.norm(aligned_points[:, :2], axis=1)
    threshold = np.percentile(distances, percentile)

    # Select only the central points
    central_points = points[distances < threshold]
    return central_points, mean, components


def fit_surface_to_pointcloud_rbf_pca(points: np.ndarray, smooth=0.2, percentile=50):
    """Fits an RBF surface to the central part of a point cloud.

    Uses PCA for rotation invariance.

    Parameters
    ----------
    points : np.ndarray
        (n_points, 3) array of 3D points.
    smooth : float
        RBF smoothing factor.
    percentile : float
        How much of the central region to retain (default 50%).

    Returns
    -------
    vertices, faces : np.ndarray
        Mesh vertices and faces.
    """
    # Extract central region
    central_points, mean, components = extract_central_region(points, percentile)

    # Transform central points into PCA space
    pca_space_points = (central_points - mean) @ components.T  # Manual projection

    # Fit RBF surface on the central points in PCA space
    x, y, z = pca_space_points[:, 0], pca_space_points[:, 1], pca_space_points[:, 2]
    rbf = Rbf(x, y, z, function="multiquadric", smooth=smooth)

    # Generate a grid in PCA space
    x_grid, y_grid = np.meshgrid(
        np.linspace(np.min(x), np.max(x), 50), np.linspace(np.min(y), np.max(y), 50)
    )
    z_grid = rbf(x_grid, y_grid)

    # Reconstruct the surface back into original space
    grid_points_pca = np.vstack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel())).T
    original_grid_points = (grid_points_pca @ components) + mean
    # Build mesh
    vertices = original_grid_points
    tri = Delaunay(vertices[:, :2])
    faces = tri.simplices

    return vertices, faces


def get_normal_of_closest_surface_point(mesh: trimesh.Trimesh, points: np.ndarray):
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
    # Find the closest point on the surface
    _, distance, closest_triangle = mesh.nearest.on_surface(points)
    normals = trimesh.triangles.normals(mesh.triangles)[0]
    normal_dict = {}
    distance_dict = {}
    for i in range(len(points)):
        normal_dict[tuple(points[i])] = normals[closest_triangle[i]]
        distance_dict[tuple(points[i])] = distance[i]

    return normal_dict, distance_dict


def distance_to_surface(vertices, faces, points):
    """
    Computes the distance of a set of points to a surface defined by vertices and faces.

    Parameters
    ----------
    vertices : np.ndarray
        An array of shape (n_vertices, 3) containing the vertices of the surface.
    faces : np.ndarray
        An array of shape (n_faces, 3) containing the faces of the surface.
    points : np.ndarray
        An array of shape (n_points, 3) containing the 3D coordinates of the points.

    Returns
    -------
    distances : np.ndarray
        An array of shape (n_points,) containing the distances of each point
        to the surface.
    """
    mesh = tri.Trimesh(vertices, faces)
    closest_point, distance, triangles = mesh.nearest.on_surface(points)
    return closest_point, distance, triangles


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
    normals = tri.triangles.normals(mesh.triangles)[0]
    normal_dict = {}
    distance_dict = {}
    for i in range(len(points)):
        normal_dict[tuple(points[i])] = normals[closest_triangle[i]]
        distance_dict[tuple(points[i])] = distance[i]

    return normal_dict, distance_dict
