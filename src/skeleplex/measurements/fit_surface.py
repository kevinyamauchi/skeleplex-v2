import networkx as nx  # noqa: D100
import numpy as np
import trimesh
import trimesh as tri
from scipy.interpolate import Rbf
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA

from skeleplex.graph.constants import (
    EDGE_SPLINE_KEY,
    LOBE_NAME_KEY,
    NODE_COORDINATE_KEY,
)


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


def fit_surface_to_pointcloud_rbf_pca(points: np.ndarray, smooth=0.2, percentile=100):
    """Fits an radial basis function surface to a pointcloud using PCA.

    Can be fitted to a central region of the pointcloud.
    PCA is used to find the principle axis of the point cloud at which axis the
    surface is fitted.

    See Also
    --------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html

    Parameters
    ----------
    points : np.ndarray
        (n_points, 3) array of 3D points.
    smooth : float
        RBF smoothing factor. 0 would fit the surface exactly to the points.
        Thus to fit a surface through a pointcloud, a high value is recommended.
    percentile : float
        How much of the central region to retain (default 100%).

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
        np.linspace(np.min(x) - 50, np.max(x) + 50, 50),
        np.linspace(np.min(y) - 50, np.max(y) + 50, 50),
    )
    z_grid = rbf(x_grid, y_grid)

    # Reconstruct the surface back into original space
    grid_points_pca = np.vstack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel())).T
    original_grid_points = (
        grid_points_pca @ components
    ) + mean  # Correct transformation

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
    if len(points.shape) == 1:
        points = np.expand_dims(points, axis=0)

    _, distance, closest_triangle = mesh.nearest.on_surface(points)
    normals = tri.triangles.normals(mesh.triangles)[0]
    normal_dict = {}
    distance_dict = {}

    for i in range(len(points)):
        normal_dict[tuple(points[i])] = normals[closest_triangle[i]]
        distance_dict[tuple(points[i])] = distance[i]

    return normal_dict, distance_dict


def fit_surface_and_get_surface_normal_of_nodes(
    graph: nx.DiGraph, lobe_name: str, smooth=1000
):
    """
    Fits a surface to the central region of a point cloud and calculate normals.

    The normals is the surface normal of the closest point on the surface to the node.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph containing the nodes.
    lobe_name : str
        The lobe to process.
    smooth : float
        RBF smoothing factor.

    Returns
    -------
    normals : dict
        A dictionary mapping each node to its corresponding normal vector.
    distances : dict
        A dictionary mapping each node to its distance to the surface.
    (vertices, faces) : tuple
        The vertices and faces of the fitted surface.
    """
    # Extract the point cloud of the lobe
    lobe_node_dict = {}
    node_coords = nx.get_node_attributes(graph, NODE_COORDINATE_KEY)
    for u, v, lobe in graph.edges(data=LOBE_NAME_KEY):
        if lobe not in lobe_node_dict:
            lobe_node_dict[lobe] = []
        lobe_node_dict[lobe].extend([node_coords[u], node_coords[v]])

    lobe_points = np.array(lobe_node_dict[lobe_name])
    vertices, faces = fit_surface_to_pointcloud_rbf_pca(
        lobe_points, smooth=smooth, percentile=100
    )
    mesh = tri.Trimesh(vertices, faces)
    normals, distances = get_normal_of_closest_point(mesh, lobe_points)
    return normals, distances, (vertices, faces)


def fit_surface_and_get_surface_normal_of_branches(
    graph: nx.DiGraph, lobe_name: str, smooth=1000
):
    """
    Fits a surface to the central region of a point cloud and calculate normals.

    The normals is the surface normal of the closest point on the
    surface to the branch midpoint.

    Parameters
    ----------
    graph : nx.DiGraph
         The graph containing the nodes.
    lobe_name : str
         The lobe to process.
    smooth : float
         RBF smoothing factor.

    Returns
    -------
    normals : dict
         A dictionary mapping each node to its corresponding normal vector.
    distances : dict
         A dictionary mapping each node to its distance to the surface.
    (vertices, faces) : tuple
         The vertices and faces of the fitted surface.
    """
    # Extract the point cloud of the lobe
    lobe_node_dict = {}
    lobe_edge_ID_dict = {}

    node_coords = nx.get_node_attributes(graph, NODE_COORDINATE_KEY)
    for u, v, lobe in graph.edges(data=LOBE_NAME_KEY):
        if lobe not in lobe_node_dict:
            lobe_node_dict[lobe] = []
            lobe_edge_ID_dict[lobe] = []
        lobe_node_dict[lobe].extend([node_coords[u], node_coords[v]])
        lobe_edge_ID_dict[lobe].append((u, v))

    lobe_coordinates = np.array(lobe_node_dict[lobe_name])
    lobe_edges = lobe_edge_ID_dict[lobe_name]
    vertices, faces = fit_surface_to_pointcloud_rbf_pca(
        lobe_coordinates, smooth=smooth, percentile=100
    )

    mesh = tri.Trimesh(vertices, faces)

    if not all("branch_midpoint" in graph.edges[u, v] for u, v in graph.edges):
        splines = nx.get_edge_attributes(graph, EDGE_SPLINE_KEY)
        spline_midpoint_dict = {}
        # get the midpoint of each spline
        for edge, spline in splines.items():
            spline_midpoint_dict[edge] = spline.eval(0.1, approx=True)
    else:
        spline_midpoint_dict = nx.get_edge_attributes(graph, "branch_midpoint")

    lobe_midpoints = np.array([spline_midpoint_dict[key] for key in lobe_edges])

    normal_dict, distance_dict = get_normal_of_closest_point(mesh, lobe_midpoints)

    edge_normal_dict = {}
    edge_distance_dict = {}

    # Assign the normals and distances to the edges
    for edge in lobe_edges:
        edge_normal_dict[edge] = normal_dict[tuple(spline_midpoint_dict[edge])]
        edge_distance_dict[edge] = distance_dict[tuple(spline_midpoint_dict[edge])]

    return edge_normal_dict, edge_distance_dict, (vertices, faces)

    # return normals, distances, (vertices, faces)
