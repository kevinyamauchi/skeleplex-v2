import networkx as nx
import numpy as np
import trimesh

from skeleplex.graph.constants import (
    BRANCH_ANGLE_EDGE_KEY,
    EDGE_COORDINATES_KEY,
    EDGE_SPLINE_KEY,
    LOBE_NAME_KEY,
    NODE_COORDINATE_KEY,
    ROTATION_ANGLE_EDGE_KEY,
    SIBLING_ANGLE_EDGE_KEY,
    SURFACE_ANGLE_EDGE_KEY,
)
from skeleplex.graph.skeleton_graph import SkeletonGraph
from skeleplex.measurements.angles import (
    compute_midline_branch_angle_branch_nodes,
    compute_rotation_angle,
    compute_sibling_angle,
    compute_surface_normals_and_angles,
)
from skeleplex.measurements.fit_surface import get_normal_of_closest_point
from skeleplex.measurements.utils import rad2deg, unit_vector


def project_points_on_surface(points, surface_mesh):
    """
    Projects points onto a surface along the z-axis.

    Parameters:
    points (np.ndarray): Array of points to be projected.
    surface_mesh (tuple): A tuple containing vertices and faces of the surface.

    Returns:
    np.ndarray: Array of projected points.
    """
    vertices, faces = surface_mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Define ray origins (same x, y as points, but high z)
    ray_origins = points.copy()
    ray_origins[:, 2] = np.max(vertices[:, 2]) + 50  # Start above the surface

    # Define ray directions (negative z-direction)
    ray_directions = np.zeros_like(points)
    ray_directions[:, 2] = -1

    # Perform ray-mesh intersection
    locations, index_ray, _ = mesh.ray.intersects_location(ray_origins, ray_directions)

    # Map intersections back to input points order
    projected_points = points.copy()
    projected_points[index_ray, 2] = locations[:, 2]

    return projected_points


def generate_curved_surface(
    curvature=0.3, n_points=100, x_lim=(0, 200), y_lim=(0, 200)
):
    """Generate a curved surface with cosine curvature in the x-direction.

    Parameters:
    curvature (float): Curvature of the surface.
    n_points (int): Number of points in each direction.
    x_lim (list): Range of x-values.
    y_lim (list): Range of y-values.

    Returns:
    tuple: Tuple containing vertices and faces of the surface.
    """

    # Generate grid of points in y direction
    y = np.linspace(y_lim[0], y_lim[1], n_points)
    x = np.linspace(x_lim[0], x_lim[1], n_points)
    X, Y = np.meshgrid(x, y)

    # Define curvature function for the middle part
    center_x = (x_lim[1] + x_lim[0]) / 2
    curve_amount = (x_lim[1] - x_lim[0]) * curvature / 2
    Z = curve_amount * np.cos((X - center_x) / curve_amount * np.pi)

    # Flatten arrays for vertices
    vertices = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

    # Create faces
    faces = []
    for i in range(n_points - 1):
        for j in range(n_points - 1):
            faces.append(
                [i * n_points + j, i * n_points + j + 1, (i + 1) * n_points + j]
            )
            faces.append(
                [
                    (i + 1) * n_points + j,
                    i * n_points + j + 1,
                    (i + 1) * n_points + j + 1,
                ]
            )
    faces = np.array(faces)

    return vertices, faces


def test_surface_angles_without_surface_fitting(
    generate_toy_skeleton_graph_symmetric_branch_angle,
):
    """Test calculating angles between edges and a curved surface

    Without fitting a new surface on the edges.
    """

    skeleton_toy = generate_toy_skeleton_graph_symmetric_branch_angle
    nx.set_edge_attributes(skeleton_toy.graph, "LeftLobe", LOBE_NAME_KEY)

    # Define a simple curved surface
    node_coords = nx.get_node_attributes(skeleton_toy.graph, NODE_COORDINATE_KEY)
    edge_coords = nx.get_edge_attributes(skeleton_toy.graph, EDGE_COORDINATES_KEY)
    min_x = min([node[0] for node in node_coords.values()])
    max_x = max([node[0] for node in node_coords.values()])
    min_y = min([node[1] for node in node_coords.values()])
    max_y = max([node[1] for node in node_coords.values()])

    surface = generate_curved_surface(
        curvature=3,
        n_points=100,
        x_lim=[min_x - 30, max_x + 30],
        y_lim=[min_y - 30, max_y + 30],
    )
    mesh = trimesh.Trimesh(vertices=surface[0], faces=surface[1])

    # project the nodes on the surface
    node_coords_projected = project_points_on_surface(
        np.array(list(node_coords.values())), surface
    )

    node_coords_projected = {
        i - 1: node_coords_projected[i] for i in range(len(node_coords_projected))
    }
    nx.set_node_attributes(
        skeleton_toy.graph, node_coords_projected, NODE_COORDINATE_KEY
    )

    # project the edges on the surface
    edge_coords_projected = {}
    for edge in edge_coords:
        edge_coords_projected[edge] = project_points_on_surface(
            edge_coords[edge], surface
        )
    nx.set_edge_attributes(
        skeleton_toy.graph, edge_coords_projected, EDGE_COORDINATES_KEY
    )

    skeleton_toy = SkeletonGraph.from_graph(
        skeleton_toy.graph, EDGE_COORDINATES_KEY, NODE_COORDINATE_KEY
    )

    graph = skeleton_toy.graph.copy()

    node_coords = nx.get_node_attributes(graph, NODE_COORDINATE_KEY)
    splines = nx.get_edge_attributes(graph, EDGE_SPLINE_KEY)
    for edge in graph.edges():
        parent_node = edge[0]
        edge_point = splines[edge].eval(0.01)
        edge_vector = edge_point - node_coords[parent_node]
        edge_vector = unit_vector(edge_vector)
        surface_normal, _ = get_normal_of_closest_point(mesh, edge_point)
        surface_normal = np.array(list(surface_normal.values()))
        angle = rad2deg(np.arccos(np.dot(surface_normal, edge_vector)))

        np.testing.assert_allclose(angle, 90, atol=3)


def test_surface_angles_with_surface_fitting(
    generate_toy_skeleton_graph_symmetric_branch_angle,
):
    """Test calculating angles between edges and a straighht surface

    With fitting a new surface on the edges.
    """
    skeleton_toy = generate_toy_skeleton_graph_symmetric_branch_angle
    nx.set_edge_attributes(skeleton_toy.graph, "LeftLobe", LOBE_NAME_KEY)

    compute_surface_normals_and_angles(
        [skeleton_toy], ["Test"], ["LeftLobe"], smooth=0.1
    )

    np.testing.assert_allclose(
        np.array(
            [
                attr
                for _, _, attr in skeleton_toy.graph.edges(data=SURFACE_ANGLE_EDGE_KEY)
            ]
        ),
        0,
        atol=1e-5,
    )


def test_branch_angle(generate_toy_skeleton_graph_symmetric_branch_angle):
    skeleton_toy = generate_toy_skeleton_graph_symmetric_branch_angle

    graph, _, _ = compute_midline_branch_angle_branch_nodes(skeleton_toy.graph)

    np.testing.assert_allclose(
        np.array(
            [
                attr
                for _, _, attr in graph.edges(data=BRANCH_ANGLE_EDGE_KEY)
                if attr is not None
            ]
        ),
        36,
        atol=1e-5,
    )


def test_sibling_angle(generate_toy_skeleton_graph_symmetric_branch_angle):
    skeleton_toy = generate_toy_skeleton_graph_symmetric_branch_angle

    graph = compute_sibling_angle(skeleton_toy.graph)

    np.testing.assert_allclose(
        np.array(
            [
                attr
                for _, _, attr in graph.edges(data=SIBLING_ANGLE_EDGE_KEY)
                if attr is not None
            ]
        ),
        72,
        atol=1e-5,
    )


def test_rotation_angle(generate_toy_skeleton_graph_symmetric_branch_angle):
    skeleton_toy = generate_toy_skeleton_graph_symmetric_branch_angle

    graph = compute_rotation_angle(skeleton_toy.graph)

    np.testing.assert_allclose(
        np.array(
            [
                attr
                for _, _, attr in graph.edges(data=ROTATION_ANGLE_EDGE_KEY)
                if attr is not None
            ]
        ),
        0,
        atol=1e-5,
    )
