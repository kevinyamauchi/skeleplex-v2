import random  # noqa
import networkx as nx
import numba
import numpy as np
import trimesh

import os

try:
    if os.name == "nt":
        # windows does not support pysdf
        from igl import signed_distance
    if os.name == "posix":
        from pysdf import SDF
except ImportError as err:
    raise ImportError(
        'Please install the "synthetic_data" extra '
        "to use this functionality: pip install skeleplex-v2[synthetic_data]"
        "For windows, please install 'libigl>=2.6.1' instead of 'pysdf'."
    ) from err

from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R
from skimage.filters import gaussian
from skimage.measure import marching_cubes
from skimage.morphology import ball, binary_dilation

from skeleplex.graph.constants import (
    EDGE_COORDINATES_KEY,
    NODE_COORDINATE_KEY,
)
from skeleplex.graph.skeleton_graph import SkeletonGraph


def generate_low_freq_noise(shape, scale=30, smooth_sigma=10):
    """Generate low-frequency noise for structured displacement.

    Parameters
    ----------
    shape : tuple
        Shape of the noise array (Z, Y, X).
    scale : float
        Scale of the noise.
    smooth_sigma : float
        Standard deviation for Gaussian smoothing.

    Returns
    -------
    np.ndarray
        Low-frequency noise array with the specified shape and scale.
    """
    noise = np.random.normal(size=shape)
    smooth_noise = gaussian_filter(noise, sigma=smooth_sigma)
    return smooth_noise / np.max(np.abs(smooth_noise)) * scale


def apply_structured_noise(mesh, noise_field):
    """Apply structured noise to a mesh by displacing its vertices.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh to which noise will be applied.
    noise_field : np.ndarray
        3D array of noise values to displace the mesh vertices.

    Returns
    -------
    trimesh.Trimesh
        The mesh with applied noise.
    """
    verts = mesh.vertices
    # Round and clip coordinates to index into volume
    coords = np.round(verts).astype(int)
    coords = np.clip(coords, [0, 0, 0], np.array(noise_field.shape) - 1)

    # Use the noise as displacement along normals
    displacement = noise_field[coords[:, 0], coords[:, 1], coords[:, 2]]
    normals = mesh.vertex_normals
    verts += normals * displacement[:, np.newaxis]

    mesh.vertices = verts
    return mesh


def add_noise_to_image_surface(image, noise_magnitude=15):
    """
    Adds noise to the surface of a binary 3D image and voxelizes the result.

    This function uses marching cubes to extract the surface mesh of the binary image,
    adds Gaussian noise to the mesh, and then voxelizes the result.

    Parameters
    ----------
    image : (Z, Y, X) ndarray
        Binary image where the object is labeled as 1.
    noise_magnitude : float
        Magnitude of Gaussian displacement applied to the surface mesh.

    Returns
    -------
    ndarray
        Binary voxel volume (Z, Y, X) of the noisy shape.
    """
    # Get surface mesh
    # time_start = time()
    verts, faces, _, _ = marching_cubes(image.astype(float), level=0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # add Gaussian noise to vertices
    noise_field = generate_low_freq_noise(
        image.shape, scale=noise_magnitude, smooth_sigma=10
    )
    mesh = apply_structured_noise(mesh, noise_field)
    mesh.vertices += np.random.normal(
        scale=noise_magnitude / 4, size=mesh.vertices.shape
    )
    # Smooth
    mesh = trimesh.smoothing.filter_taubin(mesh, lamb=0.7, nu=-0.3, iterations=10)
    mesh.fix_normals()

    x, y, z = np.indices(image.shape)
    coords = np.stack((x, y, z), axis=-1)  # shape:
    coords = coords.reshape(-1, 3)  # Flatten to (N, 3)

    # Compute signed distances
    # if on windows, use igl signed distance
    if os.name == "nt":
        signed_distances, _, _, _ = signed_distance(
            coords, np.array(mesh.vertices), np.array(mesh.faces), sign_type=1
        )

        noisey_image = signed_distances.reshape(image.shape).copy()
        noisey_image[noisey_image >= 0] = 0  # Set inside points to 1
        noisey_image[noisey_image < 0] = 1
    else:
        sdgf = SDF(mesh.vertices, mesh.faces)
        noisey_image = sdgf.contains(coords).reshape(image.shape)

    return noisey_image.astype(np.uint8)


def draw_ellipsoid_at_point(mask, point, radii):
    """
    Draw an ellipsoid at a point in a 3D binary mask.

    Parameters
    ----------
    mask : np.ndarray
        3D binary mask (Z, Y, X)
    point : tuple or np.ndarray
        Center of the ellipsoid in (x, y, z) coordinates
    radii : tuple or list
        Radii (rx, ry, rz) along X, Y, Z
    """
    px, py, pz = point[[2, 1, 0]]
    rx, ry, rz = radii
    shape = mask.shape

    zmin = max(0, int(pz - rz) - 1)
    zmax = min(shape[0], int(pz + rz) + 1)
    ymin = max(0, int(py - ry) - 1)
    ymax = min(shape[1], int(py + ry) + 1)
    xmin = max(0, int(px - rx) - 1)
    xmax = min(shape[2], int(px + rx) + 1)

    zz, yy, xx = np.meshgrid(
        np.arange(zmin, zmax),
        np.arange(ymin, ymax),
        np.arange(xmin, xmax),
        indexing="ij",
    )

    # Compute normalized squared distance
    norm_sq = ((xx - px) / rx) ** 2 + ((yy - py) / ry) ** 2 + ((zz - pz) / rz) ** 2
    local_mask = norm_sq <= 1

    mask[zmin:zmax, ymin:ymax, xmin:xmax] |= local_mask.astype(np.uint8)


def select_points_in_bounding_box(
    points: np.ndarray,
    lower_left_corner: np.ndarray,
    upper_right_corner: np.ndarray,
) -> np.ndarray:
    """Select all points inside a specified axis-aligned bounding box.

    Parameters
    ----------
    points : np.ndarray
        The n x d array containing the n, d-dimensional points to check.
    lower_left_corner : np.ndarray
        The point corresponding to the corner of the bounding box
        with lowest coordinate values.
    upper_right_corner : np.ndarray
        The point corresponding to the corner of the bounding box
        with the highest coordinate values.

    Returns
    -------
    points_in_box : np.ndarray
        The n x d array containing the n points inside of the
        specified bounding box.
    """
    in_box_mask = np.all(
        np.logical_and(lower_left_corner <= points, upper_right_corner >= points),
        axis=1,
    )
    return points[in_box_mask]


def draw_line_segment(
    start_point: np.ndarray,
    end_point: np.ndarray,
    skeleton_image: np.ndarray,
    fill_value: int = 1,
):
    """Draw a line segment in-place.

    Note: line will be clipped if it extends beyond the
    bounding box of the skeleton_image.

    Parameters
    ----------
    start_point : np.ndarray
        (d,) array containing the starting point of the line segment.
        Must be an integer index.
    end_point : np.ndarray
        (d,) array containing the end point of the line segment.
        Most be an integer index
    skeleton_image : np.ndarray
        The image in which to draw the line segment.
        Must be the same dimensionality as start_point and end_point.
    fill_value : int
        The value to use for the line segment.
        Default value is 1.
    """
    branch_length = np.linalg.norm(end_point - start_point)
    n_skeleton_points = int(2 * branch_length)
    skeleton_points = np.linspace(start_point, end_point, n_skeleton_points)

    # filter for points within the image
    image_bounds = np.asarray(skeleton_image.shape) - 1
    skeleton_points = select_points_in_bounding_box(
        points=skeleton_points,
        lower_left_corner=np.array([0, 0, 0]),
        upper_right_corner=image_bounds,
    ).astype(int)
    skeleton_image[
        skeleton_points[:, 0], skeleton_points[:, 1], skeleton_points[:, 2]
    ] = fill_value


def draw_line_segment_wiggle(
    start_point: np.ndarray,
    end_point: np.ndarray,
    skeleton_image: np.ndarray,
    fill_value: int = 1,
    wiggle_factor: float = 0.01,
    axis: int | None = None,
):
    """Draw a line segment in-place.

    Note: line will be clipped if it extends beyond the
    bounding box of the skeleton_image.

    Parameters
    ----------
    start_point : np.ndarray
        (d,) array containing the starting point of the line segment.
        Must be an integer index.
    end_point : np.ndarray
        (d,) array containing the end point of the line segment.
        Most be an integer index
    skeleton_image : np.ndarray
        The image in which to draw the line segment.
        Must be the same dimensionality as start_point and end_point.
    fill_value : int
        The value to use for the line segment.
        Default value is 1.
    wiggle_factor : float
        The factor by which to wiggle the line segment.
        Default value is 0.01.
    axis : int, optional
        The axis along which to apply the wiggle.
        If None, a random axis will be chosen.
        Default is None.
    """
    branch_length = np.linalg.norm(end_point - start_point)
    n_skeleton_points = int(2 * branch_length)
    frequency = wiggle_factor * branch_length
    amplitude = wiggle_factor * branch_length

    skeleton_points = np.linspace(start_point, end_point, n_skeleton_points)

    # Generate sinusoidal noise
    time_points = np.linspace(0, 1, n_skeleton_points)
    sinusoidal_noise = amplitude * np.sin(2 * np.pi * frequency * time_points)

    # Add sinusoidal noise to the line
    if axis is None:
        axis = np.random.randint(0, 3)
    skeleton_points[1:-1, axis] += (
        np.random.uniform(-0.5, 0.5, size=n_skeleton_points - 2)
        + sinusoidal_noise[1:-1]
    )

    for i in range(1, len(skeleton_points) - 1):
        draw_line_segment(
            skeleton_points[i],
            skeleton_points[i + 1],
            skeleton_image,
            fill_value=fill_value,
        )


def crop_to_content(branch_image, skeleton_image):
    """Crop the branch and skeleton images to the content."""
    branch_nonzero = np.argwhere(branch_image)
    if branch_nonzero.size == 0:
        return branch_image, skeleton_image

    min_coords = np.array(
        [
            np.min(np.where(branch_image)[0]),
            np.min(np.where(branch_image)[1]),
            np.min(np.where(branch_image)[2]),
        ]
    )
    max_coords = np.array(
        [
            np.max(np.where(branch_image)[0]),
            np.max(np.where(branch_image)[1]),
            np.max(np.where(branch_image)[2]),
        ]
    )

    branch_cropped = branch_image[
        min_coords[0] : max_coords[0] + 1,
        min_coords[1] : max_coords[1] + 1,
        min_coords[2] : max_coords[2] + 1,
    ]

    skeleton_cropped = skeleton_image[
        min_coords[0] : max_coords[0] + 1,
        min_coords[1] : max_coords[1] + 1,
        min_coords[2] : max_coords[2] + 1,
    ]

    return branch_cropped, skeleton_cropped


@numba.njit(parallel=True)
def draw_cylinder_segment(mask, a, b, radius):
    """Draw a cylinder segment in a 3D binary mask.

    Parameters
    ----------
    mask : np.ndarray
        3D binary mask (Z, Y, X) to draw the cylinder into.
    a : np.ndarray
        Starting point of the cylinder in (z, y, x) coordinates.
    b : np.ndarray
        Ending point of the cylinder in (z, y, x) coordinates.
    radius : float
        Radius of the cylinder.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    d = b - a
    length = np.linalg.norm(d)
    if length == 0:
        return

    d_unit = d / length
    radius_sq = radius * radius
    margin = int(radius) + 2

    shape = mask.shape
    zmin = max(0, int(min(a[2], b[2]) - margin))
    zmax = min(shape[0], int(max(a[2], b[2]) + margin + 1))
    ymin = max(0, int(min(a[1], b[1]) - margin))
    ymax = min(shape[1], int(max(a[1], b[1]) + margin + 1))
    xmin = max(0, int(min(a[0], b[0]) - margin))
    xmax = min(shape[2], int(max(a[0], b[0]) + margin + 1))

    for z in numba.prange(zmin, zmax):
        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                p = np.array([x, y, z], dtype=np.float32)
                pa = p - a
                t = np.dot(pa, d_unit)
                if t < 0:
                    t = 0
                elif t > length:
                    t = length
                closest = a + t * d_unit
                dist_sq = np.sum((p - closest) ** 2)
                if dist_sq <= radius_sq:
                    mask[z, y, x] = 1


def draw_wiggly_cylinder_3d(
    mask, start_point, end_point, radius=2, wiggle_factor=0.01, axis=None
):
    """Draw a wiggly cylinder in a 3D binary mask.

    Parameters
    ----------
    mask : np.ndarray
        3D binary mask (Z, Y, X) to draw the cylinder into.
    start_point : np.ndarray
        Starting point of the cylinder in (z, y, x) coordinates.
    end_point : np.ndarray
        Ending point of the cylinder in (z, y, x) coordinates.
    radius : float
        Radius of the cylinder.
    wiggle_factor : float
        Factor controlling the amount of wiggle.
    axis : int, optional
        Axis along which to apply the wiggle (0: x, 1: y, 2: z).

    """
    start_point = np.array(start_point)[[2, 1, 0]]
    end_point = np.array(end_point)[[2, 1, 0]]
    axis = np.random.randint(0, 3) if axis is None else axis
    axis = 2 - axis  # Adjust axis for z,y,x order
    branch_length = np.linalg.norm(end_point - start_point)
    n_skeleton_points = int(2 * branch_length)
    frequency = wiggle_factor * branch_length
    amplitude = wiggle_factor * branch_length

    # Linear skeleton path
    skeleton_points = np.linspace(start_point, end_point, n_skeleton_points)

    # Generate sinusoidal noise
    time_points = np.linspace(0, 1, n_skeleton_points)
    sinusoidal_noise = amplitude * np.sin(2 * np.pi * frequency * time_points)

    # Wiggle in a random axis

    skeleton_points[1:-1, axis] += (
        np.random.uniform(-0.5, 0.5, size=n_skeleton_points - 2)
        + sinusoidal_noise[1:-1]
    )

    # Draw cylinders along wiggly skeleton
    for i in range(len(skeleton_points) - 1):
        a = skeleton_points[i]
        b = skeleton_points[i + 1]
        draw_cylinder_segment(mask, a, b, radius=radius)


@numba.njit(parallel=True)
def _draw_elliptic_cylinder_segment(mask, a, b, rx, ry):
    """Draw an elliptic cylinder segment in a 3D binary mask.

    Parameters
    ----------
    mask : np.ndarray
        3D binary mask (Z, Y, X) to draw the segment into.
    a : np.ndarray
        Starting point of the segment in (z, y, x) coordinates.
    b : np.ndarray
        Ending point of the segment in (z, y, x) coordinates.
    rx : float
        Radius along the x-axis of the ellipse.
    ry : float
        Radius along the y-axis of the ellipse.

    """
    d = b - a
    length = np.sqrt(np.sum(d**2))
    if length == 0.0:
        return

    d_unit = d / length
    margin = int(max(rx, ry)) + 2
    shape = mask.shape

    zmin = max(0, int(min(a[2], b[2]) - margin))
    zmax = min(shape[0], int(max(a[2], b[2]) + margin + 1))
    ymin = max(0, int(min(a[1], b[1]) - margin))
    ymax = min(shape[1], int(max(a[1], b[1]) + margin + 1))
    xmin = max(0, int(min(a[0], b[0]) - margin))
    xmax = min(shape[2], int(max(a[0], b[0]) + margin + 1))

    # Create orthonormal frame (u, v) using manual dot products
    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(d_unit[0]) > 0.9:  # d_unit is too close to x-axis
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Just a cross product, but numba complained about np.cross...
    u = np.array(
        [
            tmp[1] * d_unit[2] - tmp[2] * d_unit[1],
            tmp[2] * d_unit[0] - tmp[0] * d_unit[2],
            tmp[0] * d_unit[1] - tmp[1] * d_unit[0],
        ],
        dtype=np.float32,
    )

    u_norm = np.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
    if u_norm > 0:
        u = u / u_norm

    v = np.array(
        [
            d_unit[1] * u[2] - d_unit[2] * u[1],
            d_unit[2] * u[0] - d_unit[0] * u[2],
            d_unit[0] * u[1] - d_unit[1] * u[0],
        ],
        dtype=np.float32,
    )

    v_norm = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if v_norm > 0:
        v = v / v_norm

    for z in numba.prange(zmin, zmax):
        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                p = np.array(
                    [np.float32(x), np.float32(y), np.float32(z)], dtype=np.float32
                )
                pa = p - a
                # Just a dot product, but numba complained about np.dot...
                t = pa[0] * d_unit[0] + pa[1] * d_unit[1] + pa[2] * d_unit[2]
                if t < 0 or t > length:
                    continue  #
                # if t < 0:
                #     t = 0
                # elif t > length:
                #     t = length
                closest = a + t * d_unit
                offset = p - closest
                # Manual dot products to ensure consistent dtypes
                u_proj = (offset[0] * u[0] + offset[1] * u[1] + offset[2] * u[2]) / rx
                v_proj = (offset[0] * v[0] + offset[1] * v[1] + offset[2] * v[2]) / ry
                if u_proj * u_proj + v_proj * v_proj <= 1.0:
                    mask[z, y, x] = 1


def draw_elliptic_cylinder_segment(mask, a, b, rx, ry):
    """Wrapper to ensure clean Numba-compatible input types."""
    a = np.array(a, dtype=np.float32)
    a = a[[2, 1, 0]]  # Convert to z,y,x order
    b = np.array(b, dtype=np.float32)
    b = b[[2, 1, 0]]  # Convert to z,y,x order
    _draw_elliptic_cylinder_segment(mask, a, b, np.float32(rx), np.float32(ry))


def make_skeleton_blur_image(
    skeleton_image: np.ndarray, dilation_size: float, gaussian_size: float
) -> np.ndarray:
    """Create a blurred skeleton image from a binary skeleton image.

    Parameters
    ----------
    skeleton_image : np.ndarray
        Binary image where the skeleton is represented by non-zero values.
        Should be a 3D array.
    dilation_size : float
        Size of the structuring element used for dilation.
        This controls how much the skeleton is dilated before blurring.
    gaussian_size : float
        Standard deviation for the Gaussian filter used for blurring.
        This controls the amount of blurring applied to the skeleton.

    Returns
    -------
    np.ndarray
        Blurred skeleton image, where the skeleton voxels have a value of 1,
        and the background is normalized to 0.
    """
    # make a blurred skeleton
    dilated_skeleton = binary_dilation(skeleton_image, ball(dilation_size))
    skeleton_blur = gaussian(dilated_skeleton, gaussian_size)

    # normalize the values
    skeleton_blur /= skeleton_blur.max()

    # ensure the skeleton voxels have a value of 1
    skeleton_coordinates = np.where(skeleton_image)
    skeleton_blur[skeleton_coordinates] = 1

    return skeleton_blur


def generate_toy_graph_symmetric_branch_angle(num_nodes, angle=72, edge_length=20):
    """Generate a toy skeleton graph with a symmetric branch angle.

    Parameters
    ----------
    num_nodes : int
        The total number of nodes in the graph.
    angle : float
        The angle in degrees between branches at each node.
        Default is 72 degrees, which corresponds to a 5-fold symmetry.
    edge_length : int
        The length of the start edge in the graph. Each generation shrinks by 80%.
        Default is 20.

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
        if np.log2(i - 1).is_integer():
            edge_length = int(
                0.79 * edge_length
            )  # Decrease edge length for subsequent branches
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
            # Decrease edge length for the next generation

        parent_nodes = new_parents  # Update parent nodes for the next iteration

    # Set node attributes for the graph
    nx.set_node_attributes(graph, node_pos_dic, NODE_COORDINATE_KEY)

    # Create a SkeletonGraph from the graph
    skeleton_graph = SkeletonGraph.from_graph(
        graph, EDGE_COORDINATES_KEY, NODE_COORDINATE_KEY
    )
    return skeleton_graph


def add_rotation_to_tree(tree: nx.DiGraph, rotation_angle: float | None = None):
    """Adds a random rotation between -90 and 90 degrees to each level of the tree.

    Each node needs to contain the 3D positional coordinates (array with shape (3,))
    in attribute NODE_COORDINATE_KEY.
    Rotation will happen along axis 0 (x-axis).

    NOTE: This function does not rotate edge coordinates.
    Might implement this in the future.

    Parameters
    ----------
    tree: nx.DiGraph
        The tree to rotate.
        Eg. to graph from generate_toy_graph_symmetric_branch_angle.
    rotation_angle: float | None
        If provided, the rotation angle in degrees to apply to the tree.
        If None, a random angle between -90 and 90 degrees will be used.
    """
    in_edges = list(tree.in_edges())
    in_edges_flatt = [node for edge in in_edges for node in edge]

    for nodes, degree in tree.degree():
        if rotation_angle is not None:
            rot_degree = -np.radians(rotation_angle)
        else:
            rot_degree = np.radians(random.sample(list(np.linspace(-90, 90, 10)), 1))[0]

        updated_positions = {}
        if degree == 3:
            if nodes in in_edges_flatt:
                sub_tree = tree.subgraph(nx.dfs_tree(tree, source=nodes).nodes())

                root = np.array(
                    dict(sub_tree.nodes(data=True))[nodes][NODE_COORDINATE_KEY]
                )

                for node in sub_tree.nodes():
                    p = tree.nodes(data=NODE_COORDINATE_KEY)[node]
                    p = p - root

                    # Rotation matrix around X-axis (axis 0)
                    R_matrix = np.array(
                        [
                            [1, 0, 0],
                            [0, np.cos(rot_degree), -np.sin(rot_degree)],
                            [0, np.sin(rot_degree), np.cos(rot_degree)],
                        ]
                    )

                    r = R.from_matrix(R_matrix)
                    p_rot = r.apply(p)
                    p_rot = p_rot + root
                    updated_positions[node] = p_rot.flatten()

            nx.set_node_attributes(tree, updated_positions, name=NODE_COORDINATE_KEY)

    # avoid negative coordinates
    pos_dict = nx.get_node_attributes(tree, NODE_COORDINATE_KEY)
    pos_values = np.array(list(pos_dict.values()))

    x_shift = np.abs(np.min(pos_values[:, 0])) + 10
    y_shift = np.abs(np.min(pos_values[:, 1])) + 30
    z_shift = np.abs(np.min(pos_values[:, 2])) + 10

    pos = {k: v + np.array([x_shift, y_shift, z_shift]) for k, v in pos_dict.items()}
    nx.set_node_attributes(tree, pos, NODE_COORDINATE_KEY)


def augment_tree(tree: nx.DiGraph, rotation_angles: list | None = None) -> None:
    """
    Augment a tree by rotating it along the trachea.

    Each node needs to contain the 3D positional coordinates (array with shape (3,))

    NOTE: This function does not rotate edge coordinates.
    Might implement this in the future.

    Parameters
    ----------
    tree : nx.DiGraph
        The tree to augment.
        Eg. to graph from generate_toy_graph_symmetric_branch_angle.
    rotation_angles : float | None
        If provided, the rotation angle in degrees to apply to the tree.
        If None, a random angle between -30 and 30 degrees for each axis will be used.

    """
    # Rotation matrices
    if rotation_angles is not None:
        rot_degree = np.radians(rotation_angles)
    else:
        # Randomly sample a rotation degree from -30 to 30 degrees
        # for each axis
        # 3 rotations, one for each axis
        rot_degree = np.radians(random.sample(list(np.linspace(-30, 30, 60)), 3))
    R_matrix_x = np.array(
        (
            [1, 0, 0],
            [0, np.cos(rot_degree[0]), -np.sin(rot_degree[0])],
            [0, np.sin(rot_degree[0]), np.cos(rot_degree[0])],
        )
    )

    R_matrix_y = np.array(
        (
            [np.cos(rot_degree[1]), 0, np.sin(rot_degree[1])],
            [0, 1, 0],
            [-np.sin(rot_degree[1]), 0, np.cos(rot_degree[1])],
        )
    )

    R_matrix_z = np.array(
        (
            [np.cos(rot_degree[2]), -np.sin(rot_degree[2]), 0],
            [np.sin(rot_degree[2]), np.cos(rot_degree[2]), 0],
            [0, 0, 1],
        )
    )
    r_x = R.from_matrix(R_matrix_x)
    r_y = R.from_matrix(R_matrix_y)
    r_z = R.from_matrix(R_matrix_z)

    # Augment tree by rotating it along the trachea
    pos = nx.get_node_attributes(tree, NODE_COORDINATE_KEY)

    update_pos = {}
    for node, node_pos in pos.items():
        root = pos[0]
        p = node_pos - root

        p_rot = r_x.apply(p)
        p_rot = r_y.apply(p_rot)
        p_rot = r_z.apply(p_rot)
        p_rot = p_rot + root
        update_pos[node] = p_rot.flatten()
    nx.set_node_attributes(tree, update_pos, name=NODE_COORDINATE_KEY)
