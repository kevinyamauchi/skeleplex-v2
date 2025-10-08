import dask.array as da  # noqa: D100
import networkx as nx
import numpy as np

from skeleplex.data.utils import (
    add_noise_to_image_surface,
    add_rotation_to_tree,
    augment_tree,
    crop_to_content,
    draw_ellipsoid_at_point,
    draw_elliptic_cylinder_segment,
    draw_line_segment_wiggle,
    draw_wiggly_cylinder_3d,
    generate_toy_graph_symmetric_branch_angle,
    make_skeleton_blur_image,
)
from skeleplex.graph.constants import (
    LENGTH_KEY,
    NODE_COORDINATE_KEY,
)
from skeleplex.measurements.graph_properties import compute_branch_length, compute_level
from skeleplex.skeleton.distance_field import (
    local_normalized_distance,
    local_normalized_distance_gpu,
)


def compute_distance_between_edge_nodes(
    graph: nx.DiGraph,
):
    """Compute the distance between the start and end nodes of each edge in the graph.

    For the toy graph, this is the length of the edge.

    Parameters
    ----------
    graph : nx.DiGraph
        The input graph.

    Returns
    -------
    nx.DiGraph
        The graph with edge lengths annotated.
    """
    distances = {}
    for edge in graph.edges():
        start, end = edge
        start_pos = graph.nodes[start][NODE_COORDINATE_KEY]
        end_pos = graph.nodes[end][NODE_COORDINATE_KEY]
        distance = np.linalg.norm(end_pos - start_pos)
        distances[edge] = distance

    nx.set_edge_attributes(graph, distances, LENGTH_KEY)
    return graph


def generate_synthetic_fractal_tree(
    num_nodes: int = 19,
    edge_length: int = 100,
    branch_angle: float = 45,
    wiggle_factor: float = 0.01,
    noise_magnitude: float = 5,
    dilation_size: int = 3,
    tip_dilation_size: int = 1.05,
    ellipse_ratio: float | None = None,
    use_gpu: bool = True,
    seed: int = 42,
    inplane_rotation: float | None = None,
    outplane_rotation: list[float, float, float] | None = None,
):
    """Generate a fractal tree structure in a 3D skeleton image.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the tree.
    edge_length : int
        Total length of the initial tree edge.
        Following generations will shrink by a factor of 0.8.
        See `generate_toy_graph_symmetric_branch_angle` for details.
    branch_angle : float
        Angle between branches in degrees.
    wiggle_factor : float, optional
        Factor to control the amount of wiggle in the branches.
        Default is 0.01.
    noise_magnitude : float, optional
        Magnitude of noise to add to the surface of the branches.
        Default is 5.
    dilation_size : int, optional
        Size of the dilation applied to the skeleton image.
        Default is 3.
    tip_dilation_size : float, optional
        Factor to control the size of the dilation applied to the tips of the branches.
        Default is 1.05.
    ellipse_ratio : float, optional
        Ratio of the radii of the elliptic cylinder segments.
        If None, the branches will be cylindrical.
        Default is None.
    use_gpu : bool, optional
        Whether to use GPU acceleration for distance transform computation.
        Default is True.
    seed : int, optional
        Seed for random number generation.
        Default is 42.
    inplane_rotation : float | None, optional
        If provided, the in-plane rotation angle in degrees to apply to the tree.
        If None, a random angle between -90 and 90 degrees will be used.
    outplane_rotation : list[float, float, float] | None, optional
        If provided, the rotation angles in degrees to apply to the tree.
        If None, random angles between -30 and 30 degrees for each axis will be used
    """
    seed_gen = np.random.default_rng(seed)
    # build tree
    g = generate_toy_graph_symmetric_branch_angle(
        num_nodes=num_nodes, angle=branch_angle, edge_length=edge_length
    )
    g.graph = compute_level(g.graph, origin=-1)
    g.graph = compute_branch_length(g.graph)
    augment_tree(g.graph, rotation_angles=outplane_rotation)
    add_rotation_to_tree(g.graph, rotation_angle=inplane_rotation)

    g.graph = compute_distance_between_edge_nodes(g.graph)
    pos = nx.get_node_attributes(g.graph, NODE_COORDINATE_KEY)

    # build images
    pos_values = np.array(list(pos.values()))
    x_shift = np.abs(np.min(pos_values[:, 0])) + 20
    y_shift = np.abs(np.min(pos_values[:, 1])) + 30
    z_shift = np.abs(np.min(pos_values[:, 2])) + 20
    for node in g.graph.nodes():
        p = g.graph.nodes[node][NODE_COORDINATE_KEY]
        p = p + np.array([x_shift, y_shift, z_shift])
        g.graph.nodes[node][NODE_COORDINATE_KEY] = p

    pos = np.asarray(list(dict(g.nodes(data=NODE_COORDINATE_KEY)).values()))

    lengths = np.asarray([g.graph.edges[edge][LENGTH_KEY] for edge in g.graph.edges()])
    radii = lengths / 3.2
    pad_size = 2 * np.max(radii) + 10
    # Calculate the dimensions of the skeleton image
    x_offset = 0
    y_offset = 0
    z_offset = 0

    x_coord = int(np.ceil(np.max(pos[:, 0]) - x_offset))
    y_coord = int(np.ceil(np.max(pos[:, 1]) - y_offset))
    z_coord = int(np.ceil(np.max(pos[:, 2]) - z_offset))

    skeleton_img = np.pad(
        np.zeros((x_coord, y_coord, z_coord), dtype=np.uint8), pad_width=int(pad_size)
    )
    branch_img = skeleton_img.copy()
    pos_dict = nx.get_node_attributes(g.graph, NODE_COORDINATE_KEY)

    # Fill in the structures
    axis = seed_gen.integers(0, 3)  # Randomly choose an axis for wiggle
    for i, (u, v) in enumerate(g.graph.edges()):
        a = pos_dict[u]
        b = pos_dict[v]
        radius = radii[i]
        if not ellipse_ratio:
            draw_wiggly_cylinder_3d(
                branch_img,
                start_point=a,
                end_point=b,
                radius=int(radius),
                wiggle_factor=wiggle_factor,
                axis=axis,
            )
        else:
            # Draw an elliptic cylinder segment
            draw_elliptic_cylinder_segment(
                branch_img, a=a, b=b, rx=radius, ry=radius / ellipse_ratio
            )
        draw_line_segment_wiggle(
            start_point=a,
            end_point=b,
            skeleton_image=skeleton_img,
            fill_value=1,
            wiggle_factor=wiggle_factor,
            axis=axis,
        )
    # Dilute the tips
    length_dict = nx.get_edge_attributes(g.graph, LENGTH_KEY)
    for node, _ in g.graph.degree():
        # Get the position of the node
        pos = pos_dict[node]
        if node == -1:
            edge = next(iter(list(g.graph.edges(node))))
        else:
            edge = next(iter(list(g.graph.in_edges(node))))
        length = length_dict[edge]
        radius = length / 3
        # Dilute the tip by drawing a small cylinder
        draw_ellipsoid_at_point(
            branch_img,
            pos,
            radii=(
                radius * tip_dilation_size,
                radius * tip_dilation_size,
                radius * tip_dilation_size,
            ),
        )

    # Add noise to the branch image
    branch_img = add_noise_to_image_surface(branch_img, noise_magnitude=noise_magnitude)
    # Crop to content
    branch_img, skeleton_img = crop_to_content(branch_img, skeleton_img)
    # Compute the distance field
    branch_img_dask = da.from_array(branch_img, chunks=(200, 200, 200))
    depth = np.min([*list(branch_img_dask.shape), 30])

    if use_gpu:
        distance_field = branch_img_dask.map_overlap(
            local_normalized_distance_gpu, max_ball_radius=30, depth=depth
        ).compute()
    else:
        distance_field = branch_img_dask.map_overlap(
            local_normalized_distance, max_ball_radius=30, depth=depth
        ).compute()

    # Create the skeletonization target
    skel_target = make_skeleton_blur_image(
        skeleton_img, dilation_size=dilation_size, gaussian_size=1.5
    )
    skel_target = skel_target > 0.7
    skel_target = skel_target.astype(int)
    skel_target += branch_img

    return skel_target, distance_field


def generate_random_parameters_for_fractal_tree(
    num_nodes_range: tuple[int, int] = (15, 33),
    edge_length_factor: tuple[int, int] = (4, 7),
    branch_angle_range: tuple[float, float] = (30, 90),
    wiggle_factor_range: tuple[float, float] = (0.01, 0.03),
    noise_magnitude_range: tuple[float, float] = (5, 15),
    ellipse_ratio_range: tuple[float, float] = (1.1, 1.5),
    dilation_size: int = 3,
    tip_dilation_size: float = (1.05, 1.2),
    use_gpu=True,
    seed: int = 42,
    inplane_rotation: float | None = (-20, 20),
    outplane_rotation: list[float, float, float] | None = None,
):
    """Generate random parameters for fractal tree generation."""
    seed_gen = np.random.default_rng(seed)
    num_nodes = seed_gen.choice(np.arange(*num_nodes_range, 2))
    edge_length = num_nodes * seed_gen.integers(*edge_length_factor)
    branch_angle = seed_gen.uniform(*branch_angle_range)
    wiggle_factor = seed_gen.uniform(*wiggle_factor_range)
    noise_magnitude = seed_gen.uniform(*noise_magnitude_range)
    ellipse_ratio = (
        seed_gen.uniform(*ellipse_ratio_range) if seed_gen.random() > 0.5 else None
    )
    tip_dilation_size = seed_gen.uniform(*tip_dilation_size)
    if inplane_rotation is not None:
        inplane_rotation = seed_gen.uniform(*inplane_rotation)
    if outplane_rotation is not None:
        outplane_rotation = [
            seed_gen.uniform(*outplane_rotation[0]),
            seed_gen.uniform(*outplane_rotation[1]),
            seed_gen.uniform(*outplane_rotation[2]),
        ]
    return (
        num_nodes,
        edge_length,
        branch_angle,
        wiggle_factor,
        noise_magnitude,
        dilation_size,
        tip_dilation_size,
        ellipse_ratio,
        use_gpu,
        seed,
        inplane_rotation,
        outplane_rotation,
    )
