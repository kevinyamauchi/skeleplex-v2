import dask.array as da  # noqa: D100
import numpy as np

from skeleplex.measurements.utils import unit_vector
from skeleplex.skeleton.distance_field import (
    local_normalized_distance,
    local_normalized_distance_gpu,
)
from skeleplex.synthetic_data.utils import (
    add_noise_to_image_surface,
    crop_to_content,
    draw_ellipsoid_at_point,
    draw_elliptic_cylinder_segment,
    draw_line_segment_wiggle,
    draw_wiggly_cylinder_3d,
    make_skeleton_blur_image,
)


def generate_y_junction(
    length_parent,
    length_d1,
    length_d2,
    radius_parent,
    radius_d1,
    radius_d2,
    d1_angle,
    d2_angle,
    wiggle_factor=0.022,
    noise_magnitude=5,
    ellipse_ratio: int | None = None,
    use_gpu=True,
):
    """Generate a Y-junction structure in a 3D skeleton image.

    Parameters
    ----------
    length_parent : int
        Length of the parent branch.
    length_d1 : int
        Length of the first daughter branch.
    length_d2 : int
        Length of the second daughter branch.
    radius_parent : int
        Radius of the parent branch.
    radius_d1 : int
        Radius of the first daughter branch.
    radius_d2 : int
        Radius of the second daughter branch.
    d1_angle : float cecc
        Angle of the first daughter branch relative to the parent branch in degrees.
    d2_angle : float
        Angle of the second daughter branch relative to the parent branch in degrees.
    wiggle_factor : float, optional
        Factor to control the amount of wiggle in the branches.
        Default is 0.022.
    noise_magnitude : float, optional
        Magnitude of noise to add to the surface of the branches.
        Default is 5.
    ellipse_ratio : float, optional
        Ratio of the radii of the elliptic cylinder segments.
        If None, the branches will be cylindrical.
        Default is None.
    use_gpu : bool, optional
        Whether to use GPU acceleration for distance transform computation.
        Default is True.

    Returns
    -------
    skeleton : np.ndarray
        A 3D numpy array representing the generated Y-junction structure.
    """
    theta = np.radians(d1_angle)
    iota = np.radians(d2_angle)
    M = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    N = np.array([[np.cos(iota), -np.sin(iota)], [np.sin(iota), np.cos(iota)]])
    max_radius = max(radius_parent, radius_d1, radius_d2)
    max_length = length_parent + np.max([length_d1, length_d2])
    skeleton = np.zeros([max_length] * 3, dtype=np.uint8)
    skeleton = np.pad(
        skeleton, pad_width=max_radius * 2, mode="constant", constant_values=0
    )
    branch = skeleton.copy()

    # Draw the main branch
    origin = np.array(
        [skeleton.shape[0] // 2, max_radius * 2]
    )  # Start at the middle of the image
    p1 = origin + np.array([0, length_parent])

    # translate parent vector
    parent_vector = p1 - origin

    # rotate parent unit vector
    d1 = M @ parent_vector
    d1 = unit_vector(d1) * length_d1
    d1 = p1 + d1

    d2 = N @ parent_vector
    d2 = unit_vector(d2) * length_d2
    d2 = p1 + d2

    # transform to 3D
    origin = np.array([origin[0], origin[1], skeleton.shape[2] // 2])
    p1 = np.array([p1[0], p1[1], skeleton.shape[2] // 2])
    d1 = np.array([d1[0], d1[1], skeleton.shape[2] // 2])
    d2 = np.array([d2[0], d2[1], skeleton.shape[2] // 2])

    # rotate d2 around y axis
    d2_angle_z = np.radians(np.random.uniform(-90, 90))
    d2_pivot = d2 - p1
    rotation_matrix = np.array(
        [
            [np.cos(d2_angle_z), 0, np.sin(d2_angle_z)],
            [0, 1, 0],
            [-np.sin(d2_angle_z), 0, np.cos(d2_angle_z)],
        ]
    )
    d2 = rotation_matrix @ d2_pivot
    d2 = d2 + p1

    # get wiggle axis
    axis = np.random.randint(0, 3)

    for i, line in enumerate([[origin, p1], [p1, d1], [p1, d2]]):
        radius = [radius_parent, radius_d1, radius_d2][i]
        # Draw the line segments with wiggle
        draw_line_segment_wiggle(
            line[0], line[1], skeleton, wiggle_factor=wiggle_factor, axis=axis
        )
        # Draw the cylinders for the branches
        if not ellipse_ratio:
            draw_wiggly_cylinder_3d(
                branch,
                line[0],
                line[1],
                radius=radius,
                wiggle_factor=wiggle_factor,
                axis=axis,
            )
        else:
            # Draw an elliptic cylinder segment
            draw_elliptic_cylinder_segment(
                branch, line[0], line[1], rx=radius, ry=radius / ellipse_ratio
            )

    # dilute the tips
    if not ellipse_ratio:
        for i, point in enumerate([origin, p1, d1, d2]):
            r_tip = [radius_parent, radius_parent, radius_d1, radius_d2][i]
            draw_ellipsoid_at_point(
                branch,
                point,
                radii=(
                    r_tip * np.random.uniform(1, 1.1),
                    r_tip * np.random.uniform(1, 1.1),
                    r_tip * np.random.uniform(1, 1.1),
                ),
            )

    branch_noisey = add_noise_to_image_surface(branch, noise_magnitude=noise_magnitude)

    branch_noisey = add_noise_to_image_surface(branch, noise_magnitude=noise_magnitude)

    # crop to content
    branch_noisey, skeleton = crop_to_content(branch_noisey, skeleton)

    branch_noisey_dask = da.from_array(branch_noisey, chunks=(100, 100, 100))
    if use_gpu:
        distance_field = da.map_overlap(
            local_normalized_distance_gpu,
            branch_noisey_dask,
            max_ball_radius=30,
            depth=30,
        ).compute()
    else:
        distance_field = da.map_overlap(
            local_normalized_distance, branch_noisey_dask, max_ball_radius=30, depth=30
        ).compute()

    skeletonization_blur = make_skeleton_blur_image(
        skeleton,
        dilation_size=8,
        gaussian_size=1.5,
    )
    skeletonization_target = skeletonization_blur > 0.7
    skeletonization_target = skeletonization_target.astype(int)
    skeletonization_target += branch_noisey

    return skeletonization_target, distance_field


def random_parameters_y_junctions(
    length_parent_range=(80, 120),
    length_d1_range=(70, 100),
    length_d2_range=(70, 100),
    radius_parent_range=(60, 90),
    radius_d1_range=(30, 55),
    radius_d2_range=(30, 55),
    d1_angle_range=(-90, 30),
    d2_angle_range=(20, 100),
    wiggle_factor_range=(0.01, 0.03),
    noise_magnitude_range=(8, 25),
    ellipse_ratio_range=(1.1, 1.5),
    use_gpu=False,
):
    """Generate random parameters for Y-junction generation."""
    length_parent = np.random.randint(*length_parent_range)
    length_d1 = np.random.randint(*length_d1_range)
    length_d2 = np.random.randint(*length_d2_range)
    radius_parent = np.random.randint(*radius_parent_range)
    radius_d1 = np.random.randint(*radius_d1_range)
    radius_d2 = np.random.randint(*radius_d2_range)
    d1_angle = np.random.uniform(*d1_angle_range)
    d2_angle = np.random.uniform(*d2_angle_range)
    wiggle_factor = np.random.uniform(*wiggle_factor_range)
    noise_magnitude = np.random.uniform(*noise_magnitude_range)
    ellipse_ratio = (
        np.random.uniform(*ellipse_ratio_range) if np.random.rand() > 0.5 else None
    )
    return (
        length_parent,
        length_d1,
        length_d2,
        radius_parent,
        radius_d1,
        radius_d2,
        d1_angle,
        d2_angle,
        wiggle_factor,
        noise_magnitude,
        ellipse_ratio,
        use_gpu,
    )
