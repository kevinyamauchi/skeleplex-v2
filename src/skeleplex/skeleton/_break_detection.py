"""Functions to fix breaks in skeletonized structures."""

from typing import Literal

import numpy as np
from numba import njit
from numba.typed import List
from scipy.ndimage import convolve, label
from scipy.spatial import KDTree


@njit
def _line_3d_numba(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Generate coordinates for a line in 3 dimensional space.

    Parameters
    ----------
    start : np.ndarray
        (n,) array of starting coordinates.
    end : np.ndarray
        (n,) array of ending coordinates.

    Returns
    -------
    coords : np.ndarray
        (n_points, n) array of integer coordinates along the line.
    """
    # Compute deltas
    delta = end - start

    # Find the dimension with the maximum absolute delta
    max_delta = 0.0
    for i in range(3):
        abs_delta = abs(delta[i])
        if abs_delta > max_delta:
            max_delta = abs_delta

    # Number of points is based on the maximum delta
    n_points = int(max_delta) + 1

    if n_points == 1:
        # Start and end are the same point
        coords = np.zeros((1, 3), dtype=np.int64)
        for i in range(3):
            coords[0, i] = int(round(start[i]))
        return coords

    # Generate coordinates along the line
    coords = np.zeros((n_points, 3), dtype=np.int64)

    for i in range(n_points):
        t = i / (n_points - 1)
        for dim in range(3):
            coords[i, dim] = int(round(start[dim] + t * delta[dim]))

    return coords


@njit
def _flatten_candidates(
    repair_candidates: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a list of candidate arrays into a single array with offsets.

    Parameters
    ----------
    repair_candidates : list of np.ndarray
        List where each element is an (n_candidates, 3) array.

    Returns
    -------
    flat_candidates : np.ndarray
        (total_candidates, 3) array of all candidate coordinates.
    candidate_to_endpoint : np.ndarray
        (total_candidates,) array mapping each candidate to its
        endpoint index.
    offsets : np.ndarray
        (n_endpoints + 1,) array of start indices for each endpoint's
        candidates. The last element is the total number of candidates.
    """
    n_endpoints = len(repair_candidates)

    # First pass: count total candidates
    total_candidates = 0
    for i in range(n_endpoints):
        total_candidates += repair_candidates[i].shape[0]

    # Allocate arrays
    flat_candidates = np.zeros((total_candidates, 3), dtype=np.float64)
    candidate_to_endpoint = np.zeros(total_candidates, dtype=np.int64)
    offsets = np.zeros(n_endpoints + 1, dtype=np.int64)

    # Second pass: fill arrays
    current_idx = 0
    for i in range(n_endpoints):
        offsets[i] = current_idx
        n_candidates = repair_candidates[i].shape[0]

        for j in range(n_candidates):
            flat_candidates[current_idx] = repair_candidates[i][j]
            candidate_to_endpoint[current_idx] = i
            current_idx += 1

    offsets[n_endpoints] = total_candidates

    return flat_candidates, candidate_to_endpoint, offsets


@njit
def _find_break_repairs(
    end_point_coordinates: np.ndarray,
    flat_candidates: np.ndarray,
    offsets: np.ndarray,
    label_map: np.ndarray,
    segmentation: np.ndarray,
    endpoint_directions: np.ndarray,
    w_distance: float,
    w_angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the best voxel (if any) to connect an end point to.

    This is a numba implementation. Use find_break_repairs() as it
    handles data structure conversion.

    Uses a weighted cost function combining normalised distance and
    angle deviation from the local skeleton tangent.

    Parameters
    ----------
    end_point_coordinates : np.ndarray
        (n_end_points, 3) array of coordinates of the end points
        to check.
    flat_candidates : np.ndarray
        (total_candidates, 3) array of all candidate coordinates
        (flattened).
    offsets : np.ndarray
        (n_end_points + 1,) array marking where each endpoint's
        candidates start and end in flat_candidates.
    label_map : np.ndarray
        The connected components label map image of the skeleton.
    segmentation : np.ndarray
        The 3D binary image of the segmentation.
    endpoint_directions : np.ndarray
        (n_end_points, 3) array of unit direction vectors for each
        endpoint. Zero vectors indicate degenerate cases where the
        angle term is skipped.
    w_distance : float
        Weight for the normalised distance term in the cost function.
    w_angle : float
        Weight for the normalised angle term in the cost function.

    Returns
    -------
    repair_start : np.ndarray
        (n_end_points, 3) array of repair start coordinates.
        Contains -1 for endpoints with no valid repair.
    repair_end : np.ndarray
        (n_end_points, 3) array of repair end coordinates.
        Contains -1 for endpoints with no valid repair.
    """
    n_end_points = end_point_coordinates.shape[0]
    seg_shape = segmentation.shape
    eps = 1e-8

    # Initialize output arrays with sentinel values
    repair_start = np.full((n_end_points, 3), -1, dtype=np.int64)
    repair_end = np.full((n_end_points, 3), -1, dtype=np.int64)

    # Process each endpoint
    for ep_idx in range(n_end_points):
        endpoint = end_point_coordinates[ep_idx]
        endpoint_label = label_map[int(endpoint[0]), int(endpoint[1]), int(endpoint[2])]

        # Get this endpoint's candidates
        start_idx = offsets[ep_idx]
        end_idx = offsets[ep_idx + 1]
        n_cand = end_idx - start_idx

        if n_cand == 0:
            continue

        # Check if endpoint direction is degenerate
        ep_dir = endpoint_directions[ep_idx]
        dir_norm = 0.0
        for dim in range(3):
            dir_norm += ep_dir[dim] * ep_dir[dim]
        dir_norm = np.sqrt(dir_norm)
        has_direction = dir_norm > 1e-6

        # --- First pass: collect valid candidates, distances, angles ---
        valid_indices = np.empty(n_cand, dtype=np.int64)
        valid_distances = np.empty(n_cand, dtype=np.float64)
        valid_angles = np.empty(n_cand, dtype=np.float64)
        n_valid = 0

        for cand_idx in range(start_idx, end_idx):
            candidate = flat_candidates[cand_idx]

            # Skip if same connected component
            candidate_label = label_map[
                int(candidate[0]),
                int(candidate[1]),
                int(candidate[2]),
            ]
            if candidate_label == endpoint_label:
                continue

            # Draw line between endpoint and candidate
            line_coords = _line_3d_numba(endpoint, candidate)

            # Check if line stays within segmentation
            valid = True
            for i in range(line_coords.shape[0]):
                z, y, x = line_coords[i]
                if (
                    z < 0
                    or z >= seg_shape[0]
                    or y < 0
                    or y >= seg_shape[1]
                    or x < 0
                    or x >= seg_shape[2]
                ):
                    valid = False
                    break
                if not segmentation[z, y, x]:
                    valid = False
                    break

            if not valid:
                continue

            # Euclidean distance
            distance = 0.0
            for dim in range(3):
                diff = endpoint[dim] - candidate[dim]
                distance += diff * diff
            distance = np.sqrt(distance)

            # Angle between endpoint direction and connection vector
            angle = 0.0
            if has_direction:
                # Connection vector
                conn_norm = 0.0
                conn = np.empty(3, dtype=np.float64)
                for dim in range(3):
                    conn[dim] = candidate[dim] - endpoint[dim]
                    conn_norm += conn[dim] * conn[dim]
                conn_norm = np.sqrt(conn_norm)

                if conn_norm > eps:
                    dot_val = 0.0
                    for dim in range(3):
                        dot_val += ep_dir[dim] * (conn[dim] / conn_norm)
                    # Clip to [-1, 1]
                    if dot_val > 1.0:
                        dot_val = 1.0
                    elif dot_val < -1.0:
                        dot_val = -1.0
                    angle = np.arccos(dot_val)
                    # Resolve 180-degree sign ambiguity
                    angle = min(angle, np.pi - angle)

            valid_indices[n_valid] = cand_idx
            valid_distances[n_valid] = distance
            valid_angles[n_valid] = angle
            n_valid += 1

        if n_valid == 0:
            continue

        # --- Normalise across valid candidates ---
        d_min = valid_distances[0]
        d_max = valid_distances[0]
        for vi in range(1, n_valid):
            if valid_distances[vi] < d_min:
                d_min = valid_distances[vi]
            if valid_distances[vi] > d_max:
                d_max = valid_distances[vi]

        d_range = d_max - d_min + eps
        half_pi = np.pi / 2.0

        # --- Second pass: compute costs and find minimum ---
        best_cost = np.inf
        best_idx = -1

        for vi in range(n_valid):
            norm_distance = (valid_distances[vi] - d_min) / d_range

            if has_direction:
                norm_angle = valid_angles[vi] / half_pi
                cost = w_distance * norm_distance + w_angle * norm_angle
            else:
                cost = w_distance * norm_distance

            if cost < best_cost:
                best_cost = cost
                best_idx = valid_indices[vi]

        # Store result
        if best_idx >= 0:
            repair_start[ep_idx] = endpoint.astype(np.int64)
            repair_end[ep_idx] = flat_candidates[best_idx].astype(np.int64)

    return repair_start, repair_end


def find_break_repairs(
    end_point_coordinates: np.ndarray,
    repair_candidates: list[np.ndarray],
    label_map: np.ndarray,
    segmentation: np.ndarray,
    endpoint_directions: np.ndarray | None = None,
    w_distance: float = 1.0,
    w_angle: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the best voxel (if any) to connect an end point to.

    This is a wrapper function that handles the data structure conversion
    before calling the numba-jitted core function.

    Parameters
    ----------
    end_point_coordinates : np.ndarray
        (n_end_points, 3) array of coordinates of the end points
        to check.
    repair_candidates : list[np.ndarray]
        A list of length n_end_points where each element is a
        (n_repair_candidates_for_end_point, 3) array of the
        coordinates of potential voxels to connect the end point to.
        The list is index matched to end_point_coordinates.
    label_map : np.ndarray
        The connected components label map image of the skeleton.
    segmentation : np.ndarray
        The 3D binary image of the segmentation.
    endpoint_directions : np.ndarray or None, optional
        (n_end_points, 3) array of unit direction vectors for each
        endpoint. When None, a zero-vector array is used, which
        disables the angle term and falls back to distance-only
        selection. Default is None.
    w_distance : float, optional
        Weight for the normalised distance term. Default is 1.0.
    w_angle : float, optional
        Weight for the normalised angle term. Default is 1.0.

    Returns
    -------
    repair_start : np.ndarray
        (n_end_points, 3) array of repair start coordinates.
        Contains -1 for endpoints with no valid repair.
    repair_end : np.ndarray
        (n_end_points, 3) array of repair end coordinates.
        Contains -1 for endpoints with no valid repair.
    """
    n_end_points = end_point_coordinates.shape[0]

    if endpoint_directions is None:
        endpoint_directions = np.zeros((n_end_points, 3), dtype=np.float64)

    # Convert list of arrays to flattened structure
    flat_candidates, _, offsets = _flatten_candidates(List(repair_candidates))

    # Call the numba-jitted function
    return _find_break_repairs(
        end_point_coordinates,
        flat_candidates,
        offsets,
        label_map,
        segmentation,
        endpoint_directions,
        w_distance,
        w_angle,
    )


@njit
def draw_lines(
    skeleton: np.ndarray,
    repair_start: np.ndarray,
    repair_end: np.ndarray,
) -> None:
    """Draw repair lines in the skeleton image in-place.

    This function modifies the skeleton array in-place by drawing lines
    between repair start and end coordinates. Lines are generated using
    the existing _line_3d_numba function.

    Parameters
    ----------
    skeleton : np.ndarray
        The 3D binary skeleton array to modify in-place.
        Shape (nz, ny, nx) where True indicates skeleton voxels.
    repair_start : np.ndarray
        (n_repairs, 3) array of repair start coordinates.
        Contains -1 for rows with no valid repair (which are skipped).
    repair_end : np.ndarray
        (n_repairs, 3) array of repair end coordinates.
        Contains -1 for rows with no valid repair (which are skipped).

    Returns
    -------
    None
        The skeleton array is modified in-place.
    """
    n_repairs = repair_start.shape[0]

    for i in range(n_repairs):
        # Check if this is a valid repair (not a sentinel value)
        if repair_start[i, 0] == -1:
            continue

        # Generate line coordinates between start and end
        line_coords = _line_3d_numba(repair_start[i], repair_end[i])

        # Set all voxels along the line to True
        for j in range(line_coords.shape[0]):
            z, y, x = line_coords[j]
            skeleton[z, y, x] = True


def get_endpoint_directions(
    skeleton_image: np.ndarray,
    degree_one_coordinates: np.ndarray,
    degree_map: np.ndarray,
    n_fit_voxels: int = 10,
) -> np.ndarray:
    """Estimate the local tangent direction at each skeleton endpoint.

    For each endpoint, walks along the skeleton away from the tip
    collecting up to ``n_fit_voxels`` positions, then fits a line via
    SVD to determine the principal direction.

    Parameters
    ----------
    skeleton_image : np.ndarray
        The 3D binary skeleton array.
    degree_one_coordinates : np.ndarray
        (n_endpoints, 3) array of endpoint coordinates.
    degree_map : np.ndarray
        3D array of neighbour counts for each skeleton voxel.
    n_fit_voxels : int, optional
        Maximum number of voxels to collect along the skeleton for
        the line fit. Default is 10.

    Returns
    -------
    directions : np.ndarray
        (n_endpoints, 3) array of unit direction vectors. Zero
        vectors are returned for degenerate cases (fewer than 2
        voxels collected).
    """
    n_endpoints = degree_one_coordinates.shape[0]
    directions = np.zeros((n_endpoints, 3), dtype=np.float64)

    skeleton_binary = skeleton_image.astype(bool)
    shape = skeleton_binary.shape

    # 26-connectivity offsets
    offsets = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                offsets.append((dz, dy, dx))

    for ep_idx in range(n_endpoints):
        ep = degree_one_coordinates[ep_idx]
        coords = [ep.copy().astype(np.float64)]

        current = ep.copy()
        prev = np.array([-1, -1, -1], dtype=np.intp)

        for _ in range(n_fit_voxels - 1):
            cz, cy, cx = int(current[0]), int(current[1]), int(current[2])
            found = False

            for dz, dy, dx in offsets:
                nz, ny, nx = cz + dz, cy + dy, cx + dx

                # Bounds check
                if (
                    nz < 0
                    or nz >= shape[0]
                    or ny < 0
                    or ny >= shape[1]
                    or nx < 0
                    or nx >= shape[2]
                ):
                    continue

                # Must be skeleton voxel
                if not skeleton_binary[nz, ny, nx]:
                    continue

                # Must not be previous voxel
                if nz == prev[0] and ny == prev[1] and nx == prev[2]:
                    continue

                # Move to this neighbour
                prev[:] = current
                current = np.array([nz, ny, nx], dtype=np.intp)
                coords.append(np.array([nz, ny, nx], dtype=np.float64))
                found = True
                break

            if not found:
                break

            # Stop at branch points (degree > 2)
            cz2, cy2, cx2 = (
                int(current[0]),
                int(current[1]),
                int(current[2]),
            )
            if degree_map[cz2, cy2, cx2] > 2:
                break

        if len(coords) < 2:
            # Degenerate: return zero vector
            continue

        pts = np.array(coords, dtype=np.float64)
        centroid = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centroid)
        directions[ep_idx] = vt[0]

    return directions


def get_skeleton_data_cpu(
    skeleton_image: np.ndarray,
    endpoint_bounding_box: (
        tuple[tuple[int, int, int], tuple[int, int, int]] | None
    ) = None,
    label_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract skeleton topology data needed for break repair.

    This function computes the degree map (number of neighbors for each
    skeleton voxel), identifies endpoints (degree-1 voxels), collects all
    skeleton coordinates, and labels connected components of the skeleton.

    Parameters
    ----------
    skeleton_image : np.ndarray
        The 3D binary array containing the skeleton.
        The skeleton voxels are True or non-zero.
    endpoint_bounding_box : tuple of tuple of int, optional
        Tuple of ((z_min, y_min, x_min), (z_max, y_max, x_max))
        defining a bounding box within which to consider endpoints.
        If None, all endpoints are considered. Default is None.
    label_map : np.ndarray or None, optional
        Pre-computed connected component label map for the skeleton.
        When provided, the local ``scipy.ndimage.label`` call is
        skipped and this array is used directly. Must have the same
        shape as ``skeleton_image``. When None, connected components
        are computed locally. Default is None.

    Returns
    -------
    degree_map : np.ndarray
        3D array of the same shape as skeleton_image where each
        skeleton voxel contains the count of its neighboring skeleton
        voxels.
    degree_one_coordinates : np.ndarray
        (n, 3) array of coordinates for skeleton voxels with exactly
        one neighbor (endpoints). If no endpoints exist, returns
        empty (0, 3) array.
    all_skeleton_coordinates : np.ndarray
        (m, 3) array of coordinates for all skeleton voxels. If
        skeleton is empty, returns empty (0, 3) array.
    skeleton_label_map : np.ndarray
        3D array of the same shape as skeleton_image with connected
        components labeled using full 26-connectivity. Background
        is 0, connected skeleton components are labeled with positive
        integers.

    Raises
    ------
    ValueError
        If skeleton_image is not 3D or if a provided label_map has
        a different shape than skeleton_image.
    """
    # Ensure skeleton is binary
    skeleton_binary = skeleton_image.astype(bool)

    # Create 3x3x3 kernel with all ones except center
    ndim = skeleton_binary.ndim
    if ndim != 3:
        raise ValueError(f"Expected 3D skeleton image, got {ndim}D")

    degree_kernel = np.ones((3, 3, 3), dtype=np.uint8)
    degree_kernel[1, 1, 1] = 0

    # Compute degree map: count neighbors for each skeleton voxel
    degree_map = convolve(
        skeleton_binary.astype(np.uint8),
        degree_kernel,
        mode="constant",
        cval=0,
    )
    # Mask to only skeleton voxels (zero out background)
    degree_map = degree_map * skeleton_binary

    # Find degree-1 voxels (endpoints)
    degree_one_mask = degree_map == 1
    degree_one_coordinates = np.argwhere(degree_one_mask)

    # Filter by bounding box if provided
    if endpoint_bounding_box is not None:
        (z_min, y_min, x_min), (z_max, y_max, x_max) = endpoint_bounding_box

        mask = (
            (degree_one_coordinates[:, 0] >= z_min)
            & (degree_one_coordinates[:, 0] < z_max)
            & (degree_one_coordinates[:, 1] >= y_min)
            & (degree_one_coordinates[:, 1] < y_max)
            & (degree_one_coordinates[:, 2] >= x_min)
            & (degree_one_coordinates[:, 2] < x_max)
        )

        degree_one_coordinates = degree_one_coordinates[mask]

    # Get all skeleton coordinates
    all_skeleton_coordinates = np.argwhere(skeleton_binary)

    # Label skeleton with full connectivity (26-connectivity in 3D)
    if label_map is not None:
        if label_map.shape != skeleton_image.shape:
            raise ValueError(
                f"label_map shape {label_map.shape} does not match "
                f"skeleton_image shape {skeleton_image.shape}"
            )
        skeleton_label_map = label_map
    else:
        # Structure for full connectivity: 3x3x3 of all True
        structure = np.ones((3, 3, 3), dtype=bool)
        skeleton_label_map, _ = label(skeleton_binary, structure=structure)

    return (
        degree_map,
        degree_one_coordinates,
        all_skeleton_coordinates,
        skeleton_label_map,
    )


def get_skeleton_data_cupy(
    skeleton_image: np.ndarray,
    endpoint_bounding_box: (
        tuple[tuple[int, int, int], tuple[int, int, int]] | None
    ) = None,
    label_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract skeleton topology data needed for break repair using GPU.

    This is a GPU-accelerated version of get_skeleton_data_cpu that uses
    CuPy for parallel computation. It computes the degree map (number of
    neighbors for each skeleton voxel), identifies endpoints (degree-1
    voxels), collects all skeleton coordinates, and labels connected
    components of the skeleton.

    Parameters
    ----------
    skeleton_image : np.ndarray
        The 3D binary array containing the skeleton.
        The skeleton voxels are True or non-zero.
    endpoint_bounding_box : tuple of tuple of int, optional
        Tuple of ((z_min, y_min, x_min), (z_max, y_max, x_max))
        defining a bounding box within which to consider endpoints.
        If None, all endpoints are considered. Default is None.
    label_map : np.ndarray or None, optional
        Pre-computed connected component label map for the skeleton.
        When provided, the GPU ``label`` call is skipped and this
        array is used directly. Must have the same shape as
        ``skeleton_image``. When None, connected components are
        computed locally on the GPU. Default is None.

    Returns
    -------
    degree_map : np.ndarray
        3D array of the same shape as skeleton_image where each
        skeleton voxel contains the count of its neighboring skeleton
        voxels.
    degree_one_coordinates : np.ndarray
        (n, 3) array of coordinates for skeleton voxels with exactly
        one neighbor (endpoints). If no endpoints exist, returns
        empty (0, 3) array.
    all_skeleton_coordinates : np.ndarray
        (m, 3) array of coordinates for all skeleton voxels. If
        skeleton is empty, returns empty (0, 3) array.
    skeleton_label_map : np.ndarray
        3D array of the same shape as skeleton_image with connected
        components labeled using full 26-connectivity. Background
        is 0, connected skeleton components are labeled with positive
        integers.

    Raises
    ------
    ImportError
        If CuPy is not installed.
    ValueError
        If skeleton_image is not 3D or if a provided label_map has
        a different shape than skeleton_image.
    """
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import convolve, label
    except ImportError as err:
        raise ImportError(
            "get_skeleton_data_cupy requires CuPy. "
            "Please install CuPy for GPU acceleration."
        ) from err

    # Validate dimensions
    ndim = skeleton_image.ndim
    if ndim != 3:
        raise ValueError(f"Expected 3D skeleton image, got {ndim}D")

    # Transfer to GPU and ensure binary
    skeleton_gpu = cp.asarray(skeleton_image, dtype=bool)

    # Create 3x3x3 kernel with all ones except center
    degree_kernel = cp.ones((3, 3, 3), dtype=cp.uint8)
    degree_kernel[1, 1, 1] = 0

    # Compute degree map: count neighbors for each skeleton voxel
    degree_map_gpu = convolve(
        skeleton_gpu.astype(cp.uint8),
        degree_kernel,
        mode="constant",
        cval=0,
    )
    # Mask to only skeleton voxels (zero out background)
    degree_map_gpu = degree_map_gpu * skeleton_gpu

    # Find degree-1 voxels (endpoints)
    degree_one_mask_gpu = degree_map_gpu == 1
    degree_one_coordinates_gpu = cp.argwhere(degree_one_mask_gpu)

    # Filter by bounding box if provided
    if endpoint_bounding_box is not None:
        (z_min, y_min, x_min), (z_max, y_max, x_max) = endpoint_bounding_box

        mask = (
            (degree_one_coordinates_gpu[:, 0] >= z_min)
            & (degree_one_coordinates_gpu[:, 0] < z_max)
            & (degree_one_coordinates_gpu[:, 1] >= y_min)
            & (degree_one_coordinates_gpu[:, 1] < y_max)
            & (degree_one_coordinates_gpu[:, 2] >= x_min)
            & (degree_one_coordinates_gpu[:, 2] < x_max)
        )

        degree_one_coordinates_gpu = degree_one_coordinates_gpu[mask]

    # Get all skeleton coordinates
    all_skeleton_coordinates_gpu = cp.argwhere(skeleton_gpu)

    # Label skeleton with full connectivity (26-connectivity in 3D)
    if label_map is not None:
        if label_map.shape != skeleton_image.shape:
            raise ValueError(
                f"label_map shape {label_map.shape} does not match "
                f"skeleton_image shape {skeleton_image.shape}"
            )
        skeleton_label_map = label_map
    else:
        # Structure for full connectivity: 3x3x3 of all True
        structure_gpu = cp.ones((3, 3, 3), dtype=bool)
        skeleton_label_map_gpu, _ = label(skeleton_gpu, structure=structure_gpu)
        skeleton_label_map = cp.asnumpy(skeleton_label_map_gpu)

    # Transfer results back to CPU
    degree_map = cp.asnumpy(degree_map_gpu)
    degree_one_coordinates = cp.asnumpy(degree_one_coordinates_gpu)
    all_skeleton_coordinates = cp.asnumpy(all_skeleton_coordinates_gpu)

    return (
        degree_map,
        degree_one_coordinates,
        all_skeleton_coordinates,
        skeleton_label_map,
    )


def repair_breaks(
    skeleton_image: np.ndarray,
    segmentation: np.ndarray,
    repair_radius: float = 10.0,
    endpoint_bounding_box: (
        tuple[tuple[int, int, int], tuple[int, int, int]] | None
    ) = None,
    label_map: np.ndarray | None = None,
    n_fit_voxels: int = 10,
    w_distance: float = 1.0,
    w_angle: float = 1.0,
    backend: Literal["cpu", "cupy"] = "cpu",
) -> np.ndarray:
    """Repair breaks in a skeleton.

    This function identifies endpoints in the skeleton (voxels with only
    one neighbor) and attempts to connect them to other skeleton voxels
    within a specified radius. A repair is only made if the connecting
    line stays entirely within the segmentation and connects different
    connected components.

    Parameters
    ----------
    skeleton_image : np.ndarray
        The 3D binary array containing the skeleton.
        The skeleton voxels are True or non-zero.
    segmentation : np.ndarray
        The 3D binary array containing the segmentation.
        Foreground voxels are True or non-zero.
    repair_radius : float, optional
        The maximum Euclidean distance an endpoint can be connected
        within the segmentation. Default is 10.0.
    endpoint_bounding_box : tuple of tuple of int, optional
        Tuple of ((z_min, y_min, x_min), (z_max, y_max, x_max))
        defining a bounding box within which to consider endpoints
        for repair. If None, all endpoints are considered.
        Default is None.
    label_map : np.ndarray or None, optional
        Pre-computed global connected component label map for the
        skeleton. When provided, the local connected component
        computation is skipped, preventing false positives at chunk
        boundaries. Must have the same shape as ``skeleton_image``.
        When None, connected components are computed locally.
        Default is None.
    n_fit_voxels : int, optional
        Number of voxels to walk along the skeleton from each
        endpoint to estimate the local tangent direction.
        Default is 10.
    w_distance : float, optional
        Weight for the normalised distance term in the repair cost
        function. Default is 1.0.
    w_angle : float, optional
        Weight for the normalised angle deviation term in the repair
        cost function. Default is 1.0.
    backend : Literal["cpu", "cupy"], optional
        The backend to use for calculation. Default is "cpu".

    Returns
    -------
    repaired_skeleton : np.ndarray
        The 3D binary array containing the repaired skeleton.
        Same shape and dtype as input skeleton_image.

    Raises
    ------
    ValueError
        If skeleton_image or segmentation are not 3D arrays.
    ValueError
        If skeleton_image and segmentation have different shapes.

    Notes
    -----
    The repair process:

    1. Identifies all endpoints (degree-1 voxels) in the skeleton.
    2. For each endpoint, finds candidate skeleton voxels within
       ``repair_radius``.
    3. Tests each candidate by drawing a line and checking if it stays
       in the segmentation.
    4. Selects the best candidate from a different connected component
       using a weighted cost of normalised distance and angle deviation
       from the local skeleton tangent.
    5. Draws the repair lines in the skeleton.
    """
    # Validate inputs
    if skeleton_image.ndim != 3:
        raise ValueError(
            f"Expected 3D skeleton_image, " f"got {skeleton_image.ndim}D array"
        )

    if skeleton_image.shape != segmentation.shape:
        raise ValueError(
            f"skeleton_image and segmentation must have the same "
            f"shape. Got {skeleton_image.shape} and "
            f"{segmentation.shape}"
        )

    # Convert to boolean arrays
    skeleton_binary = skeleton_image.astype(bool)
    segmentation_binary = segmentation.astype(bool)

    # Create a working copy to modify
    repaired_skeleton = skeleton_binary.copy()

    # Extract skeleton topology data
    if backend == "cpu":
        (
            degree_map,
            degree_one_coordinates,
            all_skeleton_coordinates,
            skeleton_label_map,
        ) = get_skeleton_data_cpu(
            skeleton_binary,
            endpoint_bounding_box=endpoint_bounding_box,
            label_map=label_map,
        )
    elif backend == "cupy":
        (
            degree_map,
            degree_one_coordinates,
            all_skeleton_coordinates,
            skeleton_label_map,
        ) = get_skeleton_data_cupy(
            skeleton_binary,
            endpoint_bounding_box=endpoint_bounding_box,
            label_map=label_map,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Early exit if no endpoints
    if degree_one_coordinates.shape[0] == 0:
        return repaired_skeleton

    # Estimate local tangent directions at each endpoint
    endpoint_directions = get_endpoint_directions(
        skeleton_binary,
        degree_one_coordinates,
        degree_map,
        n_fit_voxels=n_fit_voxels,
    )

    # Build KDTree for efficient spatial queries
    kdtree = KDTree(all_skeleton_coordinates)

    # Find repair candidates within radius for each endpoint
    repair_candidates = []
    for endpoint in degree_one_coordinates:
        # Query KDTree for all points within repair_radius
        indices = kdtree.query_ball_point(endpoint, repair_radius)

        # Get the coordinates of the candidate voxels
        if len(indices) > 0:
            candidates = all_skeleton_coordinates[indices]
        else:
            # No candidates found, create empty array
            candidates = np.empty((0, 3), dtype=all_skeleton_coordinates.dtype)

        repair_candidates.append(candidates)

    # Find valid repairs using the existing function
    repair_start, repair_end = find_break_repairs(
        end_point_coordinates=degree_one_coordinates,
        repair_candidates=repair_candidates,
        label_map=skeleton_label_map,
        segmentation=segmentation_binary,
        endpoint_directions=endpoint_directions,
        w_distance=w_distance,
        w_angle=w_angle,
    )

    # Draw the repairs in-place
    draw_lines(
        skeleton=repaired_skeleton,
        repair_start=repair_start,
        repair_end=repair_end,
    )

    return repaired_skeleton
