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
        (total_candidates,) array mapping each candidate to its endpoint index.
    offsets : np.ndarray
        (n_endpoints + 1,) array of start indices for each endpoint's candidates.
        The last element is the total number of candidates.
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
) -> tuple[np.ndarray, np.ndarray]:
    """Find the best voxel (if any) to connect an end point to.

    This is a numba implementation. Use find_break_repairs() as it
    handles data structure conversion.

    Parameters
    ----------
    end_point_coordinates : np.ndarray
        (n_end_points, 3) array of coordinates of the end points to check.
    flat_candidates : np.ndarray
        (total_candidates, 3) array of all candidate coordinates (flattened).
    offsets : np.ndarray
        (n_end_points + 1,) array marking where each endpoint's candidates
        start and end in flat_candidates.
    label_map : np.ndarray
        The connected components label map image of the skeleton.
    segmentation : np.ndarray
        The 3D binary image of the segmentation.

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

        best_distance = np.inf
        best_candidate_idx = -1

        # Check each candidate
        for cand_idx in range(start_idx, end_idx):
            candidate = flat_candidates[cand_idx]

            # Skip if same connected component (self-loop check)
            candidate_label = label_map[
                int(candidate[0]), int(candidate[1]), int(candidate[2])
            ]
            if candidate_label == endpoint_label:
                continue

            # Draw line between endpoint and candidate
            line_coords = _line_3d_numba(endpoint, candidate)

            # Check if line stays within segmentation
            valid = True
            for i in range(line_coords.shape[0]):
                z, y, x = line_coords[i]

                # Check bounds
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

                # Check if inside segmentation
                if not segmentation[z, y, x]:
                    valid = False
                    break

            if not valid:
                continue

            # Calculate euclidean distance
            distance = 0.0
            for dim in range(3):
                diff = endpoint[dim] - candidate[dim]
                distance += diff * diff
            distance = np.sqrt(distance)

            # Update best candidate if this is closer
            if distance < best_distance:
                best_distance = distance
                best_candidate_idx = cand_idx

        # Store result if valid candidate found
        if best_candidate_idx >= 0:
            repair_start[ep_idx] = endpoint.astype(np.int64)
            repair_end[ep_idx] = flat_candidates[best_candidate_idx].astype(np.int64)

    return repair_start, repair_end


def find_break_repairs(
    end_point_coordinates: np.ndarray,
    repair_candidates: list[np.ndarray],
    label_map: np.ndarray,
    segmentation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the best voxel (if any) to connect an end point to.

    This is a wrapper function that handles the data structure conversion
    before calling the numba-jitted core function.

    Parameters
    ----------
    end_point_coordinates : np.ndarray
        (n_end_points, 3) array of coordinates of the end points to check.
    repair_candidates : list[np.ndarray]
        A list of length n_end_points where each element is a
        (n_repair_candidates_for_end_point, 3) array of the coordinates
        of potential voxels to connect the end point to. The list is
        index matched to end_point_coordinates.
    label_map : np.ndarray
        The connected components label map image of the skeleton.
    segmentation : np.ndarray
        The 3D binary image of the segmentation.

    Returns
    -------
    repair_start : np.ndarray
        (n_end_points, 3) array of repair start coordinates.
        Contains -1 for endpoints with no valid repair.
    repair_end : np.ndarray
        (n_end_points, 3) array of repair end coordinates.
        Contains -1 for endpoints with no valid repair.
    """
    # Convert list of arrays to flattened structure
    flat_candidates, _, offsets = _flatten_candidates(List(repair_candidates))

    # Call the numba-jitted function
    return _find_break_repairs(
        end_point_coordinates,
        flat_candidates,
        offsets,
        label_map,
        segmentation,
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


def get_skeleton_data_cpu(
    skeleton_image: np.ndarray,
    endpoint_bounding_box: tuple[tuple[int, int, int], tuple[int, int, int]]
    | None = None,
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
    endpoint_bounding_box : tuple[tuple[int, int, int], tuple[int, int, int]]
        Tuple of ((z_min, y_min, x_min), (z_max, y_max, x_max)) defining a bounding box
        within which to consider endpoints. If None, all endpoints are considered.
        Default is None.

    Returns
    -------
    degree_map : np.ndarray
        3D array of the same shape as skeleton_image where each skeleton
        voxel contains the count of its neighboring skeleton voxels.
    degree_one_coordinates : np.ndarray
        (n, 3) array of coordinates for skeleton voxels with exactly one
        neighbor (endpoints). If no endpoints exist, returns empty (0, 3)
        array.
    all_skeleton_coordinates : np.ndarray
        (m, 3) array of coordinates for all skeleton voxels. If skeleton is
        empty, returns empty (0, 3) array.
    skeleton_label_map : np.ndarray
        3D array of the same shape as skeleton_image with connected
        components labeled using full 26-connectivity. Background is 0,
        connected skeleton components are labeled with positive integers.
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
        skeleton_binary.astype(np.uint8), degree_kernel, mode="constant", cval=0
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
    endpoint_bounding_box: tuple[tuple[int, int, int], tuple[int, int, int]]
    | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract skeleton topology data needed for break repair using GPU.

    This is a GPU-accelerated version of get_skeleton_data_cpu that uses CuPy
    for parallel computation. It computes the degree map (number of neighbors
    for each skeleton voxel), identifies endpoints (degree-1 voxels), collects
    all skeleton coordinates, and labels connected components of the skeleton.

    Parameters
    ----------
    skeleton_image : np.ndarray
        The 3D binary array containing the skeleton.
        The skeleton voxels are True or non-zero.
    endpoint_bounding_box : tuple[tuple[int, int, int], tuple[int, int, int]]
        Tuple of ((z_min, y_min, x_min), (z_max, y_max, x_max)) defining a
        bounding box within which to consider endpoints. If None, all endpoints
        are considered. Default is None.

    Returns
    -------
    degree_map : np.ndarray
        3D array of the same shape as skeleton_image where each skeleton
        voxel contains the count of its neighboring skeleton voxels.
    degree_one_coordinates : np.ndarray
        (n, 3) array of coordinates for skeleton voxels with exactly one
        neighbor (endpoints). If no endpoints exist, returns empty (0, 3)
        array.
    all_skeleton_coordinates : np.ndarray
        (m, 3) array of coordinates for all skeleton voxels. If skeleton is
        empty, returns empty (0, 3) array.
    skeleton_label_map : np.ndarray
        3D array of the same shape as skeleton_image with connected
        components labeled using full 26-connectivity. Background is 0,
        connected skeleton components are labeled with positive integers.

    Raises
    ------
    ImportError
        If CuPy is not installed.
    ValueError
        If skeleton_image is not 3D.
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
        skeleton_gpu.astype(cp.uint8), degree_kernel, mode="constant", cval=0
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
    # Structure for full connectivity: 3x3x3 of all True
    structure_gpu = cp.ones((3, 3, 3), dtype=bool)
    skeleton_label_map_gpu, _ = label(skeleton_gpu, structure=structure_gpu)

    # Transfer results back to CPU
    degree_map = cp.asnumpy(degree_map_gpu)
    degree_one_coordinates = cp.asnumpy(degree_one_coordinates_gpu)
    all_skeleton_coordinates = cp.asnumpy(all_skeleton_coordinates_gpu)
    skeleton_label_map = cp.asnumpy(skeleton_label_map_gpu)

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
    endpoint_bounding_box: tuple[tuple[int, int, int], tuple[int, int, int]]
    | None = None,
    backend: Literal["cpu", "cupy"] = "cpu",
) -> np.ndarray:
    """Repair breaks in a skeleton.

    This function identifies endpoints in the skeleton (voxels with only one
    neighbor) and attempts to connect them to other skeleton voxels within
    a specified radius. A repair is only made if the connecting line stays
    entirely within the segmentation and connects different connected
    components.

    Parameters
    ----------
    skeleton_image : np.ndarray
        The 3D binary array containing the skeleton.
        The skeleton voxels are True or non-zero.
    segmentation : np.ndarray
        The 3D binary array containing the segmentation.
        Foreground voxels are True or non-zero.
    repair_radius : float, default=10.0
        The maximum Euclidean distance an endpoint can be connected
        within the segmentation.
    endpoint_bounding_box : tuple[tuple[int, int, int], tuple[int, int, int]]
        Tuple of ((z_min, y_min, x_min), (z_max, y_max, x_max)) defining a bounding box
        within which to consider endpoints for repair.
        If None, all endpoints are considered. Default is None.
    backend : Literal["cpu", "cupy"]
        The backend to use for calculation. Default is cpu.

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
    1. Identifies all endpoints (degree-1 voxels) in the skeleton
    2. For each endpoint, finds candidate skeleton voxels within repair_radius
    3. Tests each candidate by drawing a line and checking if it stays in
       the segmentation
    4. Selects the closest valid candidate from a different connected component
    5. Draws the repair lines in the skeleton
    """
    # Validate inputs
    if skeleton_image.ndim != 3:
        raise ValueError(
            f"Expected 3D skeleton_image, got {skeleton_image.ndim}D array"
        )

    if skeleton_image.shape != segmentation.shape:
        raise ValueError(
            f"skeleton_image and segmentation must have the same shape. "
            f"Got {skeleton_image.shape} and {segmentation.shape}"
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
            skeleton_binary, endpoint_bounding_box=endpoint_bounding_box
        )
    elif backend == "cupy":
        (
            degree_map,
            degree_one_coordinates,
            all_skeleton_coordinates,
            skeleton_label_map,
        ) = get_skeleton_data_cupy(
            skeleton_binary, endpoint_bounding_box=endpoint_bounding_box
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Early exit if no endpoints
    if degree_one_coordinates.shape[0] == 0:
        return repaired_skeleton

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
    )

    # Draw the repairs in-place
    draw_lines(
        skeleton=repaired_skeleton,
        repair_start=repair_start,
        repair_end=repair_end,
    )

    return repaired_skeleton
