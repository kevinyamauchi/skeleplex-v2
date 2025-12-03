"""Functions to fix breaks in skeletonized structures."""

import numpy as np
from numba import njit
from numba.typed import List
from scipy.ndimage import convolve, label


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
