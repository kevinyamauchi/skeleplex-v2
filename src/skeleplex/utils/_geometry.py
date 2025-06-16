import numpy as np


def points_in_aabb(
    coordinates: np.ndarray,
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
) -> np.ndarray:
    """Create a boolean mask for coordinates within an axis-aligned bounding box.

    Parameters
    ----------
    coordinates : np.ndarray
        (n_coordinates, n_dim) array of point coordinates.
    min_bounds : np.ndarray
        (n_dim,) array with the minimum bounds of the bounding box.
    max_bounds : np.ndarray
        (n_dim,) array with the maximum bounds of the bounding box.

    Returns
    -------
    mask : np.ndarray
        Boolean array with shape (n_coordinates,) where True indicates
        coordinates within the bounding box (inclusive of boundaries).

    """
    # Check that min_bounds <= max_bounds for each dimension
    if np.any(min_bounds > max_bounds):
        raise ValueError("min_bounds must be <= max_bounds for all dimensions")

    # Vectorized comparison: coordinates >= min_bounds and coordinates <= max_bounds
    # Broadcasting handles the comparison across all coordinates simultaneously
    within_min = np.all(coordinates >= min_bounds, axis=1)
    within_max = np.all(coordinates <= max_bounds, axis=1)

    # Return mask where both conditions are satisfied
    return within_min & within_max


def line_segments_in_aabb(
    line_segments: np.ndarray,
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
) -> np.ndarray:
    """Get a boolean mask for line segments completely within an AABB.

    A line segment is considered completely inside the bounding box if both
    its start and end points are within the bounding box (inclusive of boundaries).

    Parameters
    ----------
    line_segments : np.ndarray
        (2 * n_line_segments, n_dim) array of line segment coordinates.
        Line segment n goes from line_segments[2*n] to line_segments[2*n + 1].
    min_bounds : np.ndarray
        (n_dim,) array of the minimum bounds of the bounding box.
    max_bounds : np.ndarray
        (n_dim,) array of the maximum bounds of the bounding box.

    Returns
    -------
    mask : np.ndarray
        Boolean array with shape (n_line_segments,) where True indicates
        line segments completely within the bounding box.
    """
    # Check that min_bounds <= max_bounds for each dimension
    if np.any(min_bounds > max_bounds):
        raise ValueError("min_bounds must be <= max_bounds for all dimensions")

    # Calculate number of line segments
    n_line_segments = line_segments.shape[0] // 2

    # Reshape to separate start and end points: (n_line_segments, 2, 3)
    segments_reshaped = line_segments.reshape(n_line_segments, 2, 3)

    # Check if each point is within bounds
    # Broadcasting: (n_line_segments, 2, 3) >= (3,) -> (n_line_segments, 2, 3)
    within_min = np.all(segments_reshaped >= min_bounds, axis=2)  # (n_line_segments, 2)
    within_max = np.all(segments_reshaped <= max_bounds, axis=2)  # (n_line_segments, 2)

    # Both start and end points must be within bounds
    points_in_bounds = within_min & within_max  # (n_line_segments, 2)

    # Line segment is completely inside if both points are inside
    segments_completely_inside = np.all(points_in_bounds, axis=1)  # (n_line_segments,)

    expanded_mask = np.repeat(segments_completely_inside, 2)  # (2 * n_line_segments,)

    return expanded_mask
