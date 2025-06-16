import numpy as np
import pytest

from skeleplex.utils import line_segments_in_aabb, points_in_aabb


def test_points_in_aabb():
    """Test finding points in an axis-aligned bounding box."""
    points = np.array([[-1, -3, -2], [-1, -1, -1], [3, 4, 5], [9, 10, 11]])

    min_bounds = np.array([-2, -2, -2])
    max_bound = np.array([10, 10, 10])

    mask = points_in_aabb(points, min_bounds, max_bound)

    np.testing.assert_array_equal(mask, np.array([False, True, True, False]))


def test_points_in_aabb_invalid_bounds():
    """Test invalid bounds for points in an axis-aligned bounding box."""
    points = np.array([[-1, -3, -2], [-1, -1, -1], [3, 4, 5], [9, 10, 11]])

    min_bounds = np.array([-2, -2, -2])
    max_bounds = np.array([-3, -3, -3])  # Invalid bounds

    with pytest.raises(ValueError):
        points_in_aabb(points, min_bounds, max_bounds)


def test_line_segments_in_aabb():
    """Test finding line segments in an axis-aligned bounding box."""
    line_segments = np.array(
        [
            [-5, -3, -2],  # segment 0 start (outside)
            [0, 0, 0],  # segment 0 end (inside)
            [0, 0, 0],  # segment 1 start (inside)
            [1, 1, 1],  # segment 1 end (inside)
            [5, 5, 5],  # segment 2 start (inside)
            [15, 15, 15],  # segment 2 end (outside)
        ]
    )

    min_bounds = np.array([-2, -2, -2])
    max_bounds = np.array([10, 10, 10])

    mask = line_segments_in_aabb(line_segments, min_bounds, max_bounds)

    np.testing.assert_array_equal(
        mask, np.array([False, False, True, True, False, False])
    )


def test_line_segments_in_aabb_invalid_bounds():
    """Test invalid bounds for line segments in an axis-aligned bounding box."""
    line_segments = np.array(
        [[-5, -3, -2], [0, 0, 0], [0, 0, 0], [1, 1, 1], [5, 5, 5], [15, 15, 15]]
    )

    min_bounds = np.array([-2, -2, -2])
    max_bounds = np.array([-3, -3, -3])  # Invalid bounds

    with pytest.raises(ValueError):
        line_segments_in_aabb(line_segments, min_bounds, max_bounds)
