from importlib.util import find_spec

import numpy as np
import pytest
from numba.typed import List
from scipy.ndimage import convolve

from skeleplex.skeleton import find_break_repairs, repair_breaks
from skeleplex.skeleton._break_detection import (
    _flatten_candidates,
    _line_3d_numba,
    draw_lines,
    get_endpoint_directions,
    get_skeleton_data_cpu,
)

# True if CuPy is available
CUPY_AVAILABLE = find_spec("cupy") is not None


def test_line_3d_numba():
    """Test the _line_3d_numba function."""
    start = np.array([0, 0, 0])
    end = np.array([3, 3, 3])
    expected_points = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ]
    )

    points = _line_3d_numba(start, end)
    np.testing.assert_allclose(points, expected_points)


def test_flatten_candidates():
    """Test the _flatten_candidates function."""
    candidates = List(
        [
            np.array([[0, 0, 0], [1, 1, 1]]),
            np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]]),
        ]
    )
    expected_flattened = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
        ]
    )
    expected_candidate_to_endpoint = np.array([0, 0, 1, 1, 1])
    expected_offsets = np.array([0, 2, 5])

    flattened, candidate_to_endpoint, offsets = _flatten_candidates(candidates)
    np.testing.assert_allclose(flattened, expected_flattened)
    np.testing.assert_allclose(candidate_to_endpoint, expected_candidate_to_endpoint)
    np.testing.assert_allclose(offsets, expected_offsets)


def test_find_break_repairs_straight():
    """Test find_break_repairs with a simple straight-line case."""
    segmentation = np.zeros((30, 30, 30))
    segmentation[5:25, 10:14, 10:14] = 1
    end_points = np.array([[7, 12, 12], [15, 12, 12], [20, 12, 12], [23, 12, 12]])
    repair_candidates = [
        np.array(
            [
                [15, 12, 12],
            ]
        ),
        np.array(
            [
                [7, 12, 12],
                [20, 12, 12],
            ]
        ),
        np.array(
            [
                [15, 12, 12],
                [23, 12, 12],
            ]
        ),
        np.array(
            [
                [20, 12, 12],
            ]
        ),
    ]

    label_map = np.zeros((30, 30, 30))
    label_map[7:16, 12, 12] = 1
    label_map[20:24, 12, 12] = 2

    repair_start, repair_end = find_break_repairs(
        end_point_coordinates=end_points,
        repair_candidates=repair_candidates,
        label_map=label_map,
        segmentation=segmentation,
    )

    expected_repair_start = np.array(
        [
            [-1, -1, -1],
            [15, 12, 12],
            [20, 12, 12],
            [-1, -1, -1],
        ]
    )
    expected_repair_end = np.array(
        [
            [-1, -1, -1],
            [20, 12, 12],
            [15, 12, 12],
            [-1, -1, -1],
        ]
    )

    np.testing.assert_array_equal(repair_start, expected_repair_start)
    np.testing.assert_array_equal(repair_end, expected_repair_end)


def test_find_break_repairs_tee():
    """Test find_break_repairs with a T-junction case."""
    segmentation = np.zeros((30, 30, 30))
    segmentation[5:28, 10:14, 10:14] = 1
    segmentation[16:19, 2:14, 10:14] = 1
    end_points = np.array(
        [
            [7, 12, 12],
            [26, 12, 12],
            [17, 4, 12],
            [17, 7, 12],
        ]
    )
    repair_candidates = [
        np.array(
            [
                [26, 12, 12],
                [17, 4, 12],
                [17, 7, 12],
            ]
        ),
        np.array(
            [
                [7, 12, 12],
                [17, 4, 12],
                [17, 7, 12],
            ]
        ),
        np.array(
            [
                [7, 12, 12],
                [26, 12, 12],
                [17, 7, 12],
            ]
        ),
        np.linspace([7, 12, 12], [26, 12, 12], num=19, dtype=np.int64),
    ]

    label_map = np.zeros((30, 30, 30))
    label_map[7:27, 12, 12] = 1
    label_map[17, 4:8, 12] = 2

    repair_start, repair_end = find_break_repairs(
        end_point_coordinates=end_points,
        repair_candidates=repair_candidates,
        label_map=label_map,
        segmentation=segmentation,
    )

    expected_repair_start = np.array(
        [
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
            [17, 7, 12],
        ]
    )
    expected_repair_end = np.array(
        [
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
            [17, 12, 12],
        ]
    )

    np.testing.assert_array_equal(repair_start, expected_repair_start)
    np.testing.assert_array_equal(repair_end, expected_repair_end)


def test_find_break_repairs_no_repairs():
    """Test find_break_repairs where there are no breaks to fix."""
    # make the segmentation that has a gap between the skeleton pieces
    # this break shouldn't be repaired because the two skeleton pieces
    # are in different segments
    segmentation = np.zeros((30, 30, 30))
    segmentation[5:18, 10:14, 10:14] = 1
    segmentation[19:25, 10:14, 10:14] = 1
    end_points = np.array([[7, 12, 12], [15, 12, 12], [20, 12, 12], [23, 12, 12]])
    repair_candidates = [
        np.array(
            [
                [15, 12, 12],
            ]
        ),
        np.array(
            [
                [7, 12, 12],
                [20, 12, 12],
            ]
        ),
        np.array(
            [
                [15, 12, 12],
                [23, 12, 12],
            ]
        ),
        np.array(
            [
                [20, 12, 12],
            ]
        ),
    ]

    label_map = np.zeros((30, 30, 30))
    label_map[7:16, 12, 12] = 1
    label_map[20:24, 12, 12] = 2

    repair_start, repair_end = find_break_repairs(
        end_point_coordinates=end_points,
        repair_candidates=repair_candidates,
        label_map=label_map,
        segmentation=segmentation,
    )

    expected_repair_start = np.array(
        [
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
        ]
    )
    expected_repair_end = np.array(
        [
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
        ]
    )

    np.testing.assert_array_equal(repair_start, expected_repair_start)
    np.testing.assert_array_equal(repair_end, expected_repair_end)


def test_draw_repairs_axis_aligned_and_diagonal():
    """Test drawing repair lines for axis-aligned and diagonal connections."""
    skeleton = np.zeros((10, 10, 10), dtype=bool)

    # Set up two repairs:
    # 1. Axis-aligned repair along z-axis from (2, 5, 5) to (6, 5, 5)
    # 2. Diagonal repair from (5, 2, 2) to (7, 4, 4)
    repair_start = np.array(
        [
            [2, 5, 5],  # axis-aligned start
            [5, 2, 2],  # diagonal start
        ],
        dtype=np.float64,
    )

    repair_end = np.array(
        [
            [6, 5, 5],  # axis-aligned end
            [7, 4, 4],  # diagonal end
        ],
        dtype=np.float64,
    )

    # Draw the repairs
    draw_lines(skeleton, repair_start, repair_end)

    expected = np.zeros((10, 10, 10), dtype=bool)

    # Expected axis-aligned line (z from 2 to 6, y=5, x=5)
    expected[2, 5, 5] = True
    expected[3, 5, 5] = True
    expected[4, 5, 5] = True
    expected[5, 5, 5] = True
    expected[6, 5, 5] = True

    # Expected diagonal line from (5, 2, 2) to (7, 4, 4)
    expected[5, 2, 2] = True
    expected[6, 3, 3] = True
    expected[7, 4, 4] = True

    # Verify the result
    np.testing.assert_array_equal(skeleton, expected)


def test_get_skeleton_data_cpu():
    """Test with one axis-aligned line and one diagonal line.

    This creates a simple, interpretable test case with known structure:
    - Line 1: Axis-aligned (along z-axis) from [2,2,2] to [5,2,2]
    - Line 2: Diagonal from [5,5,5] to [7,7,7]

    We can manually verify all expected outputs.
    """
    # Create empty skeleton
    skeleton = np.zeros((10, 10, 10), dtype=bool)

    # Line 1: Axis-aligned vertical line (4 voxels)
    # z=2,3,4,5 at y=2, x=2
    skeleton[2, 2, 2] = True  # Bottom endpoint
    skeleton[3, 2, 2] = True  # Middle
    skeleton[4, 2, 2] = True  # Middle
    skeleton[5, 2, 2] = True  # Top endpoint

    # Line 2: Diagonal line (3 voxels)
    # Moving diagonally in all dimensions
    skeleton[5, 5, 5] = True  # Start endpoint
    skeleton[6, 6, 6] = True  # Middle
    skeleton[7, 7, 7] = True  # End endpoint

    # Get skeleton data
    degree_map, degree_one_coords, all_coords, label_map = get_skeleton_data_cpu(
        skeleton
    )

    # Check the degree map
    # Expected degrees:
    # - Line 1 endpoints (2,2,2) and (5,2,2): degree 1
    # - Line 1 middle voxels (3,2,2) and (4,2,2): degree 2
    # - Line 2 endpoints (5,5,5) and (7,7,7): degree 1
    # - Line 2 middle voxel (6,6,6): degree 2
    # - All other voxels: degree 0

    expected_degree_map = np.zeros((10, 10, 10), dtype=np.uint8)
    expected_degree_map[2, 2, 2] = 1  # Line 1 bottom endpoint
    expected_degree_map[3, 2, 2] = 2  # Line 1 middle
    expected_degree_map[4, 2, 2] = 2  # Line 1 middle
    expected_degree_map[5, 2, 2] = 1  # Line 1 top endpoint
    expected_degree_map[5, 5, 5] = 1  # Line 2 start endpoint
    expected_degree_map[6, 6, 6] = 2  # Line 2 middle
    expected_degree_map[7, 7, 7] = 1  # Line 2 end endpoint

    np.testing.assert_array_equal(
        degree_map,
        expected_degree_map,
        err_msg="Degree map does not match expected values",
    )

    # Check the degree one coordinates
    # Expected: 4 endpoints total (2 per line)
    expected_endpoints = np.array(
        [
            [2, 2, 2],  # Line 1 bottom
            [5, 2, 2],  # Line 1 top
            [5, 5, 5],  # Line 2 start
            [7, 7, 7],  # Line 2 end
        ]
    )

    # Sort both arrays to ensure consistent ordering for comparison
    degree_one_coords_sorted = degree_one_coords[
        np.lexsort(
            (degree_one_coords[:, 2], degree_one_coords[:, 1], degree_one_coords[:, 0])
        )
    ]
    expected_endpoints_sorted = expected_endpoints[
        np.lexsort(
            (
                expected_endpoints[:, 2],
                expected_endpoints[:, 1],
                expected_endpoints[:, 0],
            )
        )
    ]

    np.testing.assert_array_equal(
        degree_one_coords_sorted,
        expected_endpoints_sorted,
        err_msg="Endpoint coordinates do not match expected values",
    )

    # Check the skeleton voxel coordinates
    # Expected: 7 total skeleton voxels
    expected_all_coords = np.array(
        [
            [2, 2, 2],  # Line 1
            [3, 2, 2],
            [4, 2, 2],
            [5, 2, 2],
            [5, 5, 5],  # Line 2
            [6, 6, 6],
            [7, 7, 7],
        ]
    )

    # Sort both arrays for consistent ordering
    all_coords_sorted = all_coords[
        np.lexsort((all_coords[:, 2], all_coords[:, 1], all_coords[:, 0]))
    ]
    expected_all_coords_sorted = expected_all_coords[
        np.lexsort(
            (
                expected_all_coords[:, 2],
                expected_all_coords[:, 1],
                expected_all_coords[:, 0],
            )
        )
    ]

    np.testing.assert_array_equal(
        all_coords_sorted,
        expected_all_coords_sorted,
        err_msg="All skeleton coordinates do not match expected values",
    )

    # Check the label map
    # Expected: Two separate connected components
    # Line 1 should have one label, Line 2 should have a different label

    # Check that background is 0
    assert label_map[0, 0, 0] == 0, "Background should be labeled 0"

    # Get labels for each line
    line1_label = label_map[2, 2, 2]
    line2_label = label_map[5, 5, 5]

    # Check that labels are positive
    assert line1_label > 0, "Line 1 should have positive label"
    assert line2_label > 0, "Line 2 should have positive label"

    # Check that the two lines have different labels
    assert line1_label != line2_label, "Two separate lines should have different labels"

    # Create expected label map
    expected_label_map = np.zeros((10, 10, 10), dtype=label_map.dtype)
    expected_label_map[2, 2, 2] = line1_label  # Line 1
    expected_label_map[3, 2, 2] = line1_label
    expected_label_map[4, 2, 2] = line1_label
    expected_label_map[5, 2, 2] = line1_label
    expected_label_map[5, 5, 5] = line2_label  # Line 2
    expected_label_map[6, 6, 6] = line2_label
    expected_label_map[7, 7, 7] = line2_label

    np.testing.assert_array_equal(
        label_map,
        expected_label_map,
        err_msg="Label map does not match expected values",
    )


def test_get_skeleton_data_cpu_with_bounding_box():
    """Test get_skeleton_data_cpu with endpoint_bounding_box filtering.

    This creates the same structure as test_get_skeleton_data_cpu:
    - Line 1: Axis-aligned (along z-axis) from [2,2,2] to [5,2,2]
    - Line 2: Diagonal from [5,5,5] to [7,7,7]

    But uses a bounding box to exclude Line 2 endpoints.
    """
    # Create empty skeleton
    skeleton = np.zeros((10, 10, 10), dtype=bool)

    # Line 1: Axis-aligned vertical line (4 voxels)
    # z=2,3,4,5 at y=2, x=2
    skeleton[2, 2, 2] = True  # Bottom endpoint
    skeleton[3, 2, 2] = True  # Middle
    skeleton[4, 2, 2] = True  # Middle
    skeleton[5, 2, 2] = True  # Top endpoint

    # Line 2: Diagonal line (3 voxels)
    # Moving diagonally in all dimensions
    skeleton[5, 5, 5] = True  # Start endpoint
    skeleton[6, 6, 6] = True  # Middle
    skeleton[7, 7, 7] = True  # End endpoint

    # Define bounding box that includes Line 1 but excludes Line 2
    # Line 1 is at (z=2-5, y=2, x=2)
    # Line 2 is at (z=5-7, y=5-7, x=5-7)
    # Bounding box: z=[0,10), y=[0,4), x=[0,4) will include Line 1 only
    endpoint_bounding_box = ((0, 0, 0), (10, 4, 4))

    # Get skeleton data with bounding box
    degree_map, degree_one_coords, all_coords, label_map = get_skeleton_data_cpu(
        skeleton, endpoint_bounding_box=endpoint_bounding_box
    )

    # Check the degree map (should be same as without bounding box)
    expected_degree_map = np.zeros((10, 10, 10), dtype=np.uint8)
    expected_degree_map[2, 2, 2] = 1  # Line 1 bottom endpoint
    expected_degree_map[3, 2, 2] = 2  # Line 1 middle
    expected_degree_map[4, 2, 2] = 2  # Line 1 middle
    expected_degree_map[5, 2, 2] = 1  # Line 1 top endpoint
    expected_degree_map[5, 5, 5] = 1  # Line 2 start endpoint
    expected_degree_map[6, 6, 6] = 2  # Line 2 middle
    expected_degree_map[7, 7, 7] = 1  # Line 2 end endpoint

    np.testing.assert_array_equal(
        degree_map,
        expected_degree_map,
        err_msg="Degree map does not match expected values",
    )

    # Check the degree one coordinates
    # Expected: Only 2 endpoints from Line 1 (Line 2 endpoints excluded by bbox)
    expected_endpoints = np.array(
        [
            [2, 2, 2],  # Line 1 bottom
            [5, 2, 2],  # Line 1 top
            # Line 2 endpoints [5,5,5] and [7,7,7] are excluded
        ]
    )

    # Sort both arrays to ensure consistent ordering for comparison
    degree_one_coords_sorted = degree_one_coords[
        np.lexsort(
            (degree_one_coords[:, 2], degree_one_coords[:, 1], degree_one_coords[:, 0])
        )
    ]
    expected_endpoints_sorted = expected_endpoints[
        np.lexsort(
            (
                expected_endpoints[:, 2],
                expected_endpoints[:, 1],
                expected_endpoints[:, 0],
            )
        )
    ]

    np.testing.assert_array_equal(
        degree_one_coords_sorted,
        expected_endpoints_sorted,
        err_msg="Endpoint coordinates do not match expected values",
    )

    # Check the skeleton voxel coordinates (should include all voxels, not filtered)
    expected_all_coords = np.array(
        [
            [2, 2, 2],  # Line 1
            [3, 2, 2],
            [4, 2, 2],
            [5, 2, 2],
            [5, 5, 5],  # Line 2
            [6, 6, 6],
            [7, 7, 7],
        ]
    )

    # Sort both arrays for consistent ordering
    all_coords_sorted = all_coords[
        np.lexsort((all_coords[:, 2], all_coords[:, 1], all_coords[:, 0]))
    ]
    expected_all_coords_sorted = expected_all_coords[
        np.lexsort(
            (
                expected_all_coords[:, 2],
                expected_all_coords[:, 1],
                expected_all_coords[:, 0],
            )
        )
    ]

    np.testing.assert_array_equal(
        all_coords_sorted,
        expected_all_coords_sorted,
        err_msg="All skeleton coordinates do not match expected values",
    )

    # Check the label map (should be same as without bounding box)
    # Expected: Two separate connected components
    # Line 1 should have one label, Line 2 should have a different label

    # Check that background is 0
    assert label_map[0, 0, 0] == 0, "Background should be labeled 0"

    # Get labels for each line
    line1_label = label_map[2, 2, 2]
    line2_label = label_map[5, 5, 5]

    # Check that labels are positive
    assert line1_label > 0, "Line 1 should have positive label"
    assert line2_label > 0, "Line 2 should have positive label"

    # Check that the two lines have different labels
    assert line1_label != line2_label, "Two separate lines should have different labels"

    # Create expected label map
    expected_label_map = np.zeros((10, 10, 10), dtype=label_map.dtype)
    expected_label_map[2, 2, 2] = line1_label  # Line 1
    expected_label_map[3, 2, 2] = line1_label
    expected_label_map[4, 2, 2] = line1_label
    expected_label_map[5, 2, 2] = line1_label
    expected_label_map[5, 5, 5] = line2_label  # Line 2
    expected_label_map[6, 6, 6] = line2_label
    expected_label_map[7, 7, 7] = line2_label

    np.testing.assert_array_equal(
        label_map,
        expected_label_map,
        err_msg="Label map does not match expected values",
    )


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy is not available")
def test_get_skeleton_data_cupy():
    """Test with one axis-aligned line and one diagonal line.

    This creates a simple, interpretable test case with known structure:
    - Line 1: Axis-aligned (along z-axis) from [2,2,2] to [5,2,2]
    - Line 2: Diagonal from [5,5,5] to [7,7,7]

    We can manually verify all expected outputs.
    """
    from skeleplex.skeleton._break_detection import get_skeleton_data_cupy

    # Create empty skeleton
    skeleton = np.zeros((10, 10, 10), dtype=bool)

    # Line 1: Axis-aligned vertical line (4 voxels)
    # z=2,3,4,5 at y=2, x=2
    skeleton[2, 2, 2] = True  # Bottom endpoint
    skeleton[3, 2, 2] = True  # Middle
    skeleton[4, 2, 2] = True  # Middle
    skeleton[5, 2, 2] = True  # Top endpoint

    # Line 2: Diagonal line (3 voxels)
    # Moving diagonally in all dimensions
    skeleton[5, 5, 5] = True  # Start endpoint
    skeleton[6, 6, 6] = True  # Middle
    skeleton[7, 7, 7] = True  # End endpoint

    # Get skeleton data
    degree_map, degree_one_coords, all_coords, label_map = get_skeleton_data_cupy(
        skeleton
    )

    # Check the degree map
    # Expected degrees:
    # - Line 1 endpoints (2,2,2) and (5,2,2): degree 1
    # - Line 1 middle voxels (3,2,2) and (4,2,2): degree 2
    # - Line 2 endpoints (5,5,5) and (7,7,7): degree 1
    # - Line 2 middle voxel (6,6,6): degree 2
    # - All other voxels: degree 0

    expected_degree_map = np.zeros((10, 10, 10), dtype=np.uint8)
    expected_degree_map[2, 2, 2] = 1  # Line 1 bottom endpoint
    expected_degree_map[3, 2, 2] = 2  # Line 1 middle
    expected_degree_map[4, 2, 2] = 2  # Line 1 middle
    expected_degree_map[5, 2, 2] = 1  # Line 1 top endpoint
    expected_degree_map[5, 5, 5] = 1  # Line 2 start endpoint
    expected_degree_map[6, 6, 6] = 2  # Line 2 middle
    expected_degree_map[7, 7, 7] = 1  # Line 2 end endpoint

    np.testing.assert_array_equal(
        degree_map,
        expected_degree_map,
        err_msg="Degree map does not match expected values",
    )

    # Check the degree one coordinates
    # Expected: 4 endpoints total (2 per line)
    expected_endpoints = np.array(
        [
            [2, 2, 2],  # Line 1 bottom
            [5, 2, 2],  # Line 1 top
            [5, 5, 5],  # Line 2 start
            [7, 7, 7],  # Line 2 end
        ]
    )

    # Sort both arrays to ensure consistent ordering for comparison
    degree_one_coords_sorted = degree_one_coords[
        np.lexsort(
            (degree_one_coords[:, 2], degree_one_coords[:, 1], degree_one_coords[:, 0])
        )
    ]
    expected_endpoints_sorted = expected_endpoints[
        np.lexsort(
            (
                expected_endpoints[:, 2],
                expected_endpoints[:, 1],
                expected_endpoints[:, 0],
            )
        )
    ]

    np.testing.assert_array_equal(
        degree_one_coords_sorted,
        expected_endpoints_sorted,
        err_msg="Endpoint coordinates do not match expected values",
    )

    # Check the skeleton voxel coordinates
    # Expected: 7 total skeleton voxels
    expected_all_coords = np.array(
        [
            [2, 2, 2],  # Line 1
            [3, 2, 2],
            [4, 2, 2],
            [5, 2, 2],
            [5, 5, 5],  # Line 2
            [6, 6, 6],
            [7, 7, 7],
        ]
    )

    # Sort both arrays for consistent ordering
    all_coords_sorted = all_coords[
        np.lexsort((all_coords[:, 2], all_coords[:, 1], all_coords[:, 0]))
    ]
    expected_all_coords_sorted = expected_all_coords[
        np.lexsort(
            (
                expected_all_coords[:, 2],
                expected_all_coords[:, 1],
                expected_all_coords[:, 0],
            )
        )
    ]

    np.testing.assert_array_equal(
        all_coords_sorted,
        expected_all_coords_sorted,
        err_msg="All skeleton coordinates do not match expected values",
    )

    # Check the label map
    # Expected: Two separate connected components
    # Line 1 should have one label, Line 2 should have a different label

    # Check that background is 0
    assert label_map[0, 0, 0] == 0, "Background should be labeled 0"

    # Get labels for each line
    line1_label = label_map[2, 2, 2]
    line2_label = label_map[5, 5, 5]

    # Check that labels are positive
    assert line1_label > 0, "Line 1 should have positive label"
    assert line2_label > 0, "Line 2 should have positive label"

    # Check that the two lines have different labels
    assert line1_label != line2_label, "Two separate lines should have different labels"

    # Create expected label map
    expected_label_map = np.zeros((10, 10, 10), dtype=label_map.dtype)
    expected_label_map[2, 2, 2] = line1_label  # Line 1
    expected_label_map[3, 2, 2] = line1_label
    expected_label_map[4, 2, 2] = line1_label
    expected_label_map[5, 2, 2] = line1_label
    expected_label_map[5, 5, 5] = line2_label  # Line 2
    expected_label_map[6, 6, 6] = line2_label
    expected_label_map[7, 7, 7] = line2_label

    np.testing.assert_array_equal(
        label_map,
        expected_label_map,
        err_msg="Label map does not match expected values",
    )


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy is not available")
def test_get_skeleton_data_cupy_with_bounding_box():
    """Test get_skeleton_data_cupy with endpoint_bounding_box filtering.

    This creates the same structure as test_get_skeleton_data_cpu:
    - Line 1: Axis-aligned (along z-axis) from [2,2,2] to [5,2,2]
    - Line 2: Diagonal from [5,5,5] to [7,7,7]

    But uses a bounding box to exclude Line 2 endpoints.
    """
    from skeleplex.skeleton._break_detection import get_skeleton_data_cupy

    # Create empty skeleton
    skeleton = np.zeros((10, 10, 10), dtype=bool)

    # Line 1: Axis-aligned vertical line (4 voxels)
    # z=2,3,4,5 at y=2, x=2
    skeleton[2, 2, 2] = True  # Bottom endpoint
    skeleton[3, 2, 2] = True  # Middle
    skeleton[4, 2, 2] = True  # Middle
    skeleton[5, 2, 2] = True  # Top endpoint

    # Line 2: Diagonal line (3 voxels)
    # Moving diagonally in all dimensions
    skeleton[5, 5, 5] = True  # Start endpoint
    skeleton[6, 6, 6] = True  # Middle
    skeleton[7, 7, 7] = True  # End endpoint

    # Define bounding box that includes Line 1 but excludes Line 2
    # Line 1 is at (z=2-5, y=2, x=2)
    # Line 2 is at (z=5-7, y=5-7, x=5-7)
    # Bounding box: z=[0,10), y=[0,4), x=[0,4) will include Line 1 only
    endpoint_bounding_box = ((0, 0, 0), (10, 4, 4))

    # Get skeleton data with bounding box
    degree_map, degree_one_coords, all_coords, label_map = get_skeleton_data_cupy(
        skeleton, endpoint_bounding_box=endpoint_bounding_box
    )

    # Check the degree map (should be same as without bounding box)
    expected_degree_map = np.zeros((10, 10, 10), dtype=np.uint8)
    expected_degree_map[2, 2, 2] = 1  # Line 1 bottom endpoint
    expected_degree_map[3, 2, 2] = 2  # Line 1 middle
    expected_degree_map[4, 2, 2] = 2  # Line 1 middle
    expected_degree_map[5, 2, 2] = 1  # Line 1 top endpoint
    expected_degree_map[5, 5, 5] = 1  # Line 2 start endpoint
    expected_degree_map[6, 6, 6] = 2  # Line 2 middle
    expected_degree_map[7, 7, 7] = 1  # Line 2 end endpoint

    np.testing.assert_array_equal(
        degree_map,
        expected_degree_map,
        err_msg="Degree map does not match expected values",
    )

    # Check the degree one coordinates
    # Expected: Only 2 endpoints from Line 1 (Line 2 endpoints excluded by bbox)
    expected_endpoints = np.array(
        [
            [2, 2, 2],  # Line 1 bottom
            [5, 2, 2],  # Line 1 top
            # Line 2 endpoints [5,5,5] and [7,7,7] are excluded
        ]
    )

    # Sort both arrays to ensure consistent ordering for comparison
    degree_one_coords_sorted = degree_one_coords[
        np.lexsort(
            (degree_one_coords[:, 2], degree_one_coords[:, 1], degree_one_coords[:, 0])
        )
    ]
    expected_endpoints_sorted = expected_endpoints[
        np.lexsort(
            (
                expected_endpoints[:, 2],
                expected_endpoints[:, 1],
                expected_endpoints[:, 0],
            )
        )
    ]

    np.testing.assert_array_equal(
        degree_one_coords_sorted,
        expected_endpoints_sorted,
        err_msg="Endpoint coordinates do not match expected values",
    )

    # Check the skeleton voxel coordinates (should include all voxels, not filtered)
    expected_all_coords = np.array(
        [
            [2, 2, 2],  # Line 1
            [3, 2, 2],
            [4, 2, 2],
            [5, 2, 2],
            [5, 5, 5],  # Line 2
            [6, 6, 6],
            [7, 7, 7],
        ]
    )

    # Sort both arrays for consistent ordering
    all_coords_sorted = all_coords[
        np.lexsort((all_coords[:, 2], all_coords[:, 1], all_coords[:, 0]))
    ]
    expected_all_coords_sorted = expected_all_coords[
        np.lexsort(
            (
                expected_all_coords[:, 2],
                expected_all_coords[:, 1],
                expected_all_coords[:, 0],
            )
        )
    ]

    np.testing.assert_array_equal(
        all_coords_sorted,
        expected_all_coords_sorted,
        err_msg="All skeleton coordinates do not match expected values",
    )

    # Check the label map (should be same as without bounding box)
    # Expected: Two separate connected components
    # Line 1 should have one label, Line 2 should have a different label

    # Check that background is 0
    assert label_map[0, 0, 0] == 0, "Background should be labeled 0"

    # Get labels for each line
    line1_label = label_map[2, 2, 2]
    line2_label = label_map[5, 5, 5]

    # Check that labels are positive
    assert line1_label > 0, "Line 1 should have positive label"
    assert line2_label > 0, "Line 2 should have positive label"

    # Check that the two lines have different labels
    assert line1_label != line2_label, "Two separate lines should have different labels"

    # Create expected label map
    expected_label_map = np.zeros((10, 10, 10), dtype=label_map.dtype)
    expected_label_map[2, 2, 2] = line1_label  # Line 1
    expected_label_map[3, 2, 2] = line1_label
    expected_label_map[4, 2, 2] = line1_label
    expected_label_map[5, 2, 2] = line1_label
    expected_label_map[5, 5, 5] = line2_label  # Line 2
    expected_label_map[6, 6, 6] = line2_label
    expected_label_map[7, 7, 7] = line2_label

    np.testing.assert_array_equal(
        label_map,
        expected_label_map,
        err_msg="Label map does not match expected values",
    )


def test_repair_breaks_value_error_not_3d():
    """Test that a ValueError is raised when skeleton is not 3D."""

    skeleton = np.zeros((10, 10), dtype=bool)  # 2D skeleton
    segmentation = np.zeros((10, 10, 10), dtype=bool)

    with pytest.raises(ValueError, match="Expected 3D skeleton_image"):
        repair_breaks(skeleton, segmentation)


def test_repair_breaks_value_error_shape_mismatch():
    """Test that a ValueError is raised when shapes don't match."""

    skeleton = np.zeros((10, 10, 10), dtype=bool)
    segmentation = np.zeros((10, 10, 5), dtype=bool)  # Different shape

    with pytest.raises(ValueError, match="must have the same shape"):
        repair_breaks(skeleton, segmentation)


def test_repair_breaks_simple_line():
    """Test repair_breaks with a simple axis-aligned line with a break."""

    # Create skeleton with a break
    skeleton = np.zeros((30, 30, 30), dtype=bool)
    skeleton[7:16, 12, 12] = True  # First segment
    skeleton[20:24, 12, 12] = True  # Second segment (break from 16 to 20)

    # Segmentation includes the gap
    segmentation = np.zeros((30, 30, 30), dtype=bool)
    segmentation[5:25, 10:14, 10:14] = True

    # Repair the break
    repaired = repair_breaks(skeleton, segmentation, repair_radius=10.0)

    # Expected: complete line from 7 to 23
    expected = np.zeros((30, 30, 30), dtype=bool)
    expected[7:24, 12, 12] = True

    np.testing.assert_array_equal(repaired, expected)


def test_repair_breaks_line_break_too_long():
    """Test that breaks longer than repair_radius are not fixed."""

    # Create skeleton with a large break
    skeleton = np.zeros((30, 30, 30), dtype=bool)
    skeleton[7:16, 12, 12] = True  # First segment
    skeleton[20:24, 12, 12] = True  # Second segment (break of 4 voxels)

    # Segmentation includes the gap
    segmentation = np.zeros((30, 30, 30), dtype=bool)
    segmentation[5:25, 10:14, 10:14] = True

    # Try to repair with small radius (should not connect)
    repaired = repair_breaks(skeleton, segmentation, repair_radius=3.0)

    # Expected: no change (break is too long)
    expected = skeleton.copy()

    np.testing.assert_array_equal(repaired, expected)


def test_repair_breaks_tee():
    """Test repair_breaks with a T-junction skeleton with a break."""

    # Create T-shaped skeleton with a break in the vertical stem
    skeleton = np.zeros((30, 30, 30), dtype=bool)
    # Vertical stem (with break)
    skeleton[7:17, 12, 12] = True  # Top part of stem
    skeleton[20:27, 12, 12] = True  # Bottom part of stem (break from 17 to 20)
    # Horizontal crossbar
    skeleton[17, 4:8, 12] = True  # Left part of crossbar

    # Segmentation allows the repair
    segmentation = np.zeros((30, 30, 30), dtype=bool)
    segmentation[5:28, 10:14, 10:14] = True
    segmentation[16:19, 2:14, 10:14] = True

    # Repair the break
    repaired = repair_breaks(skeleton, segmentation, repair_radius=5.0)

    # Expected: complete T-shape
    expected = np.zeros((30, 30, 30), dtype=bool)
    expected[7:27, 12, 12] = True  # Complete vertical stem
    expected[17, 4:8, 12] = True  # Horizontal crossbar

    np.testing.assert_array_equal(repaired, expected)


def test_get_skeleton_data_cpu_uses_provided_label_map():
    """When label_map is provided, it is returned instead of local labelling."""
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[2:6, 2, 2] = True
    skeleton[5:8, 5, 5] = True

    # Custom label map where both lines share a label
    custom_label_map = np.zeros((10, 10, 10), dtype=np.int32)
    custom_label_map[2:6, 2, 2] = 42
    custom_label_map[5:8, 5, 5] = 42

    _, _, _, returned_label_map = get_skeleton_data_cpu(
        skeleton, label_map=custom_label_map
    )

    np.testing.assert_array_equal(returned_label_map, custom_label_map)


def test_get_skeleton_data_cpu_label_map_shape_mismatch():
    """ValueError is raised when label_map shape differs from skeleton."""
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[2:6, 2, 2] = True

    wrong_shape_map = np.zeros((5, 5, 5), dtype=np.int32)

    with pytest.raises(ValueError, match="label_map shape"):
        get_skeleton_data_cpu(skeleton, label_map=wrong_shape_map)


def test_get_skeleton_data_cpu_none_label_map_computes_locally():
    """When label_map is None, local connected component labelling is used."""
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[2:6, 2, 2] = True  # Line 1
    skeleton[7:9, 7, 7] = True  # Line 2

    _, _, _, label_map = get_skeleton_data_cpu(skeleton, label_map=None)

    # Two disconnected lines should get different labels
    label_1 = label_map[2, 2, 2]
    label_2 = label_map[7, 7, 7]
    assert label_1 > 0
    assert label_2 > 0
    assert label_1 != label_2


def test_repair_breaks_global_label_map_prevents_false_repair():
    """Global label map prevents spurious repairs between same-component voxels.

    Two skeleton segments appear locally disconnected, but the global
    label map marks them as the same connected component (connected
    through a path outside this region). No repair should be drawn.
    """
    skeleton = np.zeros((30, 30, 30), dtype=bool)
    skeleton[7:16, 12, 12] = True  # Segment A
    skeleton[20:24, 12, 12] = True  # Segment B

    segmentation = np.zeros((30, 30, 30), dtype=bool)
    segmentation[5:25, 10:14, 10:14] = True

    # Both segments belong to the SAME global component
    global_label_map = np.zeros((30, 30, 30), dtype=np.int32)
    global_label_map[7:16, 12, 12] = 1
    global_label_map[20:24, 12, 12] = 1

    repaired = repair_breaks(
        skeleton,
        segmentation,
        repair_radius=10.0,
        label_map=global_label_map,
    )

    # Skeleton should be unchanged
    np.testing.assert_array_equal(repaired, skeleton)


def test_repair_breaks_without_global_label_map_makes_repair():
    """Without global label map, local labelling sees two components.

    Counterpart to test_repair_breaks_global_label_map_prevents_false_repair.
    Local labelling assigns different labels, so a repair is drawn.
    """
    skeleton = np.zeros((30, 30, 30), dtype=bool)
    skeleton[7:16, 12, 12] = True
    skeleton[20:24, 12, 12] = True

    segmentation = np.zeros((30, 30, 30), dtype=bool)
    segmentation[5:25, 10:14, 10:14] = True

    repaired = repair_breaks(
        skeleton,
        segmentation,
        repair_radius=10.0,
        label_map=None,
    )

    expected = np.zeros((30, 30, 30), dtype=bool)
    expected[7:24, 12, 12] = True
    np.testing.assert_array_equal(repaired, expected)


def _build_degree_map(skeleton):
    """Compute the degree map from a binary skeleton."""
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    degree_map = convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)
    return (degree_map * skeleton).astype(np.uint8)


@pytest.mark.parametrize(
    "skeleton_slices, endpoint, expected_axis_index",
    [
        pytest.param(np.s_[3:14, 10, 10], [3, 10, 10], 0, id="z-axis"),
        pytest.param(np.s_[10, 3:14, 10], [10, 3, 10], 1, id="y-axis"),
        pytest.param(np.s_[10, 10, 3:14], [10, 10, 3], 2, id="x-axis"),
    ],
)
def test_get_endpoint_directions_axis_aligned(
    skeleton_slices, endpoint, expected_axis_index
):
    """Direction for a straight axis-aligned line is parallel to that axis."""
    skeleton = np.zeros((20, 20, 20), dtype=bool)
    skeleton[skeleton_slices] = True
    degree_map = _build_degree_map(skeleton)

    endpoints = np.array([endpoint])
    directions = get_endpoint_directions(
        skeleton, endpoints, degree_map, n_fit_voxels=10
    )

    d = directions[0]
    norm = np.linalg.norm(d)
    assert norm > 0.99, f"Should be unit vector, got norm={norm}"
    assert (
        abs(abs(d[expected_axis_index]) - 1.0) < 0.01
    ), f"Expected direction along axis {expected_axis_index}, got {d}"


def test_get_endpoint_directions_single_voxel():
    """A single-voxel skeleton returns the zero vector (degenerate)."""
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[5, 5, 5] = True

    degree_map = np.zeros((10, 10, 10), dtype=np.uint8)
    endpoints = np.array([[5, 5, 5]])

    directions = get_endpoint_directions(
        skeleton, endpoints, degree_map, n_fit_voxels=10
    )

    np.testing.assert_array_equal(directions[0], [0.0, 0.0, 0.0])


def test_get_endpoint_directions_two_voxel_segment():
    """A two-voxel segment produces a valid unit direction vector."""
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[5, 5, 5] = True
    skeleton[6, 5, 5] = True

    degree_map = np.zeros((10, 10, 10), dtype=np.uint8)
    degree_map[5, 5, 5] = 1
    degree_map[6, 5, 5] = 1

    endpoints = np.array([[5, 5, 5]])
    directions = get_endpoint_directions(
        skeleton, endpoints, degree_map, n_fit_voxels=10
    )

    d = directions[0]
    assert np.linalg.norm(d) > 0.99, "Should return a unit vector"
    assert abs(abs(d[0]) - 1.0) < 0.01, f"Direction should be along z, got {d}"


def test_get_endpoint_directions_diagonal_line():
    """Direction for a diagonal line is along the (1,1,1) diagonal."""
    skeleton = np.zeros((20, 20, 20), dtype=bool)
    for i in range(10):
        skeleton[3 + i, 3 + i, 3 + i] = True

    degree_map = np.zeros((20, 20, 20), dtype=np.uint8)
    degree_map[3, 3, 3] = 1
    for i in range(1, 9):
        degree_map[3 + i, 3 + i, 3 + i] = 2
    degree_map[12, 12, 12] = 1

    endpoints = np.array([[3, 3, 3]])
    directions = get_endpoint_directions(
        skeleton, endpoints, degree_map, n_fit_voxels=10
    )

    d = directions[0]
    expected_dir = np.array([1.0, 1.0, 1.0])
    expected_dir /= np.linalg.norm(expected_dir)

    cosine = abs(np.dot(d, expected_dir))
    assert cosine > 0.99, f"Expected direction along (1,1,1), got {d}"


def _setup_two_candidate_scenario():
    """Common setup for angle cost tests.

    One endpoint at (10,10,10) with two candidates in different
    components:
    - Candidate A at (10,13,10): 3 voxels away, perpendicular to z
    - Candidate B at (14,10,10): 4 voxels away, aligned with z

    Endpoint direction is along the z-axis.
    """
    endpoint = np.array([[10, 10, 10]])
    candidates = [
        np.array(
            [
                [10, 13, 10],  # A: perpendicular, closer
                [14, 10, 10],  # B: aligned, farther
            ]
        ),
    ]

    label_map = np.zeros((20, 20, 20), dtype=np.int32)
    label_map[10, 10, 10] = 1
    label_map[10, 13, 10] = 2
    label_map[14, 10, 10] = 3

    segmentation = np.ones((20, 20, 20), dtype=bool)
    z_direction = np.array([[1.0, 0.0, 0.0]])

    return endpoint, candidates, label_map, segmentation, z_direction


def test_find_break_repairs_angle_cost_selects_aligned():
    """With high w_angle, the aligned (farther) candidate is preferred."""
    endpoint, candidates, label_map, seg, direction = _setup_two_candidate_scenario()

    _, repair_end = find_break_repairs(
        end_point_coordinates=endpoint,
        repair_candidates=candidates,
        label_map=label_map,
        segmentation=seg,
        endpoint_directions=direction,
        w_distance=0.1,
        w_angle=10.0,
    )

    # Candidate B (aligned along z)
    np.testing.assert_array_equal(repair_end[0], [14, 10, 10])


def test_find_break_repairs_distance_only_selects_closer():
    """With w_angle=0, the closer candidate is always selected."""
    endpoint, candidates, label_map, seg, direction = _setup_two_candidate_scenario()

    _, repair_end = find_break_repairs(
        end_point_coordinates=endpoint,
        repair_candidates=candidates,
        label_map=label_map,
        segmentation=seg,
        endpoint_directions=direction,
        w_distance=1.0,
        w_angle=0.0,
    )

    # Candidate A (closer)
    np.testing.assert_array_equal(repair_end[0], [10, 13, 10])


def test_find_break_repairs_degenerate_direction_falls_back():
    """Zero direction vector disables angle term; closer candidate wins."""
    endpoint, candidates, label_map, seg, _ = _setup_two_candidate_scenario()

    _, repair_end = find_break_repairs(
        end_point_coordinates=endpoint,
        repair_candidates=candidates,
        label_map=label_map,
        segmentation=seg,
        endpoint_directions=np.array([[0.0, 0.0, 0.0]]),
        w_distance=1.0,
        w_angle=10.0,
    )

    # Falls back to distance-only → candidate A
    np.testing.assert_array_equal(repair_end[0], [10, 13, 10])


def test_find_break_repairs_none_directions_uses_distance():
    """When endpoint_directions is None, distance-only selection is used."""
    endpoint, candidates, label_map, seg, _ = _setup_two_candidate_scenario()

    _, repair_end = find_break_repairs(
        end_point_coordinates=endpoint,
        repair_candidates=candidates,
        label_map=label_map,
        segmentation=seg,
        endpoint_directions=None,
    )

    np.testing.assert_array_equal(repair_end[0], [10, 13, 10])


def test_repair_breaks_angle_weights_propagated():
    """Integration test: angle weights steer repair toward aligned target.

    A z-axis skeleton line with two disconnected targets:
    - Perpendicular target (along y), closer
    - Aligned target (along z), farther

    With high w_angle, the aligned candidate is selected.
    """
    skeleton = np.zeros((30, 30, 30), dtype=bool)
    skeleton[5:13, 12, 12] = True  # main line along z
    skeleton[12, 16, 12] = True  # perpendicular target (closer)
    skeleton[17, 12, 12] = True  # aligned target (farther)

    segmentation = np.ones((30, 30, 30), dtype=bool)

    # All three in different components
    label_map = np.zeros((30, 30, 30), dtype=np.int32)
    label_map[5:13, 12, 12] = 1
    label_map[12, 16, 12] = 2
    label_map[17, 12, 12] = 3

    repaired = repair_breaks(
        skeleton,
        segmentation,
        repair_radius=10.0,
        label_map=label_map,
        w_distance=0.1,
        w_angle=10.0,
    )

    # The z-aligned target should be connected
    assert repaired[17, 12, 12], "Aligned candidate should be connected"
    for z in range(13, 17):
        assert repaired[z, 12, 12], f"Repair line should include voxel at z={z}"
