import numpy as np
from numba.typed import List

from skeleplex.skeleton import find_break_repairs
from skeleplex.skeleton._break_fixing import (
    _flatten_candidates,
    _line_3d_numba,
    draw_lines,
    get_skeleton_data_cpu,
)


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
