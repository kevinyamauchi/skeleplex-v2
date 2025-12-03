import numpy as np
from numba.typed import List

from skeleplex.skeleton import find_break_repairs
from skeleplex.skeleton._break_fixing import (
    _flatten_candidates,
    _line_3d_numba,
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
