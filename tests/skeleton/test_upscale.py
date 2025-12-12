"""Pytest tests for upscale_skeleton function."""

import numpy as np
import pytest

from skeleplex.skeleton import upscale_skeleton


@pytest.mark.parametrize("skeleton_dtype", [np.bool_, np.uint8, np.int32])
def test_upscale_y_skeleton_with_boundary_node(skeleton_dtype):
    """Test upscaling a Y-shaped skeleton with a node on the boundary."""
    # Create a 20x20x20 skeleton with a Y shape
    skeleton = np.zeros((20, 20, 20), dtype=skeleton_dtype)

    # Create Y shape: vertical stem and two branches
    # Stem: from (10, 10, 10) to (10, 10, 15)
    skeleton[10, 10, 10:16] = 1

    # Left branch: from (10, 10, 15) to (6, 6, 19)
    skeleton[10, 10, 15] = 1
    skeleton[9, 9, 16] = 1
    skeleton[8, 8, 17] = 1
    skeleton[7, 7, 18] = 1
    skeleton[6, 6, 19] = 1  # Node on boundary (z=19)

    # Right branch: from (10, 10, 15) to (14, 14, 19)
    skeleton[11, 11, 16] = 1
    skeleton[12, 12, 17] = 1
    skeleton[13, 13, 18] = 1
    skeleton[14, 14, 19] = 1  # Node on boundary (z=19)

    # Upscale by factor of (2, 2, 2)
    upscaled = upscale_skeleton(skeleton, (2, 2, 2))

    # make the expected upscaled skeleton
    expected_upscaled = np.zeros((40, 40, 40), dtype=bool)

    # stem nodes (original scaled points)
    expected_upscaled[20, 20, 20] = True  # (10, 10, 10) * 2
    expected_upscaled[20, 20, 22] = True  # (10, 10, 11) * 2
    expected_upscaled[20, 20, 24] = True  # (10, 10, 12) * 2
    expected_upscaled[20, 20, 26] = True  # (10, 10, 13) * 2
    expected_upscaled[20, 20, 28] = True  # (10, 10, 14) * 2
    expected_upscaled[20, 20, 30] = True  # (10, 10, 15) * 2 - branch point

    # stem interpolated points (between consecutive nodes)
    expected_upscaled[20, 20, 21] = True  # between z=20 and z=22
    expected_upscaled[20, 20, 23] = True  # between z=22 and z=24
    expected_upscaled[20, 20, 25] = True  # between z=24 and z=26
    expected_upscaled[20, 20, 27] = True  # between z=26 and z=28
    expected_upscaled[20, 20, 29] = True  # between z=28 and z=30

    # left branch nodes (original scaled points)
    expected_upscaled[18, 18, 32] = True  # (9, 9, 16) * 2
    expected_upscaled[16, 16, 34] = True  # (8, 8, 17) * 2
    expected_upscaled[14, 14, 36] = True  # (7, 7, 18) * 2
    expected_upscaled[12, 12, 38] = True  # (6, 6, 19) * 2 - boundary node

    # left branch interpolated points
    expected_upscaled[19, 19, 31] = True  # between (20,20,30) and (18,18,32)
    expected_upscaled[17, 17, 33] = True  # between (18,18,32) and (16,16,34)
    expected_upscaled[15, 15, 35] = True  # between (16,16,34) and (14,14,36)
    expected_upscaled[13, 13, 37] = True  # between (14,14,36) and (12,12,38)

    # right branch nodes (original scaled points)
    expected_upscaled[22, 22, 32] = True  # (11, 11, 16) * 2
    expected_upscaled[24, 24, 34] = True  # (12, 12, 17) * 2
    expected_upscaled[26, 26, 36] = True  # (13, 13, 18) * 2
    expected_upscaled[28, 28, 38] = True  # (14, 14, 19) * 2 - boundary node

    # right branch interpolated points
    expected_upscaled[21, 21, 31] = True  # between (20,20,30) and (22,22,32)
    expected_upscaled[23, 23, 33] = True  # between (22,22,32) and (24,24,34)
    expected_upscaled[25, 25, 35] = True  # between (24,24,34) and (26,26,36)
    expected_upscaled[27, 27, 37] = True  # between (26,26,36) and (28,28,38)

    np.testing.assert_array_equal(upscaled, expected_upscaled)


def test_upscale_skeleton_raises_error_for_non_integer_scales():
    """Test that upscale_skeleton raises ValueError for non-integer scale factors."""
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[5, 5, 5] = True

    with pytest.raises(ValueError, match="must be integers"):
        upscale_skeleton(skeleton, (2.5, 2, 2))


def test_upscale_skeleton_raises_error_for_negative_scales():
    """Test that upscale_skeleton raises ValueError for negative scale factors."""
    skeleton = np.zeros((10, 10, 10), dtype=bool)
    skeleton[5, 5, 5] = True

    with pytest.raises(ValueError, match="must be positive"):
        upscale_skeleton(skeleton, (-2, 2, 2))


def test_upscale_skeleton_raises_error_for_non_3d_input():
    """Test that upscale_skeleton raises ValueError for non-3D input."""
    skeleton_2d = np.zeros((10, 10), dtype=bool)
    skeleton_2d[5, 5] = True

    with pytest.raises(ValueError, match="must be 3D"):
        upscale_skeleton(skeleton_2d, (2, 2, 2))
