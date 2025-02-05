"""Tests for the skeleplex.visualize.spline module."""

import numpy as np

from skeleplex.graph.spline import B3Spline
from skeleplex.visualize.spline import line_segment_coordinates_from_spline


def test_line_segment_coordinates_from_spline():
    """Test getting line segment coordinates from a spline."""
    points_to_fit = np.array(
        [
            [0, 0, 0],
            [0, 0, 0.2],
            [0, 0, 0.4],
            [0, 0, 0.6],
            [0, 0, 0.8],
            [0, 0, 1.0],
        ]
    )
    spline = B3Spline.from_points(points=points_to_fit)

    n_line_segments = 4
    line_coordinates = line_segment_coordinates_from_spline(
        spline=spline, n_line_segments=n_line_segments
    )

    assert line_coordinates.shape == (2 * n_line_segments, 3)

    # check that the end points are correct
    np.testing.assert_allclose(line_coordinates[0], points_to_fit[0], atol=1e-6)
    np.testing.assert_allclose(line_coordinates[-1], points_to_fit[-1], atol=1e-6)

    # check that the coordinates are correctly interleaved to make line segments
    print(line_coordinates)
    for i in range(1, n_line_segments):
        np.testing.assert_allclose(
            line_coordinates[2 * i - 1], line_coordinates[2 * i], atol=1e-6
        )
