"""Utilities for visualizing splines."""

import numpy as np

from skeleplex.graph.spline import B3Spline


def line_segment_coordinates_from_spline(
    spline: B3Spline,
    n_line_segments: int = 3,
) -> np.ndarray:
    """Get the coordinates for a line segment from a spline."""
    n_knots = len(spline.model.knots)
    t = np.linspace(0, n_knots - 1, n_line_segments + 1, endpoint=True)

    spline_coordinates = spline.model.eval(t)

    line_coordinates = np.empty((2 * n_line_segments, 3))
    line_coordinates[::2] = spline_coordinates[:-1]
    line_coordinates[1::2] = spline_coordinates[1:]

    return line_coordinates
