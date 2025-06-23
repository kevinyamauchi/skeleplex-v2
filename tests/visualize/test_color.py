import numpy as np
from cmap import Color

from skeleplex.visualize import EdgeColormap


def test_edge_colormap():
    """Test mapping an array of edge colors."""
    colormap = EdgeColormap(
        colormap={
            (0, 1): Color([0.0, 0.0, 0.0, 1.0]),
            (3, 4): Color([1.0, 0.0, 0.0, 1.0]),
        },
        default_color=Color([0.0, 1.0, 0.0, 1.0]),
    )

    edge_key_array = np.array([[0, 1], [10, 12], [3, 4]])

    # make the colors
    mapped_colors = colormap.map_array(edge_key_array)

    # check the colors
    expected_colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [1, 0, 0, 1]])
    np.testing.assert_allclose(
        mapped_colors,
        expected_colors,
    )


def test_edge_colormap_from_arrays():
    """Test creating an EdgeColormap from arrays."""
    colormap = EdgeColormap.from_arrays(
        colormap={
            (0, 1): np.array([0.0, 0.0, 0.0, 1.0]),
            (3, 4): np.array([1.0, 0.0, 0.0, 1.0]),
        },
        default_color=np.array([0.0, 1.0, 0.0, 1.0]),
    )

    edge_key_array = np.array([[0, 1], [10, 12], [3, 4]])

    # make the colors
    mapped_colors = colormap.map_array(edge_key_array)

    # check the colors
    expected_colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [1, 0, 0, 1]])
    np.testing.assert_allclose(
        mapped_colors,
        expected_colors,
    )
