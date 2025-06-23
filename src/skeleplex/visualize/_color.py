from dataclasses import dataclass

import numpy as np
from cmap import Color
from typing_extensions import Self


@dataclass
class EdgeColormap:
    """A colormap for edges in a graph.

    Parameters
    ----------
    colormap : dict[tuple[int, int], Color]
        The colormap mapping edge keys (tuples of integers) to Color objects.
        For information on how to specify the colors, see the cmap docs:
        https://cmap-docs.readthedocs.io/en/stable/colors/
    default_color : Color
        The color to use for edges not in the colormap.

    Attributes
    ----------
    colormap : dict[tuple[int, int], Color]
        A dictionary mapping edge keys (tuples of integers) to Color objects.
    default_color : Color
        The default color to use for edges not in the colormap.
    """

    colormap: dict[tuple[int, int], Color]
    default_color: Color

    def map_edge(self, edge_key: tuple[int, int]) -> tuple[float, float, float, float]:
        """Map an edge key to an RGBA color.

        Parameters
        ----------
        edge_key : tuple[int, int]
            The edge key as a tuple of two integers.

        Returns
        -------
        tuple[float, float, float, float]
            The RGBA color.
        """
        if not isinstance(edge_key, tuple):
            # coerce to tuple
            edge_key = (int(edge_key[0]), int(edge_key[1]))
        color = self.colormap.get(edge_key, self.default_color)
        return color.rgba

    def map_array(self, edge_key_array: np.ndarray) -> np.ndarray:
        """Map an array of edge keys to an RGBA array.

        Parameters
        ----------
        edge_key_array : np.ndarray
            (n_edges, 2) array of edge keys.

        Returns
        -------
        np.ndarray
            (n_edges, 4) array of RGBA colors.
        """
        return np.array(
            [self.map_edge(edge_key) for edge_key in edge_key_array],
        )

    @classmethod
    def from_arrays(
        cls, colormap: dict[tuple[int, int], np.ndarray], default_color: np.ndarray
    ) -> Self:
        """Create an EdgeColormap from arrays.

        Parameters
        ----------
        colormap : dict[tuple[int, int], np.ndarray]
            A dictionary mapping edge keys (tuples of integers) to RGBA color arrays.
            For information on how to specify the color as an RGBA, see the cmap docs:
            https://cmap-docs.readthedocs.io/en/stable/colors/
        default_color : np.ndarray
            The default RGBA color as a numpy array.
            This is the color used when an edge key is not found in the colormap.

        Returns
        -------
        EdgeColormap
            An instance of EdgeColormap.
        """
        return cls(
            colormap={k: Color(v) for k, v in colormap.items()},
            default_color=Color(default_color),
        )
