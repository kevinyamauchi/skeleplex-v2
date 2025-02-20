"""Utilities for fitting and working with splines."""

import json

import numpy as np
import splinebox
from scipy.spatial.transform import Rotation
from splinebox.spline_curves import _prepared_dict_for_constructor

from skeleplex.graph.sample import generate_2d_grid, sample_volume_at_coordinates


class B3Spline:
    """Model for a B3 spline.

    Parameters
    ----------
    model : Spline
        The spline model.
    """

    _backend = "splinebox"

    def __init__(self, model: splinebox.Spline):
        self._model = model

    @property
    def model(self) -> splinebox.Spline:
        """Return the underlying spline model."""
        return self._model

    @property
    def arc_length(self) -> float:
        """Return the arc length of the spline."""
        return self.model.arc_length()

    def eval(
        self, positions: np.ndarray, derivative: int = 0, atol: float = 1e-6
    ) -> np.ndarray:
        """Evaluate the spline at a set of positions.

        Parameters
        ----------
        positions : np.ndarray
            (n,) array of positions to evaluate the spline at.
            The positions are normalized to the range [0, 1].
        derivative : int
            The order of the derivative to evaluate.
            Default value is 0.
        atol : float
            The absolute tolerance for converting the normalized
            evaluation positions to positions along the spline.
            Default value is 1e-6.
        """
        # convert the normalized arc length coordinates to t
        positions_t = self.model.arc_length_to_parameter(
            positions * self.arc_length, atol=atol
        )
        # For single values, splinebox's eval expects a float
        # This recasts the value to a float if positions_t is a single value
        if positions_t.ndim == 0:
            positions_t = positions_t.tolist()
        return self.model.eval(positions_t, derivative=derivative)

    def moving_frame(
        self, positions: np.ndarray, method: str = "bishop", atol: float = 1e-6
    ):
        """Generate a moving frame long the spline at specified positions.

        Parameters
        ----------
        positions : np.ndarray
            (n,) array of positions to evaluate the spline at.
            The positions are normalized to the range [0, 1].
        method : str
            The method to use for generating the moving frame.
            Default value is "bishop".
        atol : float
            The absolute tolerance for converting the normalized
            evaluation positions to positions along the spline.
            Default value is 1e-6.
        """
        # convert the normalized arc length coordinates to t
        positions_t = self.model.arc_length_to_parameter(
            positions * self.arc_length, atol=atol
        )
        return self.model.moving_frame(positions_t, method=method)

    def sample_volume_2d(
        self,
        volume: np.ndarray,
        positions: np.ndarray,
        grid_shape: tuple[int, int] = (10, 10),
        grid_spacing: tuple[float, float] = (1, 1),
        moving_frame_method: str = "bishop",
        sample_interpolation_order: int = 3,
        sample_fill_value: float = np.nan,
    ):
        """Sample a 3D image with 2D planes normal to the spline at specified positions.

        Parameters
        ----------
        volume : np.ndarray
            3D image to sample.
        positions : np.ndarray
            (n,) array of positions to evaluate the spline at.
            The positions are normalized to the range [0, 1].
        grid_shape : tuple[int, int]
            The number of pixels along each axis of the resulting 2D image.
            Default value is (10, 10).
        grid_spacing : tuple[float, float]
            Spacing between points in the sampling grid.
            Default value is (1, 1).
        moving_frame_method : str
            The method to use for generating the moving frame.
            Default value is "bishop".
        sample_interpolation_order : int
            The order of the spline interpolation to use when sampling the image.
            Default value is 3.
        sample_fill_value : float
            The fill value to use when sampling the image outside
            the bounds of the array. Default value is np.nan.
        """
        moving_frame = self.moving_frame(
            positions=positions, method=moving_frame_method
        )

        # generate the grid of points for sampling the image
        # (shape (w, h, 3))
        sampling_grid = generate_2d_grid(
            grid_shape=grid_shape, grid_spacing=(grid_spacing, grid_spacing)
        )

        # reshape the sampling grid to be a list of coordinates
        grid_coords = sampling_grid.reshape(-1, 3)

        # apply each orientation to the grid for each position and store the result
        rotated = []
        for frame in moving_frame:
            rotation_matrix = np.column_stack([frame[0], frame[1], frame[2]])
            orientation = Rotation.from_matrix(rotation_matrix)
            rotated.append(orientation.apply(grid_coords))

        # get the coordinates of the points on the spline to center
        # the sampling grid for the 2D image.
        sample_centroid_coordinates = positions = self.eval(positions=positions)

        # shift the rotated points to be centered on the spline
        rotated_shifted = np.stack(rotated, axis=1) + sample_centroid_coordinates
        placed_sample_grids = rotated_shifted.reshape(-1, *sampling_grid.shape)
        return sample_volume_at_coordinates(
            volume=volume,
            coordinates=placed_sample_grids,
            interpolation_order=sample_interpolation_order,
            fill_value=sample_fill_value,
        )

    def __eq__(self, other_object) -> bool:
        """Check if two B3Spline objects are equal."""
        if not isinstance(other_object, B3Spline):
            return False
        return self.model == other_object.model

    def to_json_dict(self) -> dict:
        """Return a JSON serializable dictionary."""
        spline_model_dict = self.model._to_dict(version=2)
        if "__class__" in spline_model_dict:
            raise ValueError(
                "The Spline object to encode already has a '__class__' key."
            )
        spline_model_dict.update({"__class__": "splinebox.Spline"})
        return {
            "__class__": "skeleplex.B3Spline",
            "model": spline_model_dict,
            "backend": self._backend,
        }

    def to_json_file(self, file_path: str) -> None:
        """Save the spline to a JSON file."""
        with open(file_path, "w") as file:
            json.dump(self.to_json_dict(), file)

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> "B3Spline":
        """Return a B3Spline from a JSON serializable dictionary."""
        if json_dict["backend"] != cls._backend:
            raise ValueError(
                f"Expected backend {cls._backend}, got {json_dict['backend']}."
            )

        # load the spline model
        spline_model_dict = json_dict["model"]

        if isinstance(spline_model_dict, splinebox.Spline):
            # model has already been deserialized
            # this can happen if a this is being called
            # within another JSON decoder.
            return cls(model=spline_model_dict)

        spline_model_dict.pop("__class__")
        spline_kwargs = _prepared_dict_for_constructor(spline_model_dict)
        spline_model = splinebox.Spline(**spline_kwargs)

        # make the class
        return cls(model=spline_model)

    @classmethod
    def from_json_file(cls, file_path: str) -> "B3Spline":
        """Construct a B3Spline from a JSON file."""
        with open(file_path) as file:
            json_dict = json.load(file)
        return cls.from_json_dict(json_dict)

    @classmethod
    def from_points(cls, points: np.ndarray, n_knots: int = 4):
        """Construct a B3 spline fit to a list of points.

        Parameters
        ----------
        points : np.ndarray
            (n, d) array of points to fit the spline to.
            These must be ordered in the positive t direction
            of the spline.
        n_knots : int
            The number of knots to use in the spline.
        """
        basis_function = splinebox.B3()
        spline = splinebox.Spline(
            M=n_knots, basis_function=basis_function, closed=False
        )
        spline.fit(points)
        return cls(model=spline)

    def flip_spline(self, path: np.ndarray) -> "B3Spline":
        """Recomputes the spline inverse to the path.

        Parameters
        ----------
        path : np.ndarray
            The coordinates to fit the spline to.

        Returns
        -------
        B3Spline
            The flipped spline.
        np.ndarray
            The flipped path coordinates.
        """
        return self.from_points(path[::-1]), path[::-1]
