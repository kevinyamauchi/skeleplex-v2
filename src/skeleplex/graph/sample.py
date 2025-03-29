"""Functions for sampling images using the SkeletonGraph."""

import einops
import numpy as np
from scipy.ndimage import map_coordinates


def generate_3d_grid(
    grid_shape: tuple[int, int, int] = (10, 10, 10),
    grid_spacing: tuple[float, float, float] = (1, 1, 1),
) -> np.ndarray:
    """
    Generate a 3D sampling grid with specified shape and spacing.

    The grid generated is centered on the origin, has shape (w, h, d, 3) for
    grid_shape (w, h, d), and spacing grid_spacing between neighboring points.

    Parameters
    ----------
    grid_shape : Tuple[int, int, int]
        The number of grid points along each axis.
    grid_spacing : Tuple[float, float, float]
        Spacing between points in the sampling grid.

    Returns
    -------
    np.ndarray
        Coordinate of points forming the 3D grid.
    """
    # generate a grid of points at each integer from 0 to grid_shape for each dimension
    grid = np.indices(grid_shape).astype(float)
    grid = einops.rearrange(grid, "xyz w h d -> w h d xyz")
    # shift the grid to be centered on the origin
    grid_offset = (np.array(grid_shape)) // 2
    grid -= grid_offset
    # scale the grid to get correct spacing
    grid *= grid_spacing
    return grid


def generate_2d_grid(
    grid_shape: tuple[int, int] = (10, 10), grid_spacing: tuple[float, float] = (1, 1)
) -> np.ndarray:
    """
    Generate a 2D sampling grid with specified shape and spacing.

    The grid generated is centered on the origin, lying on the plane with normal
    vector [1, 0, 0], has shape (w, h, 3) for grid_shape (w, h), and spacing
    grid_spacing between neighboring points.

    Parameters
    ----------
    grid_shape : Tuple[int, int]
        The number of grid points along each axis.
    grid_spacing : Tuple[float, float]
        Spacing between points in the sampling grid.

    Returns
    -------
    np.ndarray
        Coordinate of points forming the 2D grid.
    """
    grid = generate_3d_grid(
        grid_shape=(1, *grid_shape), grid_spacing=(1, *grid_spacing)
    )
    return einops.rearrange(grid, "1 w h xyz -> w h xyz")


def sample_volume_at_coordinates(
    volume: np.ndarray,
    coordinates: np.ndarray,
    interpolation_order: int = 3,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Sample a volume with spline interpolation at specific coordinates.

    The output shape is determined by the input coordinate shape such that
    if coordinates have shape (batch, *grid_shape, 3), the output array will have
    shape (*grid_shape, batch).

    Parameters
    ----------
    volume : np.ndarray
        Volume to be sampled.
    coordinates : np.ndarray
        Array of coordinates at which to sample the volume. The shape of this array
        should be (batch, *grid_shape, 3) to allow reshaping back correctly
    interpolation_order : int
        Spline order for image interpolation.
    fill_value : float
        Value to fill in for sample coordinates past the edges of the volume.

    Returns
    -------
    np.ndarray
        Array of shape (*grid_shape)
    """
    batch, *grid_shape, _ = coordinates.shape
    # map_coordinates wants transposed coordinate array
    sampled_volume = map_coordinates(
        volume,
        coordinates.reshape(-1, 3).T,
        order=interpolation_order,
        cval=fill_value,
    )

    # reshape back (need to invert due to previous transposition)
    sampled_volume = sampled_volume.reshape(*grid_shape, batch)
    # and retranspose to get batch back to the 0th dimension
    return einops.rearrange(sampled_volume, "... batch -> batch ...")


def sample_volume_at_coordinates_lazy(
    volume: np.ndarray,
    coordinates: np.ndarray,
    interpolation_order: int = 3,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Sample a volume with spline interpolation at specific coordinates.

    This function is designed to work with large volumes that are too big to fit
    into memory. It extracts only the necessary chunk of the volume and samples
    that chunk.
    The output shape is determined by the input coordinate shape such that
    if coordinates have shape (batch, *grid_shape, 3), the output array will have
    shape (*grid_shape, batch).

    Parameters
    ----------
    volume : np.ndarray
        Volume to be sampled.
    coordinates : np.ndarray
        Array of coordinates at which to sample the volume. The shape of this array
        should be (batch, *grid_shape, 3) to allow reshaping back correctly
    interpolation_order : int
        Spline order for image interpolation.
    fill_value : float
        Value to fill in for sample coordinates past the edges of the volume.

    Returns
    -------
    np.ndarray
        Array of shape (*grid_shape)
    """
    batch, *grid_shape, _ = coordinates.shape

    min_coords = np.floor(coordinates.min(axis=(0, 1, 2))).astype(int)
    max_coords = np.ceil(coordinates.max(axis=(0, 1, 2))).astype(int)

    # Extract only the necessary chunk (use slicing)
    sliced_volume = volume[
        min_coords[0] : max_coords[0],
        min_coords[1] : max_coords[1],
        min_coords[2] : max_coords[2],
    ].compute()  # Convert to NumPy only for this chunk

    # Adjust coordinates relative to the extracted chunk
    coordinates -= min_coords

    # Sample the smaller volume using `map_coordinates`
    sampled_volume = map_coordinates(
        sliced_volume,
        coordinates.reshape(-1, 3).T,
        order=interpolation_order,
        cval=fill_value,
    )

    # Reshape back to (*grid_shape, batch)
    sampled_volume = sampled_volume.reshape(*grid_shape, batch)

    return einops.rearrange(sampled_volume, "... batch -> batch ...")
