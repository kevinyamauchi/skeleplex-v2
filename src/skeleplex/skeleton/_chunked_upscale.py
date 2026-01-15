from functools import partial
from multiprocessing import get_context
from typing import Literal

import numpy as np
import zarr
from scipy.ndimage import map_coordinates

from skeleplex.skeleton._chunked_label import create_chunk_slices
from skeleplex.skeleton._upscale import upscale_skeleton
from skeleplex.utils import calculate_expanded_slice


def _upscale_skeleton_chunk(
    chunk_slice: tuple[slice, ...],
    input_path: str,
    output_path: str,
    border_size: tuple[int, int, int],
    scale_factors: tuple[int, int, int],
    input_array_shape: tuple[int, int, int],
) -> None:
    """Process a single chunk with skeleton upscaling.

    This function:
    1. Loads an expanded chunk (core chunk + border) from input zarr
    2. Applies skeleton upscaling using upscale_skeleton()
    3. Extracts only the core region (excluding border) from upscaled result
    4. Writes core result to output zarr array at scaled coordinates

    The border is used to ensure skeleton connectivity that spans chunk
    boundaries is properly preserved, but the border itself is not written
    to output.

    Parameters
    ----------
    chunk_slice : tuple of slice
        Slice objects defining the core chunk region to process (in input space).
    input_path : str
        Path to input zarr array.
    output_path : str
        Path to output zarr array for upscaled skeleton.
    border_size : tuple[int, int, int]
        Size of border to add around chunk in voxels (z, y, x) in input space.
    scale_factors : tuple[int, int, int]
        Integer scaling factors for each dimension (z, y, x).
    input_array_shape : tuple[int, int, int]
        Shape of the input array (z, y, x).
    """
    # Open zarr arrays
    input_zarr = zarr.open(input_path, mode="r")
    output_zarr = zarr.open(output_path, mode="r+")

    # Calculate expanded slice and actual border size
    expanded_slice, actual_border_before = calculate_expanded_slice(
        chunk_slice, border_size, input_array_shape
    )

    # Load expanded chunk from input
    chunk_data = input_zarr[expanded_slice]

    # Apply skeleton upscaling to expanded chunk
    upscaled_chunk = upscale_skeleton(chunk_data, scale_factors)

    # Calculate core region size in input space
    core_size_input = tuple(
        chunk_slice[i].stop - chunk_slice[i].start for i in range(3)
    )

    # Calculate core region size in scaled/output space
    core_size_scaled = tuple(core_size_input[i] * scale_factors[i] for i in range(3))

    # Calculate slice to extract core from upscaled chunk
    # Start position = actual_border_before * scale_factors
    # End position = start + core_size_scaled
    core_slice_in_upscaled = tuple(
        slice(
            actual_border_before[i] * scale_factors[i],
            actual_border_before[i] * scale_factors[i] + core_size_scaled[i],
        )
        for i in range(3)
    )

    # Extract core region from upscaled chunk
    core_upscaled = upscaled_chunk[core_slice_in_upscaled]

    # Calculate output slice (scale the original chunk_slice coordinates)
    output_slice = tuple(
        slice(
            chunk_slice[i].start * scale_factors[i],
            chunk_slice[i].stop * scale_factors[i],
        )
        for i in range(3)
    )

    # Write core result to output zarr array
    output_zarr[output_slice] = core_upscaled


def upscale_skeleton_parallel(
    input_path: str,
    output_path: str,
    scale_factors: tuple[int, int, int],
    n_processing_chunks: tuple[int, int, int],
    border_size: tuple[int, int, int],
    n_processes: int,
    pool_type: Literal["spawn", "fork"],
) -> None:
    """Upscale a skeleton image in parallel chunks across multiple processes.

    This function processes a skeleton zarr image in parallel chunks across
    multiple CPU processes, applying skeleton upscaling to each chunk. The
    border around each chunk ensures skeleton connectivity that spans chunk
    boundaries is properly preserved during upscaling.

    Processing chunks are defined as multiples of the zarr file chunks to
    ensure chunk boundaries align for safe parallel writing. The output zarr
    uses the same chunk structure as the input (before scaling).

    Parameters
    ----------
    input_path : str
        Path to input zarr array (binary skeleton image).
    output_path : str
        Path to output zarr array for upscaled skeleton (will be created).
    scale_factors : tuple[int, int, int]
        Integer scaling factors for each dimension (z, y, x). Must be positive
        integers.
    n_processing_chunks : tuple[int, int, int]
        Number of zarr file chunks to process together along each axis (z, y, x).
        Processing chunk size = zarr_chunk_size * n_processing_chunks.
        Must result in processing chunks that are multiples of zarr chunks.
    border_size : tuple[int, int, int]
        Size of border to add around each chunk in voxels (z, y, x) in input
        space. Should be large enough to capture skeleton connectivity that
        might span chunk boundaries. Used to prevent incomplete upscaling at
        chunk edges but not written to output.
    n_processes : int
        Number of parallel processes to use.
    pool_type : {"spawn", "fork"}
        Type of multiprocessing context to use.
        - "spawn": Start fresh Python process (safest, works on all platforms)
        - "fork": Copy parent process (faster but can have issues with threads)

    Raises
    ------
    ValueError
        If processing chunks don't align with zarr chunks, if border size is
        too large, or if scale factors are invalid.
    """
    # Open input zarr to get metadata
    input_zarr = zarr.open(input_path, mode="r")
    input_shape = input_zarr.shape
    zarr_chunks = input_zarr.chunks
    dtype = input_zarr.dtype

    # Calculate processing chunk size (in input space)
    processing_chunk_size = tuple(
        zarr_chunks[i] * n_processing_chunks[i] for i in range(3)
    )

    # Validate that processing chunks align with zarr chunks
    for i in range(3):
        if processing_chunk_size[i] % zarr_chunks[i] != 0:
            raise ValueError(
                f"Processing chunk size {processing_chunk_size[i]} must be a "
                f"multiple of zarr chunk size {zarr_chunks[i]} along axis {i}"
            )

    # Validate border size is smaller than processing chunk size
    for i in range(3):
        if border_size[i] >= processing_chunk_size[i]:
            raise ValueError(
                f"Border size {border_size[i]} must be smaller than processing "
                f"chunk size {processing_chunk_size[i]} along axis {i}"
            )

    # Calculate output shape
    output_shape = tuple(input_shape[i] * scale_factors[i] for i in range(3))

    # Create output zarr array (same chunk structure as input)
    _ = zarr.open(
        output_path,
        mode="w",
        shape=output_shape,
        chunks=zarr_chunks,  # Same as input chunks
        dtype=dtype,
    )

    # Create list of chunk slices (in input space)
    chunk_slices_list = create_chunk_slices(input_shape, processing_chunk_size)

    print(
        f"Processing {len(chunk_slices_list)} chunks of size "
        f"{processing_chunk_size} using {n_processes} {pool_type} workers"
    )

    # Create multiprocessing pool
    ctx = get_context(pool_type)
    pool = ctx.Pool(n_processes)

    # Create the processing function with fixed arguments
    process_func = partial(
        _upscale_skeleton_chunk,
        input_path=input_path,
        output_path=output_path,
        border_size=border_size,
        scale_factors=scale_factors,
        input_array_shape=input_shape,
    )

    try:
        # Process all chunks in parallel
        pool.map(process_func, chunk_slices_list)
    finally:
        # Cleanup pool
        pool.close()
        pool.join()

    print("Skeleton upscaling complete")


def _upscale_skeleton_chunk_to_shape(
    output_chunk_slice: tuple[slice, ...],
    input_path: str,
    output_path: str,
    border_size: tuple[int, int, int],
    target_scale_factors: tuple[float, float, float],
    intermediate_scale_factors: tuple[int, int, int],
    input_array_shape: tuple[int, int, int],
) -> None:
    """Process a single output chunk with skeleton upscaling to target shape.

    This function:
    1. Maps the output chunk coordinates to input space
    2. Loads an expanded input region (with border)
    3. Applies integer upscaling to an intermediate resolution
    4. Resamples to exact target coordinates using map_coordinates
    5. Writes the result to the output zarr at the output chunk coordinates

    The border ensures skeleton connectivity spanning chunk boundaries is
    preserved, but the border itself is not written to output.

    Parameters
    ----------
    output_chunk_slice : tuple of slice
        Slice objects defining the output chunk region to produce (in output
        space).
    input_path : str
        Path to input zarr array.
    output_path : str
        Path to output zarr array.
    border_size : tuple[int, int, int]
        Size of border to add around input region in voxels (z, y, x) in
        input space.
    target_scale_factors : tuple[float, float, float]
        True scale factors to reach target shape (z, y, x). Can be non-integer.
    intermediate_scale_factors : tuple[int, int, int]
        Integer scale factors for intermediate upscaling (z, y, x).
        Must be >= target_scale_factors for each dimension.
    input_array_shape : tuple[int, int, int]
        Shape of the input array (z, y, x).
    """
    # Open zarr arrays
    input_zarr = zarr.open(input_path, mode="r")
    output_zarr = zarr.open(output_path, mode="r+")

    # Calculate which input region is needed for this output chunk
    # Map output coordinates to input coordinates
    input_start = tuple(
        int(np.floor(output_chunk_slice[i].start / target_scale_factors[i]))
        for i in range(3)
    )
    input_stop = tuple(
        int(np.ceil(output_chunk_slice[i].stop / target_scale_factors[i]))
        for i in range(3)
    )

    # Create input chunk slice (without border)
    input_chunk_slice = tuple(slice(input_start[i], input_stop[i]) for i in range(3))

    # Calculate expanded slice and actual border size
    expanded_slice, actual_border_before = calculate_expanded_slice(
        input_chunk_slice, border_size, input_array_shape
    )

    # Load expanded chunk from input
    input_data = input_zarr[expanded_slice]

    # Apply integer upscaling to intermediate resolution
    upscaled_intermediate = upscale_skeleton(input_data, intermediate_scale_factors)

    # Calculate the size of the output chunk
    output_chunk_size = tuple(
        output_chunk_slice[i].stop - output_chunk_slice[i].start for i in range(3)
    )

    # Create coordinate grid for the output chunk in output space
    output_coords_z = np.arange(output_chunk_size[0]) + output_chunk_slice[0].start
    output_coords_y = np.arange(output_chunk_size[1]) + output_chunk_slice[1].start
    output_coords_x = np.arange(output_chunk_size[2]) + output_chunk_slice[2].start

    # Map output coordinates to input space
    input_coords_z = output_coords_z / target_scale_factors[0]
    input_coords_y = output_coords_y / target_scale_factors[1]
    input_coords_x = output_coords_x / target_scale_factors[2]

    # Calculate position in expanded input space (before upscaling)
    expanded_input_start = tuple(expanded_slice[i].start for i in range(3))

    # Adjust to be relative to the expanded region
    input_coords_z_relative = input_coords_z - expanded_input_start[0]
    input_coords_y_relative = input_coords_y - expanded_input_start[1]
    input_coords_x_relative = input_coords_x - expanded_input_start[2]

    # Map to intermediate (upscaled) space
    intermediate_coords_z = input_coords_z_relative * intermediate_scale_factors[0]
    intermediate_coords_y = input_coords_y_relative * intermediate_scale_factors[1]
    intermediate_coords_x = input_coords_x_relative * intermediate_scale_factors[2]

    # Create meshgrid for map_coordinates
    intermediate_grid_z, intermediate_grid_y, intermediate_grid_x = np.meshgrid(
        intermediate_coords_z,
        intermediate_coords_y,
        intermediate_coords_x,
        indexing="ij",
    )

    # Stack coordinates for map_coordinates (needs shape (3, *output_chunk_size))
    coordinates = np.stack(
        [intermediate_grid_z, intermediate_grid_y, intermediate_grid_x], axis=0
    )

    # Resample to exact target coordinates using nearest neighbor
    output_chunk_data = map_coordinates(
        upscaled_intermediate.astype(np.float32),
        coordinates,
        order=0,  # Nearest neighbor for binary data
        mode="nearest",
        prefilter=False,
    ).astype(bool)

    # Write to output zarr
    output_zarr[output_chunk_slice] = output_chunk_data


def upscale_skeleton_to_shape_parallel(
    input_path: str,
    output_path: str,
    target_shape: tuple[int, int, int],
    output_chunk_size: tuple[int, int, int],
    border_size: tuple[int, int, int],
    n_processes: int,
    pool_type: Literal["spawn", "fork"],
) -> None:
    """Upscale a skeleton image to a target shape in parallel chunks.

    This function upscales a skeleton zarr image to a specified target shape,
    processing in parallel chunks across multiple CPU processes. It supports
    non-integer and non-uniform scale factors by using a two-step approach:
    first applying integer upscaling to preserve topology, then resampling to
    exact target coordinates.

    Processing is organized around output chunks - each worker produces one
    complete output chunk by reading the necessary input region (which may
    span multiple input chunks), upscaling it, and extracting the exact output
    chunk. This ensures safe parallel writes since each worker writes to a
    non-overlapping output chunk.

    Parameters
    ----------
    input_path : str
        Path to input zarr array (binary skeleton image).
    output_path : str
        Path to output zarr array for upscaled skeleton (will be created).
    target_shape : tuple[int, int, int]
        Desired output shape (z, y, x). Can result in non-integer scale factors.
    output_chunk_size : tuple[int, int, int]
        Zarr chunk size for output array (z, y, x).
    border_size : tuple[int, int, int]
        Size of border to add around input regions in voxels (z, y, x) in
        input space. Should be large enough to capture skeleton connectivity
        that might span chunk boundaries. Used to prevent incomplete upscaling
        but not written to output.
    n_processes : int
        Number of parallel processes to use.
    pool_type : {"spawn", "fork"}
        Type of multiprocessing context to use.
        - "spawn": Start fresh Python process (safest, works on all platforms)
        - "fork": Copy parent process (faster but can have issues with threads)

    Raises
    ------
    ValueError
        If target_shape results in scale factors < 1 (downsampling not
        supported), or if border size is too large.

    Notes
    -----
    Scale factors are computed as target_shape / input_shape. When these are
    non-integer, the function uses ceiling of scale factors for intermediate
    integer upscaling, then resamples to exact target coordinates using
    nearest neighbor interpolation.

    The output zarr array is organized by output_chunk_size, which is
    independent of input chunk structure. Each output chunk is processed
    independently and can be written in parallel.
    """
    # Open input zarr to get metadata
    input_zarr = zarr.open(input_path, mode="r")
    input_shape = input_zarr.shape
    dtype = input_zarr.dtype

    # Validate input is 3D
    if len(input_shape) != 3:
        raise ValueError(f"Input array must be 3D, got {len(input_shape)}D")

    if len(target_shape) != 3:
        raise ValueError(f"target_shape must be 3D, got {len(target_shape)}D")

    if len(output_chunk_size) != 3:
        raise ValueError(f"output_chunk_size must be 3D, got {len(output_chunk_size)}D")

    if len(border_size) != 3:
        raise ValueError(f"border_size must be 3D, got {len(border_size)}D")

    # Calculate true scale factors (can be non-integer)
    target_scale_factors = tuple(target_shape[i] / input_shape[i] for i in range(3))

    # Validate scale factors are >= 1 (no downsampling)
    for i in range(3):
        if target_scale_factors[i] < 1.0:
            raise ValueError(
                f"Scale factor {target_scale_factors[i]} < 1 along axis {i}. "
                f"Downsampling not supported. "
                f"Input shape: {input_shape}, Target shape: {target_shape}"
            )

    # Calculate intermediate integer scale factors (round up)
    intermediate_scale_factors = tuple(
        int(np.ceil(target_scale_factors[i])) for i in range(3)
    )

    # Validate border size is reasonable
    # Border in intermediate space should not exceed output chunk size
    max_border_intermediate = tuple(
        border_size[i] * intermediate_scale_factors[i] for i in range(3)
    )
    for i in range(3):
        if max_border_intermediate[i] >= output_chunk_size[i]:
            raise ValueError(
                f"Border size {border_size[i]} when scaled by intermediate "
                f"factor {intermediate_scale_factors[i]} = "
                f"{max_border_intermediate[i]} must be smaller than output "
                f"chunk size {output_chunk_size[i]} along axis {i}"
            )

    # Create output zarr array
    _ = zarr.open(
        output_path,
        mode="w",
        shape=target_shape,
        chunks=output_chunk_size,
        dtype=dtype,
    )

    # Create list of output chunk slices
    output_chunk_slices_list = create_chunk_slices(target_shape, output_chunk_size)

    print(
        f"Upscaling from {input_shape} to {target_shape} "
        f"(scale factors: {target_scale_factors})"
    )
    print(f"Using intermediate scale factors: {intermediate_scale_factors}")
    print(
        f"Processing {len(output_chunk_slices_list)} output chunks of size "
        f"{output_chunk_size} using {n_processes} {pool_type} workers"
    )

    # Create multiprocessing pool
    ctx = get_context(pool_type)
    pool = ctx.Pool(n_processes)

    # Create the processing function with fixed arguments
    process_func = partial(
        _upscale_skeleton_chunk_to_shape,
        input_path=input_path,
        output_path=output_path,
        border_size=border_size,
        target_scale_factors=target_scale_factors,
        intermediate_scale_factors=intermediate_scale_factors,
        input_array_shape=input_shape,
    )

    try:
        # Process all output chunks in parallel
        pool.map(process_func, output_chunk_slices_list)
    finally:
        # Cleanup pool
        pool.close()
        pool.join()

    print("Skeleton upscaling complete")
