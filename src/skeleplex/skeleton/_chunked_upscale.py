from functools import partial
from multiprocessing import get_context
from typing import Literal

import zarr

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
