"""Utilities for working with chunked arrays."""

from collections.abc import Callable

import dask.array as da
import numpy as np
import zarr
from tqdm import tqdm


def iteratively_process_chunks_3d(
    input_array: da.Array,
    output_zarr: zarr.Array,
    function_to_apply: Callable[[np.ndarray], np.ndarray],
    chunk_shape: tuple[int, int, int],
    extra_border: tuple[int, int, int],
    *args,
    **kwargs,
):
    """Apply a function to each chunk of a Dask array with extra border handling.

    no
    ----------
    input_array : dask.array.Array
        The input Dask array to process. Must be 3D.
    output_zarr : zarr.Array
        The output Zarr array to write results to.
        Must have the same shape as input_array.
    function_to_apply : Callable[[np.ndarray], np.ndarray]
        The function to apply to each chunk.
    chunk_shape : tuple[int, int, int]
        The shape of each chunk to process.
    extra_border : tuple[int, int, int]
        The extra border to include around each chunk.
    *args
        Additional positional arguments to pass to function_to_apply.
    **kwargs
        Additional keyword arguments to pass to function_to_apply.
    """
    # validate inputs before processing
    if input_array.ndim != 3:
        raise ValueError(f"Input array must be 3D, got {input_array.ndim}D")

    if len(chunk_shape) != 3:
        raise ValueError(
            f"chunk_shape must be a 3-tuple, got length {len(chunk_shape)}"
        )

    if len(extra_border) != 3:
        raise ValueError(
            f"extra_border must be a 3-tuple, got length {len(extra_border)}"
        )

    # calculate the chunk grid
    array_shape = input_array.shape
    n_chunks = tuple(int(np.ceil(array_shape[i] / chunk_shape[i])) for i in range(3))


    total_chunks = n_chunks[0] * n_chunks[1] * n_chunks[2]
    with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
        for i in range(n_chunks[0]):
            for j in range(n_chunks[1]):
                for k in range(n_chunks[2]):
                    pbar.update(1)

                    # calculate core chunk slice
                    core_start = (
                        i * chunk_shape[0],
                        j * chunk_shape[1],
                        k * chunk_shape[2],
                    )
                    core_end = (
                        min((i + 1) * chunk_shape[0], array_shape[0]),
                        min((j + 1) * chunk_shape[1], array_shape[1]),
                        min((k + 1) * chunk_shape[2], array_shape[2]),
                    )
                    core_slice = tuple(
                        slice(core_start[dim], core_end[dim]) for dim in range(3)
                    )

                    # calculate expanded slice (chunk + border)
                    # clipped to array boundaries
                    expanded_start = tuple(
                        max(0, core_start[dim] - extra_border[dim]) for dim in range(3)
                    )
                    expanded_end = tuple(
                        min(array_shape[dim], core_end[dim] + extra_border[dim])
                        for dim in range(3)
                    )
                    expanded_slice = tuple(
                        slice(expanded_start[dim], expanded_end[dim])
                        for dim in range(3)
                    )

                    # calculate actual border used (may be smaller at edges)
                    actual_border_before = tuple(
                        core_start[dim] - expanded_start[dim] for dim in range(3)
                    )

                    # extract chunk + border and compute
                    chunk_with_border = input_array[expanded_slice].compute()

                    # apply function
                    processed = function_to_apply(chunk_with_border, *args, **kwargs)

                    #extend slice to match output_array_shape array dimensions
                    core_in_result_slice = [
                        slice(
                            actual_border_before[dim],
                            actual_border_before[dim] +
                            (core_end[dim] - core_start[dim]),
                        )
                        for dim in range(3)
                    ]

                    # if the processed array has extra dims (e.g., channels/features),
                    # extend the slice with full slices for those dimensions
                    n_extra_dims = processed.ndim - 3
                    # dimensions beyond the first 3
                    if n_extra_dims > 0:

                        extra_slices = [
                            slice(0, processed.shape[dim_idx])
                            for dim_idx in range(n_extra_dims)
                        ]

                        #this is used to slice the processed array
                        core_in_result_slice =  extra_slices + core_in_result_slice
                        #this is used slice the output array into which we write
                        core_slice_extended = extra_slices + list(core_slice)
                    else:
                        #if no extra dims, just use the 3D slices
                        core_slice_extended = list(core_slice)

                    # convert back to tuple
                    core_in_result_slice = tuple(core_in_result_slice)
                    core_slice_extended = tuple(core_slice_extended)

                    #check if end dimensions match input
                    if processed.ndim != len(core_in_result_slice):
                        raise ValueError(
                            "The output of function_to_apply has "
                            "incompatible number of dimensions."
                        )

                    core_result = processed[core_in_result_slice]

                    # write to Zarr
                    output_zarr[core_slice_extended] = core_result
