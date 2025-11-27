import sys  # noqa: D100
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import zarr


def get_chunking_dicts_with_overlap(
    input_array: da.Array,
    output_path: str,
    output_shape: tuple[int, ...],
    chunk_shape: tuple[int, int, int],
    extra_border: tuple[int, int, int],

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

    if len(extra_border) != 3:
        raise ValueError(
            f"extra_border must be a 3-tuple, got length {len(extra_border)}"
        )

    if len(chunk_shape) != 3:
        raise ValueError(
            f"chunk_shape must be a 3-tuple, got length {len(chunk_shape)}"
        )
    # calculate the chunk grid
    array_shape = input_array.shape


    n_chunks = tuple(int(np.ceil(array_shape[i] / chunk_shape[i])) for i in range(3))

    input_slices_dict = {}
    expanded_slices_dict = {}
    core_in_result_slices_dict = {}
    core_in_result_slices_extended_dict = {}
    chunk_counter =0
    for i in range(n_chunks[0]):
        for j in range(n_chunks[1]):
            for k in range(n_chunks[2]):

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
                n_extra_dims = len(output_shape) - 3
                # dimensions beyond the first 3
                if n_extra_dims > 0:

                    extra_slices = [
                        slice(0, output_shape[dim_idx])
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

                input_slices_dict[chunk_counter] = core_slice
                expanded_slices_dict[chunk_counter] = expanded_slice
                core_in_result_slices_dict[chunk_counter] = core_in_result_slice
                core_in_result_slices_extended_dict[chunk_counter] = core_slice_extended
                chunk_counter +=1

    job_dataframe = pd.DataFrame({
        'input_slices': input_slices_dict,
        'expanded_slices': expanded_slices_dict,
        'core_in_result_slices': core_in_result_slices_dict,
        'core_in_result_slices_extended': core_in_result_slices_extended_dict
    })

    job_dataframe['job_id'] = job_dataframe.index
    job_dataframe.to_csv(output_path)



if __name__ == "__main__":

    input_path = Path(sys.argv[1])
    df_path = Path(sys.argv[2])
    zarr_path = Path(sys.argv[3])
    chunks = (800, 800, 800)
    extra_border = (100, 100, 100)

    image = da.from_zarr(input_path)

    #if function adds extra dims, adapt output shape here!
    #This is critical for correct slicing later on
    output_shape = image.shape
    output_chunks = chunks

    get_chunking_dicts_with_overlap(
        input_array=image,
        output_path=df_path,
        output_shape=output_shape,
        chunk_shape=chunks,
        extra_border=extra_border,
    )

    #create zarr array
    if zarr_path.exists():
        print(f"Zarr path {zarr_path} already exists.")
    else:
        zarr_store = zarr.open(
            zarr_path,
            mode="w",
            shape=output_shape,
            chunks=output_chunks,
            dtype=image.dtype,
        )
        print(f"Created zarr store at {zarr_path}.")


