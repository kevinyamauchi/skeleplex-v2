"""Utilities for performing inference on large arrays.

These are intended to be used with zarr arrays on a SLURM cluster.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from skeleplex.utils._chunked import calculate_expanded_slice


def initialize_parallel_inference(
    input_zarr_path: Path,
    output_zarr_path: Path,
    chunk_shape: tuple[int, int, int],
    border_size: tuple[int, int, int],
    chunk_table_path: Path,
) -> int:
    """Initialize a parallel inference job for on a SLURM cluster.

    This function generates a chunk_table CSV file that describes all chunks
    to be processed during parallel inference. Each row in the CSV represents
    one chunk's core region (without border) along with the input and output
    zarr paths. The manifest is intended to be used with SLURM job arrays
    where each task processes one row.

    The function also creates the output zarr array with the same shape and
    chunk structure as the input.

    Parameters
    ----------
    input_zarr_path : Path
        Path to the input zarr array (must be 3D).
    output_zarr_path : Path
        Path where the output zarr array will be created.
    chunk_shape : tuple[int, int, int]
        Shape of each processing chunk in voxels (z, y, x).
        Must be a multiple of the input zarr chunk size for all axes.
    border_size : tuple[int, int, int]
        Size of border to add around each chunk in voxels (z, y, x).
        Used to prevent edge artifacts during inference.
    chunk_table_path : Path
        Path where the chunk_table CSV manifest will be written.

    Returns
    -------
    n_chunks : int
        Total number of chunks that will be processed.

    Raises
    ------
    ValueError
        If input array is not 3D, if chunk_shape is not a 3-tuple,
        if border_size is not a 3-tuple, or if chunk_shape is not
        a multiple of zarr chunk size.

    Notes
    -----
    The output CSV (chunk_table) has the following columns:
    - input_zarr_path: Path to the input zarr array
    - output_zarr_path: Path to the output zarr array
    - z_start, z_end: Start and end indices for z dimension (core region)
    - y_start, y_end: Start and end indices for y dimension (core region)
    - x_start, x_end: Start and end indices for x dimension (core region)
    - border_z, border_y, border_x: Border size for each dimension

    The row index (0-based) serves as the chunk identifier and should
    correspond to the SLURM_ARRAY_TASK_ID.

    The output zarr array is created with the same shape, chunks, and dtype
    as the input array. Chunks are aligned to ensure safe parallel writing.
    """
    # Open input zarr and validate
    input_zarr = zarr.open(input_zarr_path, mode="r")

    # Validate input is 3D
    if input_zarr.ndim != 3:
        raise ValueError(f"Input array must be 3D, got {input_zarr.ndim}D")

    # Validate chunk_shape is 3-tuple
    if len(chunk_shape) != 3:
        raise ValueError(
            f"chunk_shape must be a 3-tuple, got length {len(chunk_shape)}"
        )

    # Validate border_size is 3-tuple
    if len(border_size) != 3:
        raise ValueError(
            f"border_size must be a 3-tuple, got length {len(border_size)}"
        )

    # Get input metadata
    input_shape = input_zarr.shape
    input_chunks = input_zarr.chunks
    input_dtype = input_zarr.dtype

    # Validate that chunk_shape is a multiple of zarr chunk size
    for i, dim_name in enumerate(["z", "y", "x"]):
        if chunk_shape[i] % input_chunks[i] != 0:
            raise ValueError(
                f"Processing chunk size {chunk_shape[i]} must be a multiple of "
                f"zarr chunk size {input_chunks[i]} along {dim_name} axis (axis {i})"
            )

    # Create output zarr array with same shape, chunks, and dtype
    _ = zarr.open(
        output_zarr_path,
        mode="w",
        shape=input_shape,
        chunks=input_chunks,
        dtype=input_dtype,
    )

    # Calculate number of chunks along each dimension
    n_chunks_per_dim = tuple(
        int(np.ceil(input_shape[i] / chunk_shape[i])) for i in range(3)
    )

    # Generate chunk information
    chunk_records = []

    # Convert paths to strings for CSV storage
    input_path_str = str(input_zarr_path)
    output_path_str = str(output_zarr_path)

    for i in range(n_chunks_per_dim[0]):
        for j in range(n_chunks_per_dim[1]):
            for k in range(n_chunks_per_dim[2]):
                # Calculate core chunk boundaries
                z_start = i * chunk_shape[0]
                z_end = min((i + 1) * chunk_shape[0], input_shape[0])

                y_start = j * chunk_shape[1]
                y_end = min((j + 1) * chunk_shape[1], input_shape[1])

                x_start = k * chunk_shape[2]
                x_end = min((k + 1) * chunk_shape[2], input_shape[2])

                # Create record for this chunk
                chunk_records.append(
                    {
                        "input_zarr_path": input_path_str,
                        "output_zarr_path": output_path_str,
                        "z_start": z_start,
                        "z_end": z_end,
                        "y_start": y_start,
                        "y_end": y_end,
                        "x_start": x_start,
                        "x_end": x_end,
                        "border_z": border_size[0],
                        "border_y": border_size[1],
                        "border_x": border_size[2],
                    }
                )

    # Create DataFrame and write to CSV
    chunk_table = pd.DataFrame(chunk_records)
    chunk_table.to_csv(chunk_table_path, index=False)

    n_chunks = len(chunk_records)

    return n_chunks


def infer_on_chunk(
    input_zarr_path: str,
    output_zarr_path: str,
    inference_function: Callable[[np.ndarray], np.ndarray],
    z_start: int,
    z_end: int,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
    border_z: int,
    border_y: int,
    border_x: int,
) -> None:
    """Process a single chunk by running inference with a provided model.

    This function loads the expanded chunk (core + border) from the input zarr,
    runs inference using the provided model callable, extracts the core region
    from the result, and writes it to the output zarr.

    The border is used to prevent edge artifacts during inference but is
    not written to the output.

    Parameters
    ----------
    input_zarr_path : str
        Path to the input zarr array.
    output_zarr_path : str
        Path to the output zarr array (must already exist).
    inference_function : Callable[[np.ndarray], np.ndarray]
        A callable that takes a 3D numpy array (the expanded chunk with border)
        and returns a 3D numpy array of predictions. The model should be
        pre-loaded and ready for inference (e.g., already moved to GPU if needed).
    z_start : int
        Start index for z dimension (core region).
    z_end : int
        End index for z dimension (core region).
    y_start : int
        Start index for y dimension (core region).
    y_end : int
        End index for y dimension (core region).
    x_start : int
        Start index for x dimension (core region).
    x_end : int
        End index for x dimension (core region).
    border_z : int
        Border size for z dimension in voxels.
    border_y : int
        Border size for y dimension in voxels.
    border_x : int
        Border size for x dimension in voxels.

    Raises
    ------
    ValueError
        If model output shape is incompatible.

    Notes
    -----
    This function is designed to be called independently by each task in a
    SLURM job array. Each task processes exactly one chunk with no communication
    between tasks.

    The function handles cases where the border may be truncated at array
    boundaries (e.g., chunks at the edge of the image).
    """
    # Create slice objects for core chunk
    core_chunk_slice = (
        slice(z_start, z_end),
        slice(y_start, y_end),
        slice(x_start, x_end),
    )

    # Open input zarr
    input_zarr = zarr.open(input_zarr_path, mode="r")
    array_shape = input_zarr.shape

    # Calculate expanded slice (core + border, clipped to array boundaries)
    border_size = (border_z, border_y, border_x)
    expanded_slice, actual_border_before = calculate_expanded_slice(
        core_chunk_slice, border_size, array_shape
    )

    # Load the expanded chunk from input
    expanded_chunk = np.array(input_zarr[expanded_slice])

    # Run inference on the expanded chunk
    prediction = inference_function(expanded_chunk)

    # Validate prediction shape matches expanded chunk shape
    if prediction.ndim != 3:
        raise ValueError(
            f"Model output must be 3D, got {prediction.ndim}D array. "
            f"Input shape was {expanded_chunk.shape}"
        )

    # Calculate the slice within the prediction that corresponds to the core region
    # This accounts for the actual border that was added (may be smaller at edges)
    core_z_size = z_end - z_start
    core_y_size = y_end - y_start
    core_x_size = x_end - x_start

    core_in_prediction_slice = (
        slice(actual_border_before[0], actual_border_before[0] + core_z_size),
        slice(actual_border_before[1], actual_border_before[1] + core_y_size),
        slice(actual_border_before[2], actual_border_before[2] + core_x_size),
    )

    # Extract the core region from the prediction (remove border)
    core_prediction = prediction[core_in_prediction_slice]

    # Validate core prediction shape
    expected_core_shape = (core_z_size, core_y_size, core_x_size)
    if core_prediction.shape != expected_core_shape:
        raise ValueError(
            f"Core prediction shape {core_prediction.shape} does not match "
            f"expected shape {expected_core_shape}"
        )

    # Write the core prediction to the output zarr
    output_zarr = zarr.open(output_zarr_path, mode="r+")
    output_zarr[core_chunk_slice] = core_prediction


def build_sbatch_command(
    n_array_jobs: int,
    job_name: str,
    time_limit: str,
    memory: str,
    cpus_per_task: int,
    n_gpus: int,
    gpu_name: str,
    run_command: str,
    output_pattern: str | None = None,
    error_pattern: str | None = None,
) -> str:
    """Build the sbatch command for submitting the job array.

    Parameters
    ----------
    n_array_jobs : int
        Total number array jobs (e.g., number of chunks to process).
    job_name : str
        Name for the SLURM job.
    time_limit : str
        Time limit for each task (HH:MM:SS format).
    memory : str
        Memory allocation per CPU (e.g., "32G").
    cpus_per_task : int
        Number of CPUs to allocate per task.
    n_gpus : int
        Number of GPUs to allocate per task.
    gpu_name : str
        Name of GPU type to request (e.g., "rtx_4090").
    output_pattern : str | None
        Pattern for output log files.
    error_pattern : str | None
        Pattern for error log files.
    run_command : str
        The command to be called for each array task.
        This should include the necessary arguments.

    Returns
    -------
    sbatch_cmd : list[str]
        List of command arguments to pass to subprocess.run().
    """
    # Build GPU specification
    gpu_spec = f"{gpu_name}:{n_gpus}"

    # Build sbatch command
    sbatch_cmd = (
        "sbatch"
        + f" --job-name={job_name}"
        + f" --array=0-{n_array_jobs - 1}"
        + f" --time={time_limit}"
        + f" --cpus-per-task={cpus_per_task}"
        + f" --mem-per-cpu={memory}"
        + f" --gpus={gpu_spec}"
    )

    if output_pattern is not None:
        sbatch_cmd += f" --output={output_pattern}"
    if error_pattern is not None:
        sbatch_cmd += f" --error={error_pattern}"

    # add the run command
    sbatch_cmd += f" --wrap='{run_command}'"

    return sbatch_cmd
