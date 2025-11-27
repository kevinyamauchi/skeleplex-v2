"""Benchmark script for chunked skeleton upscaling with zarr round-trip."""

import tempfile
import time
from pathlib import Path

import numpy as np
import zarr
from skimage.draw import line_nd

from skeleplex.skeleton import upscale_skeleton_parallel


def create_y_skeleton(
    zarr_array: zarr.Array, n_repeats: int = 1
) -> int:
    """Create a Y-shaped skeleton for benchmarking, writing directly to zarr.

    The Y skeleton spans from z=0.2*shape[0] to z=shape[0]-1, with a vertical
    stem and two diagonal branches forming the Y shape. The skeleton is repeated
    n_repeats times along the y-axis.

    Parameters
    ----------
    zarr_array : zarr.Array
        Zarr array to write skeleton into. Must be boolean type.
    n_repeats : int, optional
        Number of times to repeat the Y skeleton along the y-axis. Default is 1.

    Returns
    -------
    n_voxels : int
        Number of True voxels written (skeleton voxels).
    """
    shape = zarr_array.shape
    size_z, size_y, size_x = shape

    # Calculate z range: from 0.2 * size_z to size_z - 1
    z_start = int(0.2 * size_z)
    z_end = size_z - 1

    # Calculate center x
    center_x = size_x // 2

    # Branch point at approximately 60% of the z range
    branch_z = z_start + int(0.6 * (z_end - z_start))

    # Branch offset in x direction
    x_offset = size_x // 6

    # Calculate y positions for repeats
    y_positions = np.linspace(0, size_y - 1, n_repeats, dtype=int)

    # Count total voxels written
    total_voxels = 0

    # Draw Y skeleton at each y position
    for y_pos in y_positions:
        # Define key points for this Y
        stem_start = np.array([z_start, y_pos, center_x])
        branch_point = np.array([branch_z, y_pos, center_x])
        left_end = np.array([z_end, y_pos, center_x - x_offset])
        right_end = np.array([z_end, y_pos, center_x + x_offset])

        # Draw stem: from z_start to branch point
        stem_indices = line_nd(stem_start, branch_point, endpoint=True)
        zarr_array[stem_indices] = True
        total_voxels += len(stem_indices[0])

        # Draw left branch: from branch point to left end
        left_indices = line_nd(branch_point, left_end, endpoint=True)
        zarr_array[left_indices] = True
        total_voxels += len(left_indices[0])

        # Draw right branch: from branch point to right end
        right_indices = line_nd(branch_point, right_end, endpoint=True)
        zarr_array[right_indices] = True
        total_voxels += len(right_indices[0])

    return total_voxels


if __name__ == "__main__":
    # Configuration
    skeleton_shape = (3000, 3000, 3000)
    skeleton_upscale_factor = (3, 3, 3)
    n_skeletons = 500
    zarr_chunk_size = (100, 100, 100)
    n_processing_chunks = (2, 2, 2)
    border_size = (10, 10, 10)
    n_processes = 4
    pool_type = "spawn"

    # Create temporary directory for zarr files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = str(tmp_path / "input.zarr")
        output_path = str(tmp_path / "output.zarr")

        # Create input zarr array (not timed)
        print("Creating input zarr array...")
        input_zarr = zarr.open(
            input_path,
            mode="w",
            shape=skeleton_shape,
            chunks=zarr_chunk_size,
            dtype=bool,
        )
        print(f"Input zarr created at: {input_path}")
        print(f"Input zarr chunks: {input_zarr.chunks}")

        # Write skeleton directly to zarr (not timed)
        print("\nWriting skeleton to zarr...")
        n_skeleton_voxels = create_y_skeleton(input_zarr, n_skeletons)
        print(f"Original skeleton shape: {skeleton_shape}")

        # Benchmark the upscaling function
        print("\nRunning upscale_skeleton_parallel...")
        start_time = time.time()

        upscale_skeleton_parallel(
            input_path=input_path,
            output_path=output_path,
            scale_factors=skeleton_upscale_factor,
            n_processing_chunks=n_processing_chunks,
            border_size=border_size,
            n_processes=n_processes,
            pool_type=pool_type,
        )

        run_time = time.time() - start_time

        # Print results
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Input shape: {skeleton_shape}")
        print(f"N skeleton voxels: {n_skeleton_voxels:,}")
        print(f"Scale factors: {skeleton_upscale_factor}")
        print(f"Zarr chunk size: {zarr_chunk_size}")
        print(f"Processing chunks: {n_processing_chunks}")
        print(f"Border size: {border_size}")
        print(f"Number of processes: {n_processes}")
        print(f"Pool type: {pool_type}")
        print(f"\nInput voxels: {n_skeleton_voxels:,}")
        print(f"\nUpscaling took: {run_time:.2f} seconds")
        print("=" * 60)