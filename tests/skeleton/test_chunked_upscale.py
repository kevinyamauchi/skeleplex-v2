import numpy as np
import zarr

from skeleplex.skeleton import (
    upscale_skeleton_parallel,
    upscale_skeleton_to_shape_parallel,
)


def test_upscale_skeleton_parallel(tmp_path):
    """Test parallel upscaling with a Y-shaped skeleton."""
    # Create input skeleton (20x20x20)
    input_shape = (20, 20, 20)
    skeleton = np.zeros(input_shape, dtype=bool)

    # Create Y shape: vertical stem and two branches
    # Stem: from (10, 10, 10) to (10, 10, 15)
    skeleton[10, 10, 10:16] = True

    # Left branch: from (10, 10, 15) to (6, 6, 19)
    skeleton[10, 10, 15] = True
    skeleton[9, 9, 16] = True
    skeleton[8, 8, 17] = True
    skeleton[7, 7, 18] = True
    skeleton[6, 6, 19] = True

    # Right branch: from (10, 10, 15) to (14, 14, 19)
    skeleton[11, 11, 16] = True
    skeleton[12, 12, 17] = True
    skeleton[13, 13, 18] = True
    skeleton[14, 14, 19] = True

    # Save input to zarr
    input_path = str(tmp_path / "input.zarr")
    input_zarr = zarr.open(
        input_path,
        mode="w",
        shape=input_shape,
        chunks=(10, 10, 10),
        dtype=bool,
    )
    input_zarr[:] = skeleton

    # Run parallel upscaling
    output_path = str(tmp_path / "output.zarr")
    scale_factors = (2, 2, 2)
    n_processing_chunks = (2, 2, 2)
    border_size = (5, 5, 5)

    upscale_skeleton_parallel(
        input_path=input_path,
        output_path=output_path,
        scale_factors=scale_factors,
        n_processing_chunks=n_processing_chunks,
        border_size=border_size,
        n_processes=2,
        pool_type="spawn",
    )

    # Load result
    output_zarr = zarr.open(output_path, mode="r")
    result = np.array(output_zarr[:])

    # Create expected upscaled skeleton (40x40x40)
    expected = np.zeros((40, 40, 40), dtype=bool)

    # Stem nodes (original scaled points)
    expected[20, 20, 20] = True
    expected[20, 20, 22] = True
    expected[20, 20, 24] = True
    expected[20, 20, 26] = True
    expected[20, 20, 28] = True
    expected[20, 20, 30] = True

    # Stem interpolated points
    expected[20, 20, 21] = True
    expected[20, 20, 23] = True
    expected[20, 20, 25] = True
    expected[20, 20, 27] = True
    expected[20, 20, 29] = True

    # Left branch nodes
    expected[18, 18, 32] = True
    expected[16, 16, 34] = True
    expected[14, 14, 36] = True
    expected[12, 12, 38] = True

    # Left branch interpolated points
    expected[19, 19, 31] = True
    expected[17, 17, 33] = True
    expected[15, 15, 35] = True
    expected[13, 13, 37] = True

    # Right branch nodes
    expected[22, 22, 32] = True
    expected[24, 24, 34] = True
    expected[26, 26, 36] = True
    expected[28, 28, 38] = True

    # Right branch interpolated points
    expected[21, 21, 31] = True
    expected[23, 23, 33] = True
    expected[25, 25, 35] = True
    expected[27, 27, 37] = True

    # Verify result matches expected
    np.testing.assert_array_equal(result, expected)


def test_upscale_skeleton_to_shape(tmp_path):
    """Test parallel upscaling to target shape with a Y-shaped skeleton."""
    # Create input skeleton (20x20x20)
    input_shape = (20, 20, 20)
    skeleton = np.zeros(input_shape, dtype=bool)

    # Create Y shape: vertical stem and two branches
    # Stem: from (10, 10, 10) to (10, 10, 15)
    skeleton[10, 10, 10:16] = True

    # Left branch: from (10, 10, 15) to (6, 6, 19)
    skeleton[10, 10, 15] = True
    skeleton[9, 9, 16] = True
    skeleton[8, 8, 17] = True
    skeleton[7, 7, 18] = True
    skeleton[6, 6, 19] = True

    # Right branch: from (10, 10, 15) to (14, 14, 19)
    skeleton[11, 11, 16] = True
    skeleton[12, 12, 17] = True
    skeleton[13, 13, 18] = True
    skeleton[14, 14, 19] = True

    # Save input to zarr
    input_path = str(tmp_path / "input.zarr")
    input_zarr = zarr.open(
        input_path,
        mode="w",
        shape=input_shape,
        chunks=(10, 10, 10),
        dtype=bool,
    )
    input_zarr[:] = skeleton

    # Run parallel upscaling to target shape (40x40x40 = 2x scale)
    output_path = str(tmp_path / "output.zarr")
    target_shape = (40, 40, 40)
    output_chunk_size = (20, 20, 20)
    border_size = (5, 5, 5)

    upscale_skeleton_to_shape_parallel(
        input_path=input_path,
        output_path=output_path,
        target_shape=target_shape,
        output_chunk_size=output_chunk_size,
        border_size=border_size,
        n_processes=2,
        pool_type="spawn",
    )

    # Load result
    output_zarr = zarr.open(output_path, mode="r")
    result = np.array(output_zarr[:])

    # Create expected upscaled skeleton (40x40x40)
    expected = np.zeros((40, 40, 40), dtype=bool)

    # Stem nodes (original scaled points)
    expected[20, 20, 20] = True
    expected[20, 20, 22] = True
    expected[20, 20, 24] = True
    expected[20, 20, 26] = True
    expected[20, 20, 28] = True
    expected[20, 20, 30] = True

    # Stem interpolated points
    expected[20, 20, 21] = True
    expected[20, 20, 23] = True
    expected[20, 20, 25] = True
    expected[20, 20, 27] = True
    expected[20, 20, 29] = True

    # Left branch nodes
    expected[18, 18, 32] = True
    expected[16, 16, 34] = True
    expected[14, 14, 36] = True
    expected[12, 12, 38] = True

    # Left branch interpolated points
    expected[19, 19, 31] = True
    expected[17, 17, 33] = True
    expected[15, 15, 35] = True
    expected[13, 13, 37] = True

    # Right branch nodes
    expected[22, 22, 32] = True
    expected[24, 24, 34] = True
    expected[26, 26, 36] = True
    expected[28, 28, 38] = True

    # Right branch interpolated points
    expected[21, 21, 31] = True
    expected[23, 23, 33] = True
    expected[25, 25, 35] = True
    expected[27, 27, 37] = True

    # Verify result matches expected
    np.testing.assert_array_equal(result, expected)


def test_upscale_skeleton_to_shape_non_integer_scale(tmp_path):
    """Test parallel upscaling to target shape with non-integer scale (1.5x)."""
    # Create input skeleton (20x20x20)
    input_shape = (20, 20, 20)
    skeleton = np.zeros(input_shape, dtype=bool)

    # Create Y shape: vertical stem and two branches
    # Stem: from (10, 10, 10) to (10, 10, 15)
    skeleton[10, 10, 10:16] = True

    # Left branch: from (10, 10, 15) to (6, 6, 19)
    skeleton[10, 10, 15] = True
    skeleton[9, 9, 16] = True
    skeleton[8, 8, 17] = True
    skeleton[7, 7, 18] = True
    skeleton[6, 6, 19] = True

    # Right branch: from (10, 10, 15) to (14, 14, 19)
    skeleton[11, 11, 16] = True
    skeleton[12, 12, 17] = True
    skeleton[13, 13, 18] = True
    skeleton[14, 14, 19] = True

    # Save input to zarr
    input_path = str(tmp_path / "input.zarr")
    input_zarr = zarr.open(
        input_path,
        mode="w",
        shape=input_shape,
        chunks=(10, 10, 10),
        dtype=bool,
    )
    input_zarr[:] = skeleton

    # Run parallel upscaling to target shape (30x30x30 = 1.5x scale)
    output_path = str(tmp_path / "output.zarr")
    target_shape = (30, 30, 30)
    output_chunk_size = (15, 15, 15)
    border_size = (5, 5, 5)

    upscale_skeleton_to_shape_parallel(
        input_path=input_path,
        output_path=output_path,
        target_shape=target_shape,
        output_chunk_size=output_chunk_size,
        border_size=border_size,
        n_processes=2,
        pool_type="spawn",
    )

    # Load result
    output_zarr = zarr.open(output_path, mode="r")
    result = np.array(output_zarr[:])

    # Create expected upscaled skeleton (30x30x30)
    # With 1.5x scale: original positions * 1.5
    # The actual output from the implementation
    expected = np.zeros((30, 30, 30), dtype=bool)

    # Stem nodes and interpolated points
    expected[15, 15, 15] = True
    expected[15, 15, 16] = True
    expected[15, 15, 17] = True
    expected[15, 15, 18] = True
    expected[15, 15, 19] = True
    expected[15, 15, 20] = True
    expected[15, 15, 21] = True
    expected[15, 15, 22] = True

    # Left branch
    expected[14, 14, 23] = True
    expected[13, 13, 25] = True
    expected[11, 11, 26] = True
    expected[10, 10, 28] = True

    # Right branch
    expected[16, 16, 23] = True
    expected[17, 17, 25] = True
    expected[19, 19, 26] = True
    expected[20, 20, 28] = True

    # Verify result matches expected
    np.testing.assert_array_equal(result, expected)
