"""Tests for inference utilities."""

import numpy as np
import pandas as pd
import zarr

from skeleplex.utils import (
    infer_on_chunk,
    initialize_parallel_inference,
)


def test_create_inference_job_manifest(tmp_path):
    """Test that create_inference_job_manifest creates correct chunk table.

    This test creates a small 3D zarr array and validates that the manifest
    CSV contains the correct number of chunks and that each chunk's boundaries
    are calculated correctly.
    """
    # Create a small input zarr array (12x12x12)
    input_zarr_path = tmp_path / "input.zarr"
    input_shape = (12, 12, 12)
    input_chunks = (4, 4, 4)
    input_data = np.random.rand(*input_shape).astype(np.float32)

    input_zarr = zarr.open(
        input_zarr_path,
        mode="w",
        shape=input_shape,
        chunks=input_chunks,
        dtype=np.float32,
    )
    input_zarr[:] = input_data

    # Set processing parameters
    # Process in 6x6x6 chunks with 2-voxel border
    chunk_shape = (8, 8, 8)  # Multiple of zarr chunks (4)
    border_size = (2, 2, 2)
    output_zarr_path = tmp_path / "output.zarr"
    chunk_table_path = tmp_path / "chunk_table.csv"

    # Create the manifest
    n_chunks = initialize_parallel_inference(
        input_zarr_path=input_zarr_path,
        output_zarr_path=output_zarr_path,
        chunk_shape=chunk_shape,
        border_size=border_size,
        chunk_table_path=chunk_table_path,
    )

    # Expected: ceil(12/8) = 2 chunks per dimension, so 2*2*2 = 8 total chunks
    expected_n_chunks = 8
    assert n_chunks == expected_n_chunks

    # Load and validate the chunk table
    chunk_table = pd.read_csv(chunk_table_path)
    assert len(chunk_table) == expected_n_chunks

    # Validate each chunk's boundaries
    # Expected chunks (in ZYX order):
    # Chunk 0: z[0:8], y[0:8], x[0:8]
    # Chunk 1: z[0:8], y[0:8], x[8:12]
    # Chunk 2: z[0:8], y[8:12], x[0:8]
    # Chunk 3: z[0:8], y[8:12], x[8:12]
    # Chunk 4: z[8:12], y[0:8], x[0:8]
    # Chunk 5: z[8:12], y[0:8], x[8:12]
    # Chunk 6: z[8:12], y[8:12], x[0:8]
    # Chunk 7: z[8:12], y[8:12], x[8:12]

    expected_chunks = [
        # chunk_id, z_start, z_end, y_start, y_end, x_start, x_end
        (0, 0, 8, 0, 8, 0, 8),
        (1, 0, 8, 0, 8, 8, 12),
        (2, 0, 8, 8, 12, 0, 8),
        (3, 0, 8, 8, 12, 8, 12),
        (4, 8, 12, 0, 8, 0, 8),
        (5, 8, 12, 0, 8, 8, 12),
        (6, 8, 12, 8, 12, 0, 8),
        (7, 8, 12, 8, 12, 8, 12),
    ]

    for chunk_id, z_start, z_end, y_start, y_end, x_start, x_end in expected_chunks:
        row = chunk_table.iloc[chunk_id]
        assert row["z_start"] == z_start
        assert row["z_end"] == z_end
        assert row["y_start"] == y_start
        assert row["y_end"] == y_end
        assert row["x_start"] == x_start
        assert row["x_end"] == x_end
        assert row["border_z"] == border_size[0]
        assert row["border_y"] == border_size[1]
        assert row["border_x"] == border_size[2]

    # Validate that output zarr was created with correct properties
    output_zarr = zarr.open(output_zarr_path, mode="r")
    assert output_zarr.shape == input_shape
    assert output_zarr.chunks == input_chunks
    assert output_zarr.dtype == np.float32


def test_infer_on_chunk(tmp_path):
    """Test that infer_on_chunk correctly processes a chunk with borders.

    This test creates a test image where the core chunk region has a different
    value than the border region. It uses a passthrough model and verifies
    that only the core chunk values (not border values) are written to the
    output.
    """
    # Create input zarr array
    input_zarr_path = tmp_path / "input.zarr"
    input_shape = (20, 20, 20)
    input_chunks = (5, 5, 5)

    input_zarr = zarr.open(
        input_zarr_path,
        mode="w",
        shape=input_shape,
        chunks=input_chunks,
        dtype=np.float32,
    )

    # Initialize with zeros
    input_data = np.zeros(input_shape, dtype=np.float32)

    # Define border region around core (2 voxels on each side)
    # Set border region to value 2.0
    border_slice = (slice(3, 12), slice(3, 12), slice(3, 12))
    input_data[border_slice] = 2.0

    # Define a core chunk region: z[5:10], y[5:10], x[5:10]
    # Set core region to value 1.0
    core_slice = (slice(5, 10), slice(5, 10), slice(5, 10))
    input_data[core_slice] = 1.0

    # Write to input zarr
    input_zarr[:] = input_data

    # Create output zarr array (initially zeros)
    output_zarr_path = tmp_path / "output.zarr"
    output_zarr = zarr.open(
        output_zarr_path,
        mode="w",
        shape=input_shape,
        chunks=input_chunks,
        dtype=np.float32,
    )
    output_zarr[:] = np.zeros(input_shape, dtype=np.float32)

    # Create manifest for this single chunk
    border_size = (2, 2, 2)
    chunk_table_path = tmp_path / "chunk_table.csv"

    chunk_records = [
        {
            "z_start": 5,
            "z_end": 10,
            "y_start": 5,
            "y_end": 10,
            "x_start": 5,
            "x_end": 10,
            "border_z": border_size[0],
            "border_y": border_size[1],
            "border_x": border_size[2],
        }
    ]
    chunk_table = pd.DataFrame(chunk_records)
    chunk_table.to_csv(chunk_table_path, index=False)

    # Define a passthrough model function
    def passthrough_model(x: np.ndarray) -> np.ndarray:
        """Simple passthrough model that returns input unchanged."""
        return x

    # Process the chunk
    infer_on_chunk(
        chunk_id=0,
        manifest_path=str(chunk_table_path),
        input_zarr_path=str(input_zarr_path),
        output_zarr_path=str(output_zarr_path),
        model=passthrough_model,
    )

    # Load output and verify
    output_zarr = zarr.open(output_zarr_path, mode="r")
    output_data = np.array(output_zarr[:])

    # Only the core region should be written (value 1.0)
    core_output = output_data[core_slice]
    assert np.all(core_output == 1.0), "Core chunk should contain value 1.0"

    # Verify that the entire rest of the array is still zeros
    # this includes the border region
    mask = np.ones(input_shape, dtype=bool)
    mask[core_slice] = False
    non_core_data = output_data[mask]
    assert np.all(non_core_data == 0.0), "Non-core regions should remain zeros"
