import numpy as np
import zarr

from skeleplex.skeleton import label_chunks_parallel


def test_label_chunks_parallel_simple_cubes(tmp_path):
    """
    Test labeling with a simple image containing 2 cubes per chunk.
    Image is (20, 20, 20) with 2 chunks of (10, 20, 20).
    Each chunk contains 2 non-overlapping cubes.
    """
    # Define an array with 2 chunks
    array_shape = (20, 20, 20)
    chunk_shape = (10, 20, 20)

    # Define cube regions
    cube_regions = [
        (slice(2, 5), slice(2, 5), slice(2, 5)),  # Chunk 1, Cube 1
        (slice(6, 9), slice(6, 9), slice(6, 9)),  # Chunk 1, Cube 2
        (slice(12, 15), slice(2, 5), slice(2, 5)),  # Chunk 2, Cube 3
        (slice(16, 19), slice(6, 9), slice(6, 9)),  # Chunk 2, Cube 4
    ]

    # Create input array with 4 cubes total (2 per chunk)
    input_array = np.zeros(array_shape, dtype=np.uint8)
    for region in cube_regions:
        input_array[region] = 1

    # Save input to zarr
    input_path = str(tmp_path / "input.zarr")
    input_zarr = zarr.open(
        input_path, mode="w", shape=array_shape, chunks=chunk_shape, dtype=np.uint8
    )
    input_zarr[:] = input_array

    # Run labeling
    output_path = str(tmp_path / "output.zarr")
    total_labels = label_chunks_parallel(
        input_path=input_path,
        output_path=output_path,
        chunk_shape=chunk_shape,
        n_processes=2,
        pool_type="spawn",
    )

    # Load output
    output_zarr = zarr.open(output_path, mode="r")
    output_array = output_zarr[:]

    # Check 1: Correct label values (0 for background + 4 cubes)
    unique_labels = set(np.unique(output_array))
    assert unique_labels == {
        0,
        1,
        2,
        3,
        4,
    }, f"Expected labels {{0, 1, 2, 3, 4}}, got {unique_labels}"
    assert total_labels == 4, f"Expected 4 total labels, got {total_labels}"

    # Check 2 & 3: All voxels in each cube have the same unique label
    cube_labels = []
    for i, region in enumerate(cube_regions):
        cube_data = output_array[region]
        unique_in_cube = np.unique(cube_data)
        assert (
            len(unique_in_cube) == 1
        ), f"Cube {i} has multiple labels: {unique_in_cube}"
        assert unique_in_cube[0] > 0, f"Cube {i} has background label"
        cube_labels.append(unique_in_cube[0])

    # Verify each cube has a unique label value
    assert len(set(cube_labels)) == 4, f"Cubes don't have unique labels: {cube_labels}"
    assert set(cube_labels) == {
        1,
        2,
        3,
        4,
    }, f"Expected cube labels {{1, 2, 3, 4}}, got {set(cube_labels)}"
