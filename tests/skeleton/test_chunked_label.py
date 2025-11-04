import numpy as np
import zarr

from skeleplex.skeleton import label_chunks_parallel
from skeleplex.skeleton._chunked_label import _find_touching_labels, _make_label_mapping


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


def test_find_touching_labels(tmp_path):
    """Test that labels spanning multiple chunks are detected as touching."""
    array_shape = (20, 20, 20)
    chunk_shape = (8, 8, 8)

    # Create zarr array
    label_path = str(tmp_path / "labels.zarr")
    label_image = zarr.open(
        str(label_path),
        mode="w",
        shape=array_shape,
        chunks=chunk_shape,
        dtype=np.uint16,
    )

    # Initialize with zeros (background)
    label_image[:] = 0

    # Create an object spanning multiple chunks with different labels in each chunk
    # Object spans z=(5:18), which crosses chunk boundary at z=8 and z=16
    # Label it as 1 in first chunk (z=5:8), label 2 in middle chunk (z=8:16),
    # label 3 in last chunk (z=16:18)
    label_image[5:8, 2:4, 2:4] = 1
    label_image[8:16, 2:4, 2:4] = 2
    label_image[16:18, 2:4, 2:4] = 3

    # Create a non-touching object in a different location (label 4)
    label_image[10:15, 10:12, 10:12] = 4

    # Find the z-boundary at z=8 (between first and second chunk)
    z_boundary_8 = (slice(7, 9), slice(None), slice(None))

    # Check touching labels at z=8 boundary
    touching_at_8 = _find_touching_labels(z_boundary_8, label_path)

    # Labels 1 and 2 should be touching at this boundary
    touching_8_set = {tuple(sorted(pair)) for pair in touching_at_8.tolist()}
    assert {(1, 2)} == touching_8_set

    # Find the z-boundary at z=16 (between second and third chunk)
    z_boundary_16 = (slice(15, 17), slice(None), slice(None))

    # Check touching labels at z=16 boundary
    touching_at_16 = _find_touching_labels(z_boundary_16, label_path)

    # Labels 2 and 3 should be touching at this boundary
    touching_16_set = {tuple(sorted(pair)) for pair in touching_at_16.tolist()}
    assert {(2, 3)} == touching_16_set


def test_make_label_mapping():
    """Test label mapping with single pair and multi-pair connected component."""
    # Create touching pairs:
    # - Single pair: 10 and 15 (should map 10 -> 15)
    # - Connected component: 2 -> 5 -> 8 (should map 2 -> 8 and 5 -> 8)
    touching_pairs = np.array(
        [
            [10, 15],  # Single pair
            [2, 5],  # Connected component part 1
            [5, 8],  # Connected component part 2 (connects 2, 5, 8)
        ]
    )

    max_label_value = 20

    mapping = _make_label_mapping(touching_pairs, max_label_value)

    # Expected mapping:
    # - 10 -> 15 (max of {10, 15})
    # - 2 -> 8 (max of {2, 5, 8})
    # - 5 -> 8 (max of {2, 5, 8})
    # - 15 and 8 are not in mapping (already max in their components)

    expected = {
        10: 15,
        2: 8,
        5: 8,
    }

    assert mapping == expected, f"Expected {expected}, got {mapping}"
