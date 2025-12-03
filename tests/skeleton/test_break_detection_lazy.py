"""Tests for lazy chunk-based skeleton break repair."""

import numpy as np
import pytest
import zarr

from skeleplex.skeleton import repair_breaks_lazy


def test_repair_breaks_lazy_shape_mismatch(tmp_path):
    """Test that ValueError is raised when input shapes don't match."""
    # Create skeleton zarr
    skeleton_path = tmp_path / "skeleton.zarr"
    skeleton_zarr = zarr.open(
        str(skeleton_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=bool,
    )
    skeleton_zarr[:] = False

    # Create segmentation zarr with different shape
    segmentation_path = tmp_path / "segmentation.zarr"
    segmentation_zarr = zarr.open(
        str(segmentation_path),
        mode="w",
        shape=(40, 30, 30),  # Different shape
        chunks=(10, 10, 10),
        dtype=bool,
    )
    segmentation_zarr[:] = False

    # Create output path
    output_path = tmp_path / "output.zarr"

    # Should raise ValueError
    with pytest.raises(ValueError, match="Input and segmentation shapes must match"):
        repair_breaks_lazy(
            skeleton_path=skeleton_path,
            segmentation_path=segmentation_path,
            output_path=output_path,
            repair_radius=10.0,
            chunk_shape=(10, 10, 10),
        )


def test_repair_breaks_lazy_simple_line(tmp_path):
    """Test repair_breaks_lazy with a simple line skeleton.

    The break crosses the chunk boundary,
    but is within the repair radius.
    """
    # Create skeleton with a break
    # Position the break so it crosses the chunk boundary
    skeleton_data = np.zeros((30, 30, 30), dtype=bool)
    skeleton_data[7:16, 12, 12] = True  # First segment
    skeleton_data[20:24, 12, 12] = True  # Second segment (break from 16 to 20)

    # Segmentation includes the gap
    segmentation_data = np.zeros((30, 30, 30), dtype=bool)
    segmentation_data[5:25, 10:14, 10:14] = True

    # Create zarr arrays
    skeleton_path = tmp_path / "skeleton.zarr"
    skeleton_zarr = zarr.open(
        str(skeleton_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=bool,
    )
    skeleton_zarr[:] = skeleton_data

    segmentation_path = tmp_path / "segmentation.zarr"
    segmentation_zarr = zarr.open(
        str(segmentation_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=bool,
    )
    segmentation_zarr[:] = segmentation_data

    output_path = tmp_path / "output.zarr"

    # Set chunk size to 15 so the break at z=16-20 crosses into the boundary
    # Chunk 0: z=0-15 (endpoint at z=15)
    # Chunk 1: z=15-30 (endpoint at z=20)
    # The break spans these two chunks
    repair_breaks_lazy(
        skeleton_path=skeleton_path,
        segmentation_path=segmentation_path,
        output_path=output_path,
        repair_radius=10.0,
        chunk_shape=(15, 30, 30),
    )

    # Load result
    result = np.array(zarr.open(str(output_path), mode="r")[:])

    # Expected: complete line from 7 to 23
    expected = np.zeros((30, 30, 30), dtype=bool)
    expected[7:24, 12, 12] = True

    np.testing.assert_array_equal(result, expected)


def test_repair_breaks_lazy_tee(tmp_path):
    """Test repair_breaks_lazy with a T-junction skeleton.

    The break crosses the chunk boundary,
    but is within the repair radius.
    """
    # Create T-shaped skeleton with a break in the vertical stem
    skeleton_data = np.zeros((30, 30, 30), dtype=bool)
    # Vertical stem (with break)
    skeleton_data[7:17, 12, 12] = True  # Top part of stem
    skeleton_data[20:27, 12, 12] = True  # Bottom part of stem (break from 17 to 20)
    # Horizontal crossbar
    skeleton_data[17, 4:8, 12] = True  # Left part of crossbar

    # Segmentation allows the repair
    segmentation_data = np.zeros((30, 30, 30), dtype=bool)
    segmentation_data[5:28, 10:14, 10:14] = True
    segmentation_data[16:19, 2:14, 10:14] = True

    # Create zarr arrays
    skeleton_path = tmp_path / "skeleton.zarr"
    skeleton_zarr = zarr.open(
        str(skeleton_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=bool,
    )
    skeleton_zarr[:] = skeleton_data

    segmentation_path = tmp_path / "segmentation.zarr"
    segmentation_zarr = zarr.open(
        str(segmentation_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=bool,
    )
    segmentation_zarr[:] = segmentation_data

    output_path = tmp_path / "output.zarr"

    # Set chunk size to 15 so the break at z=17-20 crosses into the boundary
    repair_breaks_lazy(
        skeleton_path=skeleton_path,
        segmentation_path=segmentation_path,
        output_path=output_path,
        repair_radius=5.0,
        chunk_shape=(15, 30, 30),
    )

    # Load result
    result = np.array(zarr.open(str(output_path), mode="r")[:])

    # Expected: complete T-shape
    expected = np.zeros((30, 30, 30), dtype=bool)
    expected[7:27, 12, 12] = True  # Complete vertical stem
    expected[17, 4:8, 12] = True  # Horizontal crossbar

    np.testing.assert_array_equal(result, expected)


def test_repair_breaks_lazy_break_too_long(tmp_path):
    """Test that breaks longer than repair_radius remain unrepaired."""
    # Create skeleton with a large break
    skeleton_data = np.zeros((30, 30, 30), dtype=bool)
    skeleton_data[7:16, 12, 12] = True  # First segment
    skeleton_data[20:24, 12, 12] = True  # Second segment (break of 4 voxels)

    # Segmentation includes the gap
    segmentation_data = np.zeros((30, 30, 30), dtype=bool)
    segmentation_data[5:25, 10:14, 10:14] = True

    # Create zarr arrays
    skeleton_path = tmp_path / "skeleton.zarr"
    skeleton_zarr = zarr.open(
        str(skeleton_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=bool,
    )
    skeleton_zarr[:] = skeleton_data

    segmentation_path = tmp_path / "segmentation.zarr"
    segmentation_zarr = zarr.open(
        str(segmentation_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=bool,
    )
    segmentation_zarr[:] = segmentation_data

    output_path = tmp_path / "output.zarr"

    # Try to repair with small radius (should not connect)
    repair_breaks_lazy(
        skeleton_path=skeleton_path,
        segmentation_path=segmentation_path,
        output_path=output_path,
        repair_radius=3.0,
        chunk_shape=(15, 30, 30),
    )

    # Load result
    result = np.array(zarr.open(str(output_path), mode="r")[:])

    # Expected: no change (break is too long)
    expected = skeleton_data.copy()

    np.testing.assert_array_equal(result, expected)
