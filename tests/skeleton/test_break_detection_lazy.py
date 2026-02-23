"""Tests for lazy chunk-based skeleton break repair."""

from importlib.util import find_spec

import numpy as np
import pytest
import zarr

from skeleplex.skeleton import repair_breaks_lazy, repair_fusion_breaks_lazy
from skeleplex.skeleton._break_detection_lazy import (
    repair_breaks_chunk,
    repair_fusion_breaks_chunk,
)

# True if cupy is installed
CUPY_AVAILABLE = find_spec("cupy") is not None


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


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="cupy not installed")
def test_repair_breaks_lazy_simple_line_gpu(tmp_path):
    """Test repair_breaks_lazy with a simple line skeleton.

    The break crosses the chunk boundary,
    but is within the repair radius.

    This uses the GPU backend.
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
        backend="cupy",
    )

    # Load result
    result = np.array(zarr.open(str(output_path), mode="r")[:])

    # Expected: complete line from 7 to 23
    expected = np.zeros((30, 30, 30), dtype=bool)
    expected[7:24, 12, 12] = True

    np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="cupy not installed")
def test_repair_breaks_lazy_tee_gpu(tmp_path):
    """Test repair_breaks_lazy with a T-junction skeleton.

    The break crosses the chunk boundary,
    but is within the repair radius.

    This uses the GPU backend.
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
        backend="cupy",
    )

    # Load result
    result = np.array(zarr.open(str(output_path), mode="r")[:])

    # Expected: complete T-shape
    expected = np.zeros((30, 30, 30), dtype=bool)
    expected[7:27, 12, 12] = True  # Complete vertical stem
    expected[17, 4:8, 12] = True  # Horizontal crossbar

    np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="cupy not installed")
def test_repair_breaks_lazy_break_too_long_gpu(tmp_path):
    """Test that breaks longer than repair_radius remain unrepaired.

    This uses the GPU backend
    """
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
        backend="cupy",
    )

    # Load result
    result = np.array(zarr.open(str(output_path), mode="r")[:])

    # Expected: no change (break is too long)
    expected = skeleton_data.copy()

    np.testing.assert_array_equal(result, expected)


def test_repair_breaks_lazy_label_map_shape_mismatch(tmp_path):
    """ValueError is raised when label_map_path points to wrong shape."""
    skeleton_path = tmp_path / "skeleton.zarr"
    zarr.open(
        str(skeleton_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=bool,
    )

    segmentation_path = tmp_path / "segmentation.zarr"
    zarr.open(
        str(segmentation_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=bool,
    )

    label_map_path = tmp_path / "label_map.zarr"
    zarr.open(
        str(label_map_path),
        mode="w",
        shape=(20, 20, 20),  # Wrong shape
        chunks=(10, 10, 10),
        dtype=np.int32,
    )

    output_path = tmp_path / "output.zarr"

    with pytest.raises(ValueError, match="label_map shape"):
        repair_breaks_lazy(
            skeleton_path=skeleton_path,
            segmentation_path=segmentation_path,
            output_path=output_path,
            label_map_path=label_map_path,
        )


def test_repair_breaks_lazy_global_label_map_prevents_false_repair(
    tmp_path,
):
    """Global label map prevents false repairs in chunked processing.

    Two segments that are locally disconnected in the chunk but globally
    connected should NOT be repaired when a global label map says they
    are the same component.
    """
    # Skeleton with a gap between z=16 and z=20
    skeleton_data = np.zeros((30, 30, 30), dtype=bool)
    skeleton_data[7:16, 12, 12] = True
    skeleton_data[20:24, 12, 12] = True

    segmentation_data = np.zeros((30, 30, 30), dtype=bool)
    segmentation_data[5:25, 10:14, 10:14] = True

    # Global label map: same component
    label_data = np.zeros((30, 30, 30), dtype=np.int32)
    label_data[7:16, 12, 12] = 1
    label_data[20:24, 12, 12] = 1

    # Write zarr arrays
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

    label_map_path = tmp_path / "label_map.zarr"
    label_map_zarr = zarr.open(
        str(label_map_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=np.int32,
    )
    label_map_zarr[:] = label_data

    output_path = tmp_path / "output.zarr"

    repair_breaks_lazy(
        skeleton_path=skeleton_path,
        segmentation_path=segmentation_path,
        output_path=output_path,
        repair_radius=10.0,
        chunk_shape=(15, 30, 30),
        label_map_path=label_map_path,
    )

    result = np.array(zarr.open(str(output_path), mode="r")[:])

    # No repair should have been made
    np.testing.assert_array_equal(result, skeleton_data)


def test_repair_breaks_chunk_with_label_map_zarr(tmp_path):
    """Label map slice is read from zarr and forwarded to repair_breaks.

    Two segments in the same global component should not be repaired,
    even when processed through repair_breaks_chunk.
    """
    skeleton_data = np.zeros((20, 20, 20), dtype=bool)
    skeleton_data[3:8, 10, 10] = True
    skeleton_data[12:16, 10, 10] = True

    segmentation_data = np.ones((20, 20, 20), dtype=bool)

    # Same component in global label map
    label_data = np.zeros((20, 20, 20), dtype=np.int32)
    label_data[3:8, 10, 10] = 1
    label_data[12:16, 10, 10] = 1

    # Write zarr arrays (single chunk covers entire volume)
    skeleton_path = tmp_path / "skeleton.zarr"
    skeleton_zarr = zarr.open(
        str(skeleton_path),
        mode="w",
        shape=(20, 20, 20),
        chunks=(20, 20, 20),
        dtype=bool,
    )
    skeleton_zarr[:] = skeleton_data

    output_path = tmp_path / "output.zarr"
    output_zarr = zarr.open(
        str(output_path),
        mode="w",
        shape=(20, 20, 20),
        chunks=(20, 20, 20),
        dtype=bool,
    )

    segmentation_path = tmp_path / "segmentation.zarr"
    segmentation_zarr = zarr.open(
        str(segmentation_path),
        mode="w",
        shape=(20, 20, 20),
        chunks=(20, 20, 20),
        dtype=bool,
    )
    segmentation_zarr[:] = segmentation_data

    label_map_path = tmp_path / "label_map.zarr"
    label_map_zarr = zarr.open(
        str(label_map_path),
        mode="w",
        shape=(20, 20, 20),
        chunks=(20, 20, 20),
        dtype=np.int32,
    )
    label_map_zarr[:] = label_data

    full_slice = (slice(0, 20), slice(0, 20), slice(0, 20))
    no_border = (0, 0, 0)

    repair_breaks_chunk(
        skeleton=skeleton_zarr,
        output_skeleton=output_zarr,
        segmentation=segmentation_zarr,
        expanded_slice=full_slice,
        actual_border=no_border,
        repair_radius=10.0,
        label_map_zarr=label_map_zarr,
    )

    result = np.array(output_zarr[:])

    # Same component → no repair
    np.testing.assert_array_equal(result, skeleton_data)


def test_repair_breaks_lazy_angle_params(tmp_path):
    """Test break detection when the angle params are used.

    A simple line with a break should still be repaired when the
    angle parameters are explicitly provided.
    """
    skeleton_data = np.zeros((30, 30, 30), dtype=bool)
    skeleton_data[7:16, 12, 12] = True
    skeleton_data[20:24, 12, 12] = True

    segmentation_data = np.zeros((30, 30, 30), dtype=bool)
    segmentation_data[5:25, 10:14, 10:14] = True

    # Write zarr arrays
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

    # Provide explicit angle parameters
    repair_breaks_lazy(
        skeleton_path=skeleton_path,
        segmentation_path=segmentation_path,
        output_path=output_path,
        repair_radius=10.0,
        chunk_shape=(15, 30, 30),
        n_fit_voxels=5,
        w_distance=1.0,
        w_angle=0.5,
    )

    result = np.array(zarr.open(str(output_path), mode="r")[:])

    # Repair should be made (aligned candidates)
    expected = np.zeros((30, 30, 30), dtype=bool)
    expected[7:24, 12, 12] = True
    np.testing.assert_array_equal(result, expected)


def _make_two_tile_scale_map(shape, boundary_z):
    """Return a scale map with tile 1 below boundary_z and tile 2 above.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Shape of the volume (z, y, x).
    boundary_z : int
        z-index at which the tile boundary falls.
        Voxels with z < boundary_z receive tile ID 1;
        voxels with z >= boundary_z receive tile ID 2.

    Returns
    -------
    scale_map : np.ndarray
        Integer array of shape ``shape``.
    """
    scale_map = np.ones(shape, dtype=np.int32)
    scale_map[boundary_z:] = 2
    return scale_map


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_repair_fusion_breaks_lazy_skeleton_segmentation_shape_mismatch(tmp_path):
    """ValueError is raised when skeleton and segmentation shapes differ."""
    skeleton_path = tmp_path / "skeleton.zarr"
    skeleton_zarr = zarr.open(
        str(skeleton_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=bool,
    )
    skeleton_zarr[:] = False

    segmentation_path = tmp_path / "segmentation.zarr"
    segmentation_zarr = zarr.open(
        str(segmentation_path),
        mode="w",
        shape=(40, 30, 30),  # Different shape
        chunks=(10, 10, 10),
        dtype=bool,
    )
    segmentation_zarr[:] = False

    scale_map_path = tmp_path / "scale_map.zarr"
    scale_map_zarr = zarr.open(
        str(scale_map_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=np.int32,
    )
    scale_map_zarr[:] = 0

    output_path = tmp_path / "output.zarr"

    with pytest.raises(ValueError, match="Skeleton and segmentation shapes must match"):
        repair_fusion_breaks_lazy(
            skeleton_path=skeleton_path,
            segmentation_path=segmentation_path,
            scale_map_path=scale_map_path,
            output_path=output_path,
        )


def test_repair_fusion_breaks_lazy_scale_map_shape_mismatch(tmp_path):
    """ValueError is raised when the scale map shape differs from the skeleton."""
    skeleton_path = tmp_path / "skeleton.zarr"
    skeleton_zarr = zarr.open(
        str(skeleton_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=bool,
    )
    skeleton_zarr[:] = False

    segmentation_path = tmp_path / "segmentation.zarr"
    segmentation_zarr = zarr.open(
        str(segmentation_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=bool,
    )
    segmentation_zarr[:] = False

    scale_map_path = tmp_path / "scale_map.zarr"
    scale_map_zarr = zarr.open(
        str(scale_map_path),
        mode="w",
        shape=(40, 30, 30),  # Different shape
        chunks=(10, 10, 10),
        dtype=np.int32,
    )
    scale_map_zarr[:] = 0

    output_path = tmp_path / "output.zarr"

    with pytest.raises(ValueError, match="Skeleton and scale map shapes must match"):
        repair_fusion_breaks_lazy(
            skeleton_path=skeleton_path,
            segmentation_path=segmentation_path,
            scale_map_path=scale_map_path,
            output_path=output_path,
        )


def test_repair_fusion_breaks_lazy_label_map_shape_mismatch(tmp_path):
    """ValueError is raised when label_map_path points to an array of wrong shape."""
    skeleton_path = tmp_path / "skeleton.zarr"
    skeleton_zarr = zarr.open(
        str(skeleton_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=bool,
    )
    skeleton_zarr[:] = False

    segmentation_path = tmp_path / "segmentation.zarr"
    segmentation_zarr = zarr.open(
        str(segmentation_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=bool,
    )
    segmentation_zarr[:] = False

    scale_map_path = tmp_path / "scale_map.zarr"
    scale_map_zarr = zarr.open(
        str(scale_map_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=np.int32,
    )
    scale_map_zarr[:] = 0

    label_map_path = tmp_path / "label_map.zarr"
    label_map_zarr = zarr.open(
        str(label_map_path),
        mode="w",
        shape=(20, 20, 20),  # Wrong shape
        chunks=(10, 10, 10),
        dtype=np.int32,
    )
    label_map_zarr[:] = 0

    output_path = tmp_path / "output.zarr"

    with pytest.raises(ValueError, match="label_map shape"):
        repair_fusion_breaks_lazy(
            skeleton_path=skeleton_path,
            segmentation_path=segmentation_path,
            scale_map_path=scale_map_path,
            output_path=output_path,
            label_map_path=label_map_path,
        )


def test_repair_fusion_breaks_lazy_simple_line(tmp_path):
    """A break straddling a fusion boundary is repaired.

    The skeleton has two segments along z separated by a gap at z=15.
    The scale map places a tile boundary at z=15, so the endpoint of
    the first segment (z=14, tile 1) is adjacent to tile 2 and the
    endpoint of the second segment (z=15, tile 2) is adjacent to tile 1.
    Both endpoints lie directly on the fusion boundary and are candidates
    for repair.
    """
    # Skeleton: two segments along z with a gap at z=15
    skeleton_data = np.zeros((30, 30, 30), dtype=bool)
    skeleton_data[5:15, 12, 12] = True  # First segment, endpoint at z=14
    skeleton_data[15:24, 14, 12] = True  # Second segment, endpoint at z=15

    # Segmentation covers both y positions and the gap between them
    segmentation_data = np.zeros((30, 30, 30), dtype=bool)
    segmentation_data[5:25, 10:16, 10:16] = True

    # Tile boundary at z=15: endpoint at z=14 is in tile 1 adjacent to
    # tile 2, endpoint at z=15 is in tile 2 adjacent to tile 1
    scale_map_data = _make_two_tile_scale_map((30, 30, 30), boundary_z=15)

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

    scale_map_path = tmp_path / "scale_map.zarr"
    scale_map_zarr = zarr.open(
        str(scale_map_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=np.int32,
    )
    scale_map_zarr[:] = scale_map_data

    output_path = tmp_path / "output.zarr"

    # Single chunk covers the full volume to isolate fusion boundary
    # repair logic from chunking behaviour
    repair_fusion_breaks_lazy(
        skeleton_path=skeleton_path,
        segmentation_path=segmentation_path,
        scale_map_path=scale_map_path,
        output_path=output_path,
        repair_radius=10.0,
        chunk_shape=(30, 30, 30),
    )

    result = np.array(zarr.open(str(output_path), mode="r")[:])

    expected = np.zeros((30, 30, 30), dtype=bool)
    expected[5:15, 12, 12] = True
    expected[14, 13, 12] = True  # Diagonal bridging voxel
    expected[15:24, 14, 12] = True

    np.testing.assert_array_equal(result, expected)


def test_repair_fusion_breaks_lazy_break_not_on_boundary(tmp_path):
    """A break not on a fusion boundary is NOT repaired.

    The skeleton has a gap identical to the simple-line test, but the
    scale map is uniform (single tile everywhere). Since neither endpoint
    lies on a fusion boundary, no repair should be made.
    """
    skeleton_data = np.zeros((30, 30, 30), dtype=bool)
    skeleton_data[7:16, 12, 12] = True
    skeleton_data[20:24, 12, 12] = True

    segmentation_data = np.zeros((30, 30, 30), dtype=bool)
    segmentation_data[5:25, 10:14, 10:14] = True

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

    # Uniform tile map — no fusion boundaries exist
    scale_map_path = tmp_path / "scale_map.zarr"
    scale_map_zarr = zarr.open(
        str(scale_map_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=np.int32,
    )
    scale_map_zarr[:] = 1

    output_path = tmp_path / "output.zarr"

    repair_fusion_breaks_lazy(
        skeleton_path=skeleton_path,
        segmentation_path=segmentation_path,
        scale_map_path=scale_map_path,
        output_path=output_path,
        repair_radius=10.0,
        chunk_shape=(15, 30, 30),
    )

    result = np.array(zarr.open(str(output_path), mode="r")[:])

    # No repair should have been made
    np.testing.assert_array_equal(result, skeleton_data)


def test_repair_fusion_breaks_lazy_break_too_long(tmp_path):
    """A break on a fusion boundary that exceeds repair_radius is not repaired.

    Both endpoints sit directly on the fusion boundary at z=14 and z=15.
    They are offset in y so the Euclidean distance between them is
    sqrt(1 + 36) ≈ 6.1, which exceeds repair_radius=5.
    """
    skeleton_data = np.zeros((30, 30, 30), dtype=bool)
    skeleton_data[7:15, 12, 12] = True  # Endpoint at (z=14, y=12), tile 1
    skeleton_data[15:22, 18, 12] = True  # Endpoint at (z=15, y=18), tile 2

    # Segmentation covers both y positions
    segmentation_data = np.zeros((30, 30, 30), dtype=bool)
    segmentation_data[5:23, 10:20, 10:14] = True

    scale_map_data = _make_two_tile_scale_map((30, 30, 30), boundary_z=15)

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

    scale_map_path = tmp_path / "scale_map.zarr"
    scale_map_zarr = zarr.open(
        str(scale_map_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=np.int32,
    )
    scale_map_zarr[:] = scale_map_data

    output_path = tmp_path / "output.zarr"

    repair_fusion_breaks_lazy(
        skeleton_path=skeleton_path,
        segmentation_path=segmentation_path,
        scale_map_path=scale_map_path,
        output_path=output_path,
        repair_radius=5.0,
        chunk_shape=(30, 30, 30),
    )

    result = np.array(zarr.open(str(output_path), mode="r")[:])

    np.testing.assert_array_equal(result, skeleton_data)


def test_repair_fusion_breaks_lazy_global_label_map_prevents_false_repair(tmp_path):
    """Global label map prevents repair of endpoints in the same component.

    Two segments sit directly on either side of the fusion boundary
    (endpoints at z=14 and z=15) and are within repair_radius. Without
    the label map they would be repaired. With the label map marking
    them as the same connected component, no repair should be made.
    """
    # Same geometry as test_repair_fusion_breaks_lazy_simple_line so
    # that the label map is the only thing preventing the repair
    skeleton_data = np.zeros((30, 30, 30), dtype=bool)
    skeleton_data[5:15, 12, 12] = True  # Endpoint at (z=14, y=12)
    skeleton_data[15:24, 14, 12] = True  # Endpoint at (z=15, y=14)

    segmentation_data = np.zeros((30, 30, 30), dtype=bool)
    segmentation_data[5:25, 10:16, 10:16] = True

    scale_map_data = _make_two_tile_scale_map((30, 30, 30), boundary_z=15)

    # Both segments share the same global label
    label_data = np.zeros((30, 30, 30), dtype=np.int32)
    label_data[5:15, 12, 12] = 1
    label_data[15:24, 14, 12] = 1

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

    scale_map_path = tmp_path / "scale_map.zarr"
    scale_map_zarr = zarr.open(
        str(scale_map_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=np.int32,
    )
    scale_map_zarr[:] = scale_map_data

    label_map_path = tmp_path / "label_map.zarr"
    label_map_zarr = zarr.open(
        str(label_map_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=np.int32,
    )
    label_map_zarr[:] = label_data

    output_path = tmp_path / "output.zarr"

    repair_fusion_breaks_lazy(
        skeleton_path=skeleton_path,
        segmentation_path=segmentation_path,
        scale_map_path=scale_map_path,
        output_path=output_path,
        repair_radius=10.0,
        chunk_shape=(30, 30, 30),
        label_map_path=label_map_path,
    )

    result = np.array(zarr.open(str(output_path), mode="r")[:])

    # Same component → no repair
    np.testing.assert_array_equal(result, skeleton_data)


def test_repair_fusion_breaks_lazy_endpoint_mask_dilation(tmp_path):
    """endpoint_mask_dilation controls whether near-boundary endpoints are repaired.

    The skeleton has a gap whose endpoints are 2 voxels away from the
    fusion boundary. With no dilation the endpoints fall outside the
    boundary mask and no repair is made. With sufficient dilation the
    mask expands to include them and the break is repaired.
    """
    # Endpoints at z=12 and z=18 — both 2 voxels from the boundary band
    skeleton_data = np.zeros((30, 30, 30), dtype=bool)
    skeleton_data[7:13, 12, 12] = True  # Endpoint at z=12
    skeleton_data[18:24, 12, 12] = True  # Endpoint at z=18

    segmentation_data = np.zeros((30, 30, 30), dtype=bool)
    segmentation_data[5:25, 10:14, 10:14] = True

    # Make the scale map have a boundary at z=15
    scale_map_data = _make_two_tile_scale_map((30, 30, 30), boundary_z=15)

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

    scale_map_path = tmp_path / "scale_map.zarr"
    scale_map_zarr = zarr.open(
        str(scale_map_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=np.int32,
    )
    scale_map_zarr[:] = scale_map_data

    # --- No dilation: break should NOT be repaired ---
    output_no_dilation_path = tmp_path / "output_no_dilation.zarr"
    repair_fusion_breaks_lazy(
        skeleton_path=skeleton_path,
        segmentation_path=segmentation_path,
        scale_map_path=scale_map_path,
        output_path=output_no_dilation_path,
        repair_radius=10.0,
        chunk_shape=(15, 30, 30),
        endpoint_mask_dilation=0,
    )
    result_no_dilation = np.array(zarr.open(str(output_no_dilation_path), mode="r")[:])
    np.testing.assert_array_equal(result_no_dilation, skeleton_data)

    # --- Sufficient dilation: break SHOULD be repaired ---
    output_dilated_path = tmp_path / "output_dilated.zarr"
    repair_fusion_breaks_lazy(
        skeleton_path=skeleton_path,
        segmentation_path=segmentation_path,
        scale_map_path=scale_map_path,
        output_path=output_dilated_path,
        repair_radius=10.0,
        chunk_shape=(15, 30, 30),
        endpoint_mask_dilation=4,
    )
    result_dilated = np.array(zarr.open(str(output_dilated_path), mode="r")[:])

    expected = np.zeros((30, 30, 30), dtype=bool)
    expected[7:24, 12, 12] = True
    np.testing.assert_array_equal(result_dilated, expected)


def test_repair_fusion_breaks_chunk_with_label_map_zarr(tmp_path):
    """Label map slice is loaded from zarr and forwarded to repair_fusion_breaks.

    Two segments straddle the fusion boundary directly: the first ends
    at z=9 (tile 1, adjacent to tile 2) and the second starts at z=10
    (tile 2, adjacent to tile 1). They share the same global component
    label, so no repair should be made.
    """
    skeleton_data = np.zeros((20, 20, 20), dtype=bool)
    skeleton_data[3:10, 10, 10] = True  # Endpoint at (z=9, y=10)
    skeleton_data[10:16, 12, 10] = True  # Endpoint at (z=10, y=12)

    segmentation_data = np.ones((20, 20, 20), dtype=bool)

    scale_map_data = _make_two_tile_scale_map((20, 20, 20), boundary_z=10)

    # Both segments share the same global component label
    label_data = np.zeros((20, 20, 20), dtype=np.int32)
    label_data[3:10, 10, 10] = 1
    label_data[10:16, 12, 10] = 1

    skeleton_zarr = zarr.open(
        str(tmp_path / "skeleton.zarr"),
        mode="w",
        shape=(20, 20, 20),
        chunks=(20, 20, 20),
        dtype=bool,
    )
    skeleton_zarr[:] = skeleton_data

    segmentation_zarr = zarr.open(
        str(tmp_path / "segmentation.zarr"),
        mode="w",
        shape=(20, 20, 20),
        chunks=(20, 20, 20),
        dtype=bool,
    )
    segmentation_zarr[:] = segmentation_data

    scale_map_zarr = zarr.open(
        str(tmp_path / "scale_map.zarr"),
        mode="w",
        shape=(20, 20, 20),
        chunks=(20, 20, 20),
        dtype=np.int32,
    )
    scale_map_zarr[:] = scale_map_data

    label_map_zarr = zarr.open(
        str(tmp_path / "label_map.zarr"),
        mode="w",
        shape=(20, 20, 20),
        chunks=(20, 20, 20),
        dtype=np.int32,
    )
    label_map_zarr[:] = label_data

    output_zarr = zarr.open(
        str(tmp_path / "output.zarr"),
        mode="w",
        shape=(20, 20, 20),
        chunks=(20, 20, 20),
        dtype=bool,
    )

    full_slice = (slice(0, 20), slice(0, 20), slice(0, 20))
    no_border = (0, 0, 0)

    repair_fusion_breaks_chunk(
        skeleton=skeleton_zarr,
        output_skeleton=output_zarr,
        segmentation=segmentation_zarr,
        scale_map=scale_map_zarr,
        expanded_slice=full_slice,
        actual_border=no_border,
        repair_radius=10.0,
        label_map_zarr=label_map_zarr,
    )

    result = np.array(output_zarr[:])

    # Same component → no repair
    np.testing.assert_array_equal(result, skeleton_data)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="cupy not installed")
def test_repair_fusion_breaks_lazy_simple_line_gpu(tmp_path):
    """A break straddling a fusion boundary is repaired using the GPU backend."""
    skeleton_data = np.zeros((30, 30, 30), dtype=bool)
    skeleton_data[5:15, 12, 12] = True  # Endpoint at (z=14, y=12), tile 1
    skeleton_data[15:24, 14, 12] = True  # Endpoint at (z=15, y=14), tile 2

    segmentation_data = np.zeros((30, 30, 30), dtype=bool)
    segmentation_data[5:25, 10:16, 10:16] = True

    scale_map_data = _make_two_tile_scale_map((30, 30, 30), boundary_z=15)

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

    scale_map_path = tmp_path / "scale_map.zarr"
    scale_map_zarr = zarr.open(
        str(scale_map_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=np.int32,
    )
    scale_map_zarr[:] = scale_map_data

    output_path = tmp_path / "output.zarr"

    repair_fusion_breaks_lazy(
        skeleton_path=skeleton_path,
        segmentation_path=segmentation_path,
        scale_map_path=scale_map_path,
        output_path=output_path,
        repair_radius=10.0,
        chunk_shape=(30, 30, 30),
        backend="cupy",
    )

    result = np.array(zarr.open(str(output_path), mode="r")[:])

    expected = np.zeros((30, 30, 30), dtype=bool)
    expected[5:15, 12, 12] = True
    expected[14, 13, 12] = True  # Diagonal bridging voxel
    expected[15:24, 14, 12] = True

    np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="cupy not installed")
def test_repair_fusion_breaks_lazy_break_too_long_gpu(tmp_path):
    """A break exceeding repair_radius on a fusion boundary is not repaired (GPU).

    Both endpoints sit directly on the fusion boundary at z=14 and z=15.
    They are offset in y so the Euclidean distance between them is
    sqrt(1 + 36) ≈ 6.1, which exceeds repair_radius=5.
    """
    skeleton_data = np.zeros((30, 30, 30), dtype=bool)
    skeleton_data[7:15, 12, 12] = True  # Endpoint at (z=14, y=12), tile 1
    skeleton_data[15:22, 18, 12] = True  # Endpoint at (z=15, y=18), tile 2

    segmentation_data = np.zeros((30, 30, 30), dtype=bool)
    segmentation_data[5:23, 10:20, 10:14] = True

    scale_map_data = _make_two_tile_scale_map((30, 30, 30), boundary_z=15)

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

    scale_map_path = tmp_path / "scale_map.zarr"
    scale_map_zarr = zarr.open(
        str(scale_map_path),
        mode="w",
        shape=(30, 30, 30),
        chunks=(10, 10, 10),
        dtype=np.int32,
    )
    scale_map_zarr[:] = scale_map_data

    output_path = tmp_path / "output.zarr"

    repair_fusion_breaks_lazy(
        skeleton_path=skeleton_path,
        segmentation_path=segmentation_path,
        scale_map_path=scale_map_path,
        output_path=output_path,
        repair_radius=5.0,
        chunk_shape=(30, 30, 30),
        backend="cupy",
    )

    result = np.array(zarr.open(str(output_path), mode="r")[:])

    np.testing.assert_array_equal(result, skeleton_data)
