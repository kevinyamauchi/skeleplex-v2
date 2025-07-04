"""
Tests for distance transform functions.
"""

import numpy as np
import dask.array as da
import pytest
from skeleplex.data.bifurcating_tree import generate_tree_3d, apply_dilation_3d
from skeleplex.skeleton.distance_field import local_normalized_distance, local_normalized_distance_gpu


def test_local_normalized_distance_tree():
    """Test distance transform on synthetic 3D tree."""
    
    # create synthetic 3D tree
    image = generate_tree_3d(
        shape=(50, 50, 50),
        nodes=2,
        branch_length=10,
        z_layer=25,
    )
    
    image_dilated = apply_dilation_3d(image, dilation_radius=2)
    
    normalized_distance = local_normalized_distance(image_dilated)
    
    # check shape and dtype
    assert normalized_distance.shape == image_dilated.shape
    assert normalized_distance.dtype == np.float32
    
    # background stays zero
    assert np.all(normalized_distance[image_dilated == 0] == 0)
    
    # foreground contains positive values
    assert np.all(normalized_distance[image_dilated > 0] > 0)
    
    # values should be ≤ 1
    assert np.all(normalized_distance[image_dilated > 0] <= 1.0)



# run this test only if CuPy is available
def _is_cupy_available():
    try:
        import cupy
        return True
    except ImportError:
        return False

@pytest.mark.skipif(
    not _is_cupy_available(),
    reason="CuPy is not installed"
)
def test_local_normalized_distance_tree_gpu():
    """Test distance transform on synthetic 3D tree using GPU with CuPy."""
    
    # create synthetic 3D tree
    image = generate_tree_3d(
        shape=(50, 50, 50),
        nodes=2,
        branch_length=10,
        z_layer=25,
    )
    
    image_dilated = apply_dilation_3d(image, dilation_radius=2)
    
    normalized_distance = local_normalized_distance_gpu(image_dilated)
    
    # check shape and dtype
    assert normalized_distance.shape == image_dilated.shape
    assert normalized_distance.dtype == np.float32
    
    # background stays zero
    assert np.all(normalized_distance[image_dilated == 0] == 0)
    
    # foreground contains positive values
    assert np.all(normalized_distance[image_dilated > 0] > 0)
    
     # values should be ≤ 1
    assert np.all(normalized_distance[image_dilated > 0] <= 1.0)




def test_local_normalized_distance_dask():
    """Test distance transform on synthetic 3D tree with Dask and map_overlap."""

    image = generate_tree_3d(
        shape=(50, 50, 50),
        nodes=2,
        branch_length=10,
        z_layer=25,
    )
    
    image_dilated = apply_dilation_3d(image, dilation_radius=2)
    
    dask_image = da.from_array(image_dilated, chunks=(20, 20, 20))
    
    normalized_distance = dask_image.map_overlap(
        local_normalized_distance,
        depth=10,
        boundary=0,
        dtype="float32",
    )
    
    normalized_distance_array = normalized_distance.compute()
    
    # check shape and dtype
    assert normalized_distance_array.shape == image_dilated.shape
    assert normalized_distance_array.dtype == np.float32
    
    # background stays zero
    assert np.all(normalized_distance_array[image_dilated == 0] == 0)
    
    # foreground contains positive values
    assert np.all(normalized_distance_array[image_dilated > 0] > 0)
    
    # values should be ≤ 1
    assert np.all(normalized_distance_array[image_dilated > 0] <= 1.0)
    
