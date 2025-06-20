from skeleplex.graph.sample import (    generate_2d_grid,
    sample_volume_at_coordinates_lazy,
    sample_volume_at_coordinates
)
import numpy as np
import dask.array as da

def test_sample_volume_at_coordinates():
    """Test sampling a volume at specific coordinates."""
    # Create a simple 3D volume
    volume = np.zeros((10, 10, 10), dtype=np.float32)
    volume[2:8, 2:8, 2:8] = 1  # a solid cube of 1s

    sampling_grid = generate_2d_grid((12,12),(1,1))
    sampling_grid += 5

    slice = sample_volume_at_coordinates(volume, sampling_grid,
                                            interpolation_order=1,
                                            fill_value = 5)
    assert slice[5,5] == 1
    assert slice[1,5] == 0
    assert slice[0,5] == 5


def test_sample_volume_at_coordinates_lazy():
    """Test lazy sampling a volume at specific coordinates."""
    # Create a simple 3D volume
    volume = np.zeros((10, 10, 10), dtype=np.float32)
    volume[2:8, 2:8, 2:8] = 1  # a solid cube of 1s
    volume = da.from_array(volume, chunks=(5, 5, 5))

    sampling_grid = generate_2d_grid((12,12),(1,1))
    sampling_grid += 5

    slice = sample_volume_at_coordinates_lazy(volume, sampling_grid,
                                            interpolation_order=1,
                                            fill_value = 5)
    assert slice[5,5] == 1
    assert slice[1,5] == 0
    assert slice[0,5] == 5 