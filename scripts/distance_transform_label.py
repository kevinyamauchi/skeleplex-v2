import dask.array as da  # noqa
from dask.diagnostics import ProgressBar
import numpy as np
import pyclesperanto as cle
from skeleplex.preprocessing.distance_transform import normalized_distance_transform


# prepare for skeleplex with dask and GPU

if __name__ == "__main__":
    img = np.zeros((100, 100, 100), dtype=np.uint8)
    img[30:70, 20:40, 20:70] = 1

    img = da.from_array(img, chunks=(20, 20, 20))

    # check your backends
    cle.list_available_backends()

    cle.select_backend("opencl")

    img_normalized = normalized_distance_transform(img, depth=10, min_ball_radius=6)

    with ProgressBar():
        da.to_hdf5(
            "test.h5",
            {
                "normalized_vector_background_image": img_normalized,
                "labels": img,
            },
            compression="gzip",
        )
