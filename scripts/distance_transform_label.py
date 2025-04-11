import dask.array as da  # noqa
import numpy as np
import pyclesperanto as cle
import h5py
from skeleplex.preprocessing.distance_transform import normalized_distance_transform


# prepare for skeleplex with dask and GPU

if __name__ == "__main__":
    img = np.zeros((100, 100, 100), dtype=np.uint8)
    img[30:70, 200:40, 200:70] = 1

    img = da.from_array(img, chunks=(20, 20, 20))

    # check your backends
    cle.list_available_backends()

    cle.select_backend("opencl")

    img_normalized = normalized_distance_transform(img, depth=10, min_ball_radius=6)

    # computed image
    img_normalized_np = img_normalized.compute()

    # store as input for nn
    with h5py.File("test.h5", "w") as f:
        f.create_dataset(
            "normalized_vector_background_image",
            data=img_normalized_np,
            compression="gzip",
        )
        f.create_dataset("labels", data=img, compression="gzip")
