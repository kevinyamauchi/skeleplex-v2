import dask.array as da  # noqa
import numpy as np
import pyclesperanto as cle

from skeleplex.preprocessing import normalized_distance_transform


# prepare for skeleplex with dask and GPU

img = np.zeros((100, 100, 100), dtype=np.uint8)
img[30:70, 200:40, 200:70] = 1

img = da.from_array(img, chunks=(20, 20, 20))

# check your backends
cle.list_available_backends()

cle.select_backend("opencl")

img_normalized = normalized_distance_transform(img, depth=10, min_ball_radius=6)

# computed image
img_normalized_np = img_normalized.compute()
