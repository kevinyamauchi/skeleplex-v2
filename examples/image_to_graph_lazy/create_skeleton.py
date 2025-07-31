"""Example script to threshold and skeletonize a prediction volume."""

import dask.array as da
import numpy as np
from skimage.morphology import skeletonize

from skeleplex.skeleton._skeletonize import threshold_skeleton

path_to_skeleton = "examples/image_to_graph_lazy/skeleton_prediction.zarr"

skeleton_prediction = da.from_zarr(path_to_skeleton)

binary_skeleton = threshold_skeleton(skeleton_prediction, threshold=0.6)


factor = np.array([4, 4, 4])  # even numbers
desired_chunksize = np.array(binary_skeleton.shape) // factor
skeleton_dask = binary_skeleton.rechunk(desired_chunksize)

depth = 20

skeleton = binary_skeleton.map_overlap(
    skeletonize, depth=depth, boundary="none", dtype=np.uint8
)

skeleton.to_zarr("examples/image_to_graph_lazy/skeleton.zarr", overwrite=True)
