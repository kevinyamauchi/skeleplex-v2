"""Example script to compute a distance field and run chunkwise skeletonization."""

import dask.array as da

from skeleplex.skeleton._skeletonize import (
    filter_skeleton_by_segmentation,
    skeletonize_chunkwise,
)
from skeleplex.skeleton.distance_field import local_normalized_distance

path_to_segmentation = "../example_data/bifurcating_tree.zarr"

segmentation = da.from_zarr(path_to_segmentation)

distance_field = segmentation.map_overlap(
    local_normalized_distance,
    depth=10,
    boundary=0,
)

distance_field.to_zarr("../example_data/distance_field.zarr", overwrite=True)

skeleton_prediction = skeletonize_chunkwise(
    distance_field,
    model="pretrained",
    chunk_size=(100, 100, 100),
    roi_size=(64, 64, 64),
    overlap=0.5,
    padding=(10, 10, 10),
)

skeleton_prediction = filter_skeleton_by_segmentation(
    skeleton=skeleton_prediction, segmentation=segmentation
)

skeleton_prediction.to_zarr("../example_data/skeleton_prediction.zarr", overwrite=True)
